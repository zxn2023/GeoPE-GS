# GeoPE-GS adaptation of Luminance-GS. Modified to use the local gsplat package
# and to save generated images under the configured result directory.
import json
import math
import os
import time
from dataclasses import dataclass, field
from turtle import color
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

def save_debug_image(image_tensor, save_path):
    """保存调试图像
    
    Args:
        image_tensor: 图像张量，形状为 (H, W, C) 或 (H, W)
        save_path: 保存路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 处理不同形状的输入
    if image_tensor.dim() == 2:
        # 单通道图像，扩展为3通道
        image_np = image_tensor.numpy()
        image_np = np.stack([image_np, image_np, image_np], axis=-1)
    elif image_tensor.dim() == 3:
        image_np = image_tensor.numpy()
    else:
        print(f"Warning: Unsupported tensor shape {image_tensor.shape}")
        return
    
    # 确保值在 [0, 1] 范围内
    image_np = np.clip(image_np, 0, 1)
    
    # 转换为 uint8 并保存
    image_np = (image_np * 255).astype(np.uint8)
    imageio.imwrite(save_path, image_np)

def detect_exposure_condition(image):
    """检测曝光条件
    
    Args:
        image: 输入图像张量
    
    Returns:
        exposure_condition: 曝光条件 ('low', 'normal', 'over')
    """
    brightness = image.mean().item()
    if brightness < 0.3:
        return 'low'
    elif brightness > 0.7:
        return 'over'
    else:
        return 'normal'

def detect_dataset_type(data_dir: str) -> str:
    """自动检测数据集类型
    
    Args:
        data_dir: 数据目录路径
    
    Returns:
        dataset_type: 数据集类型 ('LOM' 或 'NeRF360')
    """
    if 'LOM' in data_dir or 'lom' in data_dir.lower():
        return 'LOM'
    elif 'NeRF_360' in data_dir or 'nerf360' in data_dir.lower() or '360' in data_dir:
        return 'NeRF360'
    else:
        return 'LOM'  # 默认认为是LOM
import tqdm
import tyro
import viser
import nerfview

from datasets.traj import generate_interpolated_path
import torchvision
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils import (
    AppearanceOptModule,
    CameraOptModule,
    CrossAttention,
    GeoEncoder,
    knn,
    normalized_quat_to_rotmat,
    rgb_to_sh,
    set_random_seed,
)

# Curve visualization tools for paper
#from curve_visualization import TrainingCurveVisualizer

import matplotlib.pyplot as plt
from gsplat.rendering_double import rasterization_dual

from tools import pixel_project, pixel_project_back, LUT_mapping
from losses import L_spa, HistogramPriorLoss, gamma_curve, s_curve, loss_structure, loss_noise, loss_multi_view


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = True
    # Path to the .pt file. If provide, it will skip training and render a video
    ckpt: Optional[str] = None

    # Path to the dataset
    data_dir: str = "../data/LOM_full/bike"
    # data_dir: str = "../data/NeRF_360/bicycle"

    exp_name: str = "low"   # Switch Conditions Here. overexposure: str = "over_exp"; varying exposure: str = "variance"
    
    # Downsample factor for the dataset
    data_factor: int = 1    # data_factor 8 for Mip360 dataset
    # Directory to save results
    result_dir: str = "results_low/ours/bike"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps, max steps 10000 for LOM dataset training
    max_steps: int = 10_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [5_000, 7_000, 10_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [5_000, 7_000, 10_000])

    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # GSs with opacity below this value will be pruned
    prune_opa: float = 0.005
    # GSs with image plane gradient above this value will be split/duplicated
    grow_grad2d: float = 0.0002
    # GSs with scale below this value will be duplicated. Above will be split
    grow_scale3d: float = 0.01
    # GSs with scale above this value will be pruned.
    prune_scale3d: float = 0.1

    # Start refining GSs after this iteration
    refine_start_iter: int = 500
    # Stop refining GSs after this iteration
    #refine_stop_iter: int = 15_000
    refine_stop_iter: int = 8_000
    # Reset opacities every this steps
    reset_every: int = 3000
    # Refine GSs every this steps
    refine_every: int = 100
    # Contrast Level
    constrast_level: float = 0.5

    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use absolute gradient for pruning. This typically requires larger --grow_grad2d, e.g., 0.0008 or 0.0006
    absgrad: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False
    
    # Enable curve analysis visualization (for paper)
    enable_curve_analysis: bool = True
    # Enable debug steps saving (geometric visualization for paper)
    enable_debug_steps: bool = True
    # Debug steps save interval
    debug_steps_interval: int = 2000
    
    # Dataset type for adaptive adjustment
    # Options: "LOM", "NeRF360", or "auto" (auto-detect from data_dir)
    dataset_type: str = "auto"
    
    # Enable NeRF360-specific tuning (default False for backward compatibility)
    enable_nerf360_tuning: bool = False
    
    # Enable checkpoint saving
    enable_checkpoint: bool = True

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        self.refine_start_iter = int(self.refine_start_iter * factor)
        self.refine_stop_iter = int(self.refine_stop_iter * factor)
        self.reset_every = int(self.reset_every * factor)
        self.refine_every = int(self.refine_every * factor)

cfg = tyro.cli(Config)

# Auto-detect dataset type if set to "auto"
if cfg.dataset_type == "auto":
    cfg.dataset_type = detect_dataset_type(cfg.data_dir)
print(f"Detected dataset type: {cfg.dataset_type}")

# Set dataset-specific hyperparameters (only when NeRF360 tuning is explicitly enabled)
if cfg.dataset_type == "NeRF360" and cfg.enable_nerf360_tuning:
    # NeRF-360 specific adjustments - 保持与旧版一致的设置
    cfg.ssim_lambda = 0.2  # 与旧版一致，保持训练稳定
else:
    # LOM-Full / NeRF360 without tuning: default settings
    pass

if cfg.exp_name in ["low", "over_exp"]:
    from datasets.colmap import Dataset, Parser
else:
    from datasets.colmap_mip360 import Dataset, Parser


class WarmupCosineAnnealingLR:
    """带热身的余弦退火学习率调度器"""
    def __init__(self, optimizer, warmup_steps, max_steps, eta_min=1e-8):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eta_min = eta_min
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.last_step = 0

    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step
        
        if step < self.warmup_steps:
            # Warmup: 线性增长
            lr_factor = (step + 1) / self.warmup_steps
        else:
            # CosineAnnealing: 余弦衰减
            cosine_steps = step - self.warmup_steps
            max_cosine_steps = self.max_steps - self.warmup_steps
            # 防止除零错误：当 max_steps == warmup_steps 时
            if max_cosine_steps == 0:
                lr_factor = self.eta_min
            else:
                lr_factor = self.eta_min + 0.5 * (1 - self.eta_min) * (1 + math.cos(math.pi * cosine_steps / max_cosine_steps))
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_factor
    
    def state_dict(self):
        return {
            'last_step': self.last_step,
            'warmup_steps': self.warmup_steps,
            'max_steps': self.max_steps,
            'eta_min': self.eta_min,
            'base_lrs': self.base_lrs,
        }
    
    def load_state_dict(self, state_dict):
        self.last_step = state_dict['last_step']
        self.warmup_steps = state_dict['warmup_steps']
        self.max_steps = state_dict['max_steps']
        self.eta_min = state_dict['eta_min']
        self.base_lrs = state_dict['base_lrs']


def create_splats_with_optimizers(
    points: Tensor,  # [N, 3]
    rgbs: Tensor,  # [N, 3]
    frame_nums: int, # Training Frame Number
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    init_opacity: float = 0.1,
    sparse_grad: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
) -> Tuple[torch.nn.ParameterDict, torch.optim.Optimizer]:
    N = points.shape[0]

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]   point cloud position
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]
    params = [
        # name, value, lr
        ("means3d", torch.nn.Parameter(points), 1e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]

    if feature_dim is None:    # Color is Here     
        # color is SH coefficients.     
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, 4**2, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)   # rgb to sh
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
        
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    # Eq.3 in our paper, a least-squares formula
    adjust_k = torch.nn.Parameter(torch.ones_like(colors[:, :1, :]), requires_grad=True)    # enhance, for multiply
    adjust_b = torch.nn.Parameter(torch.zeros_like(colors[:, :1, :]), requires_grad=True)   # bias, for add

    params.append(("adjust_k", adjust_k, 2.5e-3))
    params.append(("adjust_b", adjust_b, 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)    # parameter dict

    optimizers = [
        (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
            [{"params": splats[name], "lr": lr * math.sqrt(batch_size), "name": name}],
            eps=1e-15 / math.sqrt(batch_size),
            betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
        )
        for name, _, lr in params
    ]
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(self, cfg: Config) -> None:
        set_random_seed(42)

        self.cfg = cfg
        self.device = "cuda"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.render_dir_depth = f"{cfg.result_dir}/renders_depth"
        os.makedirs(self.render_dir_depth, exist_ok=True)
        # Directory for best model renders
        self.render_dir_best = f"{cfg.result_dir}/renders_best"
        os.makedirs(self.render_dir_best, exist_ok=True)
        self.render_dir_best_depth = f"{cfg.result_dir}/renders_best_depth"
        os.makedirs(self.render_dir_best_depth, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")
        
        # Best model tracking
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        self.best_step = 0

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            exp_name = cfg.exp_name,
            factor=cfg.data_factor, # down scale ratio
            normalize=True,
            test_every=cfg.test_every,
        )
        self.trainset = Dataset(    # Training Set
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        self.valset = Dataset(self.parser, split="val") # Validation Set
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        
        # Model
        feature_dim = 32 if cfg.app_opt else None
        # return GS-parameters & optimizers
        self.splats, self.optimizers = create_splats_with_optimizers(   # basic gaussian splatting
            torch.from_numpy(self.parser.points).float(),
            torch.from_numpy(self.parser.points_rgb / 255.0).float(),
            frame_nums = len(self.trainset),
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            init_opacity=cfg.init_opa,
            sparse_grad=cfg.sparse_grad,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
        )
        print("Model initialized. Number of GS:", len(self.splats["means3d"]))
        
        # 根据数据集类型设置对比度级别
        if cfg.dataset_type == "NeRF360" and cfg.enable_nerf360_tuning:
            # NeRF360特调时使用与旧版一致的对比度级别，保持训练稳定
            self.constrast_level = cfg.constrast_level  # 使用配置值(默认0.5)，与旧版一致
        else:
            self.constrast_level = cfg.constrast_level
        
        curve = torch.linspace(0, 1, 255).unsqueeze(0).cuda()   # Luminance Curve
        self.curve = torch.nn.Parameter(curve)
        
        # 根据数据集类型设置曲线学习率
        if cfg.dataset_type == "NeRF360" and cfg.enable_nerf360_tuning:
            # NeRF360特调时使用较低的曲线学习率，提高训练稳定性
            curve_lr = 5e-4 * math.sqrt(cfg.batch_size)
        else:
            curve_lr = 1e-3 * math.sqrt(cfg.batch_size)
        
        self.curve_optimizers = [
                torch.optim.Adam(
                    [self.curve],
                    # [self.curve, self.curve_2, self.curve_3],
                    lr=curve_lr,
                    weight_decay=1e-4,
                )
            ]
        
        
        # 调整CrossAttention的输入通道数，以适应几何信息的输入
        # 3个颜色通道 + 1个深度通道 + 3个法线通道 = 7个通道
        self.curve_adjust = CrossAttention(img_channels=7, out_dim=255).to(self.device)    # Output: curve bias parameters, L_k_b
        self.curve_adjust_gamma = CrossAttention(img_channels=7, out_dim=3).to(self.device)    # Output: curve shape control parameters, Eq.9
        
        # 轻量几何编码器：将深度和法线编码成更丰富的特征
        self.geo_encoder = GeoEncoder(in_channels=4, out_channels=4).to(self.device)    # 4通道：深度(1) + 法线(3)
        
        # 根据数据集类型设置学习率
        # NeRF360特调时使用较低学习率（暗场景增强容易不稳定，需要更慢更稳定的训练）
        if cfg.dataset_type == "NeRF360" and cfg.enable_nerf360_tuning:
            adjust_lr = 5e-6 * math.sqrt(cfg.batch_size)  # 降低学习率，提高训练稳定性
        else:
            adjust_lr = 1e-5 * math.sqrt(cfg.batch_size)
        
        self.adjust_optimizers = [
            torch.optim.Adam(
                list(self.curve_adjust.parameters()) + list(self.curve_adjust_gamma.parameters()) + list(self.geo_encoder.parameters()),
                lr=adjust_lr,
                weight_decay=1e-5,
            )
        ]
        
        self.pesdo_curve = torch.nn.Parameter(torch.linspace(0, 1, 255).unsqueeze(0).cuda(), requires_grad=False)

        self.axis1_para = [torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0]).cuda()) for _ in range(len(self.trainset))]
        self.axis2_para = [torch.nn.Parameter(torch.tensor([0.0, 0.0]).cuda()) for _ in range(len(self.trainset))]

        self.sat_optimizers = [
                torch.optim.Adam(
                    self.axis1_para + self.axis2_para,
                    lr=2e-4 * math.sqrt(cfg.batch_size),
                    weight_decay=1e-4,
                )
            ]
        
        # Curve visualizer for paper (controlled by flag)
        #self.curve_visualizer = None
        #self.curve_visualizer = None
        #if cfg.enable_curve_analysis:
            #self.curve_visualizer = TrainingCurveVisualizer(
                #save_dir=os.path.join(cfg.result_dir, 'curve_analysis')
            #)
        
        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)

        self.app_optimizers = []
        if cfg.app_opt:
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
            self.device
        )
        
        # 初始化深度可分离卷积层用于高斯模糊
        self.blur_conv = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3, bias=False).to(self.device)
        # 初始化卷积核为高斯模糊核
        kernel = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], device=self.device)
        kernel = kernel / kernel.sum()
        self.blur_conv.weight.data = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )

        # Running stats for prunning & growing.
        n_gauss = len(self.splats["means3d"])
        self.running_stats = {
            "grad2d": torch.zeros(n_gauss, device=self.device),  # norm of the gradient
            "count": torch.zeros(n_gauss, device=self.device, dtype=torch.int),
        }

    def save_checkpoint(self, step, is_best=False):
        """Save complete checkpoint including model parameters, optimizer states, scheduler states, etc."""
        # Skip checkpoint saving if disabled
        if not self.cfg.enable_checkpoint:
            return
        
        ckpt = {
            # Metadata
            "step": step,
            "cfg": vars(self.cfg),
            
            # Model parameters (essential for inference)
            "splats": self.splats.state_dict(),
            "curve": self.curve.data,
            "curve_adjust": self.curve_adjust.state_dict(),
            "curve_adjust_gamma": self.curve_adjust_gamma.state_dict(),
            "geo_encoder": self.geo_encoder.state_dict(),
            "axis1_para": [p.data for p in self.axis1_para],
            "axis2_para": [p.data for p in self.axis2_para],
            "pesdo_curve": self.pesdo_curve.data,
            
            # Conditional modules
            "pose_adjust": self.pose_adjust.state_dict() if self.cfg.pose_opt else None,
            "app_module": self.app_module.state_dict() if self.cfg.app_opt else None,
            
            # Optimizer states (only save for final checkpoint to save space)
            "optimizers": [opt.state_dict() for opt in self.optimizers] if (self.optimizers and step == self.cfg.max_steps - 1) else None,
            "curve_optimizers": [opt.state_dict() for opt in self.curve_optimizers] if (self.curve_optimizers and step == self.cfg.max_steps - 1) else None,
            "adjust_optimizers": [opt.state_dict() for opt in self.adjust_optimizers] if (self.adjust_optimizers and step == self.cfg.max_steps - 1) else None,
            "sat_optimizers": [opt.state_dict() for opt in self.sat_optimizers] if (self.sat_optimizers and step == self.cfg.max_steps - 1) else None,
            "pose_optimizers": [opt.state_dict() for opt in self.pose_optimizers] if (self.pose_optimizers and step == self.cfg.max_steps - 1) else None,
            "app_optimizers": [opt.state_dict() for opt in self.app_optimizers] if (self.app_optimizers and step == self.cfg.max_steps - 1) else None,
            
            # Scheduler states (only save for final checkpoint)
            "schedulers": [sch.state_dict() for sch in self.schedulers] if (self.schedulers and step == self.cfg.max_steps - 1) else None,
            
            # Running stats for pruning/growing (only save for final checkpoint)
            "running_stats": {
                "grad2d": self.running_stats["grad2d"].cpu(),
                "count": self.running_stats["count"].cpu(),
            } if step == self.cfg.max_steps - 1 else None,
            
            # Performance metrics (only for best model)
            "psnr": self.best_psnr if is_best else None,
            "ssim": self.best_ssim if is_best else None,
            
            # Random seed states for reproducibility (only save for final checkpoint)
            "random_state": torch.random.get_rng_state().cpu() if step == self.cfg.max_steps - 1 else None,
            "cuda_random_state": torch.cuda.get_rng_state().cpu() if (torch.cuda.is_available() and step == self.cfg.max_steps - 1) else None,
        }
        
        filename = f"best.ckpt" if is_best else f"ckpt_{step}.pt"
        # Use legacy serialization for faster save/load
        torch.save(ckpt, f"{self.ckpt_dir}/{filename}", _use_new_zipfile_serialization=False)
        print(f"Checkpoint saved to {self.ckpt_dir}/{filename}")

    def load_checkpoint(self, ckpt_path):
        """Load checkpoint and restore complete training state."""
        ckpt = torch.load(ckpt_path, map_location=self.device)
        
        ckpt_gs_count = len(ckpt["splats"]["means3d"])
        current_gs_count = len(self.splats["means3d"])
        
        if ckpt_gs_count != current_gs_count:
            print(f"Warning: Checkpoint has {ckpt_gs_count} Gaussians, but current model has {current_gs_count}. Recreating model...")
            
            # 释放旧模型占用的显存
            if hasattr(self, 'splats'):
                del self.splats
            if hasattr(self, 'optimizers'):
                del self.optimizers
            torch.cuda.empty_cache()
            
            feature_dim = 32 if self.cfg.app_opt else None
            self.splats, self.optimizers = create_splats_with_optimizers(
                ckpt["splats"]["means3d"].data.cpu(),
                torch.zeros((ckpt_gs_count, 3)).fill_(0.5),
                frame_nums=len(self.trainset),
                scene_scale=self.scene_scale,
                sh_degree=self.cfg.sh_degree,
                init_opacity=self.cfg.init_opa,
                sparse_grad=self.cfg.sparse_grad,
                batch_size=self.cfg.batch_size,
                feature_dim=feature_dim,
                device=self.device,
            )
        
        self.splats.load_state_dict(ckpt["splats"])
        self.curve.data = ckpt["curve"]
        self.curve_adjust.load_state_dict(ckpt["curve_adjust"])
        self.curve_adjust_gamma.load_state_dict(ckpt["curve_adjust_gamma"])
        self.geo_encoder.load_state_dict(ckpt["geo_encoder"])
        for i, p in enumerate(self.axis1_para):
            p.data = ckpt["axis1_para"][i]
        for i, p in enumerate(self.axis2_para):
            p.data = ckpt["axis2_para"][i]
        self.pesdo_curve.data = ckpt["pesdo_curve"]
        
        # Load conditional modules
        if self.cfg.pose_opt and ckpt["pose_adjust"] is not None:
            self.pose_adjust.load_state_dict(ckpt["pose_adjust"])
        if self.cfg.app_opt and ckpt["app_module"] is not None:
            self.app_module.load_state_dict(ckpt["app_module"])
        
        # Load optimizer states (skip if None or empty)
        if ckpt["optimizers"] is not None and self.optimizers:
            for opt, opt_state in zip(self.optimizers, ckpt["optimizers"]):
                opt.load_state_dict(opt_state)
        if ckpt["curve_optimizers"] is not None and self.curve_optimizers:
            for opt, opt_state in zip(self.curve_optimizers, ckpt["curve_optimizers"]):
                opt.load_state_dict(opt_state)
        if ckpt["adjust_optimizers"] is not None and self.adjust_optimizers:
            for opt, opt_state in zip(self.adjust_optimizers, ckpt["adjust_optimizers"]):
                opt.load_state_dict(opt_state)
        if ckpt["sat_optimizers"] is not None and self.sat_optimizers:
            for opt, opt_state in zip(self.sat_optimizers, ckpt["sat_optimizers"]):
                opt.load_state_dict(opt_state)
        if ckpt["pose_optimizers"] is not None and self.pose_optimizers:
            for opt, opt_state in zip(self.pose_optimizers, ckpt["pose_optimizers"]):
                opt.load_state_dict(opt_state)
        if ckpt["app_optimizers"] is not None and self.app_optimizers:
            for opt, opt_state in zip(self.app_optimizers, ckpt["app_optimizers"]):
                opt.load_state_dict(opt_state)
        
        # Initialize schedulers (same as in train method)
        max_steps = self.cfg.max_steps
        warmup_steps = 500
        
        self.schedulers = []
        self.schedulers.append(WarmupCosineAnnealingLR(self.optimizers[0], warmup_steps, max_steps))
        self.schedulers.append(WarmupCosineAnnealingLR(self.curve_optimizers[0], warmup_steps, max_steps))
        self.schedulers.append(WarmupCosineAnnealingLR(self.adjust_optimizers[0], warmup_steps, max_steps))
        self.schedulers.append(WarmupCosineAnnealingLR(self.sat_optimizers[0], warmup_steps, max_steps))
        if self.cfg.pose_opt:
            self.schedulers.append(WarmupCosineAnnealingLR(self.pose_optimizers[0], warmup_steps, max_steps))
        
        # Load scheduler states (skip if None or empty)
        if ckpt["schedulers"] is not None and self.schedulers:
            for sch, sch_state in zip(self.schedulers, ckpt["schedulers"]):
                sch.load_state_dict(sch_state)
        
        # Load running stats (handle None for backward compatibility)
        if ckpt["running_stats"] is not None:
            if "grad2d" in ckpt["running_stats"]:
                self.running_stats["grad2d"] = ckpt["running_stats"]["grad2d"].to(self.device)
            if "count" in ckpt["running_stats"]:
                self.running_stats["count"] = ckpt["running_stats"]["count"].to(self.device)
        
        # Restore random seed states (must be on CPU, handle None)
        if ckpt["random_state"] is not None:
            torch.random.set_rng_state(ckpt["random_state"].cpu())
        if ckpt["cuda_random_state"] is not None:
            torch.cuda.set_rng_state(ckpt["cuda_random_state"].cpu())
        
        print(f"Checkpoint loaded from {ckpt_path}, step: {ckpt['step']}")
        return ckpt["step"]

    def compute_normals(self, info):
        """从深度图计算法线图（简化版本）"""
        # 首先检查info中是否有图像尺寸信息
        height = info.get('height', 1)
        width = info.get('width', 1)
        
        # 尝试从info中获取深度图
        depths = info.get('depths', None)
        
        # 如果没有深度图，返回默认法线
        if depths is None:
            return torch.ones(1, height, width, 3, device=self.device)
        
        try:
            # 从深度图计算法线
            # 使用Sobel算子计算深度图的梯度
            depth_map = depths.squeeze(-1)  # (B, H, W)
            
            # 计算x和y方向的梯度
            # 使用卷积来计算梯度
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                   dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                   dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
            
            # 计算梯度
            grad_x = F.conv2d(depth_map.unsqueeze(1), sobel_x, padding=1).squeeze(1)  # (B, H, W)
            grad_y = F.conv2d(depth_map.unsqueeze(1), sobel_y, padding=1).squeeze(1)  # (B, H, W)
            
            # 计算法线
            # 法线 = (-grad_x, -grad_y, 1) / ||(-grad_x, -grad_y, 1)||
            normal_z = torch.ones_like(grad_x)
            normal_x = -grad_x
            normal_y = -grad_y
            
            # 归一化
            normal_length = torch.sqrt(normal_x**2 + normal_y**2 + normal_z**2 + 1e-8)
            normal_x = normal_x / normal_length
            normal_y = normal_y / normal_length
            normal_z = normal_z / normal_length
            
            # 组合法线图
            normal_map = torch.stack([normal_x, normal_y, normal_z], dim=-1)  # (B, H, W, 3)
            
            return normal_map
        except Exception as e:
            # 如果出现任何错误，返回默认法线
            return torch.ones(1, height, width, 3, device=self.device)

    def generate_struct_mask(self, normals, depths):
        """生成连续结构置信度图（替代二值掩码）
        
        Returns:
            struct_confidence: 连续置信度图（用于增强调制和损失计算）
        """
        # 基于法线梯度计算结构掩码
        if normals is None:
            if depths is not None:
                return torch.zeros_like(depths)
            else:
                # 如果depths也是None，返回一个默认形状的掩码
                return torch.zeros(1, 1, 1, device=self.device)
        
        # 确保normals和depths的形状匹配
        if depths is not None:
            if normals.shape[:3] != depths.shape[:3]:
                # 如果形状不匹配，调整normals的形状
                # 假设depths的形状是 (batch, height, width) 或 (batch, height, width, channel)
                target_shape = depths.shape[:3]
                normals = F.interpolate(normals.permute(0, 3, 1, 2), size=target_shape[1:], mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        
        # 计算法线梯度
        normal_gradient_x = normals[:, 1:, :, :] - normals[:, :-1, :, :]
        normal_gradient_y = normals[:, :, 1:, :] - normals[:, :, :-1, :]
        
        # 计算梯度幅度
        gradient_magnitude_x = torch.norm(normal_gradient_x, dim=-1)
        gradient_magnitude_y = torch.norm(normal_gradient_y, dim=-1)
        
        # 填充边界，确保尺寸匹配
        gradient_magnitude_x = F.pad(gradient_magnitude_x, (0, 0, 0, 1))  # 填充高度
        gradient_magnitude_y = F.pad(gradient_magnitude_y, (0, 1, 0, 0))  # 填充宽度
        
        # 确保梯度幅度的形状与输入匹配
        assert gradient_magnitude_x.shape[:3] == normals.shape[:3], f"Shape mismatch: {gradient_magnitude_x.shape}, {normals.shape}"
        assert gradient_magnitude_y.shape[:3] == normals.shape[:3], f"Shape mismatch: {gradient_magnitude_y.shape}, {normals.shape}"
        
        # 合并梯度
        gradient_magnitude = torch.max(gradient_magnitude_x, gradient_magnitude_y)
        
        # 确保gradient_magnitude的形状与depths匹配
        if depths is not None:
            if gradient_magnitude.shape != depths.shape:
                # 如果depths有通道维度，扩展gradient_magnitude
                if depths.dim() == 4 and gradient_magnitude.dim() == 3:
                    gradient_magnitude = gradient_magnitude.unsqueeze(-1)
                # 如果尺寸不匹配，调整大小
                elif gradient_magnitude.shape[:3] != depths.shape[:3]:
                    target_shape = depths.shape[1:3]
                    gradient_magnitude = F.interpolate(gradient_magnitude.unsqueeze(1), size=target_shape, mode='bilinear', align_corners=False).squeeze(1)
                    if depths.dim() == 4:
                        gradient_magnitude = gradient_magnitude.unsqueeze(-1)
        else:
            # 如果depths为None，确保gradient_magnitude是3D或4D张量
            if gradient_magnitude.dim() == 3:
                gradient_magnitude = gradient_magnitude.unsqueeze(-1)
        
        # 生成连续置信度图（替代二值掩码）
        # 使用原始参数，与v14版本保持一致
        struct_confidence = torch.sigmoid((gradient_magnitude - 0.03) * 50)
        
        # 应用形态学操作，去除小的噪声区域
        from torch.nn.functional import max_pool2d, avg_pool2d
        
        # 膨胀操作，增强结构区域
        struct_confidence = max_pool2d(struct_confidence.permute(0, 3, 1, 2), kernel_size=3, padding=1).permute(0, 2, 3, 1)
        # 腐蚀操作，去除小的噪声区域
        struct_confidence = avg_pool2d(struct_confidence.permute(0, 3, 1, 2), kernel_size=3, padding=1).permute(0, 2, 3, 1)
        
        return struct_confidence

    def generate_dark_mask(self, pixels):
        """生成暗区掩码"""
        # 基于像素亮度计算暗区掩码
        brightness = torch.mean(pixels, dim=-1, keepdim=True)
        dark_mask = (brightness < 0.3).float()
        return dark_mask

    def dual_branch_enhancer(self, init_enhanced, original, struct_confidence, flat_mask, dark_mask):
        """双分支自适应增强模块：修正连续置信度下的融合逻辑

        Args:
            struct_confidence: 连续置信度图（用于增强调制和损失计算）
        """
        if struct_confidence.shape[1:3] != init_enhanced.shape[1:3]:
            struct_confidence = F.interpolate(struct_confidence.permute(0, 3, 1, 2), size=init_enhanced.shape[1:3], mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            flat_mask = F.interpolate(flat_mask.permute(0, 3, 1, 2), size=init_enhanced.shape[1:3], mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            dark_mask = F.interpolate(dark_mask.permute(0, 3, 1, 2), size=init_enhanced.shape[1:3], mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

        struct_confidence = struct_confidence.expand_as(init_enhanced)
        flat_mask = flat_mask.expand_as(init_enhanced)
        dark_mask = dark_mask.expand_as(init_enhanced)

        # 结构区域增强：保留原始增强结果
        struct_branch = init_enhanced

        # 平坦区域增强：根据亮度分别处理
        flat_dark_branch = self.blur_conv(init_enhanced.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        flat_bright_branch = init_enhanced

        # 融合平坦区域
        flat_branch = flat_dark_branch * dark_mask + flat_bright_branch * (1 - dark_mask)

        # 关键修复：使用加权平均融合两个分支，确保每个像素的总权重始终为1
        enhanced = struct_branch * struct_confidence + flat_branch * (1 - struct_confidence)

        return enhanced

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        # Learnable Parameters:
        means = self.splats["means3d"]  # [N, 3]
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,], sigmoid function

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:    
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
            
        else:  # Here 
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]
        
        adjust_k = self.splats["adjust_k"]  # 1090, 1, 3
        adjust_b = self.splats["adjust_b"]  # 1090, 1, 3
        
        colors_low = colors * adjust_k + adjust_b  # least squares: x_enh=a*x+b
        
        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        
        render_colors_enh, render_colors_low, render_enh_alphas, render_low_alphas, info = rasterization_dual(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,    
            colors=colors,
            colors_low=colors_low,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=self.cfg.absgrad,
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            **kwargs,)
        
        # 添加深度信息到info中
        render_mode = kwargs.get('render_mode', 'RGB')
        if render_mode in ['RGB+D', 'RGB+ED'] and render_colors_low.shape[-1] == 4:
            # 从渲染结果中提取深度信息
            depths = render_colors_low[..., 3:4]
            # 归一化深度信息到[0, 1]范围
            depths_min = depths.min()
            depths_max = depths.max()
            if depths_max - depths_min > 1e-6:
                depths = (depths - depths_min) / (depths_max - depths_min)
            info['depths'] = depths
        elif render_low_alphas is not None:
            # 使用不透明度作为深度的近似
            info['depths'] = render_low_alphas
        elif 'means2d' in info:
            # 如果没有深度信息，创建一个默认的深度图
            height = info.get('height', 1)
            width = info.get('width', 1)
            info['depths'] = torch.ones(1, height, width, 1, device=self.device)
        
        # 添加相机姿态信息到info中，用于法线计算
        info['camtoworld'] = camtoworlds
        # 添加图像尺寸信息
        info['height'] = height
        info['width'] = width
        
        # 计算并添加法线信息
        with torch.no_grad():
            normals = self.compute_normals(info)
            info['normals'] = normals
        

        # 暂时保留，后续再处理
        # ========== 修复CUDA对齐导致的黑边问题 ==========
        # CUDA核函数为了并行计算效率，会将图像宽高向上对齐到16的倍数
        # 需要将对齐后渲染出的结果截断回原始图像尺寸，移除对齐补出来的黑边
        render_colors_enh = render_colors_enh[:, :height, :width, :]
        render_colors_low = render_colors_low[:, :height, :width, :]
        if render_enh_alphas is not None:
            render_enh_alphas = render_enh_alphas[:, :height, :width]
        if render_low_alphas is not None:
            render_low_alphas = render_low_alphas[:, :height, :width]
        # =============================================
        
        return render_colors_enh, render_colors_low, render_enh_alphas, render_low_alphas, info   # return colors and alphas

    def train(self):
        cfg = self.cfg
        device = self.device

        loss_contrast = L_spa()     # spatial consistancy loss
        loss_histo = HistogramPriorLoss()   # curve control loss


        # Dump cfg.
        with open(f"{cfg.result_dir}/cfg.json", "w") as f:
            json.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0
        warmup_steps = 500  # Warmup 500步

        # 初始化带热身的余弦退火学习率调度器
        self.schedulers = []
        # 对所有优化器应用 WarmupCosineAnnealingLR
        self.schedulers.append(WarmupCosineAnnealingLR(self.optimizers[0], warmup_steps, max_steps))
        self.schedulers.append(WarmupCosineAnnealingLR(self.curve_optimizers[0], warmup_steps, max_steps))
        self.schedulers.append(WarmupCosineAnnealingLR(self.adjust_optimizers[0], warmup_steps, max_steps))
        self.schedulers.append(WarmupCosineAnnealingLR(self.sat_optimizers[0], warmup_steps, max_steps))
        if cfg.pose_opt:
            self.schedulers.append(WarmupCosineAnnealingLR(self.pose_optimizers[0], warmup_steps, max_steps))

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            renders_enh, renders_low, alphas_enh, alphas_low, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+D",
            )
            if renders_low.shape[-1] == 4:
                colors_low, depths_low = renders_low[..., 0:3], renders_low[..., 3:4]
                colors_enh, depths_enh = renders_enh[..., 0:3], renders_enh[..., 3:4]
            else:
                colors_low, depths_low = renders_low, None
                colors_enh, depths_enh = renders_enh, None

            # Clamp colors to [0, 1] range
            colors_low = torch.clamp(colors_low, 0.0, 1.0)
            colors_enh = torch.clamp(colors_enh, 0.0, 1.0)

            if cfg.random_bkgd: # False
                bkgd = torch.rand(1, 3, device=device)
                colors_low = colors_low + bkgd * (1.0 - alphas_low)
                colors_enh = colors_enh + bkgd * (1.0 - alphas_enh)
                # Clamp again after background addition
                colors_low = torch.clamp(colors_low, 0.0, 1.0)
                colors_enh = torch.clamp(colors_enh, 0.0, 1.0)

            info["means2d"].retain_grad()  # used for running stats
            
            # 几何先验提取 - 现在从info中直接获取，因为在rasterize_splats中已经计算
            normals = info.get('normals', None)
            
            # 区域掩码生成 - 不需要梯度
            with torch.no_grad():
                struct_confidence = self.generate_struct_mask(normals, depths_low)
                # 使用连续值的平坦区域掩码（与v14版本一致）
                flat_mask = 1.0 - struct_confidence
                dark_mask = self.generate_dark_mask(pixels)
            
            # 提取几何先验信息
            depths = info.get('depths', None)
            
            # 将几何先验信息添加到曲线调整中
            # 首先将深度图和法线图调整为合适的形状
            pixels_permuted = pixels.permute(0,3,1,2)  # [B, 3, H, W]
            
            # 添加深度信息
            geo_input = None
            if depths is not None:
                # 清理深度图中的NaN和inf
                depths = torch.nan_to_num(depths, nan=0.0, posinf=0.0, neginf=0.0)
                # 调整深度图的形状以匹配网络输入
                depths_reshaped = depths.permute(0, 3, 1, 2)  # 从 (B, H, W, 1) 变为 (B, 1, H, W)
                geo_input = depths_reshaped
            
            # 添加法线信息
            if normals is not None:
                # 清理法线图中的NaN和inf
                normals = torch.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
                # 调整法线图的形状以匹配网络输入
                normals_reshaped = normals.permute(0, 3, 1, 2)  # 从 (B, H, W, 3) 变为 (B, 3, H, W)
                if geo_input is not None:
                    geo_input = torch.cat([geo_input, normals_reshaped], dim=1)  # [B, 4, H, W]
                else:
                    geo_input = normals_reshaped
            
            # 使用几何编码器将深度和法线编码成更丰富的特征
            if geo_input is not None:
                geo_feat = self.geo_encoder(geo_input)  # [B, 4, H, W]
                # 将编码后的几何特征与RGB拼接
                input_with_geo = torch.cat([pixels_permuted, geo_feat], dim=1)  # [B, 7, H, W]
            else:
                input_with_geo = pixels_permuted
            
            # 将camtoworlds转换为合适的embedding向量
            # 提取相机位置（平移部分）并归一化
            cam_pos = camtoworlds[:, :3, 3]  # (batch_size, 3)
            # 清理相机位置中的NaN和inf
            cam_pos = torch.nan_to_num(cam_pos, nan=0.0, posinf=0.0, neginf=0.0)
            # 对相机位置进行归一化
            cam_pos_norm = torch.norm(cam_pos, dim=-1, keepdim=True)
            cam_pos_normalized = cam_pos / (cam_pos_norm + 1e-8)
            # 扩展到16维
            cam_embedding = torch.zeros(camtoworlds.shape[0], 16, device=camtoworlds.device)
            cam_embedding[:, :3] = cam_pos_normalized
            # 添加旋转信息（可选）
            # cam_embedding[:, 3:12] = camtoworlds[:, :3, :3].reshape(-1, 9)
            
            # ========== 数据集自适应：仅NeRF360特调时添加数值稳定性保护 ==========
            if cfg.dataset_type == "NeRF360" and cfg.enable_nerf360_tuning:
                # 对输入进行裁剪，防止极端值导致NaN
                _input_geo = torch.clamp(input_with_geo, -10.0, 10.0)
                _cam_emb = torch.clamp(cam_embedding, -10.0, 10.0)
            else:
                _input_geo = input_with_geo
                _cam_emb = cam_embedding
            # =============================================
            
            curve_adj_bias = self.curve_adjust(_input_geo, _cam_emb) # encode low-light GT, camera position and geometry to get adjust curve
            
            gamma_alpha_beta = self.curve_adjust_gamma(_input_geo, _cam_emb)

            # ========== NaN/Inf 安全检查（所有数据集通用） ==========
            if torch.isnan(curve_adj_bias).any() or torch.isinf(curve_adj_bias).any():
                print("Warning: curve_adj_bias contains NaN or inf!")
                curve_adj_bias = torch.zeros_like(curve_adj_bias)
            
            if torch.isnan(gamma_alpha_beta).any() or torch.isinf(gamma_alpha_beta).any():
                print("Warning: gamma_alpha_beta contains NaN or inf!")
                gamma_alpha_beta = torch.zeros_like(gamma_alpha_beta)
            # =============================================
            
            # ========== NeRF360特调专用输出约束 ==========
            if cfg.dataset_type == "NeRF360" and cfg.enable_nerf360_tuning:
                # 对curve_adj_bias进行裁剪，防止极端调整
                curve_adj_bias = torch.clamp(curve_adj_bias, -0.5, 0.5)
                
                # 对gamma_alpha_beta进行约束（使用非原地操作，保护计算图）
                # gamma: [0.1, 2.0], alpha/beta: [-1.0, 1.0]
                gamma_alpha_beta_clamped = torch.clamp(gamma_alpha_beta, -1.0, 2.0)
                gamma_alpha_beta = gamma_alpha_beta_clamped.clone()
                gamma_alpha_beta[:, 0] = torch.clamp(gamma_alpha_beta_clamped[:, 0], 0.1, 2.0)  # gamma
                gamma_alpha_beta[:, 1:] = torch.clamp(gamma_alpha_beta_clamped[:, 1:], -1.0, 1.0)  # alpha, beta
            # =============================================

            curve_adj = torch.clamp(self.curve + curve_adj_bias, 0, 1)    # Clamp the curve in range of (0, 1)
            
            # ========== 数据集自适应曲线调整（仅 NeRF360 特调） ==========
            if cfg.dataset_type == "NeRF360" and cfg.enable_nerf360_tuning:
                # 计算当前输入图像的平均亮度（0-1范围）
                input_brightness = pixels.mean().item()
                
                # 低光照数据集通常需要更强的增强
                # 目标亮度：低光照场景要提亮到 0.25 左右（约4倍增强，与旧版bonsai_test相当）
                target_brightness = 0.25
                required_enhancement = target_brightness / max(input_brightness, 0.01)
                
                # 根据输入亮度动态调整约束强度
                # 输入越暗（需要增强越多），约束越松，让曲线有更大调整空间
                if input_brightness < 0.06:
                    # 非常暗的场景（需要强增强），用弱约束
                    curve_weight = 0.1
                    adj_weight = 0.9
                    clamp_min = 0.01
                    clamp_max = 0.99
                    if step % 500 == 0:
                        print(f"Step {step}: NeRF360 tuning enabled, enhancement={required_enhancement:.2f}x, using weak constraint")
                elif input_brightness > 0.12:
                    # 较亮的场景（需要弱增强），用强约束
                    curve_weight = 0.8
                    adj_weight = 0.2
                    clamp_min = 0.05
                    clamp_max = 0.95
                    if step % 500 == 0:
                        print(f"Step {step}: NeRF360 tuning enabled, enhancement={required_enhancement:.2f}x, using strong constraint")
                else:
                    # 中等亮度，插值调整
                    t = (input_brightness - 0.06) / (0.12 - 0.06)
                    curve_weight = 0.1 + 0.7 * t
                    adj_weight = 0.9 - 0.7 * t
                    clamp_min = 0.01 + 0.04 * t
                    clamp_max = 0.99 - 0.04 * t
                    if step % 500 == 0:
                        print(f"Step {step}: NeRF360 tuning enabled, enhancement={required_enhancement:.2f}x, using medium constraint")
                
                curve_adj = curve_weight * self.curve + adj_weight * curve_adj
                curve_adj = torch.clamp(curve_adj, clamp_min, clamp_max)
            # =============================================
            
            # ========== 保护机制：防止curve_adj变成全黑 ==========
            # 检查curve_adj是否过暗（平均值过低）
            curve_mean = curve_adj.mean().item()
            if curve_mean < 0.1:
                print(f"Warning: curve_adj too dark (mean={curve_mean:.4f}), resetting to default curve")
                curve_adj = torch.clamp(self.curve, 0.01, 1)
            
            # 确保曲线最小值不低于0.01，防止映射到全黑
            curve_adj = torch.clamp(curve_adj, 0.01, 1)
            # =============================================
            
            # ========== 调试代码：打印曲线信息 ==========
            if step % 500 == 0:
                print(f"Step {step}: curve min={self.curve.min().item():.4f}, max={self.curve.max().item():.4f}, mean={self.curve.mean().item():.4f}")
                print(f"Step {step}: curve_adj_bias min={curve_adj_bias.min().item():.4f}, max={curve_adj_bias.max().item():.4f}, mean={curve_adj_bias.mean().item():.4f}")
                print(f"Step {step}: curve_adj min={curve_adj.min().item():.4f}, max={curve_adj.max().item():.4f}, mean={curve_adj.mean().item():.4f}")
                print(f"Step {step}: pixels mean={pixels.mean().item():.4f}")
            # =============================================
            
            # Record curve parameters for visualization (paper, controlled by flag)
  #          if cfg.enable_curve_analysis and self.curve_visualizer is not None:
             #   if step % cfg.debug_steps_interval == 0:
            #        self.curve_visualizer.record(
            #            step=step,
              #          curve=self.curve,
              #          curve_adj_bias=curve_adj_bias,
              #          gamma_alpha_beta=gamma_alpha_beta,
              #      )

            # ========== 调试代码：保存中间结果（阶段1：预处理） ==========
            # 每隔指定间隔保存一次调试图像（由开关控制）
            if cfg.enable_debug_steps and step % cfg.debug_steps_interval == 0:
                debug_dir = f"{cfg.result_dir}/debug_steps"
                os.makedirs(debug_dir, exist_ok=True)
                # 保存原始渲染图像
                save_debug_image(pixels[0].cpu().detach(), f"{debug_dir}/step_{step}_01_original.png")
                # 保存低光输入图像
                save_debug_image(colors_low[0].cpu().detach(), f"{debug_dir}/step_{step}_02_lowlight.png")
                # 保存深度图（几何可视化）
                if depths is not None:
                    save_debug_image(depths[0].cpu().detach().repeat(1, 1, 3), f"{debug_dir}/step_{step}_03_depth.png")
                # 保存法线图（几何可视化）
                if normals is not None:
                    # 法线需要归一化到 [0,1]
                    normals_vis = (normals[0].cpu().detach() + 1) / 2  # 从 [-1,1] 转换到 [0,1]
                    save_debug_image(normals_vis, f"{debug_dir}/step_{step}_04_normal.png")
                # 保存结构置信度图（几何可视化）
                save_debug_image(struct_confidence[0].cpu().detach().repeat(1, 1, 3), f"{debug_dir}/step_{step}_05_struct_confidence.png")
                # 保存暗区掩码（几何可视化）
                save_debug_image(dark_mask[0].cpu().detach().repeat(1, 1, 3), f"{debug_dir}/step_{step}_06_dark_mask.png")
                # 保存平坦区域掩码
                save_debug_image(flat_mask[0].cpu().detach().repeat(1, 1, 3), f"{debug_dir}/step_{step}_07_flat_mask.png")
            # =============================================

            normal= (self.axis1_para[image_ids] + torch.Tensor([1, 0, 0]).to(colors_low.device)).unsqueeze(0)
            
            normal2 = (self.axis2_para[image_ids] + torch.Tensor([1, 0]).to(colors_low.device)).unsqueeze(0)
            
            bias = torch.zeros([1, 3]).to(colors_low.device)
            
            t1s, t2s, t3s, bias  = pixel_project(pixels.permute(0,3,1,2), normal, normal2, bias)
            t1s_out = [LUT_mapping(t1s, curve_adj), t1s[1], t1s[2], t1s[3]] 
            t2s_out = [LUT_mapping(t2s, curve_adj), t2s[1], t2s[2], t2s[3]] 
            t3s_out = [LUT_mapping(t3s, curve_adj), t3s[1], t3s[2], t3s[3]] 

            pixels_enh = pixel_project_back(t1s_out, t2s_out, t3s_out, bias).permute(0,2,3,1)
            pixels_enh = torch.clamp(pixels_enh, 0.0, 1.0)
            
            # 双分支自适应增强
            pixels_enh_refined = self.dual_branch_enhancer(pixels_enh, pixels, struct_confidence, flat_mask, dark_mask)
            pixels_enh_refined = torch.clamp(pixels_enh_refined, 0.0, 1.0)
            
            # ========== 调试代码：保存中间结果（阶段2：增强后） ==========
            # 每隔指定间隔保存一次调试图像（由开关控制）
            if cfg.enable_debug_steps and step % cfg.debug_steps_interval == 0:
                debug_dir = f"{cfg.result_dir}/debug_steps"
                # 保存曲线调整后的图像（After curve）
                save_debug_image(pixels_enh[0].cpu().detach(), f"{debug_dir}/step_{step}_08_after_curve.png")
                # 保存双分支增强后的图像（After dual-branch refinement）
                save_debug_image(pixels_enh_refined[0].cpu().detach(), f"{debug_dir}/step_{step}_09_after_dual_branch.png")
                # 保存最终输出（Ours）
                save_debug_image(pixels_enh_refined[0].cpu().detach(), f"{debug_dir}/step_{step}_10_ours.png")
                # 保存目标增强图像（GT）
                save_debug_image(colors_enh[0].cpu().detach(), f"{debug_dir}/step_{step}_11_gt.png")
                print(f"Debug images saved at step {step}")
            # =============================================

            # Save enhanced images under the configured result directory.
            if step % 1000 == 0:
                enh_refined_dir = os.path.join(cfg.result_dir, "renders_enh_refined")
                os.makedirs(enh_refined_dir, exist_ok=True)
                save_debug_image(pixels_enh_refined[0].cpu().detach(), f"{enh_refined_dir}/step_{step}_enh_refined.png")
            
            gamma = gamma_alpha_beta[:,0]
            alpha, beta = gamma_alpha_beta[:,1], gamma_alpha_beta[:,2]
            
            gamma = torch.Tensor([1.0]).to(device) + 0.1*gamma
            alpha = torch.Tensor([0.5]).to(device) + 0.002*alpha 
            beta = torch.Tensor([1.0]).to(device) + 0.002*beta
            
            pesdo_curve = gamma_curve(self.pesdo_curve, gamma)  # Pseudo-gamma curve
            pesdo_curve = s_curve(pesdo_curve, alpha, beta) # Pseudo-scurve curve
            

            con_degree = (self.constrast_level/torch.mean(pixels)).item()   # frame-adaptive contrast degree, Eq.8 in paper
            loss_co = loss_contrast(pixels.permute(0,3,1,2), colors_enh.permute(0,3,1,2), contrast=con_degree)
            
            l1loss = F.l1_loss(colors_low, pixels)
            ssimloss = 1.0 - self.ssim(pixels.permute(0,3,1,2), colors_low.permute(0,3,1,2))
            loss_regress_low = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

            l1loss_enh = F.l1_loss(colors_enh, pixels_enh_refined)  # enhancement loss constrain
            ssimloss_enh = 1.0 - self.ssim(pixels_enh_refined.permute(0,3,1,2), colors_enh.permute(0,3,1,2))
            loss_regress_enh = l1loss_enh * (1.0 - cfg.ssim_lambda) + ssimloss_enh * cfg.ssim_lambda
            
            hist_loss = loss_histo(curve_adj, pixels, pesdo_curve, step)
            
            # 计算新增的损失函数
            # 使用最终的增强图像pixels_enh_refined而不是colors_enh
            loss_struct = loss_structure(struct_confidence, pixels_enh_refined, pixels)
            loss_noise_val = loss_noise(flat_mask, dark_mask, pixels_enh_refined)
            
            # 多视图一致性损失 - 这里需要多个视图，暂时使用当前视图和增强视图的一致性
            # 实际应用中应该使用不同视角的图像
            loss_multi = loss_multi_view([colors_enh, pixels_enh_refined], weight=0.2)
            
            # ========== 总损失 - 根据数据集调整权重 ==========
            if cfg.dataset_type == "NeRF360" and cfg.enable_nerf360_tuning:
                # NeRF360特调：动态调整损失权重
                input_brightness = pixels.mean().item()
                if input_brightness < 0.06:
                    # 非常暗的场景：弱正则化，允许强增强
                    hist_weight = 5.0
                    struct_weight = 0.2
                    noise_weight = 0.1
                    multi_weight = 0.1
                elif input_brightness > 0.12:
                    # 较亮的场景：强正则化，避免过度增强
                    hist_weight = 20.0
                    struct_weight = 0.5
                    noise_weight = 0.3
                    multi_weight = 0.3
                else:
                    # 中等亮度：插值调整
                    t = (input_brightness - 0.06) / (0.12 - 0.06)
                    hist_weight = 5.0 + 15.0 * t
                    struct_weight = 0.2 + 0.3 * t
                    noise_weight = 0.1 + 0.2 * t
                    multi_weight = 0.1 + 0.2 * t
            else:
                # LOM-Full / NeRF360无特调: 默认设置
                hist_weight = 10.0
                struct_weight = 0.3
                noise_weight = 0.2
                multi_weight = 0.2
            
            loss = loss_regress_low + 0.5*loss_regress_enh + loss_co + hist_weight * hist_loss + struct_weight * loss_struct + noise_weight * loss_noise_val + multi_weight * loss_multi
            
            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "

            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            if cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar(
                    "train/num_GS", len(self.splats["means3d"]), step
                )
                self.writer.add_scalar("train/mem", mem, step)
                
                if cfg.tb_save_image:
                    canvas = torch.cat([colors_enh, pixels_enh], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)

                    canvas_low = torch.cat([colors_low, pixels], dim=2).detach().cpu().numpy()
                    canvas_low = canvas_low.reshape(-1, *canvas_low.shape[2:])
                    self.writer.add_image("train/render_low", canvas_low, step)

                self.writer.flush()

            # update running stats for prunning & growing
            if step < cfg.refine_stop_iter:
                self.update_running_stats(info)

                if step > cfg.refine_start_iter and step % cfg.refine_every == 0:
                    grads = self.running_stats["grad2d"] / self.running_stats[
                        "count"
                    ].clamp_min(1)
                    
                    # ========== 自适应GS数量控制 ==========
                    current_gs = len(self.splats['means3d'])
                    # 根据GS数量动态调整grow_grad2d阈值
                    # GS数量超过150万时，提高阈值减少分裂
                    dynamic_grow_grad2d = cfg.grow_grad2d
                    if current_gs > 2000000:
                        dynamic_grow_grad2d = cfg.grow_grad2d * 1.5
                        if step % 500 == 0:
                            print(f"Step {step}: High GS count ({current_gs:,}), increased grow_grad2d to {dynamic_grow_grad2d:.6f}")
                    elif current_gs > 1500000:
                        dynamic_grow_grad2d = cfg.grow_grad2d * 1.2
                    
                    # 动态调整prune_opa，更积极清理
                    dynamic_prune_opa = cfg.prune_opa
                    if current_gs > 2000000:
                        dynamic_prune_opa = cfg.prune_opa * 1.5
                    # =====================================

                    # grow GSs
                    is_grad_high = grads >= dynamic_grow_grad2d
                    is_small = (
                        torch.exp(self.splats["scales"]).max(dim=-1).values
                        <= cfg.grow_scale3d * self.scene_scale
                    )
                    is_dupli = is_grad_high & is_small
                    n_dupli = is_dupli.sum().item()
                    self.refine_duplicate(is_dupli)

                    is_split = is_grad_high & ~is_small
                    is_split = torch.cat(
                        [
                            is_split,
                            # new GSs added by duplication will not be split
                            torch.zeros(n_dupli, device=device, dtype=torch.bool),
                        ]
                    )
                    n_split = is_split.sum().item()
                    self.refine_split(is_split)
                    print(
                        f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                        f"Now having {len(self.splats['means3d'])} GSs."
                    )

                    # prune GSs
                    is_prune = torch.sigmoid(self.splats["opacities"]) < dynamic_prune_opa
                    if step > cfg.reset_every:
                        # The official code also implements sreen-size pruning but
                        # it's actually not being used due to a bug:
                        # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
                        is_too_big = (
                            torch.exp(self.splats["scales"]).max(dim=-1).values
                            > cfg.prune_scale3d * self.scene_scale
                        )
                        is_prune = is_prune | is_too_big
                    n_prune = is_prune.sum().item()
                    self.refine_keep(~is_prune)
                    print(
                        f"Step {step}: {n_prune} GSs pruned. "
                        f"Now having {len(self.splats['means3d'])} GSs."
                    )

                    # reset running stats
                    self.running_stats["grad2d"].zero_()
                    self.running_stats["count"].zero_()

                if step % cfg.reset_every == 0:
                    self.reset_opa(cfg.prune_opa * 2.0)

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            # optimize
            for optimizer in self.optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.curve_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.adjust_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.sat_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            # 更新学习率调度器
            for scheduler in self.schedulers:
                scheduler.step()

            # save checkpoint
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means3d"]),
                }
                print("Step: ", step, stats)
                with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
                    json.dump(stats, f)
                # Save complete checkpoint
                self.save_checkpoint(step)

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps] or step == max_steps - 1:
                self.eval(step)
                self.render_traj(step)
                
                # Generate curve visualizations at training end (paper, controlled by flag)
                if step == max_steps - 1 and cfg.enable_curve_analysis and getattr(self, "curve_visualizer", None) is not None:
                    self.curve_visualizer.finalize()

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def update_running_stats(self, info: Dict):
        """Update running stats."""
        cfg = self.cfg

        # normalize grads to [-1, 1] screen space
        if cfg.absgrad:
            grads = info["means2d"].absgrad.clone()
        else:
            grads = info["means2d"].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * cfg.batch_size
        grads[..., 1] *= info["height"] / 2.0 * cfg.batch_size
        if cfg.packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz] or None
            self.running_stats["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
            self.running_stats["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids))
        else:
            # grads is [C, N, 2]
            sel = info["radii"] > 0.0  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            self.running_stats["grad2d"].index_add_(0, gs_ids, grads[sel].norm(dim=-1))
            self.running_stats["count"].index_add_(
                0, gs_ids, torch.ones_like(gs_ids).int()
            )

    @torch.no_grad()
    def reset_opa(self, value: float = 0.01):
        """Utility function to reset opacities."""
        opacities = torch.clamp(
            self.splats["opacities"], max=torch.logit(torch.tensor(value)).item()
        )
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                if param_group["name"] != "opacities":
                    continue
                p = param_group["params"][0]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        p_state[key] = torch.zeros_like(p_state[key])
                p_new = torch.nn.Parameter(opacities)
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[param_group["name"]] = p_new
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_split(self, mask: Tensor):
        """Utility function to grow GSs."""
        device = self.device

        sel = torch.where(mask)[0]
        rest = torch.where(~mask)[0]

        scales = torch.exp(self.splats["scales"][sel])  # [N, 3]
        quats = F.normalize(self.splats["quats"][sel], dim=-1)  # [N, 4]
        rotmats = normalized_quat_to_rotmat(quats)  # [N, 3, 3]
        samples = torch.einsum(
            "nij,nj,bnj->bni",
            rotmats,
            scales,
            torch.randn(2, len(scales), 3, device=device),
        )  # [2, N, 3]

        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                # create new params
                if name == "means3d":
                    p_split = (p[sel] + samples).reshape(-1, 3)  # [2N, 3]
                elif name == "scales":
                    p_split = torch.log(scales / 1.6).repeat(2, 1)  # [2N, 3]
                else:
                    repeats = [2] + [1] * (p.dim() - 1)
                    p_split = p[sel].repeat(repeats)
                p_new = torch.cat([p[rest], p_split])
                p_new = torch.nn.Parameter(p_new)
                # update optimizer
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key == "step":
                        continue
                    v = p_state[key]
                    # new params are assigned with zero optimizer states
                    # (worth investigating it)
                    v_split = torch.zeros((2 * len(sel), *v.shape[1:]), device=device)
                    p_state[key] = torch.cat([v[rest], v_split])
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        for k, v in self.running_stats.items():
            if v is None:
                continue
            repeats = [2] + [1] * (v.dim() - 1)
            v_new = v[sel].repeat(repeats)
            self.running_stats[k] = torch.cat((v[rest], v_new))
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_duplicate(self, mask: Tensor):
        """Unility function to duplicate GSs."""
        sel = torch.where(mask)[0]
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        # new params are assigned with zero optimizer states
                        # (worth investigating it as it will lead to a lot more GS.)
                        v = p_state[key]
                        v_new = torch.zeros(
                            (len(sel), *v.shape[1:]), device=self.device
                        )
                        # v_new = v[sel]
                        p_state[key] = torch.cat([v, v_new])
                p_new = torch.nn.Parameter(torch.cat([p, p[sel]]))
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        for k, v in self.running_stats.items():
            self.running_stats[k] = torch.cat((v, v[sel]))
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_keep(self, mask: Tensor):
        """Unility function to prune GSs."""
        sel = torch.where(mask)[0]
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        p_state[key] = p_state[key][sel]
                p_new = torch.nn.Parameter(p[sel])
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        for k, v in self.running_stats.items():
            self.running_stats[k] = v[sel]
        torch.cuda.empty_cache()

    @torch.no_grad()
    def eval(self, step: int):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": []}
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]
            torch.cuda.synchronize()
            tic = time.time()
            colors_enh, colors_low, _, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 3]
            
            depth_low = colors_low[:, :, :, 3:]
            depth_enh = colors_enh[:, :, :, 3:]
            colors_low = colors_low[:, :, :, :3]
            
            colors_enh = torch.clamp(colors_enh[:, :, :, :3], 0.0, 1.0)
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            # write images with step info
            
            canvas = torch.cat([colors_low, colors_enh], dim=2).squeeze(0).cpu().numpy()
            
            # Save render images with step info to avoid overwriting
            imageio.imwrite(
                f"{self.render_dir_depth}/val_{step:04d}_{i:04d}_depth_low.png", (depth_low.squeeze(0).squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
            )

            imageio.imwrite(
                f"{self.render_dir_depth}/val_{step:04d}_{i:04d}_depth_enh.png", (depth_enh.squeeze(0).squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
            )

            imageio.imwrite(
                f"{self.render_dir}/val_{step:04d}_{i:04d}_low.png", (colors_low.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            )
            imageio.imwrite(
                f"{self.render_dir}/val_{step:04d}_{i:04d}_enh.png", (colors_enh.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            )

            pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
            colors_enh = colors_enh.permute(0, 3, 1, 2)  # [1, 3, H, W]
            metrics["psnr"].append(self.psnr(colors_enh, pixels))
            metrics["ssim"].append(self.ssim(colors_enh, pixels))
            metrics["lpips"].append(self.lpips(colors_enh, pixels))

        ellipse_time /= len(valloader)

        psnr = torch.stack(metrics["psnr"]).mean()
        ssim = torch.stack(metrics["ssim"]).mean()
        lpips = torch.stack(metrics["lpips"]).mean()
        print(
            f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f} "
            f"Time: {ellipse_time:.3f}s/image "
            f"Number of GS: {len(self.splats['means3d'])}"
        )
        
        # Save best model based on PSNR
        if psnr.item() > self.best_psnr:
            self.best_psnr = psnr.item()
            self.best_ssim = ssim.item()
            self.best_step = step
            print(f"New best model found! PSNR: {self.best_psnr:.3f} at step {step}")
            
            # Save best checkpoint with complete state
            self.save_checkpoint(step, is_best=True)
            
            # Save renders for the best model
            self.save_best_renders()
        
        # save stats as json
        stats = {
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "lpips": lpips.item(),
            "ellipse_time": ellipse_time,
            "num_GS": len(self.splats["means3d"]),
        }
        with open(f"{self.stats_dir}/val_step{step:04d}.json", "w") as f:
            json.dump(stats, f)
        # save stats to tensorboard
        for k, v in stats.items():
            self.writer.add_scalar(f"val/{k}", v, step)
        self.writer.flush()

    @torch.no_grad()
    def save_best_renders(self):
        """Save renders for the best model to a dedicated directory."""
        print("Saving best model renders...")
        cfg = self.cfg
        device = self.device
        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]
            
            colors_enh, colors_low, _, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            
            depth_low = colors_low[:, :, :, 3:]
            depth_enh = colors_enh[:, :, :, 3:]
            colors_low = colors_low[:, :, :, :3]
            colors_enh = torch.clamp(colors_enh[:, :, :, :3], 0.0, 1.0)
            
            # Save to best model directory
            imageio.imwrite(
                f"{self.render_dir_best_depth}/val_{i:04d}_depth_low.png", (depth_low.squeeze(0).squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
            )
            imageio.imwrite(
                f"{self.render_dir_best_depth}/val_{i:04d}_depth_enh.png", (depth_enh.squeeze(0).squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
            )
            imageio.imwrite(
                f"{self.render_dir_best}/val_{i:04d}_low.png", (colors_low.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            )
            imageio.imwrite(
                f"{self.render_dir_best}/val_{i:04d}_enh.png", (colors_enh.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            )
        
        print(f"Best model renders saved to {self.render_dir_best}")

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        # print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device
        camtoworlds = self.parser.camtoworlds[10:60]
        camtoworlds = generate_interpolated_path(camtoworlds, 10)  # [N, 3, 4]
        
        camtoworlds = np.concatenate(
            [
                camtoworlds,
                np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]
        
        canvas_all = []
        for i in tqdm.trange(len(camtoworlds), desc="Rendering trajectory"):
            renders_enh, renders_low, _, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds[i : i + 1],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders_enh[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
            depths = renders_enh[0, ..., 3:4]  # [H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())

            canvas = colors
            canvas = (canvas.cpu().numpy() * 255).astype(np.uint8)
            canvas_all.append(canvas)

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=50)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors_enh, render_colors_low, _, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors_enh[0].cpu().numpy()


def main(cfg: Config):
    runner = Runner(cfg)

    if cfg.ckpt is not None:
        # Run evaluation with the provided checkpoint
        print(f"\n=== Loading checkpoint from {cfg.ckpt} ===")
        step = runner.load_checkpoint(cfg.ckpt)
        runner.eval(step=step)
        runner.render_traj(step=step)
    else:
        runner.train()
        
        # Load best model and run final evaluation
        best_ckpt_path = f"{runner.ckpt_dir}/best.ckpt"
        if os.path.exists(best_ckpt_path):
            print(f"\n=== Loading best model from {best_ckpt_path} ===")
            step = runner.load_checkpoint(best_ckpt_path)
            print(f"Best model achieved PSNR: {runner.best_psnr:.3f}, SSIM: {runner.best_ssim:.4f} at step {step}")
            
            # Final evaluation with best model
            print("\n=== Final evaluation with best model ===")
            runner.eval(step=step)
            runner.render_traj(step=step)
        else:
            print("Warning: No best model found!")

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)
    main(cfg)
