import random

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import einops

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, img_channels=4, hidden_dim=128, out_dim=255, embed_dim=16):
        super(CrossAttention, self).__init__()
        
        # 保存hidden_dim为类的属性
        self.hidden_dim = hidden_dim

        # 确定第一个卷积层的输入通道数
        if img_channels > 4:
            # 当输入通道大于4时，使用1x1卷积压缩到4通道
            self.channel_compress = nn.Conv2d(img_channels, 4, kernel_size=1)
            conv_in_channels = 4
        else:
            # 当输入通道<=4时，直接使用原始通道数
            self.channel_compress = None
            conv_in_channels = img_channels

        self.conv = nn.Sequential(
            nn.Conv2d(conv_in_channels, 64, kernel_size=3, padding=1),  # (1, 64, H, W)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (1, 64, H/2, W/2)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (1, 128, H/2, W/2)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # (1, 128, 4, 4)
        )

        self.query_fc = nn.Linear(embed_dim, hidden_dim)  # embedding -> query
        self.key_fc = nn.Linear(128, hidden_dim)  # image feature -> key
        self.value_fc = nn.Linear(128, hidden_dim)  # image feature -> value

        self.out_fc = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, image, embedding):
        
        # 压缩输入通道数
        if self.channel_compress is not None:
            image = self.channel_compress(image)
        
        img_feat = self.conv(image)
        
        img_feat = einops.rearrange(img_feat, 'b c h w -> b (h w) c')
        # 支持任意batch size
        embed_flat = embedding.view(embedding.shape[0], -1)  # (batch_size, 16)
        query = self.query_fc(embed_flat).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        key = self.key_fc(img_feat)  # (batch_size, 16, hidden_dim)
        value = self.value_fc(img_feat)  # (batch_size, 16, hidden_dim)

        attention_scores = torch.bmm(query, key.transpose(1, 2)) / (self.hidden_dim ** 0.5)  # (batch_size, 1, 16)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, 1, 16)

        context = torch.bmm(attention_weights, value)  # (batch_size, 1, hidden_dim)
        context = context.squeeze(1)  # (batch_size, hidden_dim)

        output = self.out_fc(context)  # (batch_size, out_dim)
        return output

class GeoEncoder(nn.Module):
    """轻量几何编码器：将深度和法线编码成更丰富的特征"""
    def __init__(self, in_channels=4, out_channels=4):
        super(GeoEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        )
        
        # 残差连接
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = None
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, geo):
        out = self.conv(geo)
        if self.residual is not None:
            out = out + self.residual(geo)
        else:
            out = out + geo
        return out


class CameraOptModule(torch.nn.Module):
    """Camera pose optimization module."""

    def __init__(self, n: int):
        super().__init__()
        # Delta positions (3D) + Delta rotations (6D)
        self.embeds = torch.nn.Embedding(n, 9)
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        """Adjust camera pose based on deltas.

        Args:
            camtoworlds: (..., 4, 4)
            embed_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """
        assert camtoworlds.shape[:-2] == embed_ids.shape
        batch_shape = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(
            drot + self.identity.expand(*batch_shape, -1)
        )  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_shape, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)


class AppearanceOptModule(torch.nn.Module):
    """Appearance optimization module."""

    def __init__(
        self,
        n: int,
        feature_dim: int,
        embed_dim: int = 16,
        sh_degree: int = 3,
        mlp_width: int = 64,
        mlp_depth: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sh_degree = sh_degree
        self.embeds = torch.nn.Embedding(n, embed_dim)
        layers = []
        layers.append(
            torch.nn.Linear(embed_dim + feature_dim + (sh_degree + 1) ** 2, mlp_width)
        )
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(mlp_width, 3))
        self.color_head = torch.nn.Sequential(*layers)

    def forward(
        self, features: Tensor, embed_ids: Tensor, dirs: Tensor, sh_degree: int
    ) -> Tensor:
        """Adjust appearance based on embeddings.

        Args:
            features: (N, feature_dim)
            embed_ids: (C,)
            dirs: (C, N, 3)

        Returns:
            colors: (C, N, 3)
        """
        from gsplat.cuda._torch_impl import _eval_sh_bases_fast

        C, N = dirs.shape[:2]
        # Camera embeddings
        if embed_ids is None:
            embeds = torch.zeros(C, self.embed_dim, device=features.device)
        else:
            embeds = self.embeds(embed_ids)  # [C, D2]
        embeds = embeds[:, None, :].expand(-1, N, -1)  # [C, N, D2]
        # GS features
        features = features[None, :, :].expand(C, -1, -1)  # [C, N, D1]
        # View directions
        dirs = F.normalize(dirs, dim=-1)  # [C, N, 3]
        num_bases_to_use = (sh_degree + 1) ** 2
        num_bases = (self.sh_degree + 1) ** 2
        sh_bases = torch.zeros(C, N, num_bases, device=features.device)  # [C, N, K]
        sh_bases[:, :, :num_bases_to_use] = _eval_sh_bases_fast(num_bases_to_use, dirs)
        # Get colors
        if self.embed_dim > 0:
            h = torch.cat([embeds, features, sh_bases], dim=-1)  # [C, N, D1 + D2 + K]
        else:
            h = torch.cat([features, sh_bases], dim=-1)
        colors = self.color_head(h)
        return colors


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def normalized_quat_to_rotmat(quat: Tensor) -> Tensor:
    assert quat.shape[-1] == 4, quat.shape
    w, x, y, z = torch.unbind(quat, dim=-1)
    mat = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return mat.reshape(quat.shape[:-1] + (3, 3))


def knn(x: Tensor, K: int = 4) -> Tensor:
    """使用PyTorch原生实现K近邻搜索，避免GPU-CPU数据传输
    
    Args:
        x: 输入张量，形状为 [N, D]
        K: 近邻数量
    
    Returns:
        距离张量，形状为 [N, K]
    """
    N = x.shape[0]
    
    # 对于大数据集，使用分块计算避免 OOM
    if N > 100000:
        chunk_size = 10000
        all_distances = []
        
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            chunk_i = x[i:end_i]  # [chunk_size, D]
            
            # 计算当前块与所有点的距离
            chunk_i_expanded = chunk_i.unsqueeze(1)  # [chunk_size, 1, D]
            x_expanded = x.unsqueeze(0)  # [1, N, D]
            distances = torch.norm(chunk_i_expanded - x_expanded, dim=-1)  # [chunk_size, N]
            
            # 找到 K+1 个最近邻（包括自身）
            dist_chunk, _ = torch.topk(distances, k=K+1, largest=False, dim=-1)  # [chunk_size, K+1]
            
            # 排除自身（第一个是自身）
            all_distances.append(dist_chunk[:, 1:])  # [chunk_size, K]
        
        return torch.cat(all_distances, dim=0)  # [N, K]
    else:
        # 对于小数据集，使用完整计算
        x_expanded = x.unsqueeze(1)  # [N, 1, D]
        x_t_expanded = x.unsqueeze(0)  # [1, N, D]
        distances = torch.norm(x_expanded - x_t_expanded, dim=-1)  # [N, N]
        
        # 找到每个点的K个最近邻
        distances, indices = torch.topk(distances, k=K+1, largest=False, dim=-1)  # [N, K+1]
        
        # 排除自身（距离为0的点）
        return distances[:, 1:]  # [N, K]


def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 测试代码
if __name__ == "__main__":
    
    image = torch.randn(1, 3, 374, 499)  # (1, 3, 200, 200)
    embedding = torch.randn(1, 4, 4)  # (1, 4, 4)展平后为16维

    
    model = CrossAttention()
    output = model(image, embedding)
    print(output)
    print(output.shape)  
    print(output[:,0].shape)
    