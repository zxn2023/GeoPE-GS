# GeoPE-GS: Geometry-Conditioned Pseudo-Enhancement for Low-Light 3D Gaussian Splatting

Official code for **GeoPE-GS: Geometry-Conditioned Pseudo-Enhancement for Low-Light 3D Gaussian Splatting**. The training entry point is [`examples/simple_trainer_ours.py`](examples/simple_trainer_ours.py).

> **Release status:** Code is available. Datasets and results will be shared separately. Pretrained weights are not currently available.

## 部署教程：按顺序复制运行

以下命令以 **AutoDL / Ubuntu 22.04 / Python 3.10 / 系统 CUDA 11.8 / NVIDIA GPU** 为例。服务器镜像请选择已有 Miniconda 或 Conda3 的 Python 3.10 环境。命令在 Linux 终端中执行；每个灰色代码框可整块复制。其他 Linux 服务器可以照做，但需先准备兼容的 NVIDIA 驱动、CUDA toolkit（包含 `nvcc`）和 Python 3.9 或 3.10。

### 1. 查看环境

```bash
python --version
nvidia-smi
nvcc --version
```

应能看到 Python 3.10、NVIDIA GPU 和 CUDA 编译器。如果 `nvcc` 找不到，需要安装 CUDA toolkit；仅安装 PyTorch 不会提供它。

### 2. 下载代码

AutoDL 的数据盘通常位于 `/root/autodl-tmp`。在其他服务器上，把第一行换成你自己的工作目录。

```bash
cd /root/autodl-tmp
git clone --recurse-submodules https://github.com/zxn2023/GeoPE-GS.git
cd GeoPE-GS
```

已经克隆过仓库时，在仓库根目录运行以下命令补齐 GLM 子模块：

```bash
git submodule update --init --recursive
```

### 3. 安装 PyTorch 和依赖

下面的 PyTorch 1.13.1 + CUDA 11.7 安装命令来自 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。这套组合对应系统 CUDA 11.8 的部署记录。请先安装 PyTorch，再编译仓库内的 gsplat；不要额外安装另一份 gsplat。

```bash
python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install -r examples/requirements.txt
python -m pip install numpy==1.24.3
python -m pip install --no-build-isolation -e .
```

最后一行会编译本仓库的 CUDA 扩展，可能需要几分钟。安装后复制下面的检查命令：

```bash
python -c "import torch, gsplat, gsplat.csrc; print('PyTorch:', torch.__version__); print('PyTorch CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available()); print('gsplat:', gsplat.__version__)"
```

`GPU available:` 应为 `True`；`import gsplat.csrc` 成功表示本地 CUDA 扩展已可导入。若你使用的机器或 CUDA 版本不同，应选择与其兼容的 PyTorch 安装包。

### 4. 放置数据

数据集**尚未随代码公开**，需要先自行放到仓库的 `data/` 目录。`data/` 已加入 `.gitignore`，训练结果也不会被提交到 GitHub。至少要满足下列目录结构；文件名大小写在 Linux 上必须匹配。

```text
GeoPE-GS/
├── data/
│   ├── LOM_full/
│   │   └── buu/
│   │       ├── colmap_sparse/0/
│   │       ├── low/
│   │       ├── high/
│   │       ├── over_exp/
│   │       ├── transforms_train.json
│   │       └── transforms_test.json
│   └── NeRF_360/
│       └── bicycle/
│           ├── sparse/0/
│           ├── images/
│           └── images_8_variance/
└── examples/
```

仅训练低光场景时，`over_exp/` 可以暂不准备；训练过曝场景时需要它。换其他场景时，把下方命令中的 `buu` 或 `bicycle` 改为对应目录名。先在仓库根目录确认数据确实存在：

```bash
ls data/LOM_full/buu
ls data/NeRF_360/bicycle
```

### 5. 训练 GeoPE-GS

先进入 `examples/`。以下三种任务**分别选择一块命令运行**；运行前确认对应数据已放好。`--result_dir` 指定输出目录，可以改成你自己的路径。

```bash
cd /root/autodl-tmp/GeoPE-GS/examples
```

**LOM 低光（buu，10,000 步）：**

```bash
python simple_trainer_ours.py \
  --data_dir ../data/LOM_full/buu \
  --exp_name low \
  --result_dir ../results/LOM/buu_low \
  --max_steps 10000 \
  --batch_size 1
```

**LOM 过曝（buu，10,000 步）：**

```bash
python simple_trainer_ours.py \
  --data_dir ../data/LOM_full/buu \
  --exp_name over_exp \
  --result_dir ../results/LOM/buu_over_exp \
  --max_steps 10000 \
  --batch_size 1
```

**MipNeRF360 变化曝光（bicycle，30,000 步）：**

```bash
python simple_trainer_ours.py \
  --data_dir ../data/NeRF_360/bicycle \
  --exp_name variance \
  --data_factor 8 \
  --result_dir ../results/NeRF_360/bicycle \
  --max_steps 30000 \
  --eval_steps 15000 20000 30000 \
  --save_steps 15000 20000 30000 \
  --batch_size 1
```

不要在反斜杠 `\` 后面加注释，否则整块复制到 Bash 时会报错。训练完成后，在各自的 `--result_dir` 下查看 `ckpts/`（模型）、`stats/`（指标）、`renders/`（渲染）和 `tb/`（TensorBoard 日志）。

### 6. 用已有 checkpoint 评估

下面以第 5 步训练出的 LOM 低光模型为例。`--ckpt` 会跳过训练，加载模型进行评估并渲染轨迹；当前代码**没有** `--eval_only` 参数。

```bash
cd /root/autodl-tmp/GeoPE-GS/examples
python simple_trainer_ours.py \
  --data_dir ../data/LOM_full/buu \
  --exp_name low \
  --result_dir ../results/eval/LOM/buu_low \
  --ckpt ../results/LOM/buu_low/ckpts/best.ckpt
```

如果找不到 `best.ckpt`，先检查训练输出目录中实际保存的文件名。仓库目前不提供预训练权重。

### 7. 查看 TensorBoard

训练时在另一个终端运行，打开服务器的 6006 端口或使用平台提供的端口映射：

```bash
cd /root/autodl-tmp/GeoPE-GS
tensorboard --logdir results/LOM/buu_low/tb --host 0.0.0.0 --port 6006
```

### 常见问题

- `No module named gsplat`：回到仓库根目录，重新执行第 3 步最后一行。
- `third_party/glm/glm/glm.hpp: No such file`：回到仓库根目录，执行 `git submodule update --init --recursive`，然后重新安装本地扩展。
- `CUDA available: False` 或编译器报错：检查 GPU 驱动、`nvcc --version`、PyTorch CUDA 版本及编译器是否匹配。
- `Image folder ... does not exist`：核对第 4 步的数据目录和大小写。过曝数据里 `.jpg` 与 `.JPG` 不同，不要未经检查批量重命名。

完整命令行参数可在 `examples/` 下查看：

```bash
python simple_trainer_ours.py --help
```

## Code map

| Path | Purpose |
| --- | --- |
| `examples/simple_trainer_ours.py` | GeoPE-GS training and evaluation |
| `examples/utils.py` | Geometry encoder and model utilities |
| `gsplat/rendering_double.py` | Dual-branch rasterization |
| `examples/datasets/` | LOM and MipNeRF360 data loaders |
| `examples/simple_trainer_3dgs.py` | Included 3DGS baseline example |

## Citation and attribution

GeoPE-GS citation metadata will be added with the paper release. Until then, use the title above when referring to this implementation. If you use the underlying Luminance-GS method or dataset, cite its [paper and repository](https://github.com/cuiziteng/Luminance-GS). This code also incorporates [gsplat](https://github.com/nerfstudio-project/gsplat); see [NOTICE.md](NOTICE.md) and [LICENSE](LICENSE).
