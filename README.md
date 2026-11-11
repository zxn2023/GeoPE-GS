
https://github.com/zxn2023/GeoPE-GS我的项目主页怎么将中文调整为英文

19:19
你的项目首页中文来自仓库根目录的 README.md，其中“部署与使用”这一节是中文。把该文件的内容翻译成英文，首页就会同步更新。查看当前 README

直接在 GitHub 上操作：

打开 README 编辑页面，登录你的账号。
将中文标题和说明改为英文，保留代码命令、路径和参数。例如：
部署与使用 → Installation and Usage
查看环境 → Check Your Environment
下载代码 → Clone the Repository
常见问题 → Troubleshooting
点击 Preview 检查排版。
点击 Commit changes…，提交说明可填 Translate README into English，提交到 main 后刷新首页即可。
这不需要修改 GitHub 的语言设置；需要修改的是 README 的正文。

19:20
# GeoPE-GS: Geometry-Conditioned Pseudo-Enhancement for Low-Light 3D Gaussian Splatting

Official code for **GeoPE-GS: Geometry-Conditioned Pseudo-Enhancement for Low-Light 3D Gaussian Splatting**. The training entry point is [`examples/simple_trainer_ours.py`](examples/simple_trainer_ours.py).

> **Release status:** Code is available. Datasets and results will be shared separately. Pretrained weights are not currently available.

## 部署与使用

以下以 **AutoDL / Ubuntu 22.04 / Python 3.10 / CUDA 11.8** 为例。需要 NVIDIA GPU、兼容的驱动和包含 `nvcc` 的 CUDA toolkit。

### 1. 查看环境

```bash
python --version
nvidia-smi
nvcc --version
```

确认 Python、GPU 驱动和 CUDA 编译器可用。

### 2. 下载代码

以下使用 AutoDL 的 `/root/autodl-tmp` 目录；其他服务器请替换为实际工作目录。

```bash
cd /root/autodl-tmp
git clone --recurse-submodules https://github.com/zxn2023/GeoPE-GS.git
cd GeoPE-GS
```

已有仓库可运行以下命令补齐 GLM 子模块：

```bash
git submodule update --init --recursive
```

### 3. 安装 PyTorch 和依赖

先安装 PyTorch 1.13.1 + CUDA 11.7，再安装项目依赖和仓库内的 gsplat。PyTorch 安装命令见[官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。

```bash
python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install -r examples/requirements.txt
python -m pip install --no-build-isolation -e .
```

最后一行会编译本仓库的 CUDA 扩展。安装后检查：

```bash
python -c "import torch, gsplat, gsplat.csrc; print('PyTorch:', torch.__version__); print('PyTorch CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available()); print('gsplat:', gsplat.__version__)"
```

`GPU available:` 应为 `True`，且 `gsplat.csrc` 应能成功导入。其他 CUDA 环境需选择兼容的 PyTorch 安装包。

### 4. 放置数据

数据集尚未随代码公开。低光数据放在仓库的 `data/` 目录，结构如下（Linux 区分文件名大小写）：

```text
GeoPE-GS/
├── data/
│   └── LOM_full/
│       └── buu/
│           ├── colmap_sparse/0/
│           ├── low/
│           ├── high/
│           ├── transforms_train.json
│           └── transforms_test.json
└── examples/
```

`low/` 是低光输入，`high/` 是评估参考图像。其他低光场景请替换命令中的 `buu`。

### 5. 训练 GeoPE-GS（LOM 低光，buu）

```bash
cd /root/autodl-tmp/GeoPE-GS/examples
python simple_trainer_ours.py \
  --data_dir ../data/LOM_full/buu \
  --exp_name low \
  --result_dir ../results/LOM/buu_low \
  --max_steps 10000 \
  --batch_size 1
```

结果写入 `--result_dir`，包括 `ckpts/`、`stats/`、`renders/` 和 `tb/`。

### 6. 用已有 checkpoint 评估

使用训练生成的 `best.ckpt` 评估并渲染：

```bash
cd /root/autodl-tmp/GeoPE-GS/examples
python simple_trainer_ours.py \
  --data_dir ../data/LOM_full/buu \
  --exp_name low \
  --result_dir ../results/eval/LOM/buu_low \
  --ckpt ../results/LOM/buu_low/ckpts/best.ckpt
```

### 7. 查看 TensorBoard

训练时可在另一个终端运行；远程访问需映射 6006 端口：

```bash
cd /root/autodl-tmp/GeoPE-GS
tensorboard --logdir results/LOM/buu_low/tb --host 0.0.0.0 --port 6006
```

### 常见问题

- `No module named gsplat`：回到仓库根目录，重新执行第 3 步最后一行。
- `third_party/glm/glm/glm.hpp: No such file`：回到仓库根目录，执行 `git submodule update --init --recursive`，然后重新安装本地扩展。
- `CUDA available: False` 或编译器报错：检查 GPU 驱动、`nvcc --version`、PyTorch CUDA 版本及编译器是否匹配。
- `Image folder ... does not exist`：核对第 4 步的数据目录、图像文件名和大小写。

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
| `examples/datasets/` | Dataset loaders; the guide above covers LOM low-light training |

## Citation and attribution

GeoPE-GS citation metadata will be added with the paper release. Until then, use the title above when referring to this implementation. If you use the underlying Luminance-GS method or dataset, cite its [paper and repository](https://github.com/cuiziteng/Luminance-GS). This code also incorporates [gsplat](https://github.com/nerfstudio-project/gsplat); see [NOTICE.md](NOTICE.md) and [LICENSE](LICENSE).
你帮我改

…
19:21
回复：the corresponding scene name. ### 5. Train GeoPE-GS on the LOM Low-Light Dataset The following example trains GeoPE-GS on the `buu` scene: ```bash cd /root/autodl-tmp/GeoPE-GS/examples python simple_trainer_ours.py \ --data_dir ../data/LOM_full/buu \ --exp_name low \ --result_dir ../results/LOM/buu_low \ --max_steps 10000 \ --batch_size 1 ``` Outputs are saved to `--result_dir`, including the `ckpts/`, `stats/`, `renders/`, and `tb/` directories. ### 6. Evaluate an Existing Checkpoint Use the `best.ckpt` checkpoint generated during training for evaluation and rendering: ```bash cd /root/autodl-tmp/GeoPE-GS/examples python simple_trainer_ours.py \ --data_dir ../data/LOM_full/buu \ --exp_name low \ --result_dir ../results/eval/LOM/buu_low \ --ckpt ../results/LOM/buu_low/ckpts/
下面是完整英文版，已清理多余的转义符，直接替换 README.md 的全部内容即可：

# GeoPE-GS: Geometry-Conditioned Pseudo-Enhancement for Low-Light 3D Gaussian Splatting

Official code for **GeoPE-GS: Geometry-Conditioned Pseudo-Enhancement for Low-Light 3D Gaussian Splatting**. The training entry point is [`examples/simple_trainer_ours.py`](examples/simple_trainer_ours.py).

> **Release status:** Code is available. Datasets and results will be shared separately. Pretrained weights are not currently available.

## Installation and Usage

The following instructions use **AutoDL / Ubuntu 22.04 / Python 3.10 / CUDA 11.8** as an example environment. An NVIDIA GPU, a compatible driver, and a CUDA toolkit that includes `nvcc` are required.

### 1. Check Your Environment

```bash
python --version
nvidia-smi
nvcc --version
```

Ensure that Python, the NVIDIA GPU driver, and the CUDA compiler are available.

### 2. Clone the Repository

The commands below use AutoDL's `/root/autodl-tmp` directory. On other servers, replace this path with your working directory.

```bash
cd /root/autodl-tmp
git clone --recurse-submodules https://github.com/zxn2023/GeoPE-GS.git
cd GeoPE-GS
```

If you have already cloned the repository, initialize the GLM submodule:

```bash
git submodule update --init --recursive
```

### 3. Install PyTorch and Dependencies

First, install PyTorch 1.13.1 with CUDA 11.7 support. Then install the project dependencies and the local gsplat package. See the [official PyTorch previous versions page](https://pytorch.org/get-started/previous-versions/) for installation commands.

```bash
python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install -r examples/requirements.txt
python -m pip install --no-build-isolation -e .
```

The final command compiles the CUDA extension included in this repository. After installation, verify your setup:

```bash
python -c "import torch, gsplat, gsplat.csrc; print('PyTorch:', torch.__version__); print('PyTorch CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available()); print('gsplat:', gsplat.__version__)"
```

`GPU available:` should be `True`, and `gsplat.csrc` should import successfully. For other CUDA environments, select a compatible PyTorch build.

### 4. Prepare the Data

The datasets are not yet publicly available with the code. Place the low-light data under the repository's `data/` directory using the following structure. File and directory names are case-sensitive on Linux.

```text
GeoPE-GS/
├── data/
│   └── LOM_full/
│       └── buu/
│           ├── colmap_sparse/0/
│           ├── low/
│           ├── high/
│           ├── transforms_train.json
│           └── transforms_test.json
└── examples/
```

`low/` contains the low-light input images, and `high/` contains the reference images for evaluation. To use another low-light scene, replace `buu` in the commands below with the corresponding scene name.

### 5. Train GeoPE-GS on the LOM Low-Light Dataset

The following example trains GeoPE-GS on the `buu` scene:

```bash
cd /root/autodl-tmp/GeoPE-GS/examples
python simple_trainer_ours.py \
  --data_dir ../data/LOM_full/buu \
  --exp_name low \
  --result_dir ../results/LOM/buu_low \
  --max_steps 10000 \
  --batch_size 1
```

Outputs are saved to `--result_dir`, including the `ckpts/`, `stats/`, `renders/`, and `tb/` directories.

### 6. Evaluate an Existing Checkpoint

Use the `best.ckpt` checkpoint generated during training for evaluation and rendering:

```bash
cd /root/autodl-tmp/GeoPE-GS/examples
python simple_trainer_ours.py \
  --data_dir ../data/LOM_full/buu \
  --exp_name low \
  --result_dir ../results/eval/LOM/buu_low \
  --ckpt ../results/LOM/buu_low/ckpts/best.ckpt
```

### 7. Monitor Training with TensorBoard

Run the following command in another terminal during training. For remote access, forward port `6006` to your local machine.

```bash
cd /root/autodl-tmp/GeoPE-GS
tensorboard --logdir results/LOM/buu_low/tb --host 0.0.0.0 



