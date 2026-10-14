# GeoPE-GS: Geometry-Conditioned Pseudo-Enhancement for Low-Light 3D Gaussian Splatting

Code release for **GeoPE-GS: Geometry-Conditioned Pseudo-Enhancement for Low-Light 3D Gaussian Splatting**.

GeoPE-GS builds on [Luminance-GS](https://github.com/cuiziteng/Luminance-GS) and includes a geometry encoder that uses rendered depth and normals to condition the image enhancement branch. The repository also includes a local [gsplat](https://github.com/nerfstudio-project/gsplat) implementation. Dual-branch rasterization is inherited from Luminance-GS. The main training entry point is [`examples/simple_trainer_ours.py`](examples/simple_trainer_ours.py).

> **Release status:** Code is included. GeoPE-GS datasets and experimental results are planned for a separate release. No pretrained GeoPE-GS checkpoint is currently available.

## Installation

The environment follows the Luminance-GS setup. A Linux machine with an NVIDIA GPU and a compatible CUDA toolkit (including `nvcc`) is recommended. The original setup used Python 3.9+, PyTorch 1.13.1, gsplat 1.0, and NumPy 1.24.3. Install a CUDA-enabled PyTorch build appropriate for your system before the commands below.

Clone with the GLM submodule required by the local CUDA extension:

```bash
git clone --recurse-submodules https://github.com/zxn2023/GeoPE-GS.git
cd GeoPE-GS
```

If you already cloned the repository, run `git submodule update --init --recursive` from its root. Then install the Python dependencies:

```bash
python -m pip install numpy==1.24.3
python -m pip install -r examples/requirements.txt
python -m pip install --no-build-isolation -e .
```

The final command installs the **local** `gsplat/` code and builds its CUDA extension. Do not install a second `gsplat` package on top of it. CUDA compilation requires a working compiler toolchain and may take several minutes. The exact environment and GPU run for this cleaned copy have not yet been independently verified.

## Data layout

Place datasets under `data/` (ignored by Git). The current code supports the LOM and MipNeRF360-varying layouts used by Luminance-GS:

```text
GeoPE-GS/
├── data/
│   ├── LOM_full/
│   │   └── buu/                 # example LOM scene
│   └── NeRF_360/
│       └── bicycle/             # example MipNeRF360-varying scene
│           ├── images/
│           ├── images_8/
│           ├── images_8_variance/
│           └── sparse/
└── examples/
```

For the baseline datasets and preparation details, see [Luminance-GS](https://github.com/cuiziteng/Luminance-GS/blob/main/Luminance-GS/README.md). GeoPE-GS-specific data will be linked when released. Dataset licenses and citations remain with their respective providers.

## Training

Run commands from `examples/` so the relative data paths resolve as shown:

```bash
cd examples

# Low-light LOM example
python simple_trainer_ours.py --data_dir ../data/LOM_full/buu --exp_name low --result_dir ../results/low/buu

# Overexposure LOM example
python simple_trainer_ours.py --data_dir ../data/LOM_full/buu --exp_name over_exp --result_dir ../results/over_exp/buu

# Varying-exposure MipNeRF360 example
python simple_trainer_ours.py --data_dir ../data/NeRF_360/bicycle --exp_name variance --data_factor 8 --result_dir ../results/variance/bicycle
```

Checkpoints, renders, TensorBoard logs, and debug images are written beneath `--result_dir`. These generated files are excluded by `.gitignore`. To see the complete set of options:

```bash
python simple_trainer_ours.py --help
```

## Evaluation

After training, pass a saved checkpoint to the same entry point. For example:

```bash
python simple_trainer_ours.py --data_dir ../data/LOM_full/buu --exp_name low --result_dir ../results/eval/buu --ckpt ../results/low/buu/ckpts/best.ckpt
```

The `--ckpt` branch loads the checkpoint, evaluates it, and renders a trajectory. No pretrained GeoPE-GS weights are included; training produces the checkpoint used here.

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
