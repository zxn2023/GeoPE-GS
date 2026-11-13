# GeoPE-GS LOM Low-Light Results

This directory contains the supplied GeoPE-GS experiments for the five LOM low-light scenes. [Download the complete result archive (ZIP)](https://github.com/zxn2023/GeoPE-GS/raw/refs/heads/main/assets/experiments/GeoPE-GS-LOM-lowlight.zip). It preserves all 136 original files: 126 PNG renders, five `cfg.json` configuration files, and five `stats/val.json` files. Final-stage previews for view `0000`, configurations, and statistics are also available below. Model checkpoints are not included.

## Browse by Scene

| Scene | Configuration | Validation statistics | Final-stage previews |
| --- | --- | --- | --- |
| bike | [cfg.json](bike/cfg.json) | [val.json](bike/stats/val.json) | [renders/](bike/renders) |
| buu | [cfg.json](buu/cfg.json) | [val.json](buu/stats/val.json) | [renders/](buu/renders) |
| chair | [cfg.json](chair/cfg.json) | [val.json](chair/stats/val.json) | [renders/](chair/renders) |
| shrub | [cfg.json](shrub/cfg.json) | [val.json](shrub/stats/val.json) | [renders/](shrub/renders) |
| sofa | [cfg.json](sofa/cfg.json) | [val.json](sofa/stats/val.json) | [renders/](sofa/renders) |

## File Names and Contents

The complete archive includes all supplied evaluation stages and views. Extract it to obtain the original `LOM-lowlight/<scene>/` directory structure.

- `cfg.json` records the supplied training configuration. Paths such as `result_dir` describe the original experiment environment.
- `renders/val_<step>_<view>_low.png` is the low-light branch render.
- `renders/val_<step>_<view>_enh.png` is the enhanced branch render of the same view.
- Steps `4999`, `6999`, and `9999` correspond to 5,000, 7,000, and 10,000 training iterations, respectively.
- `stats/val.json` contains the supplied PSNR, SSIM, LPIPS, rendering time (`ellipse_time`), and Gaussian count (`num_GS`).

The `_low.png` files are model outputs; they are not the original dataset input images. See the [project README](../../../README.md#results) for paired previews.

## Dataset Source

Obtain the datasets through the [Luminance-GS dataset setup guide](https://github.com/cuiziteng/Luminance-GS/blob/main/Luminance-GS/README.md). The LOM dataset originates from [Aleth-NeRF](https://github.com/cuiziteng/Aleth-NeRF), as linked by Luminance-GS. Follow the original providers' usage terms and citation instructions.
