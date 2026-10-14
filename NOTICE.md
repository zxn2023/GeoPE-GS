# Code provenance

GeoPE-GS is developed from [Luminance-GS: Adapting 3D Gaussian Splatting to Challenging Lighting Conditions with View-Adaptive Curve Adjustment](https://github.com/cuiziteng/Luminance-GS) by Ziteng Cui, Xuangeng Chu, and Tatsuya Harada. Training structure, dataset handling, and related components originate from that project. The [upstream source tree](https://github.com/cuiziteng/Luminance-GS/blob/main/Luminance-GS/LICENSE) includes the Apache License 2.0.

The repository includes code from [gsplat](https://github.com/nerfstudio-project/gsplat) by the Nerfstudio team. The `gsplat/` package and CUDA rasterization sources are retained to support the dual-branch implementation inherited from Luminance-GS. The Apache 2.0 `LICENSE` file was present in the provided source tree and is preserved here.

GeoPE-GS work in the provided project includes geometry-conditioned enhancement components in `examples/simple_trainer_ours.py` and `examples/utils.py`. The `rasterization_dual` implementation in `gsplat/rendering_double.py` already exists in Luminance-GS; this public copy only normalizes its package imports. A full line-by-line comparison of all model changes has not yet been completed.

Please retain upstream copyright and attribution notices when modifying or redistributing these components. GeoPE-GS datasets and result images are distributed separately and may have their own terms.
