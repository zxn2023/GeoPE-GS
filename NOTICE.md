# Code provenance

GeoPE-GS is developed from [Luminance-GS: Adapting 3D Gaussian Splatting to Challenging Lighting Conditions with View-Adaptive Curve Adjustment](https://github.com/cuiziteng/Luminance-GS) by Ziteng Cui, Xuangeng Chu, and Tatsuya Harada. Training structure, dataset handling, and related components originate from that project. The [upstream source tree](https://github.com/cuiziteng/Luminance-GS/blob/main/Luminance-GS/LICENSE) includes the Apache License 2.0.

The repository includes code from [gsplat](https://github.com/nerfstudio-project/gsplat) by the Nerfstudio team. The `gsplat/` package and CUDA rasterization sources are retained to support the dual-branch implementation inherited from Luminance-GS. The Apache 2.0 `LICENSE` file was present in the provided source tree and is preserved here.

Compared with the current Luminance-GS source tree, this release modifies `examples/datasets/colmap.py`, `examples/datasets/colmap_mip360.py`, `examples/losses.py`, `examples/simple_trainer_ours.py`, `examples/tools.py`, `examples/tools_decompose.py`, `examples/utils.py`, and `gsplat/rendering_double.py`. These modified files carry notices at the top. GeoPE-GS work in the provided project includes geometry-conditioned enhancement components in `examples/simple_trainer_ours.py` and `examples/utils.py`. The `rasterization_dual` implementation in `gsplat/rendering_double.py` already exists in Luminance-GS; this public copy normalizes its package imports. This file comparison identifies changed files but does not establish authorship of individual lines.

The GLM dependency is referenced as a Git submodule at the commit used by gsplat v1.0.0. GLM's own license and notices remain in that upstream submodule.

Please retain upstream copyright and attribution notices when modifying or redistributing these components. GeoPE-GS datasets and result images are distributed separately and may have their own terms.
