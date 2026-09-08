# Third-party software

The FTGSPP portions are governed by the research license in `LICENSE`. The
vendored VkSplat portions retain Apache-2.0 in
`LICENSES/VKSPLAT-APACHE-2.0.txt`. All other packages and assets remain under
their respective upstream terms.

## Vendored source

| Component | Role | License |
| --- | --- | --- |
| VkSplat | Vulkan projection, tile/radix processing, alpha blending, loss/backward runtime | Apache-2.0 |

## Build/runtime software

| Component | Role | Typical license |
| --- | --- | --- |
| Vulkan SDK/loader | compute API and shader tools | Apache-2.0 / platform terms |
| Slang | compilation of bundled `.slang` kernels | Apache-2.0 |
| GLM | C++ camera/vector math | MIT |
| pybind11 | native Python boundary | BSD-3-Clause |
| PyTorch | optional only: LPIPS plugin (`.[lpips]`), `.pt` checkpoint convert (`.[convert]`), UFM/MEMFOF tooling (`.[flow]`); not used by native training | BSD-style |
| Hydra/OmegaConf | retained configuration composition | MIT / BSD-3-Clause |
| OpenCV, NumPy, SciPy, imageio | SelfCap data and initialization utilities | upstream terms |
| torchmetrics, flip-evaluator | optional LPIPS / FLIP evaluation extras | Apache-2.0; BSD-3-Clause |

This repository neither imports nor installs `gsplat`, `fused-ssim`, CUDA, or
tiny-cuda-nn for native training. gsplat 1.5.3 remains a source-level
mathematical reference for relocation. Optional legacy `.pt` field conversion
runs outside the production training pipeline and emits explicit-only V5.

## Optional models and datasets

| Component | Upstream | License/terms |
| --- | --- | --- |
| UFM pretrained weights | https://huggingface.co/infinity1096/UFM-Base | CC-BY-NC-4.0 |
| MEMFOF and weights | https://github.com/msu-video-group/memfof | BSD-3-Clause |
| SelfCap Dataset | https://huggingface.co/datasets/zju3dv/SelfCap-Dataset | dataset terms |

UFM is an **offline preprocessing tool** only: run
`python -m ftgspp.data.flow <config> --out ...` to write flow/covisibility
caches. Native initialization reads those NPZ files and never embeds UFM
inference. MEMFOF is an **evaluation/render reference-flow tool** only: its
Gaussian flow feature projection/rasterization is implemented by the bundled
Vulkan kernels, while MEMFOF supplies the image-derived comparison flow and
must be installed separately when that metric is requested.
