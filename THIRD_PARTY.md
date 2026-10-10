# Third-party software

The FTGSPP portions are governed by the research license in `LICENSE`. The
vendored VkSplat portions retain Apache-2.0 in
`LICENSES/VKSPLAT-APACHE-2.0.txt`. All other packages and assets remain under
their respective upstream terms.

## Source attribution

The FTGSPP pipeline was imported from source revision
`73797291a1d490b79f27178c9c94519ce9a3ce55`; the VkSplat renderer was imported from
`harry7557558/vksplat` revision `e26c254938c81ff85998cd357a9e005e255d9b03`.
Both source trees are vendored. Their license boundaries and notices are
preserved below; development history is in [dev](dev/docs/source-history.md).

## Vendored source

| Component | Role | License |
| --- | --- | --- |
| VkSplat | Vulkan projection, tile processing, alpha blending, loss/backward runtime | Apache-2.0 |
| GPU radix sort | Bundled radix-sort shaders | MIT; see `native/shaders/radix_sort/LICENSE` |

## Build/runtime software

| Component | Role | Typical license |
| --- | --- | --- |
| Vulkan SDK/loader | compute API and shader tools | Apache-2.0 / platform terms |
| Slang | compilation of bundled `.slang` kernels | Apache-2.0 |
| GLM | C++ camera/vector math | MIT option; `LICENSES/GLM.txt` |
| pybind11 | native Python boundary | BSD-3-Clause; `LICENSES/PYBIND11-BSD-3-CLAUSE.txt` |
| PyTorch/TorchVision | BiRefNet-HR masks, optional CUDA carving, eval-time LPIPS, and DA3; not used by native training | BSD-style |
| Transformers, Safetensors, timm, Kornia, einops | BiRefNet-HR model loading and inference | upstream terms |
| Addict, evo, ImageIO/MoviePy, Pillow, plyfile, trimesh | pinned DA3 inference API support | upstream terms |
| OmegaConf | structured YAML configuration | BSD-3-Clause |
| OpenCV, NumPy, SciPy | SelfCap data and initialization utilities | upstream terms |
| torchmetrics | optional eval-time LPIPS implementation | Apache-2.0 |
| Depth Anything 3 0.1.1 | pinned optional DA3 depth/pose initialization backend | Apache-2.0 code; checkpoint-specific terms apply |

## Optional models and datasets

| Component | Upstream | License/terms |
| --- | --- | --- |
| SelfCap Dataset | https://huggingface.co/datasets/zju3dv/SelfCap-Dataset | dataset terms |
| BiRefNet-HR model | https://huggingface.co/ZhengPeng7/BiRefNet_HR | MIT model repository; review the pinned model card |
| Depth Anything 3 models | https://github.com/ByteDance-Seed/depth-anything-3 | model-card-specific terms (DA3-BASE is Apache-2.0; larger checkpoints may be CC BY-NC 4.0) |

## Optional desktop viewer

The `FTGSPP_BUILD_VIEWER=ON` build statically includes
[GLFW 3.4](https://github.com/glfw/glfw/tree/3.4) (zlib/libpng license) and
[Dear ImGui v1.91.9b](https://github.com/ocornut/imgui/tree/v1.91.9b) (MIT),
including its GLFW and Vulkan backends. Their license texts are distributed in
`LICENSES/GLFW-ZLIB.txt` and `LICENSES/IMGUI-MIT.txt`. Archive SHA-256 pins are
recorded in `native/CMakeLists.txt`. These dependencies are absent from the
headless build. EasyVolcap informed the interaction design; no EasyVolcap
runtime or source code is bundled.
