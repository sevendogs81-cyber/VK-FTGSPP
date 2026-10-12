# Research attribution and third-party software

The FTGSPP portions are governed by the research license in `LICENSE`. The
vendored VkSplat portions retain Apache-2.0 in
`LICENSES/VKSPLAT-APACHE-2.0.txt`. All other packages and assets remain under
their respective upstream terms.

## FreeTimeGS++ paper and citation

FTGSPP refers to **FreeTimeGS++**, the research work on which this community
Vulkan implementation is based:

Lucas Yunkyu Lee, Soonho Kim, Youngwook Kim, Sangmin Kim, and Jaesik Park.
**FreeTimeGS++: Secrets of Dynamic Gaussian Splatting and Their Principles.**
2026. [Paper (arXiv:2605.03337)](https://arxiv.org/abs/2605.03337) ·
[Authors' project page](https://yklcs.com/ftgspp).

When using this work in research, please cite the original paper. The BibTeX
below follows the authors' project page:

```bibtex
@misc{ftgspp,
  title={FreeTimeGS++: Secrets of Dynamic Gaussian Splatting and Their Principles},
  author={Lucas Yunkyu Lee and Soonho Kim and Youngwook Kim and Sangmin Kim and Jaesik Park},
  year={2026},
  eprint={2605.03337},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2605.03337}
}
```

VK-FTGSPP is a modified community implementation. Its supported algorithms
and experimental results are described in this repository; the paper citation
does not imply that every component of the original method is implemented.
Academic attribution and software licensing are separate: the citation does
not replace the license and source notices below.

## Source attribution

The local FTGSPP pipeline import is recorded at source revision
`73797291a1d490b79f27178c9c94519ce9a3ce55` in the
[source history](dev/docs/source-history.md). This is code provenance, not a
paper identifier or a public upstream release link.
The [VkSplat renderer](https://github.com/harry7557558/vksplat) was imported from
revision `e26c254938c81ff85998cd357a9e005e255d9b03`.
Both source trees are vendored. Their license boundaries and notices are
preserved below; development history is in [dev](dev/docs/source-history.md).

## Vendored source

| Component | Role | License |
| --- | --- | --- |
| FreeTimeGS++ / FTGSPP-derived pipeline | Dynamic Gaussian representation and training pipeline, modified for this implementation | Research and educational use; see `LICENSE` and the original-paper citation above |
| VkSplat | Vulkan projection, tile processing, alpha blending, loss/backward runtime | Apache-2.0 |
| GPU radix sort | Bundled radix-sort shaders | MIT; see `native/shaders/radix_sort/LICENSE` |

## Build/runtime software

This is an inventory of major dependencies, not a complete transitive license
manifest. Versions are recorded in `uv.lock` and `native/CMakeLists.txt`;
the license files and notices shipped with each installed dependency apply.

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
| [pycolmap / COLMAP](https://github.com/colmap/colmap) | Camera models and image undistortion | upstream terms, including bundled dependencies |
| [tqdm](https://github.com/tqdm/tqdm) | Training and preparation progress reporting | upstream license notices |
| [websockets](https://github.com/python-websockets/websockets) | Optional browser viewer transport | upstream license notices |
| [TensorBoard](https://github.com/tensorflow/tensorboard) | Optional training event logging | upstream license notices |
| Hugging Face Hub | Optional model-weight retrieval | upstream license notices |
| torchmetrics | optional eval-time LPIPS implementation | Apache-2.0 |
| Depth Anything 3 0.1.1 | pinned optional DA3 depth/pose initialization backend | Apache-2.0 code; checkpoint-specific terms apply |

Installation/build tooling also includes uv, scikit-build-core, CMake and Ninja.
Video import/export invokes an external FFmpeg installation; its licensing
depends on the build and enabled codecs. These tools retain their upstream
notices and are not relicensed by this project.

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
