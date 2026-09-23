# Third-party software

The FTGSPP portions are governed by the research license in `LICENSE`. The
vendored VkSplat portions retain Apache-2.0 in
`LICENSES/VKSPLAT-APACHE-2.0.txt`. All other packages and assets remain under
their respective upstream terms.

## Source provenance and revisions

- This community edition was assembled from the gated FTGSPP core
  (`9b2321d`) and the shared dual-device collective line (`5caa617`). Hard
  separation is included as an optional post-training phase; ICGS and the
  SAD_mcmc experimental pipeline are intentionally excluded.
- The FTGSPP Python pipeline was copied from the workspace `ftgspp` tree on
  2026-07-20; its audited source checkout was
  `73797291a1d490b79f27178c9c94519ce9a3ce55`.
- The VkSplat Vulkan renderer base was copied from the workspace `vksplat`
  tree on 2026-07-20; the upstream `harry7557558/vksplat` checkout was
  `e26c254938c81ff85998cd357a9e005e255d9b03`. Its static trainer, COLMAP
  loader, and default/MCMC densification compatibility path have since been
  removed.
- These source trees are vendored here; the repository never imports either
  sibling at runtime. Since 2026-07-22, the Torch/legacy training path has
  been removed and training is native Vulkan only. Torch remains optional
  for BiRefNet-HR mask preparation, optional CUDA carving, eval-time LPIPS,
  and DA3 initialization. MEMFOF/reference-flow evaluation remains outside
  this repository.
- Production no longer contains velocity-field/HashGrid/HyperMLP shaders or
  buffers. The V6 community refactor introduced the `src` layout, PEP 517
  native/shader builds, one view per device, per-session pair coordination,
  and a pybind11 3.0.1/NumPy 2 boundary; it also removed the offline
  Torch/field compatibility package.

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

Native training does not use `gsplat`, `fused-ssim`, CUDA, or tiny-cuda-nn.
The optional BiRefNet-HR/DA3 initialization environment may install PyTorch's
bundled CUDA runtime; the default Vulkan trainer remains independent of it. gsplat
1.5.3 remains a source-level mathematical reference for relocation.
Torch/field checkpoint conversion and FLIP evaluation dependencies are not
distributed.

## Optional models and datasets

| Component | Upstream | License/terms |
| --- | --- | --- |
| SelfCap Dataset | https://huggingface.co/datasets/zju3dv/SelfCap-Dataset | dataset terms |
| BiRefNet-HR model | https://huggingface.co/ZhengPeng7/BiRefNet_HR | MIT model repository; review the pinned model card |
| Depth Anything 3 models | https://github.com/ByteDance-Seed/depth-anything-3 | model-card-specific terms (DA3-BASE is Apache-2.0; larger checkpoints may be CC BY-NC 4.0) |

The Vulkan model-flow endpoint is a diagnostic renderer and is not
part of the native training graph or default metrics. MEMFOF and other
reference-flow evaluators are intentionally not bundled; users who need such
comparisons must run them externally.

## Optional desktop viewer

The `FTGSPP_BUILD_VIEWER=ON` build statically includes
[GLFW 3.4](https://github.com/glfw/glfw/tree/3.4) (zlib/libpng license) and
[Dear ImGui v1.91.9b](https://github.com/ocornut/imgui/tree/v1.91.9b) (MIT),
including its GLFW and Vulkan backends. Their license texts are distributed in
`LICENSES/GLFW-ZLIB.txt` and `LICENSES/IMGUI-MIT.txt`. Archive SHA-256 pins are
recorded in `native/CMakeLists.txt`. These dependencies are absent from the
headless build. EasyVolcap informed the interaction design; no EasyVolcap
runtime or source code is bundled.
