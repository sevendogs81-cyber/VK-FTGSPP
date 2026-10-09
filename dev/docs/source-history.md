# Source and implementation history

Current licensing notices are in [THIRD_PARTY.md](../../THIRD_PARTY.md).

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

Native training does not use `gsplat`, `fused-ssim`, CUDA, or tiny-cuda-nn.
The optional BiRefNet-HR/DA3 initialization environment may install PyTorch's
bundled CUDA runtime; the default Vulkan trainer remains independent of it. gsplat
1.5.3 remains a source-level mathematical reference for relocation.
Torch/field checkpoint conversion and FLIP evaluation dependencies are not
distributed.

The Vulkan model-flow endpoint is a diagnostic renderer and is not
part of the native training graph or default metrics. MEMFOF and other
reference-flow evaluators are intentionally not bundled; users who need such
comparisons must run them externally.
