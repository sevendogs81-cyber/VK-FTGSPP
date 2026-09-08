# Imported source revisions

- FTGSPP Python pipeline: copied from the workspace `ftgspp` tree on 2026-07-20; audited source checkout `73797291a1d490b79f27178c9c94519ce9a3ce55`.
- VkSplat Vulkan renderer base: copied from the workspace `vksplat` tree on 2026-07-20; upstream `harry7557558/vksplat` checkout `e26c254938c81ff85998cd357a9e005e255d9b03`. The static trainer, COLMAP loader and default/MCMC densification compatibility path have since been removed.
- The two source trees are vendored here; this repository never imports either sibling at runtime.
- Since 2026-07-22 the Torch/legacy training path (`ftgspp/train`, `backends`, `runner`, etc.) has been removed; training is native Vulkan only (`run.runtime=native`). Torch remains optional for LPIPS, `.pt` convert, and UFM/MEMFOF tooling.
- Since the explicit-only V5 transition, production no longer contains velocity-field/HashGrid/HyperMLP shaders or buffers. Legacy field layout code is retained only under the offline Torch checkpoint compatibility package so old checkpoints can be sampled into explicit velocity.

See `LICENSE`, `LICENSES/`, and `THIRD_PARTY.md` for the applicable notices.
