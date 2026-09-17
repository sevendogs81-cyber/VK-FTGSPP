# Contributing

Read LICENSE before contributing or redistributing. Contributions must preserve
third-party attribution and the research-only licensing boundary.

Use Python 3.12, NumPy 2 and pybind11 3.0.1. Run CPU tests and, for native
changes, a clean build plus the relevant GPU fixtures. Changes to shader layouts
must update the C++ push constants/descriptors and behavior tests together.

The supported surface is the installed CLI plus the YAML keys documented in
README and `docs/architecture.md`. `tools/` scripts are unsupported research
helpers and may change without notice.

Do not commit generated SPIR-V, extension binaries, datasets, weights, runs, or
the local-only notes `docs/foreground_reconstruction_plan.md`,
`docs/initialization_pipeline_improvement_plan.md`, and `docs/validation.md`.
Include numerical, sampling, checkpoint and performance evidence with algorithm
changes. Keep optional Torch imports out of the ordinary CLI/training path.

Report issues with the command/config, doctor output, package/build versions and
a small reproducer. Remove private paths/data before sharing logs. Do not upload
confidential scenes. GPU CI runs only on explicitly controlled hosts.
