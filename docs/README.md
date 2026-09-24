# Docs

Start at the repository [README](../README.md). Use this folder when the
landing page is not enough.

| Doc | Audience |
|---|---|
| [hardware.md](hardware.md) | Capability gates, estimated runnable GPU families, validated hardware, and `doctor` interpretation |
| [install.md](install.md) | Manual install, pins, and troubleshooting |
| [usage.md](usage.md) | Prepare / train / eval, foreground, DA3, research YAML |
| [viewer.md](viewer.md) | Local and SSH browser viewers, cameras, playback, screenshots and headless export |
| [performance.md](performance.md) | SelfCap train-only speed tables |
| [architecture.md](architecture.md) | CLI, native, and config boundaries |
| [migration.md](migration.md) | Breaking changes from older FTGSPP layouts |

Local design notes (`foreground_reconstruction_plan.md`,
`initialization_pipeline_improvement_plan.md`, `validation.md`, and the local
viewer plan/performance reports and raw benchmark JSON) are gitignored and are
not part of the published snapshot. Tests and research helpers live under
[dev/](../dev/README.md), not here.
