# User setup

`setup.sh` is the supported install path. It pins Slang **2026.10** and uv
**0.11.29** to the same versions as `.github/workflows/artifacts.yml`, creates
`.venv` with Python 3.12, installs the lightweight Vulkan training runtime,
builds the native wheel from the locked environment, and runs `ftgspp doctor`.
It also installs/checks `ffmpeg`, `ffprobe`, and `libx264` for SelfCap video
preparation and MP4 export. The NVIDIA driver and Vulkan ICD must already be
installed on the host; see [hardware.md](../docs/hardware.md).

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

The default requires the prepared initialization artifacts named by the
configuration: foreground masks where applicable and a complete point-cloud
set. `--with-birefnet` adds mask generation, `--with-da3` adds point generation,
and `--with-init` installs both for raw-RGB preparation. `--with-lpips` adds
TorchMetrics for optional locked final-test scoring; it does not add a training
metric. These options are composable and keep model packages out of
training-only environments. Each run synchronizes exactly the selected profile;
pass all needed flags together on later setup runs.

`--with-viewer` builds the optional GLFW/ImGui X11 desktop viewer and installs
its development headers. See [viewer.md](../docs/viewer.md) for usage. The
headless `ftgspp render` command works without this option.
`--with-remote` installs the lightweight `websockets` extra for
`ftgspp viewer --remote` on an SSH GPU server; it needs no server display or
desktop build. `--with-tensorboard` installs optional research visualization
tools. All setup flags can be combined in one first-install command.

`--skip-apt` skips `apt-get` when system build and FFmpeg tools are already
present. CMake can fetch GLM if it is not installed system-wide.
`--skip-doctor` skips the GPU check (useful on a CPU build machine).
`--device N` is the Vulkan id passed to doctor, not an `nvidia-smi` ordinal.

Research helpers under `dev/tools/` are unsupported and are not part of this
install path.
