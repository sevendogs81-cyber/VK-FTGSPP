# User setup

`setup.sh` is the supported install path. It pins Slang **2026.10** and uv
**0.11.29** to the same versions as `.github/workflows/artifacts.yml`, creates
`.venv` with Python 3.12, builds the native wheel, and runs `ftgspp doctor`.

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

`--skip-apt` skips `apt-get` when `g++` and Vulkan headers are already present.
`--skip-doctor` skips the GPU check (useful on a CPU build machine).
`--device N` is the Vulkan id passed to doctor, not an `nvidia-smi` ordinal.

Research helpers under `dev/tools/` are unsupported and are not part of this
install path.
