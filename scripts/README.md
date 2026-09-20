# User setup

`setup.sh` is the supported install path. It pins Slang **2026.10** and uv
**0.11.29** to the same versions as `.github/workflows/artifacts.yml`, creates
`.venv` with Python 3.12, installs the lightweight Vulkan training runtime,
builds the native wheel from the locked environment, and runs `ftgspp doctor`.

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

The default requires the prepared initialization artifacts named by the
configuration: foreground masks where applicable and a complete point-cloud
set. `--with-birefnet` adds mask generation, `--with-da3` adds point generation,
and `--with-init` installs both for raw-RGB preparation. These options are
composable and keep model packages out of training-only environments. Each run
synchronizes exactly the selected profile; pass both component flags together,
or use `--with-init`, when both must remain installed.

`--skip-apt` skips `apt-get` when every build dependency is already present.
`--skip-doctor` skips the GPU check (useful on a CPU build machine).
`--device N` is the Vulkan id passed to doctor, not an `nvidia-smi` ordinal.

Research helpers under `dev/tools/` are unsupported and are not part of this
install path.
