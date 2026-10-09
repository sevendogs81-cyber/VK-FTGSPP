# Setup script

Run the supported installer from the repository root:

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

Use `bash scripts/setup.sh --help` for the current options. Installation profiles
and troubleshooting are in [docs/install.md](../docs/install.md); manual build
commands are in [maintainer notes](../dev/CONTRIBUTING.md). Check the supported GPU
families and `ftgspp doctor` gate in [docs/hardware.md](../docs/hardware.md).
