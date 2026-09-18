# Install

The usual path is one script from a clone:

```bash
git clone https://github.com/sevendogs81-cyber/FastGS.git
cd FastGS
bash scripts/setup.sh
source .venv/bin/activate
```

Building the wheel compiles C++ and every shader. That step does **not** need
a GPU. `ftgspp doctor`, `train`, and `eval` do. CUDA ordinals and Vulkan
device indices can differ; `--devices` is the Vulkan id from `doctor`, not
`nvidia-smi`.

The core wheel is Torch-free. Optional extras add Torch only when you ask.

Release wheels are not published to PyPI. YAML examples live in the **clone**;
the wheel only provides the `ftgspp` command.

## What the script does

1. Installs `build-essential` and `libvulkan-dev` with `apt-get` when they are
   missing (unless `--skip-apt`).
2. Downloads Slang **2026.10** to `$HOME/.local/slang-2026.10` and checks the
   same tarball checksum as CI (`beddadd9b481…`).
3. Installs [uv](https://docs.astral.sh/uv/) **0.11.29** if it is not on
   `PATH`.
4. Creates `.venv` with Python **3.12.x**, appends Slang to that venv's
   `activate` script, then `uv pip install .`.
5. Runs `ftgspp doctor --devices 0` unless `--skip-doctor`.

`.venv` is gitignored. Keep it activated for every later `ftgspp` and
extra-install command.

`uv pip install .` needs CMake 3.24+; scikit-build-core can download that
CMake if the system copy is older. GLM is fetched during the build when it is
not installed.

## Manual install

Use this only if you cannot run `scripts/setup.sh`. The pins must stay the
same.

```bash
git clone https://github.com/sevendogs81-cyber/FastGS.git
cd FastGS

sudo apt-get update
sudo apt-get install -y build-essential libvulkan-dev

curl -fsSL https://github.com/shader-slang/slang/releases/download/v2026.10/slang-2026.10-linux-x86_64-glibc-2.27.tar.gz \
  -o /tmp/slang.tgz
echo "beddadd9b48151aa2d7b556056ea02b9826bde59d4d4fa827b33c5184e42a983  /tmp/slang.tgz" | sha256sum -c -
mkdir -p "$HOME/.local/slang-2026.10"
tar -xzf /tmp/slang.tgz -C "$HOME/.local/slang-2026.10"
export PATH="$HOME/.local/slang-2026.10/bin:$PATH"
slangc -v   # expect 2026.10

curl -LsSf https://astral.sh/uv/0.11.29/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv pip install .
ftgspp doctor --devices 0
```

`uv build --python 3.12` followed by
`uv pip install dist/ftgspp_vulkan_community-*.whl` is equivalent: it is how
the artifact workflow records a wheel. Day-to-day installs can skip the
separate `dist/` step.

## Troubleshooting

The JSON `name` from `doctor` should be an NVIDIA GPU, not `llvmpipe`. Use
that object's `device_id` as `--devices`. Doctor opens a real device; it fails
on a CPU-only build machine even when the wheel compiled. `scripts/setup.sh`
treats a doctor failure as a warning so a CPU host can still finish the
install.

If `slangc: not found`, Python/CMake cannot find 3.12, or Vulkan headers are
missing, stop and fix that step. Do not pass `--devices` with two ids: the
first release trains on one GPU.

Optional extras (`[da3]`, `[lpips]`, `[birefnet]`, `[foreground]`,
`[tensorboard]`) install into the same venv:

```bash
uv pip install '.[da3]'
```

See [usage.md](usage.md) for when each extra is required.
