# Hardware compatibility

VK-FTGSPP does not support every NVIDIA GPU. The first-release hardware contract
is capability-based because the Vulkan features exposed by the installed driver
matter as much as the marketing model name.

## Supported boundary

The supported training, evaluation, headless rendering, and browser-viewer path
requires all of the following:

- Linux x86_64 with glibc 2.28 or newer (2.31 or newer when installing the
  optional TensorBoard tools);
- one physical NVIDIA GPU selected by its Vulkan device ID;
- a driver exposing Vulkan 1.2 and the features checked by `doctor`;
- a device reported as **`VIABLE`** by `ftgspp doctor`; and
- enough device memory for the selected Gaussian count, image resolution,
  optimizer state, and rasterizer workspace.

`POSSIBLY VIABLE` is an engineering fallback classification. It does not qualify
the device for supported FTGSPP training. `NOT VIABLE`, CPU Vulkan
implementations such as `llvmpipe`, non-NVIDIA GPUs, Windows, non-x86_64 Linux,
and a user-facing multi-GPU training configuration are outside the first-release
support boundary.

## Model-level runnable estimate

The following model families are part of the documented runnable range even
though most have not been exercised in the maintained acceptance machine. Here,
**runnable (estimated)** means that the architecture and NVIDIA Linux driver are
expected to satisfy the native capability checks. A listed GPU counts as runnable when
it has at least 4 GB of VRAM, the proprietary driver exposes it as a Vulkan
physical device, and `ftgspp doctor` reports it as `VIABLE`.

| Architecture | Runnable models covered by this estimate | Status |
|---|---|---|
| Maxwell | GeForce GTX 950/960 4 GB variants, GTX 970, GTX 980, GTX 980 Ti, TITAN X, and matching mobile variants with at least 4 GB | **Runnable (estimated)** |
| Pascal | GeForce GTX 1050 Ti, GTX 1060 5/6 GB, GTX 1070/1070 Ti, GTX 1080/1080 Ti, TITAN X, TITAN Xp, and matching mobile variants with at least 4 GB | **Runnable (estimated)** |
| Turing, GTX | GeForce GTX 1630, GTX 1650/1650 SUPER/1650 Ti, GTX 1660/1660 SUPER/1660 Ti, including mobile variants | **Runnable (estimated)** |
| Turing, RTX | GeForce RTX 2060/2070/2080 families including SUPER, Ti, and mobile variants; TITAN RTX | **Runnable (estimated)** |
| Volta | TITAN V and Quadro GV100 | **Runnable (estimated)** |
| Ampere | GeForce RTX 2050 and RTX 3050 through RTX 3090 Ti, including Ti and Laptop GPU variants | **Runnable (estimated)** |
| Ada Lovelace | GeForce RTX 4050 Laptop GPU and RTX 4060 through RTX 4090, including SUPER, Ti, D, and Laptop GPU variants | **Runnable (estimated)**; RTX 4090 is validated |
| Blackwell | GeForce RTX 5050 through RTX 5090, including Ti, D, and Laptop GPU variants | **Runnable (estimated)** |
| Professional equivalents | Quadro M and P models with at least 4 GB, Quadro RTX, RTX A-series, RTX Ada Generation, and RTX PRO Blackwell cards | **Runnable (estimated)** |

Laptop variants are included for training, evaluation, headless export, and the
SSH browser viewer when the NVIDIA discrete GPU is selected. The local desktop
viewer also needs that GPU to own a presentation-capable queue for the active
display; hybrid/Optimus routing can prevent this even when compute passes.

For Maxwell, Pascal, and Volta products, use the NVIDIA R580 Linux driver branch
or another branch that still lists the exact PCI device. Turing and newer cards
may use R580 or a newer branch that supports the model. This estimate is based
on NVIDIA's [R580 supported-chip list](https://download.nvidia.com/XFree86/Linux-x86_64/580.178.04/README/supportedchips.html),
the NVIDIA [Vulkan driver history](https://developer.nvidia.com/vulkan-driver),
and the repository's native checks. NVIDIA's history records full Vulkan 1.2
support before adding `VK_EXT_shader_atomic_float`; current FTGSPP additionally
checks the required float32 atomic-add feature at startup.

The estimate does not cover GeForce cards below 4 GB, GT/MX low-end parts,
Kepler or older GPUs, Jetson/aarch64 systems, virtual GPUs, or compute-only data
center accelerators such as A100/H100/B100. Those products may differ in memory,
driver branch, Vulkan ICD exposure, or platform architecture enough that a
model-family promise would be misleading. They remain outside the documented
runnable range even if a particular machine happens to initialize.

## Validated hardware

GPU acceptance has been exercised on NVIDIA GeForce RTX 4090 24 GB devices.
Training, evaluation, rendering and browser viewing have been exercised on
this hardware.

The estimated families above are accepted as runnable hardware, while
"validated" means that the repository's GPU acceptance suite has actually run
on that model. Performance numbers in [performance.md](performance.md) are RTX
4090 measurements and must not be projected onto another GPU.

## Check a machine

Install the core environment, activate it, and run:

```bash
ftgspp doctor --devices 0
```

Use the Vulkan ID printed in the device list. It can differ from the CUDA or
`nvidia-smi` ordinal. A suitable result has all of these properties:

1. The selected device name is the intended NVIDIA GPU, rather than
   `llvmpipe` or another adapter.
2. Its classification is `VIABLE`.

Passing `doctor` does not reserve the memory required by a real run. Peak memory
depends on the model budget, input resolution, projected tile load, training
state, and viewer/export resolution. The practical estimates are:

| VRAM | Expected scope |
|---:|---|
| 4–5 GB | Small core-Vulkan runs after reducing `data.scale` and `init.num_gaussians`; optional Torch initialization may need CPU execution |
| 6–7 GB | Small to medium scenes, with workload tuning still expected |
| 8–11 GB | Practical baseline for ordinary training and viewer use |
| 12 GB or more | Preferred starting range for the default 2-million-Gaussian budget |
| 24 GB | Capacity used by the maintained RTX 4090 acceptance environment |

These bands are planning estimates rather than allocation guarantees. Treat an
out-of-memory failure as a workload-capacity failure even when the feature gate
passes.

Optional Torch components have a separate device boundary. BiRefNet-HR, DA3,
CUDA point filtering, and LPIPS may use CUDA and its ordinal mapping; their
availability does not change whether the core Vulkan device is compatible.
The locked Torch 2.6 / CUDA 12.4 stack is tested on RTX 4090. The estimated
Blackwell support in the table above applies to Vulkan, not this older CUDA
wheel: newer GPU architectures need a separately validated Torch build or CPU
preparation (`--device cpu`, and `init.da3.device: cpu`). Driver version alone
does not establish CUDA wheel architecture support.
