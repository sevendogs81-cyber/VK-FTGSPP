# Native hardware validation

For user hardware selection and doctor instructions, see
[hardware compatibility](../../docs/hardware.md).

## Native capability gates

The current native build prints its exact compiled requirements before listing
devices. With the shipped 16×16 tile configuration, the full gate is:

| Capability | Current requirement | Why it is needed |
|---|---:|---|
| Vulkan API | 1.2 or newer | The native instance and device feature chain use Vulkan 1.2 |
| Compute queue | Compute plus timestamp queries | Submission and GPU timing |
| Subgroup width | At least 32; shipped shaders use 32 lanes | Rasterization, scans, optimizer, and relocation kernels |
| Workgroup counts | At least `[786432, 256, 1]` | Worst-case dispatch ranges in the current renderer |
| Workgroup sizes | At least `[1024, 16, 1]` | Compiled compute local sizes |
| Compute shared memory | At least 21,712 bytes | Raster backward, SSIM, and radix-sort kernels |
| Push constants | At least 224 bytes | Largest native shader uniform block |
| `shaderInt16` | Required | Native shader arithmetic and storage paths |
| `shaderInt64` | Required | Exact integer construction and training kernels |
| `shaderBufferFloat32AtomicAdd` | Required | FTGSPP loss, gradient, and optimizer kernels |

Timeline semaphores improve the submission path but are not a hard compatibility
gate; the runtime has a fence fallback. The local desktop viewer additionally
needs one queue family that supports compute, graphics, timestamp queries, and
presentation to the active display, plus swapchain and
`shaderDrawParameters` support. The SSH browser viewer and `ftgspp render` do
not require a server display.

These are execution requirements, not a performance recommendation. A device
can pass every feature check and still be impractical for a large scene.

## Interpreting individual limits

- `I16`, `I64`, and `F32Atomic` are all present.
- The reported subgroup, group counts, group sizes, shared memory, and push
   constant limits meet the requirement line printed above the device list.

For example, the maintained RTX 4090 environment reports a 32-lane subgroup,
`[2147483647, 65535, 65535]` group counts,
`[1024, 1024, 64]` group sizes, 49,152 bytes of compute shared memory, and all
three required shader features.
