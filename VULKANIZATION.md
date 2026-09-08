# Vulkan 化边界与实现状态

Production 只支持 explicit velocity；field/HashGrid/HyperMLP 不属于 native
训练状态、shader、buffer 或 checkpoint ABI。旧 field 只存在于离线转换兼容
层，必须先采样成 explicit V5。

## 单卡训练 step

```text
DynamicProjection
  -> exact scan-line tile count
  -> prefix sum / exact key emit / radix sort / tile ranges
  -> raster forward
  -> optional color-correction forward
  -> L1 + valid SSIM forward/backward
  -> optional color-correction raw backward
  -> adaptive raster backward
  -> FusedDynamicBackwardAdam
  -> color Adam
```

DynamicProjection 直接读取 base mean、log-scale/opacity、time/duration、
velocity/gate、SH、quaternion 与 timestamp；普通 RGB train/eval 不再写完整
materialized Gaussian。独立 materialize API 仅用于 export、离线检查和需要两个
时间端点的 flow。

`FusedDynamicBackwardAdam` 直接消费 projection/raster gradients，执行动态
chain rule、regularizer、relocation statistics 和 Adam。单卡普通训练不分配
Gaussian raw-gradient/collective buffers。性能计时使用独立
`FusedDynamicBackwardAdam` stage，不把同一个 dispatch 重复算作 backward 和
Adam。

## 同步多卡 step

```text
每 rank: projection/raster/loss -> split raw backward
K views: 同一参数点累积并除以 K
P ranks: Adam 前 exact FP32 gradient mean
每 rank: explicit Adam -> color Adam
relocation: statistics sum -> rank 0 plan -> broadcast/replay
```

双卡使用 device-resident local raw gradient 作为 accumulator，只上传 peer
bucket；同步模式不支持三卡及以上。Gaussian 可按 visible ID 做
exact sorted-sparse，color gradient 保持 dense。双卡
`auto_exact` 每 step 交换实际 message bytes，并在至少有 5% 估计收益时选择
sparse；record V2 只传当前 active SH degree。

## 精确 tile culling

`TILE_CULLING_MODE=2` 的 count 与 emit 共用
`ellipse_scanline_tile_span`。投影被裁剪或 exact overlap 为零时，radius、rect、
xy/depth/inverse-covariance/RGB 均清零。key shader 记录 GPU invariant bits：

- bit 0：`emitted_count != tiles_touched`；
- bit 1：tile key 超出真实 grid。

状态普通 step 留在设备上，performance snapshot 才读取，避免每 step D2H
stall。

## Adaptive raster backward

候选实现按一个 renderer/device 实例内的
`(resolution, tile-density bucket, active-SH)` 建立独立 Thompson warm-up。
每个候选至少两个观测、总计至少 16 次更新后冻结该 bucket 的均值最快实现；
之后不再随机探索。device UUID 由 Vulkan renderer 实例天然隔离。

## Loss 与可选能力

非日志 step 仍执行完整 L1/SSIM/color/regularizer 梯度，但跳过
`loss_accum` clear、scalar reducer，以及 regularizer/color scalar atomics。
日志 step 返回 `l1`、`ssim`、`reg_opacity`、`reg_scale`、`gate`、`color_cc`。

- Color correction：Vulkan affine forward/backward/Adam；
- LPIPS：可选 Torch 主机插件，仅单卡；
- UFM：离线预处理并为可选速度初始化提供 flow/covis cache，不在 native train graph；
- model flow：Vulkan materialize endpoints + projected feature raster，仅按需用于运动诊断/可视化；
- MEMFOF/reference-flow：不内置，任何 flow EPE 由外部工具完成；
- gated occupancy：活动性看 \(\phi=g+(1-g)a(t)\)，persistents 不被短 duration 硬裁。

## Optimizer 与 ABI

默认 Adam 语义保持 B=1。选择
`world_size_lr_scaling=effective_batch_adam` 时，B=P×K：

```text
LR *= sqrt(B)
beta1 = 0.9^B
beta2 = 0.999^B
eps = 1e-15/sqrt(B)
```

checkpoint 固定 explicit-only V5，native ABI 为
`ftgspp-vulkan-community-explicit-v5-abi-v1`。旧 explicit V4 必须离线迁移；旧 field
必须离线 sample 为 explicit。源码、SPIR-V 与 native module 必须一起重建。

## 验证状态

源码层 contract、Python tests、全量 Slang→SPIR-V 与 C++ module build 可在无
NVIDIA GPU 的开发机完成。单卡/双卡 smoke、100/1k/5k 质量趋势和最终 30k
性能必须在满足 Vulkan subgroup/float-atomic 要求的目标 GPU 上执行；命令与
门限见 [SERVER_DEBUG.md](SERVER_DEBUG.md)。
