# 服务器编译与验收

本清单只验收 production 的 explicit-velocity V5 路径。旧 field checkpoint
只能先离线转换，不能在服务器训练命令中启用 field/HashGrid/HyperMLP。

## 1. 设备与依赖

要求 Python 3.12、C++17、CMake/Ninja、Vulkan SDK 1.2+、`slangc`、
`glslc`。训练设备还必须支持 subgroup、16/64-bit shader arithmetic、至少
192-byte push constants，以及 `shaderBufferFloat32AtomicAdd`。CAS fallback 只
覆盖继承自 VkSplat 的少量路径，不能替代 FTGSPP 训练所需的 float atomic。

```bash
vulkaninfo --summary
nvidia-smi
uv sync --extra build
```

设备号是 Vulkan physical-device ordinal，不是 CUDA ordinal。若枚举中夹有
llvmpipe，两张卡常见为 `[0,2]`。

## 2. 静态测试、shader 与 native module

```bash
uv run pytest -q
uv run python native/compile_shaders.py --force \
  --slangc /path/to/slangc --glslc /path/to/glslc
uv run python -m ftgspp.build_native --skip-shaders
git diff --check
```

必须看到全部 shader 编译成功，并确认以下关键产物来自同一次源码构建：

```text
projection_forward.spv
ftgspp_dynamic_projection.spv
generate_keys.spv
rasterize_backward_0..4.spv
ftgspp_fused_dynamic_backward_adam.spv
ftgspp_fused_backward_optimizer.spv
ftgspp_explicit_adam.spv
ftgspp_sparse_{flag,pack,prepare,scatter}.spv
```

当前 native ABI 为 `ftgspp-vulkan-community-explicit-v5-abi-v1`。不要把旧 `.so` 与新
SPIR-V 混用。

## 3. 无数据 GPU smoke

```bash
uv run python -m scripts.smoke_native --device <vk0>
uv run python -m scripts.smoke_sync_multi_gpu --devices <vk0> <vk1>
```

单卡 smoke 覆盖 native-owned image、loss reduction、color correction、flow
render 和 relocation replay 合约；同步 smoke 覆盖双卡初始化和 Host 中转
通信入口。训练结束的 performance snapshot 会读取 GPU
`tile_invariant_status`，非零必须视为失败。

## 4. 数据与 100-step 门禁

先只检查数据：

```bash
uv run python -m ftgspp.run data=<data_config> pipeline=check
```

然后固定 explicit、K=1、无 large-batch 补偿：

```bash
export FTGSPP_TRAIN_LOG_INTERVAL=10
export FTGSPP_TRAIN_EVAL_INTERVAL=50

uv run python -m ftgspp.run \
  data=<data_config> model=explicit pipeline=from_points train=30k \
  train.iterations=100 train.relocation.start=10 \
  train.relocation.stop=100 train.relocation.every=20 \
  run.output_path=/path/to/runs/single_100

uv run python -m ftgspp.run \
  data=<data_config> model=explicit pipeline=from_points train=30k \
  train.iterations=100 train.relocation.start=10 \
  train.relocation.stop=100 train.relocation.every=20 \
  distributed=sync distributed.device_ids='[<vk0>,<vk1>]' \
  distributed.accumulation.microsteps=1 \
  distributed.world_size_lr_scaling=none \
  run.view_load_mode=lazy run.output_path=/path/to/runs/sync2_100
```

双卡默认 `host_collective.bucket_mb=0`，启动时会在两 rank 上并发测量
8/16/32/64 MiB 固定载荷 D2H→H2D，并按较慢 rank 选择。实际候选、耗时与
选中值必须出现在 `performance_summary.json`；不能用理论值冒充实测。

检查：

- 每阶段 finite，loss 分项和等于 `total_loss`；
- 非日志 step 没有 scalar loss readback；
- 两卡每次 Adam/relocation 后 replica digest 一致；
- `FusedDynamicBackwardAdam` 只出现在单卡 timing；双卡保留
  `DynamicBackward` + `ExplicitAdam`；
- `tile_invariant_status=0`；
- 无持续显存增长或周期性 peer-wait 尖峰。

## 5. 1k 与 5k 门禁

100-step 通过后，把 `train.iterations` 分别改为 1000 和 5000；测试集必须
覆盖 ordinary、temporal spike 与边界/极细 ellipse view。至少运行：

1. 单卡 fused；
2. 双卡 P=2,K=1, `world_size_lr_scaling=none`；
3. 仅在通信占比明显时再测 K=2/4；
4. effective-batch 候选使用
   `distributed.world_size_lr_scaling=effective_batch_adam`。

effective-batch 的契约是 B=P×K、LR×sqrt(B)、`beta1=0.9^B`、
`beta2=0.999^B`、`eps=1e-15/sqrt(B)`。它是质量实验，不是吞吐默认值。

比较时直接读取对应 run 目录下的 `metrics.jsonl` / `train_metrics.jsonl`，
对照指定 step 的 PSNR、SSIM、最弱相机与主要 timing。完整验收报告仍需分别
覆盖 equal-step、equal-view、equal-wall-time；LPIPS/FLIP（若装了可选依赖）、
flow EPE 与 temporal stability 属于外部可选质量指标，需单独生成并随报告
归档。建议硬门限：PSNR 不低于基线 0.05 dB 以上、SSIM 下降不超过 0.001、
LPIPS 不变差超过 0.002。

## 6. Resume

从 `ckpt/latest.json` 恢复：

```bash
uv run python -m ftgspp.run data=<data_config> model=explicit \
  train.resume=true run.output_path=/path/to/existing/run
```

检查 optimizer clock、sampler、relocation history、
replica digest 和选中 bucket 均延续。自动 bucket 的 resume 复用 checkpoint
中的实测值，不因重新标定噪声制造兼容性失败。hard-separation checkpoint
支持恢复，恢复时不会重复执行 gate split。

旧 checkpoint 只允许：

```bash
ftgspp-convert-checkpoint old-field.pt explicit-v5.npz --field-mode sample
ftgspp-convert-checkpoint explicit-v4.npz explicit-v5.npz --field-mode reject
```

## 7. 最终 30k

只有组件、100-step、1k 和 5k 趋势全部通过后，才执行一次最终 30k。最终
报告至少包含 step p50/p95、steps/s、views/s、各 native stage、`native.replicas[].workload.tile_pair_count`、`native.replicas[].workload.sort_storage_bytes`、D2H/peer-wait/H2D/merge、`native.replicas[].vram_bytes`/`native.replicas[].peak_vram_bytes`/`native.host_staging_bytes` 与上述质量指标。

双卡目标 `views/s >= 1.5x` 优化后单卡只是验收目标，不得在未跑目标服务器
实测时写成结论；论文 3.3x/33% 仅作背景参考。
