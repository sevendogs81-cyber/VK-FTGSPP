# 同步多卡训练

## 当前单卡/双卡架构

Production 模型固定为 explicit velocity，采用 replicated synchronous data
parallel。每张 Vulkan GPU 持有完整模型和 Adam state；每 rank 处理不同 view，
在 Adam 前对 raw FP32 gradient 做全局 mean。

```text
P=1   distributed=off（单卡）
P=2   hierarchical_host / pair_exchange_gpu_merge
```

`distributed.backend=auto` 只解析为 `hierarchical_host`。旧 `packed_host`、
`host_mailbox`、`device_group` 不再是公开配置。配置和运行时拒绝 P≥3。

## 启动

```bash
uv run python -m ftgspp.run \
  data=<data_config> model=explicit pipeline=from_points \
  distributed=sync distributed.device_ids='[0,2]' \
  distributed.accumulation.microsteps=1 \
  distributed.world_size_lr_scaling=none \
  run.view_load_mode=lazy
```

`device_ids` 是 Vulkan ordinal。每卡 micro-batch 固定为 1；K 由
`accumulation.microsteps` 选择 1/2/4，全局 batch B=P×K。

双卡默认 `host_collective.bucket_mb=0`：两个 rank 并发用固定 64 MiB payload
实测 8/16/32/64 MiB Vulkan D2H→H2D bucket，并按较慢 rank 的 wall time 选择。
候选明细与选中值写入 `performance_summary.json`；正数是显式覆盖。resume
复用 checkpoint 中的已选值，避免标定噪声导致兼容性变化。三张及以上设备会在配置
解析和运行时直接拒绝，不再存在对应的 auto sentinel 路径。

## 数学边界

对 rank p、local view k 的 data gradient `g[p,k]`：

```text
local[p] = (1/K) * sum_k g[p,k]
global   = (1/P) * sum_p local[p]
Adam(global)
```

K-view accumulation 始终在同一参数点完成，不是 Local SGD。scale/opacity/gate
等共享 regularizer 要么在每 rank identically 加入后再求 mean，要么在 dense
fallback 前显式加一次；两种路径都只保留一个正则项。color L2 在 global
color-gradient mean 后由每 replica 的 color Adam 各加入一次。

relocation statistics 不是 optimizer gradient：`magnitude_sum` 与
`observation_count` 跨 rank 求 sum，rank 0 只生成一次 plan，然后所有 replica
按同一 map/counts 回放。每次 Adam、relocation 和 resume 后
replica 必须一致。

## 双卡 pair exchange

每 rank 的 local raw gradient 留在设备上并作为 accumulator。传输环只下载本
rank message、把 borrowed mapped pointer 发布给 peer，再把 peer bucket 上传到
本卡；GPU kernel 原位执行 `0.5*local + 0.5*peer`。local payload 不会再次 H2D。

三槽 persistent mapped readback/upload ring 有 generation、layout signature、
abort 与 finite checks。报告区分 D2H、peer wait、H2D、GPU merge、optimizer 和
host memcpy；Python 不持有完整 gradient NumPy payload。

## Exact sparse / dense

Gaussian dense ABI 是 mean 3、scale/opacity 4、time/duration 2、velocity/gate
4、SH 48、quaternion 4，共 65 float/Gaussian（SH 按 16-Gaussian float4 布局
padding）。

sorted-sparse record V2 的 word 数为：

```text
18 + 3*(active_degree+1)^2
```

degree 0/1/2/3 分别是 21/30/45/66 words。省略项
必须数学上恒为零。record 按 Gaussian ID 排序，pack/scan/scatter 均在 Vulkan
执行；color correction gradient 始终是小型 dense auxiliary segment。

双卡 `codec_policy=auto_exact` 先交换双方实际 sparse message bytes，再比较：

```text
dense_cost  = 2 * complete_dense_bytes
sparse_cost = local_sparse + peer_sparse + 2 * dense_aux
```

只有 sparse 至少节省 5% 才选择它。双方用 local+peer 对称成本，必然选择同一
codec。`dense` 与 `sorted_sparse` 是显式覆盖。active SH 会随训练 step 增长，
record width 每 generation 精确变化，无需重建最大容量 attachment。

双卡支持 exact sorted-sparse + GPU pair merge；dense fallback 仍保持完全精确。

## Optimizer batch 策略

第一轮系统性能比较固定 P=2,K=1、`world_size_lr_scaling=none`，以免同时改变
优化器。质量实验可选择：

- `none`：历史 LR、beta、epsilon；
- `sqrt_lr`：LR×sqrt(P)；
- `effective_batch_adam`：B=P×K，LR×sqrt(B)，`beta1=0.9^B`、
  `beta2=0.999^B`、`eps=1e-15/sqrt(B)`；
- `steps_div_p`：显式缩短 optimizer steps。

K=2/4 会改变每次 Adam/relocation 所观察的 view 数和梯度噪声，只在通信占比
高时测试，不能凭 views/s 直接设为默认。

## 采样与时钟

默认 `flat_epoch` 把完整 `(frame,camera)` view 表排列为可重建 epoch，用
`step*P*K + rank*K + micro` 消费。四个时钟分别记录：

- `optimizer_steps`：Adam/schedule；
- `microsteps`：local backward 次数；
- `views_seen`：P×K 样本数；
- `wall_seconds`：墙钟。

main phase seed 是 `train.seed`。checkpoint 固化 sampler ABI 与四时钟。

## Checkpoint 与恢复

安全点只在完整 optimizer/relocation generation 后。rank 0 物化一份
explicit-only V5 snapshot；relocation history 已做 global sum，恢复时只把历史
放回 leader，其他 rank 清零，避免下次重复聚合。

默认拒绝 world size、K、collective ABI、codec policy、算法和 native ABI
变化。确需改变 P，必须显式
`distributed.allow_resume_world_size_change=true`；这只确认 global batch/噪声
变化，不放宽其他不兼容状态。

## 性能与质量报告

`performance_summary.json` / `train_metrics.jsonl` 的当前 runner 字段至少覆盖：

- step p50/p95、steps/s、views/s；
- projection、tile count/key、sort、raster forward/backward、loss、dynamic
  backward/fused Adam、relocation；
- `native.replicas[].workload.tile_pair_count`、`native.replicas[].workload.sort_storage_bytes`、`native.replicas[].workload.tile_invariant_status`；
- gradient D2H、peer wait、H2D、merge、host memcpy；
- `native.replicas[].vram_bytes`、`native.replicas[].peak_vram_bytes`、`native.host_staging_bytes` / `native.replicas[].host_staging_bytes`；
- bucket calibration 与实际 codec/payload。

结果对比直接读取 `metrics.jsonl` / `train_metrics.jsonl`，覆盖 PSNR、SSIM、最弱相机和主要 timing。完整验收仍必须分别做 equal-step、equal-view、equal-wall-time；LPIPS/FLIP、flow EPE 和 temporal stability 作为外部可选质量报告补充。双卡
`views/s >= 1.5x` 是目标而非预先结论；完整 100/1k/5k/30k 顺序见
[SERVER_DEBUG.md](SERVER_DEBUG.md)。
