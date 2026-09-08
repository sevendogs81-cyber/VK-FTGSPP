# 双卡 Host Collective（当前生产契约）

本文档只描述当前接线的 pipeline。两个 Vulkan device 组成一个同步组；三卡及以上
在配置解析和运行时都会拒绝。单卡使用 `distributed.mode=off`，不创建 collective。

## 数据路径

```text
每卡 K∈{1,2,4} 个 view
        ↓
本卡 FP32 梯度累加并除以 K
        ↓
pair_exchange_gpu_merge（Host mailbox transfer ticket）
        ↓
两卡得到相同的全局 mean 梯度
        ↓
每卡执行相同的 Vulkan Adam
```

`distributed.backend=auto` 唯一解析为 `hierarchical_host`；同步模式唯一算法为
`pair_exchange_gpu_merge`。`packed_host`、`host_mailbox`、`device_group`、
`central_hierarchical_reduce_broadcast` 和 `gpu_sharded_reduce_scatter_allgather`
不再是可配置的生产路径。

## 梯度编码

- `codec_policy=auto_exact` 在 dense 与严格 ID 排序的 exact sparse 之间按实际字节成本选择。
- `dense` 强制 dense；`sorted_sparse` 强制 exact sparse。
- Gaussian 使用 active-SH record；颜色/辅助正则保持 dense。
- `bucket_mb=0` 启动时在两卡并发实测 8/16/32/64 MiB，选择较慢一端的最优值。

## 结构与专属阶段

通用 pipeline 只保留单卡和双卡执行、K-view 累加、overflow replay、relocation
统计同步、checkpoint、lookahead，以及 FTGSPP 主训练后的 hard separation 阶段。

## 失败与恢复

pair exchange 使用 generation、transfer ticket 和 native overflow 状态；发生 tile
overflow 时所有 rank 增容后重放同一批 view。任一 rank 失败都会终止当前 generation，
不会静默回退到旧的 CPU 多卡 reducer。
