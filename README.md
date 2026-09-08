# FTGSPP Vulkan Community Edition

这是一个面向开源社区的 FTGSPP Vulkan 独立实现。它把
`ftgspp-vulkan-gated-marginalization` 的 gated FTGSPP 主训练链路与
`ftgspp-vulkan-multi_gpu` 的同步多卡实现整理到一个新库中，默认使用
explicit velocity 和原生 Vulkan/Slang 内核。

社区版只保留容易复现的主线：

- explicit velocity 4D Gaussian 训练、L1+SSIM loss 和 Vulkan Adam；
- gated marginalization、relocation 和 optical-flow feature；
- 可选 per-camera color correction；
- 单卡和双卡同步训练（Host collective / GPU pair merge）；
- 可选 **hard separation**：主训练完成后，把 gate 固化为静态/动态两组，
  自动运行 5000 步轻量微调，并支持 checkpoint 恢复。

ICGS 和 SAD_mcmc 已从这个新库移除，不会参与构建、运行或 checkpoint ABI。
两个源仓库保持独立，本目录是全新的 git 仓库。

## 安装

需要 Python 3.12、C++17、CMake/Ninja、Vulkan SDK 1.2+ 和 `slangc`。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[build]"
python -m ftgspp.build_native       # 编译 C++ 扩展和全部 SPIR-V
python -m scripts.smoke_native --device 0
```

修改 `native/slang/` 后必须重新运行 `python -m ftgspp.build_native`，让
SPIR-V 与 native ABI 同步。运行时会检查所有必需 shader 是否存在。

## 最短运行路径

先准备 SelfCap/COLMAP 数据，再选择一个 `configs/data/*.yaml`：

```bash
# 检查数据和配置
python -m ftgspp.run pipeline=check data=<data_config>

# 单卡训练
python -m ftgspp.run data=<data_config> model=explicit pipeline=from_points

# 双卡同步训练
python -m ftgspp.run data=<data_config> model=explicit pipeline=from_points \
  distributed=sync distributed.backend=hierarchical_host \
  distributed.device_ids='[0,1]'
```

默认配置位于 `configs/train/30k.yaml`。开启 hard separation：

```yaml
train:
  hard_separation: true
```

训练会先执行常规 `iterations` 步，再执行 hard separation 阶段。阶段边界
会写入 checkpoint；中断后设置 `train.resume=true` 可继续，不会重复执行
破坏性的 gate split。最终模型写入 `gaussians.npz`，启用 hard separation 时
另写入 `separated.npz`。

## 配置和工具

- `configs/data/`：数据集、相机和时间戳示例；
- `configs/train/`：训练、采样、relocation 和 hard separation 参数；
- `scripts/smoke_native.py`：单卡 Vulkan 冒烟测试；
- `scripts/smoke_sync_multi_gpu.py`：双卡同步通信测试；
- `ftgspp-convert-checkpoint`：迁移旧 explicit/field checkpoint（可选
  `.[convert]` 依赖）。

更多实现边界见 [VULKANIZATION.md](VULKANIZATION.md)，双卡通信见
[MULTI_GPU.md](MULTI_GPU.md)，运维命令见 [OPERATIONS.md](OPERATIONS.md)。
来源和许可证见 [THIRD_PARTY.md](THIRD_PARTY.md)、
[SOURCE_REVISIONS.md](SOURCE_REVISIONS.md) 和 `LICENSES/`。

## 验证

```bash
pytest -q
python -m scripts.smoke_native --device 0
```
