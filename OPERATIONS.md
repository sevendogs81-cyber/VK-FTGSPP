# FTGSPP Vulkan 操作手册

本文件只维护可执行命令、配置覆盖项、数据布局和输出产物。算法说明链接到
[VULKANIZATION.md](VULKANIZATION.md)，多卡语义链接到
[MULTI_GPU.md](MULTI_GPU.md)，不在此复制推导和容量表。

面向在本机/服务器上**从零准备 SelfCap 数据并跑通 native 训练**的完整说明。  
实现边界见 [VULKANIZATION.md](VULKANIZATION.md)，环境验收见 [SERVER_DEBUG.md](SERVER_DEBUG.md)，许可证见 [THIRD_PARTY.md](THIRD_PARTY.md)。

**本仓库只支持 `run.runtime=native`（Vulkan）。** 默认训练路径不依赖 PyTorch/CUDA；LPIPS、checkpoint 转换、UFM/MEMFOF 为可选旁路。

---

## 1. 快速对照

| 目标 | 命令要点 |
|------|----------|
| 构建 + smoke | `pip install -e ".[build]"` → `python -m ftgspp.build_native` → `scripts/smoke_native.py` |
| 准备 SelfCap | `ftgspp-prepare-selfcap ...` |
| 检查配置/数据 | `python -m ftgspp.run data=<name> pipeline=check` |
| Explicit 训练 | `model=explicit pipeline=from_points` |
| 同步多卡 | `distributed=sync distributed.device_ids='[0,2]' run.view_load_mode=lazy`（Vulkan 序号；无 llvmpipe 时也可能是 `[0,1]`） |
| 恢复 | `train.resume=true` |
| 短训 | `train=debug` |
| 长训 | `train=30k` |

入口一律：

```bash
python -m ftgspp.run <hydra overrides...>
# 或：ftgspp-vulkan <overrides...>
```

---

## 2. 环境与构建

### 2.1 依赖

- Python **3.12**
- C++17、CMake、Ninja、pybind11
- Vulkan SDK **1.2+**（需 `slangc`、`glslc`、可计算的 GPU）
- 可选：`.[lpips]`（Torch + torchmetrics，**不**要求 Torch CUDA 可用）、`.[convert]`、`.[flow]`、`.[eval]`

### 2.2 安装与编译

```bash
cd /path/to/ftgspp-vulkan
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[build]"

# 确保 slangc / glslc / vulkaninfo 在 PATH 中
python -m ftgspp.build_native
uv run python -m scripts.smoke_native --device 0
```

修改 `native/slang/` 后必须重编并提交对应 `.spv`：

```bash
python -m ftgspp.build_native
# 或：python native/compile_shaders.py
```

更细的验收顺序见 [SERVER_DEBUG.md](SERVER_DEBUG.md)。

### 2.3 常用环境变量

| 变量 | 作用 |
|------|------|
| `FTGSPP_RUNS_ROOT` | 实验根目录（默认 `runs/`） |
| `FTGSPP_LOG_DIR` | 保留配置字段；当前 native runner 不单独写入该目录 |
| `FTGSPP_TRAIN_LOG_INTERVAL` | 写入 `train_metrics.jsonl` 的 step 间隔 |
| `FTGSPP_TRAIN_EVAL_INTERVAL` | 内置 RGB PSNR/SSIM eval 间隔 |
| `FTGSPP_VIEW_LOAD_MODE` | 覆盖 `run.view_load_mode`：`lazy`（默认）或 `resident`（小数据集显式 opt-in） |
| `FTGSPP_VIEW_RGB_CACHE` | 覆盖 `run.view_rgb_cache`（lazy 线程安全 LRU，默认 `64`；同步 ranks 共享） |
| `FTGSPP_VK_IMAGE_CACHE` | 覆盖 `run.image_cache_device`：`cpu`/`gpu`（仅**单卡** `resident` 有效；sync shared resident 忽略 gpu；默认 cpu） |
| `FTGSPP_VIEW_LOAD_WORKERS` | 覆盖 `run.view_load_workers`（resident 并行解码；lazy 正值启用 train/eval lookahead，`0`=关闭 lookahead 且当前批串行解码；默认 `8`） |
| `FTGSPP_TRAIN_LOG_GRAD` | 日志 step 是否额外写 geometry gradient RMS/L2（默认 `1`；设 `0` 关闭） |
| `FTGSPP_VIEW_PREFETCH_LOOKAHEAD` | 覆盖 lazy train lookahead 窗口 |
| `FTGSPP_EVAL_PREFETCH_LOOKAHEAD` | 覆盖 lazy eval lookahead 窗口 |
| `FTGSPP_VK_DEVICE` | 单卡入口的 Vulkan device override；同步多卡使用 `distributed.device_ids` 并忽略该变量 |

输出路径默认：

```text
${FTGSPP_RUNS_ROOT:-runs}/${data.name}/run_${train.iterations}
```

可用 Hydra 覆盖：`run.output_path=...`、`run.gpu=0`、`run.overwrite_completed=true`。

---

## 3. 数据：SelfCap / EasyMocap

### 3.1 发布包典型布局

```text
corgi-release/
  videos/0000.mp4 ...          # 每相机一路视频
  optimized/
    intri.yml
    extri.yml
    sync.json                  # 相机时间偏移（秒）
  pcds/005200.ply ...          # 可能是「物理帧号」命名
  dense_pcds/ ...              # 可选稀疏稠密点云
```

准备后的训练场景：

```text
<output>/
  images/<camera>/<frame:06d>.jpg
  intri.yml / extri.yml
  t_offsets.json               # 由 sync.json 写出
  points_strideN/fXXXXXX.ply   # 指向 pcds 的软链
```

### 3.2 两层「时间/帧」概念（重要）

**A. 相机间时间同步（`optimized/sync.json`）**

- 值为**秒**，不是帧号。
- `ftgspp-prepare-selfcap` 写成 `t_offsets.json`。
- Native 加载：`timestamp = logical_frame / fps − offset[camera]`。
- Native **不直接读 `sync.json`**，只认 `t_offsets.json` / `timecode.json` / `time_offsets.json`。  
  必须跑 prepare，或自行把 sync 拷成上述文件名之一。
- `data.strict_time_offsets=true` 为默认值：一旦发现上述任一 offset 文件，其中必须显式覆盖所有已发现相机；缺项会在创建 Vulkan replica 前直接报错。prepare 不再把源 `sync.json` 的缺项悄悄写成显式 `0`，而是保留缺项供 preflight 识别。只有明确设置 `false` 才会 warning 并对缺项补 `0` 秒。
- train/eval 相机集合必须互斥，且并集精确等于图像目录中发现的相机集合。不能把某些相机留在两者之外；这会导致初始化、UFM 或相机索引与训练 manifest 不一致。
- 兼容旧的平铺布局 `images/cXXX-fXXXXXX.ext`。加载器从文件名解析真实（可稀疏）相机 ID，保留每个相机自己的扩展名；同一数值 ID 的不同补零写法（如 `c7` 与 `c007`）会被视为重复并拒绝。
- V5 checkpoint 的 dataset metadata 除逐相机 offset 外，还记录 offset 的 min/max/span，以及每个逻辑帧在应用 offset 后跨相机 timestamp spread 的 min/max/mean，便于在服务器恢复前审计同步跨度。

**B. 物理帧 vs 逻辑帧（点云文件名）**

部分 SelfCap 发布（如 `corgi-release`）：

| | 范围示例 | 用途 |
|--|----------|------|
| 逻辑帧 | 视频下标 `0 … N-1` | `data.frames`、图像文件名、时间戳、`points_stride*/fXXXXXX.ply` |
| 物理帧 | 如 `5200 … 5200+N-1` | 原始 `pcds/005200.ply` 文件名 |

关系通常为：`physical = logical + FRAME0`（corgi-release 上 `FRAME0=5200`）。

- **配置里的 `data.frames` 必须用逻辑帧**。若写成 `5200–5500`，时间戳会从 ~86s 起，破坏 clip 语义。
- Init 按逻辑帧找 `f{frame:06d}.ply`，因此需要把物理 PCD **映射/软链**到逻辑名后再交给 prepare。

### 3.3 `ftgspp-prepare-selfcap`

```bash
ftgspp-prepare-selfcap \
  --scene-root /path/to/corgi-release \
  --output /path/to/prepared/corgi \
  --calibration-dir /path/to/corgi-release/optimized \
  --pcds /path/to/pcds_logical \          # 逻辑帧命名的 PCD 目录
  --start 0 --stop 1000 \                 # 逻辑帧 [start, stop)
  --scale 0.5 \
  --keyframe-stride 1 \                   # SelfCap 密点云常用 1；稀疏可用 10
  --eval-cameras 0 8 16 \                 # 或 --holdout --llffhold 8
  --name corgi_first1000_train21_eval3 \
  --write-config configs/data/corgi_first1000_train21_eval3.yaml
```

要点：

- `--start/--stop` 同时用于 **ffmpeg 抽帧** 与 **keyframe 链接**；二者共用同一索引，故 PCD 必须先逻辑化。
- `keyframe_stride=1` → 输出 `points_stride1/`；训练时必须 `init.keyframe_stride=1`（默认 init 是 10）。
- 进度条按「整路相机抽完」更新；并行 ffmpeg 时长时间停在 `0/24` **不代表卡住**，可看 `images/<cam>/` 文件数是否增长。
- 已存在完整帧时会跳过 extract；可安全重跑以补写 config / points。

相机划分：

- `--eval-cameras 0 8 16` → 3 路测试，其余为训练（24 相机时即 21/3）。
- `--holdout --llffhold 8` → 等价每隔 8 路 hold-out（有序下标 `idx % 8 == 0`）。
- `train_cameras: null` + 非空 `eval_cameras` → train = 全部相机 − eval。
- `eval_cameras: []` → 全相机训练、无 hold-out。

### 3.4 逻辑化 PCD 示例（corgi-release）

```bash
SRC=/path/to/corgi-release
REMAP=$SRC/pcds_logical
FRAME0=5200
mkdir -p "$REMAP"
for L in $(seq 0 999); do   # 与 --stop 对齐
  printf -v dst '%06d.ply' "$L"
  printf -v src '%06d.ply' "$((L + FRAME0))"
  ln -sfn "$SRC/pcds/$src" "$REMAP/$dst"
done
```

### 3.5 可选：UFM 光流 cache

仅当 `init.temporal_motion_adapted=true` 时需要：

```bash
pip install -e '.[flow]'   # 并另行安装 UFM 权重/包
python -m ftgspp.data.flow configs/data/<name>.yaml \
  --out /path/to/prepared/_flow --keyframe-stride 1 --cameras all
```

训练附加：

```text
init.temporal_motion_adapted=true init.temporal_flow_path=/path/to/prepared/_flow
```

UFM **不会**在训练进程内推理；native 只读 NPZ cache。

---

## 4. Hydra 配置一览

默认组合见 `configs/config.yaml`：`model` / `data` / `init` / `train` / `distributed` / `eval` / `run` / `pipeline`。

| 组 | 常用选项 |
|----|----------|
| `data=` | `bar_first300_scale05`、自建 yaml stem |
| `model=` | 仅 `explicit`；`model.marginal_gating=true/false` |
| `init=` | `default`；覆盖 `init.keyframe_stride`、`init.num_gaussians` |
| `train=` | `debug`（10 step）、`30k` |
| `distributed=` | `off`（默认单卡）、`sync`（复制式同步多卡） |
| `pipeline=` | `check`、`from_points`（init→train→export）、`train_only`（需已有 `init.npz` 或 resume checkpoint） |

Pipeline `from_points`：`stages: [init, train, export]`。  
若输出目录已有 `gaussians.npz` 且未设 `run.overwrite_completed=true`，会拒绝覆盖。

---

## 5. 训练配方

### 5.1 检查

```bash
python -m ftgspp.run data=<name> init.keyframe_stride=1 pipeline=check
```

日志应报告 frames / cameras / train views / eval views。例如 1000 帧 × 21/3：约 `21000 train`、`3000 eval`。

### 5.2 短训冒烟

```bash
export FTGSPP_TRAIN_LOG_INTERVAL=1
export FTGSPP_TRAIN_EVAL_INTERVAL=5

python -m ftgspp.run \
  data=<name> \
  init.keyframe_stride=1 \
  model=explicit \
  train=debug \
  pipeline=from_points
```

### 5.3 全功能长训（推荐验收路径）

覆盖：explicit velocity、marginal gating、color correction、relocation（含 gate score）。
需要固定动静角色时，将 `train.hard_separation=true` 作为主循环后的附加阶段。
LPIPS 可选（见下节）。

```bash
export FTGSPP_RUNS_ROOT=/path/to/runs
export FTGSPP_TRAIN_LOG_INTERVAL=50
export FTGSPP_TRAIN_EVAL_INTERVAL=1000

python -m ftgspp.run \
  data=<name> \
  init.keyframe_stride=1 \
  model=explicit \
  model.marginal_gating=true \
  train=30k \
  train.color_correction=true \
  train.color_correction_start=3000 \
  train.relocation.score_mode=gate_included \
  train.relocation.score_mode_start=3000 \
  pipeline=from_points \
  run.output_path='${run.experiment_root}/full_explicit'
```

主循环结束后写出 `gaussians.npz`。设置 `train.hard_separation=true` 时，主循环后
会额外执行 5000 步 hard separation，把 gate 固化为静态/动态两组，并写出
`separated.npz`。

### 5.4 Explicit-only

```bash
python -m ftgspp.run data=<name> model=explicit pipeline=from_points
```

### 5.5 恢复

```bash
python -m ftgspp.run data=<name> train.resume=true \
  run.output_path=/path/to/existing/run_dir
# 或指定编号 NPZ / 原子指针：train.resume_checkpoint=/path/to/ckpt/latest.json
```

同步多卡 checkpoint 只物化一份完整 NPZ。explicit 模型、color 与 Adam state 取 rank 0 的完整副本；relocation 的 `(magnitude_sum, count)` 会先跨 rank 求和。恢复时这份历史只放到 rank 0，其余 rank 置零，避免下次聚合重复历史。V5 checkpoint 记录数据清单、采样 ABI、算法配置、shader/Git/native ABI 与 Vulkan 设备 provenance；默认拒绝改变 world size。确需换卡数时必须显式设置 `distributed.allow_resume_world_size_change=true`。

hard-separation 阶段 checkpoint（`phase` 以 `hard_separation` 开头）支持恢复。
恢复时只重建阶段配置，不会重复执行 gate split。view sampler 使用独立的
`seed + train_iterations` 阶段种子；V5 metadata 会校验阶段游标和 sampler ABI。

### 5.6 同步多卡

```bash
# 两卡 explicit
python -m ftgspp.run \
  data=<name> model=explicit pipeline=from_points \
  distributed=sync \
  distributed.device_ids='[0,2]' \
  run.view_load_mode=lazy

# 双卡 explicit（device_ids 按本机 Vulkan 枚举填写，跳过 llvmpipe）
python -m ftgspp.run \
  data=<name> model=explicit pipeline=from_points \
  model.marginal_gating=true \
  train.color_correction=true \
  distributed=sync \
  distributed.device_ids='[0,2]' \
  run.view_load_mode=lazy
```

约束：`train.batch_size=1` 是每个 micro-step、每卡的固定 batch；global batch 为 `world_size × accumulation.microsteps`，并在 Adam 前执行精确 FP32 `mean`。单卡使用 `distributed=off`，同步模式只接受双卡；双卡内部使用 GPU pair merge，三个及以上设备会在配置解析和运行时拒绝。双卡 `codec_policy=auto_exact` 每步按两端实际 visible count 与 active-SH 字节选择 dense/sorted-sparse。`host_collective.bucket_mb=0` 会在启动时并发实测 8/16/32/64 MiB；正数可固定覆盖。旧 `packed_host`、`host_mailbox` 和 `device_group` 配置均不再接受。可选 large-batch 补偿：

```bash
  distributed.world_size_lr_scaling=effective_batch_adam \
  distributed.sampling_policy=flat_epoch \
  # 或同帧多相机：distributed.sampling_policy=co_frame \
  distributed.wall_eval_interval_seconds=300 \
```

同步模式不支持 Torch LPIPS。算法、显存和通信说明统一见 [MULTI_GPU.md](MULTI_GPU.md)。启动 bucket 标定会写入 `performance_summary.json`。结果对比直接读取 `metrics.jsonl` / `train_metrics.jsonl`，覆盖 PSNR、SSIM、最弱相机和主要 timing。

### 5.7 功能开关说明

| 开关 | 说明 |
|------|------|
| `model.marginal_gating=true` | 启用连续 gate \(\phi=g+(1-g)a(t)\)；persistents 不被短 duration 硬裁 |
| `train.color_correction=true` | 每相机仿射，Vulkan Adam |
| `train.lpips_loss=true` | Torch 主机插件；安装 `.[lpips]`（extra 已含 torchvision） |
| `train.relocation.score_mode=gate_included` | relocation 分数含 gate |

Loss 日志包含 `image_loss = L1 + SSIM`、`reg_opacity`、`reg_scale`、`gate`、`color_cc` 与 `lpips`（启用时）。native scalar 的和不含 Torch 项；最终训练日志 `total_loss` 会加上已加权的 `lpips`，与实际优化目标一致。同步多卡训练不支持该 Torch 插件。内置 `metrics.jsonl` eval 当前只写 PSNR/SSIM/per-camera；LPIPS/FLIP/flow EPE 作为服务器验收中的外部可选质量指标。

---

## 6. LPIPS 可选依赖注意

```bash
pip install -e '.[lpips]'
```

`lpips` extra 已同时声明 Torch、TorchMetrics 和 torchvision，不需要再单独补装 torchvision。

- LPIPS **不是** Vulkan shader；Native 渲 RGB → Torch 算损失与像素梯度；RGBA reference alpha 会遮罩两侧图像及反向梯度。
- 若本机 Torch CUDA 与驱动不匹配（`cuda=False`），插件会回退 **CPU**，每 step 明显变慢，但不影响 Vulkan 主路径。
- 想先跑通主训：去掉 `train.lpips_loss=true` 即可。

---

## 7. 输出产物

典型 run 目录：

```text
run_dir/
  config.composed.yaml
  config.resolved.yaml
  .hydra/                # Hydra runtime metadata
  init.npz
  train_metrics.jsonl    # 训练 loss/timing；日志 step 才写 scalar
  metrics.jsonl          # 内置 eval：PSNR/SSIM/per-camera
  performance_summary.json
  ckpt/
    latest.json          # 原子更新的小指针
    000100.npz ...       # 不可变快照；文件名是 6 位 iteration
  gaussians.npz
  gaussians_4dgs.ply     # export
```

### 7.1 「像卡住」的常见阶段

1. **抽帧**：tqdm 按相机完成度更新；看 `images/<cam>/` 计数。
2. **view 注册**：默认 `run.view_load_mode=lazy`（按 step 读盘 + LRU；同步 ranks 共享线程安全 cache 并确定性预取）。同步 `resident` 会一次性解码进**共享** pinned host RGB 池（不按卡翻倍），各 rank 只注册 meta 并逐步 bind；`image_cache_device=gpu` 在 sync resident 下无效。单卡才可对 `resident` 使用 `run.image_cache_device=cpu|gpu`。`run.view_load_workers` 控制 resident 解码并行度；lazy 模式下正值启用有界 lookahead，`0` 关闭 lookahead 并串行解码当前批。
3. **训练中**：按 `FTGSPP_TRAIN_LOG_INTERVAL` 写 `train_metrics.jsonl`。

---

## 8. Checkpoint 转换

```bash
pip install -e '.[convert]'
ftgspp-convert-checkpoint old-field.pt explicit-v5.npz --field-mode sample
ftgspp-convert-checkpoint explicit-v4.npz explicit-v5.npz --field-mode reject
```

`sample` 是旧 field checkpoint 唯一允许的迁移路径：离线采样为 explicit velocity。普通 explicit V4 使用 `reject`，确保输入意外包含 field state 时立即失败。输出固定为 explicit-only V5；`preserve` 不用于生产迁移。

日常 native 训练只读写 `.npz`。写出 `.pt` 仅用于与旧栈互操作；本仓库**不再**提供 Torch 训练器。

---

## 9. 测试

```bash
python -m unittest discover -s tests -v
```

---

## 10. 端到端范例：corgi-release 前 300 帧、21/3 划分

图片建议写到**本地 NVMe**（避免 `/mnt/merge` 小文件读极慢）。读盘方式（默认 **lazy**）：

```bash
# 默认：按需解码；同步多卡共享 cache 并按 rank 顺序预取
run.view_load_mode=lazy run.view_rgb_cache=64

# 小数据集显式常驻（起步解码全部图，需足够 RAM/VRAM）
# run.view_load_mode=resident run.image_cache_device=cpu run.view_load_workers=8
```

```bash
cd /path/to/ftgspp-vulkan
source .venv/bin/activate
# 按本机方式 export Vulkan SDK PATH

SRC=/path/to/corgi-release
OUT=/home/xiaowenhan/datasets/SelfCap/ftgspp/corgi   # 本地盘
REMAP=$SRC/pcds_logical
FRAME0=5200
CFG=configs/data/corgi_first300_train21_eval3.yaml

mkdir -p "$REMAP"
for L in $(seq 0 299); do
  printf -v dst '%06d.ply' "$L"
  printf -v src '%06d.ply' "$((L + FRAME0))"
  ln -sfn "$SRC/pcds/$src" "$REMAP/$dst"
done

ftgspp-prepare-selfcap \
  --scene-root "$SRC" \
  --output "$OUT" \
  --calibration-dir "$SRC/optimized" \
  --pcds "$REMAP" \
  --start 0 --stop 300 \
  --scale 0.5 \
  --keyframe-stride 1 \
  --eval-cameras 0 8 16 \
  --name corgi_first300_train21_eval3 \
  --write-config "$CFG"

python -m ftgspp.run data=corgi_first300_train21_eval3 init.keyframe_stride=1 pipeline=check

export FTGSPP_VIEW_RGB_CACHE=64
export FTGSPP_TRAIN_LOG_INTERVAL=50
export FTGSPP_TRAIN_EVAL_INTERVAL=1000

python -m ftgspp.run \
  data=corgi_first300_train21_eval3 \
  init.keyframe_stride=1 \
  model=explicit \
  model.marginal_gating=true \
  train=30k \
  train.color_correction=true \
  train.color_correction_start=3000 \
  train.relocation.score_mode=gate_included \
  train.relocation.score_mode_start=3000 \
  pipeline=from_points \
  run.output_path='${run.experiment_root}/full_explicit_300f_21_3'
```

同类示例还有 `bar_first300_scale05.yaml`、`corgi_first300_scale05.yaml`、`example_full_calibration.yaml`。

---

## 11. 故障速查

| 现象 | 处理 |
|------|------|
| `No module named '_ftgspp_vk'` | 在仓库根目录、已 `build_native`，或检查 `.so` 是否生成 |
| `train.lpips_loss requires ...` / 缺 torchvision | `pip install -e '.[lpips]'`，或关闭 LPIPS |
| Torch CUDA driver too old | 可忽略（Vulkan 主训不依赖）；LPIPS 将走 CPU |
| 找不到 `points_stride10` | 准备用了 stride=1 时加 `init.keyframe_stride=1` |
| PCD `FileNotFoundError: 000000.ply` | 未做物理→逻辑映射 |
| 时间全错位 | 确认存在 `t_offsets.json` 且来自 `optimized/sync.json` |
| 拒绝覆盖 `gaussians.npz` | `run.overwrite_completed=true` 或换 `run.output_path` |
| 每 step 很慢 / 读盘忙 | 把 `images/` 放到本地 NVMe、增大 `run.view_rgb_cache`；仅在小数据集内存充足时显式改为 `resident` |
| resident 启动 OOM / swap | sync 已改为共享一份 host RGB；仍 OOM 则改 `lazy` 或减少相机/帧数；单卡可改 `lazy` |
| Slang/validation 报错 | 按 [SERVER_DEBUG.md](SERVER_DEBUG.md) 从 shader → module → smoke 排查 |

---

## 12. 相关文档

| 文档 | 内容 |
|------|------|
| [README.md](README.md) | 总览与最短命令 |
| [OPERATIONS.md](OPERATIONS.md) | 本手册 |
| [SERVER_DEBUG.md](SERVER_DEBUG.md) | 服务器编译与验收清单 |
| [MULTI_GPU.md](MULTI_GPU.md) | 同步多卡算法、通信量、checkpoint 与验收 |
| [VULKANIZATION.md](VULKANIZATION.md) | 算法/模块边界 |
| [THIRD_PARTY.md](THIRD_PARTY.md) / [SOURCE_REVISIONS.md](SOURCE_REVISIONS.md) | 第三方与修订来源 |
