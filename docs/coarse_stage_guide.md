# 单图流程: `--stage coarse` 选帧/补帧/调参指南

这篇文档专门解释 3 件事:

1. `--stage coarse` 到底有没有"选到新帧并补进几何".
2. 为什么你会看到 coarse 和 fine/outpaint 的 PLY 肉眼几乎没区别.
3. 当 coarse 选不到新帧时,应该怎么调参数让它更激进,并产生更明显的几何增量.

## 先给结论(最重要的 3 句话)

1. 只有 `--stage coarse` 在选到 extra frame 后,RGBD inpaint 产生的新视角帧才会被加入最终高斯(PLY).
2. 如果 `[COARSE] Summary` 里 `selected_extra_frames=0`,coarse 实际等价于"只初始化(input+outpaint)",导出的 PLY 和 `fine/outpaint` 很像是正常现象.
3. coarse 选不到帧通常只有 3 类原因: 洞太小(< min),洞太大(> max),或被 `adjacent_exclusion` 相邻剔除掉了.

## `--stage` 语义(避免概念混淆)

单图 CLI(`tools/run_single_img.py`)目前有 4 种 stage:

- `no-outpaint`(默认): 不跑 Flux outpaint. 只构建 input frame 并训练初始化场景.
- `outpaint`: 跑一次 Flux outpaint. 初始化时会有 2 个视角: `input` + `outpaint`. 不做 coarse 补帧.
- `fine`: 在当前仓库实现里等价于 `outpaint`(初始化-only). 不做 coarse 补帧.
- `coarse`: 先做 `outpaint` 初始化,再进入 coarse 循环,尝试选新视角 -> RGBD inpaint -> 加入高斯并训练.

因此你可以把它理解成:

- `coarse` = 初始化 + 补帧(如果选得到).
- `fine/outpaint` = 只有初始化.

## coarse 到底在做什么(从结果看过程)

`--stage coarse` 的核心循环可以用一句话概括:

"在一条更密集的相机轨迹上,挑一个洞(inpaint mask)大小合适的新视角,用 Flux + 深度预测补出 RGBD,再把 `inpaint_wo_edge` 区域变成新的 splats 加进场景里."

这里涉及 3 个关键量:

- 相机动作幅度(影响洞的大小): `--traj-forward-ratio/--traj-backward-ratio/--traj-min-percentage/--traj-max-percentage`
- 选帧阈值(决定什么洞算"合适"): `--coarse-min-inpaint-ratio`, `--coarse-max-inpaint-ratio`
- "最终能加多少几何": 主要看 `inpaint_wo_edge_pixels` 和 `added_splats`,而不是只看 `inpaint_pixels`

## 怎么判断 coarse 是否真的补到了新帧

跑完 coarse 以后,先不要看 PLY,先看日志:

- `[COARSE] Summary` 里的 `selected_extra_frames=X/Y`
  - `X>0` 才说明 coarse 真的选到了新视角并补帧了.
  - `X=0` 说明 coarse 直接退化成初始化-only,PLY 和 fine/outpaint 很像是必然结果.

Summary 里每个被选中的帧还会打印:

- `dense_pose_index`: 这帧在 dense trajectory 里的索引
- `inpaint_ratio`, `inpaint_pixels`: 洞面积(比例/像素数)
- `inpaint_wo_edge_pixels`: 去掉深度边缘 + 低置信后的有效像素数
- `added_splats`: 这一帧实际新增进场景的 splats 数量

一个很常见的现象是:

- `inpaint_pixels` 看起来很大
- 但 `inpaint_wo_edge_pixels` 很小

这意味着"洞虽然大,但能转成稳定几何的像素很少".
此时即使 coarse 选到了帧,PLY 的几何增量也可能不明显.

## coarse 选不到新帧时看什么: `[COARSE][DIAG]`

当 `_next_frame()` 选不到帧时,日志会自动打印 `[COARSE][DIAG]`.
它的目的就是回答: "到底是洞太小,洞太大,还是被相邻剔除".

你会看到类似这些字段:

- `inpaint_ratio(valid) stats: min/mean/max`
- `counts(valid): below_min / in_range / above_max (in_range_without_adj=...)`
- `topK(idx:ratio)=...`

解释方式:

- `below_min` 很大且 `max < min_inpaint_ratio`: 洞太小.
- `above_max` 很大且 `min > max_inpaint_ratio`: 洞太大.
- `in_range_without_adj > 0` 但 `in_range=0`: 很可能被 `coarse_adjacent_exclusion` 剔除了.

## 调参指南(按诊断结果来,不要盲目乱试)

### 1) 洞太小(< `coarse_min_inpaint_ratio`)

目标是让相机动作更大,或者降低下限:

- 增大相机动作(通常优先):
  - `--traj-forward-ratio` / `--traj-backward-ratio`
  - 放宽 `--traj-max-percentage`(通常会让轨迹 radius 更大)
- 增大渲染边缘: `--coarse-margin`
  - 代价是更慢,也更吃显存.
- 降低下限: `--coarse-min-inpaint-ratio`

### 2) 洞太大(> `coarse_max_inpaint_ratio`)

目标是放宽上限,或者让相机动作更小:

- 放宽上限: `--coarse-max-inpaint-ratio`
- 或减小相机动作/减小 margin:
  - 下调 `--traj-forward-ratio/--traj-backward-ratio`
  - 下调 `--traj-max-percentage`
  - 下调 `--coarse-margin`

### 3) 被相邻剔除挡住了

当你看到 `in_range_without_adj > 0` 但 `in_range=0` 时,优先:

- 取消相邻剔除: `--coarse-adjacent-exclusion 0`
- 或提高 dense 采样密度: `--coarse-dense-multiplier`
  - 注意: 这会显著变慢,因为它会对每个候选姿态渲染一次来统计洞比例.

## 一键更激进: `--coarse-fallback-mode closest`

如果你现在的目标是:

- "先确认 coarse 真的会选到新帧"
- "希望 coarse 更激进,补出更明显的几何差异"

可以开启显式兜底:

- `--coarse-fallback-mode closest`

行为是:

- 严格模式找不到落在 `[min,max]` 的候选时,
- 从剩余候选里选一个"最接近阈值区间"的姿态继续推进.

注意事项:

- 这是更激进的实验开关,默认是关闭的(`none`).
- 开启后,你可能会选到 ratio 明显偏大或偏小的帧.
- 选中日志会标记 `(fallback_mode=closest)`,便于你确认"这次是兜底选中的".

## 常用命令模板(直接复制)

### 1) 标准 coarse(只看是否能选到新帧)

```bash
pixi run python tools/run_single_img.py \
  --max-resolution 384 \
  --image-path data/office/IMG_4029.jpg \
  --stage coarse \
  --export-gaussians-ply-path data/coarse
```

### 2) 更激进: 强制兜底选帧(推荐用于排查/对比 PLY 增量)

```bash
pixi run python tools/run_single_img.py \
  --max-resolution 384 \
  --image-path data/office/IMG_4029.jpg \
  --stage coarse \
  --coarse-fallback-mode closest \
  --export-gaussians-ply-path data/coarse
```

### 3) 更激进: 同时放宽阈值 + 增大动作(示例,请结合 `[COARSE][DIAG]` 调整)

```bash
pixi run python tools/run_single_img.py \
  --max-resolution 384 \
  --image-path data/office/IMG_4029.jpg \
  --stage coarse \
  --coarse-fallback-mode closest \
  --coarse-max-inpaint-ratio 0.45 \
  --coarse-min-inpaint-ratio 0.01 \
  --coarse-adjacent-exclusion 0 \
  --coarse-margin 64 \
  --traj-forward-ratio 0.5 \
  --traj-backward-ratio 0.6 \
  --traj-max-percentage 70 \
  --export-gaussians-ply-path data/coarse
```

这组参数的取舍很明确:

- 更容易选到新帧,更容易产生几何差异.
- 代价是 inpaint 区域更大,质量不确定性更高,速度更慢,显存压力更大.
