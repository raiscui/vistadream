## Context

当前 `SingleImagePipeline` 的深度主来源是单目 MoGe.
当我们通过 warp/inpaint 生成了 30+ 新视角帧后,深度在跨视角的一致性上会成为最终渲染质量的瓶颈.

同时,该流程的特殊性很强:

- 多视图不是来自真实拍摄,而是由 inpaint 生成.
- inpaint 区域存在幻觉内容,会污染匹配与几何估计.
- pipeline 内部已经有固定的相机 pose 生成逻辑(轨迹),我们希望把 pose 当作真值,而不是交给 SfM/BA 去优化.

因此,我们需要一条离线精修链路:

1. 严格只在非 inpaint 区域做匹配.
2. 在固定 pose 的前提下做 triangulation + MVS depth.
3. 用 MVS depth 作为几何主干,与 MoGe 融合补洞.
4. 以融合 depth 重建/重初始化 gaussian scene,并再训练一轮.
5. 任何步骤失败都必须可回退,不影响 MoGe-only 主流程.

## Goals / Non-Goals

**Goals:**

- 在 `stage=coarse` 生成多帧后,提供一个可选的离线精修阶段,提升最终渲染质量.
- 固定使用 pipeline pose(不跑 mapper/BA),避免 inpaint 内容把位姿带偏.
- matching 端到端 mask 掉 inpaint 区域,并叠加对极几何过滤,尽量压制幻觉/外点.
- 对 30+ 帧的规模可控:
  - pairs 数量近似线性增长.
  - 支持帧数上限与滑窗/stride 策略.
- 可诊断:
  - 输出结构化 summary(帧数,pairs 数,MVS coverage,融合 coverage,耗时).
- 可回退:
  - 缺依赖或任何 colmap 步骤失败时,继续走现有 MoGe-only 渲染/导出.

**Non-Goals:**

- 不把该流程升级为通用 SfM 工具链(不支持无序真实照片的自动 pose 恢复).
- 不追求绝对尺度精度(除非后续引入额外尺度约束,本 change 不做).
- 不要求 inpaint 区域的几何完全可靠(该区域优先以 MoGe 补洞为主,并在训练中可更保守).

## Decisions

1. **固定 pose,不跑 mapper/BA**
   - 选择: `point_triangulator` + MVS,而不是 `mapper`.
   - 理由: 该多视图包含合成内容,BA 很容易被幻觉/重复纹理带偏,导致整体位姿漂移.

2. **matcher 使用 RoMa v2,并强制 mask + 几何过滤**
   - 选择: RoMa v2 dense matching 作为主 matcher.
   - 理由: inpaint 生成的新视角往往弱纹理/大视差,SIFT 更容易失败或匹配稀疏.
   - 约束: 任何 match 只要落在 inpaint 区域必须丢弃;并用固定 pose 的对极误差阈值二次过滤.

3. **工程导出以 COLMAP 可复现为第一优先级**
   - 导出目录固定为:
     - `project/images/`
     - `project/masks/`
     - `project/sparse_manual/`
     - `project/database.db`
   - 文件命名使用零填充序号,并用 `meta.json` 记录 frame->image 的映射与关键配置(帧上限,pairs 策略,阈值等).

4. **写入 COLMAP database 采用 sqlite3 直写,避免引入 pycolmap**
   - 选择: 实现最小 COLMAP DB writer(写 cameras/images/keypoints/matches).
   - 理由:
     - pycolmap 在不同平台可能带来编译与版本耦合成本.
     - 直写 sqlite 更可控,也便于做单测(纯 Python).
   - 备选: 未来可切换为 `matches_importer` 等文件导入方式,但仍需要稳定的 feature/keypoint 表达.

5. **pairs 选择策略默认使用滑窗 + 少量长距离 pairs**
   - 选择: `pair_window` 控制局部连通性,`pair_long_stride` 提供较大基线.
   - 理由: 30+ 帧下禁止全连接 O(N^2),同时需要足够基线支撑 triangulation 与 MVS.

6. **MVS depth 采用 geometric 结果,融合策略为 "MVS 主干 + MoGe 补洞"**
   - 选择:
     - MVS 有效像素: 直接使用 MVS depth.
     - MVS 无效或 inpaint 区域: 使用对齐后的 MoGe depth.
   - 理由:
     - MVS 在非 inpaint 区域能提供更几何一致的深度.
     - MoGe 提供更好的覆盖率,适合补洞与合成区域.

## Risks / Trade-offs

- [运行耗时与磁盘爆炸] → 用 `max_frames_for_colmap` + pairs 近似线性策略 + 可选清理中间文件缓解.
- [inpaint 幻觉污染匹配] → 强制 mask(inpaint 与边缘/低置信区域) + 对极误差过滤 + 限制每对 matches 数量.
- [COLMAP 版本差异导致 depth map 格式变化] → 在实现中集中封装 depth IO,并添加 roundtrip 单测与最小 smoke(检测到 colmap 时才跑).
- [依赖缺失导致体验差] → 明确错误提示 + 自动回退 MoGe-only,并在 summary 中标记 refine 未生效原因.
