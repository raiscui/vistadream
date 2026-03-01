## Why

当前 `SingleImagePipeline` 的深度主要来自单目 MoGe,在多视角一致性与几何精度上存在天然上限.
当我们通过 warp/inpaint 生成了 30+ 视角帧后,实际上已经具备了做多视几何约束的条件.

本 change 的目标是引入一条离线精修链路.
用 RoMa v2 在非 inpaint 区域做 dense matching,再用 COLMAP 的 triangulation + MVS depth 提供更一致的几何主干,最后与 MoGe 融合补洞,从而提升最终渲染质量.

## What Changes

- 新增一个可选的离线精修阶段(默认关闭):
  - 仅在 `SingleImagePipeline` 生成多帧后启用.
  - 固定使用 pipeline 生成的相机 pose(不让 COLMAP 重新估 pose).
  - 对 RoMa matching 强制 mask 掉 inpaint 区域,避免幻觉内容污染几何.
- 新增 COLMAP 工程导出与执行器:
  - 导出 images/masks/手工 sparse 模型,并将 RoMa matches 写入 COLMAP database.
  - 运行 `point_triangulator -> image_undistorter -> patch_match_stereo -> stereo_fusion`.
- 新增深度融合策略:
  - 以 MVS depth 为非 inpaint 区域的几何主干.
  - 对齐 MoGe depth(scale+shift),并在 MVS 无效或 inpaint 区域用 MoGe 补洞.
- 新增回退策略:
  - 任一步 COLMAP/MVS 失败时,回退到 MoGe-only,不让主流程崩溃.
- 增加单元测试(不依赖外部 `colmap` 可执行文件),覆盖:
  - pairs 生成,RoMa keypoint 索引化,几何一致性过滤,depth 融合.

## Capabilities

### New Capabilities

- `colmap-project-export`: 将 pipeline 生成的多帧(Frame)导出为可复现的 COLMAP 工程(含 images/masks/sparse manual).
- `roma-v2-matching-import`: 使用 RoMa v2 在非 inpaint 区域做 dense matching,并将 keypoints/matches 写入 COLMAP database,同时基于已知 pose 做对极几何过滤.
- `colmap-mvs-depth`: 在固定 pose 的前提下执行 COLMAP triangulation + MVS depth,产出 per-view depth maps 与 fused 点云.
- `mvs-moge-depth-fusion`: 将 MVS depth 与 MoGe depth 对齐并融合,输出用于重建/训练的最终 depth 与置信度掩码.
- `singleimg-colmap-mvs-refine`: 将上述链路作为 `SingleImagePipeline` 的可选精修阶段接入,并在失败时安全回退.

### Modified Capabilities

- (无)

## Impact

- 受影响代码:
  - `src/vistadream/api/single_img_pipeline.py` 将新增可选精修阶段与配置项(默认关闭,不影响现有行为).
  - 新增 `src/vistadream/ops/colmap_mvs/` 相关模块(工程导出,matching,db,runner,depth io,融合).
- 新增依赖与运行时要求:
  - 需要 `colmap` CLI(用于 triangulation + MVS).
  - 需要 RoMa v2 匹配器实现(优先作为可选依赖,并提供明确失败提示与 fallback 路径).
- 资源消耗:
  - 该精修链路为离线重型流程,预计会显著增加运行时间与磁盘中间产物.
  - 通过 pairs 选择与帧数上限控制复杂度,避免 O(N^2) 爆炸.
