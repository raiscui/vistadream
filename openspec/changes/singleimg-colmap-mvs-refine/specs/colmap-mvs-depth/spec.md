## ADDED Requirements

### Requirement: Triangulate sparse points using fixed poses
系统 SHALL 在固定 pose 的前提下执行 COLMAP triangulation.
系统 MUST 使用 `point_triangulator`(而不是 mapper)来生成稀疏点云,避免 BA 被 inpaint 内容带偏.

#### Scenario: point_triangulator produces a sparse model
- **WHEN** `database.db` 已包含 keypoints 与 matches,且 `sparse_manual/` 已包含相机与固定 pose
- **THEN** 系统 MUST 运行 `colmap point_triangulator`
- **AND THEN** 系统 MUST 在输出目录生成可被 COLMAP 读取的 sparse 模型(例如 `cameras.bin/images.bin/points3D.bin` 或 text 等价物)

### Requirement: Run COLMAP MVS to produce per-view depth maps
系统 SHALL 在 triangulation 结果基础上运行 COLMAP MVS,产出每张图像的 depth map.
系统 MUST 运行以下步骤(顺序固定):
1. `image_undistorter`
2. `patch_match_stereo`
3. `stereo_fusion`

#### Scenario: Depth maps are generated for most input views
- **WHEN** 对一个包含多视图的工程执行 MVS
- **THEN** 系统 MUST 在 `dense/stereo/depth_maps/` 产出与输入图像同名的 depth map 文件(允许少量缺失)
- **AND THEN** 系统 MUST 产出 `stereo_fusion` 的融合点云产物(例如 `fused.ply`)

### Requirement: MVS runtime must be bounded for large frame counts
系统 SHALL 提供控制 MVS 成本的机制,以支持 30+ 帧的离线精修.

#### Scenario: Source image count is limited
- **WHEN** 输入视图数量超过阈值或用户设置了 source 限制参数
- **THEN** 系统 MUST 限制每个 reference view 的 source images 数量(例如只选最邻近的 K 个)
- **AND THEN** 整体运行时间 MUST 随帧数增长保持可控(避免 O(N^2) 直接爆炸)

### Requirement: Failures must be reported with actionable diagnostics
系统 SHALL 在任一步 COLMAP 命令失败时输出可诊断信息.

#### Scenario: COLMAP command failure includes stderr/stdout context
- **WHEN** 任一 COLMAP 子命令返回非 0
- **THEN** 系统 MUST 报错并包含该命令行与关键 stdout/stderr 片段
