## ADDED Requirements

### Requirement: Compute RoMa v2 dense matches only on valid (non-inpaint) pixels
系统 SHALL 使用 RoMa v2 在指定图像对(pairs)上进行 dense matching.
系统 MUST 仅在有效像素区域(由 `masks/*.png` 指定,255 为有效)上采样/保留匹配点.

#### Scenario: Matches are filtered by mask
- **WHEN** 对某一对图像执行 RoMa v2 matching
- **THEN** 系统 MUST 丢弃任一端落在 mask=0 区域的匹配点
- **AND THEN** 写入数据库的匹配点坐标 MUST 全部落在 mask=255 的像素上

### Requirement: Matches must be geometrically filtered using fixed poses
系统 SHALL 基于已知的内参 K 与固定相机 pose 对匹配点做几何一致性过滤.
系统 MUST 使用对极几何误差阈值剔除外点,避免 inpaint 幻觉或重复纹理污染三角化与 MVS.

#### Scenario: Epipolar outliers are rejected
- **WHEN** 某个匹配点对在固定 pose 下的对极误差超过阈值(像素域)
- **THEN** 系统 MUST 丢弃该匹配点对
- **AND THEN** 输出的 matches 集合 SHOULD 显著少于 raw matches(但不能为负或 NaN)

### Requirement: Write keypoints and matches into COLMAP database deterministically
系统 SHALL 将 keypoints 与 matches 写入 COLMAP database,供后续 `point_triangulator` 与 MVS 使用.
系统 MUST 对每张图像建立确定性的 keypoint 索引,并将 matches 写为 keypoint index pairs.

#### Scenario: Keypoint indexing is stable under duplicate raw matches
- **WHEN** 多个 raw matches 在像素上落入同一量化网格(例如 2px)
- **THEN** 系统 MUST 去重并复用同一个 keypoint index
- **AND THEN** 写入数据库的 keypoints 数量 MUST 可控(不随 raw matches 无界膨胀)

#### Scenario: Matches are written with zero-based keypoint indices
- **WHEN** 系统把一对图像的 matches 写入 COLMAP database
- **THEN** 写入的 matches MUST 使用零基的 keypoint indices(与 COLMAP 约定一致)

### Requirement: Pair selection must scale to >30 frames
系统 SHALL 支持在 30+ 帧情况下生成可控数量的 pairs,避免 O(N^2) 的匹配爆炸.

#### Scenario: Sliding-window pairs are generated
- **WHEN** 输入帧数量 N 很大且设置 `pair_window > 0`
- **THEN** 系统 MUST 生成以滑窗为主的 pairs(例如 i 与 i+1..i+W)
- **AND THEN** pairs 总数 MUST 与 N 近似线性增长
