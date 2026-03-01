## ADDED Requirements

### Requirement: Provide an opt-in offline refine stage in SingleImagePipeline
系统 SHALL 在 `SingleImagePipeline` 中提供一个可选的离线精修阶段,默认关闭,以避免改变现有行为.

#### Scenario: Disabled refine keeps current behavior
- **WHEN** 用户未启用该精修开关
- **THEN** pipeline MUST 不运行任何 RoMa/COLMAP/MVS 步骤
- **AND THEN** 输出结果 MUST 与当前版本保持一致(同输入与随机种子)

#### Scenario: Enabled refine runs after coarse multi-frame generation
- **WHEN** 用户启用精修,且 pipeline 已生成多帧(通常 >30)
- **THEN** 系统 MUST 按固定顺序执行:
  1. 导出 COLMAP 工程(`colmap-project-export`)
  2. RoMa v2 匹配并写入数据库(`roma-v2-matching-import`)
  3. COLMAP triangulation + MVS depth(`colmap-mvs-depth`)
  4. MVS 与 MoGe depth 融合(`mvs-moge-depth-fusion`)
  5. 用融合 depth 重建/重初始化 gaussian scene,并再训练一轮

### Requirement: Poses are fixed and MUST NOT be optimized by COLMAP
系统 SHALL 固定使用 pipeline 生成的 pose.
系统 MUST NOT 调用 COLMAP mapper 或任何会优化位姿的 BA 流程.

#### Scenario: point_triangulator is used instead of mapper
- **WHEN** 进入 COLMAP 阶段
- **THEN** 系统 MUST 使用 `point_triangulator` 作为稀疏三角化入口

### Requirement: Matching MUST exclude inpaint regions end-to-end
系统 SHALL 将 inpaint 区域视为不可信内容,并在 matching 与几何过滤中全程剔除.

#### Scenario: Inpaint masking is enforced
- **WHEN** 计算 RoMa matches 并写入 COLMAP database
- **THEN** 系统 MUST 确保任一端落在 inpaint 区域的 match 不会进入数据库

### Requirement: Failures MUST safely fall back to MoGe-only
系统 SHALL 在离线精修链路失败时保持主流程可用.

#### Scenario: Missing dependency does not crash the pipeline
- **WHEN** `colmap` CLI 或 RoMa v2 依赖缺失,或任一步骤运行失败
- **THEN** 系统 MUST 打印可执行的错误提示(缺什么,如何安装/启用)
- **AND THEN** 系统 MUST 回退到当前 MoGe-only 的结果继续渲染/导出

### Requirement: Refine stage MUST log summary metrics for validation
系统 SHALL 输出可验证的结构化摘要,用于判断精修是否真实生效.

#### Scenario: Coverage and frame usage are reported
- **WHEN** 精修成功完成
- **THEN** 系统 MUST 至少记录:
  - 参与精修的帧数
  - pairs 数量(匹配规模)
  - 每帧 MVS depth 的有效像素比例(非 inpaint 区域)
  - depth 融合后的有效像素比例
