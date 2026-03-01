## ADDED Requirements

### Requirement: Export frames into a deterministic COLMAP project
系统 SHALL 将 `SingleImagePipeline` 生成的多帧数据导出为可复现的 COLMAP 工程目录.
导出结果 MUST 是确定性的(同输入,同配置,输出文件名与内容一致).

#### Scenario: Export directory structure is created
- **WHEN** 给定一组包含 RGB,深度,内参,外参(固定 pose)与 mask 信息的 frames,并指定 `work_dir`
- **THEN** 系统 MUST 在 `work_dir/project/` 下创建至少以下目录与文件:
  - `images/`(按帧索引命名的图片文件)
  - `masks/`(与 images 同名的一张灰度 mask,255 表示有效像素)
  - `sparse_manual/`(COLMAP text model,包含 `cameras.txt`,`images.txt`,`points3D.txt`)

#### Scenario: Existing output is handled safely
- **WHEN** `work_dir/project/` 已存在且非空
- **THEN** 系统 MUST 默认拒绝覆盖并给出清晰错误信息
- **AND THEN** 仅当用户显式开启 `overwrite=true` 时才允许清理并重建输出

### Requirement: Camera intrinsics and fixed poses are exported correctly
系统 SHALL 使用 pipeline 的固定 pose 导出 COLMAP 可消费的相机模型.
系统 MUST 不触发 COLMAP 的 mapper/BA 重新估计相机位姿.

#### Scenario: cameras.txt and images.txt are consistent
- **WHEN** 导出包含 N 张图像的工程
- **THEN** `sparse_manual/cameras.txt` MUST 至少包含 N 条 camera 记录(允许相同内参去重复用 camera_id)
- **AND THEN** `sparse_manual/images.txt` MUST 为每张图像写入:
  - 图像名(与 `images/` 下文件名一致)
  - 与该图像对应的 camera_id
  - world->cam 的姿态(由 pipeline 的 `cam_T_world` 或其逆矩阵得到)

### Requirement: Matching mask must exclude inpaint regions
系统 SHALL 使用 mask 明确指出 "允许参与匹配/几何" 的像素区域.

#### Scenario: Outpaint/coarse frames exclude inpaint regions
- **WHEN** 某帧包含 `inpaint` mask(表示 inpaint/outpaint 生成区域)
- **THEN** 导出的 `masks/<image>.png` MUST 满足:
  - 非 inpaint 像素为 255(有效)
  - inpaint 像素为 0(无效)

#### Scenario: Input frame mask does not misuse training supervision mask
- **WHEN** 导出的帧是 input 视角,且该帧在训练中可能使用全 True 的监督 mask
- **THEN** 导出的匹配 mask MUST 仍表示 "全部像素可用于匹配"(即全 255),或使用更保守但语义正确的深度置信 mask
