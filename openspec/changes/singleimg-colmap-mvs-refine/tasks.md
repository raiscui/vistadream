## 1. Setup & Public Interfaces

- [ ] 1.1 在 `src/vistadream/api/single_img_pipeline.py` 为 `SingleImageConfig` 增加 `ColmapMvsRefineConfig`(默认关闭,含 work_dir/max_frames/pairs/阈值/overwrite/keep_intermediate 等字段)
- [ ] 1.2 新建模块目录 `src/vistadream/ops/colmap_mvs/` 并规划文件边界(导出,pairs,matcher,db,runner,depth_io,fusion)
- [ ] 1.3 增加统一的日志前缀与结构化 summary 数据结构(用于覆盖率与耗时统计)

## 2. COLMAP Project Export(colmap-project-export)

- [ ] 2.1 实现导出: 生成 `project/images/` 与 `project/masks/` 的确定性命名(零填充序号)与 `meta.json` 映射
- [ ] 2.2 实现导出: 生成 `project/sparse_manual/cameras.txt`(按内参去重 camera_id)与 `images.txt`(固定 pose,world->cam)
- [ ] 2.3 实现导出: 生成 `project/sparse_manual/points3D.txt`(允许为空占位,但格式合法)
- [ ] 2.4 实现 mask 规则: outpaint/coarse 帧 mask=~inpaint,并与 `dpt_conf_mask` 取交集; input 帧 mask 不能误用训练监督 mask(默认全 255 或更保守的置信 mask)
- [ ] 2.5 增加导出校验: 目录结构/文件数量/文件名一致性,并实现 overwrite 清理逻辑

## 3. Pair Selection + RoMa v2 Matching Import(roma-v2-matching-import)

- [ ] 3.1 实现 pairs 生成: 滑窗 `pair_window` + 少量长距离 `pair_long_stride`,并做去重与稳定排序
- [ ] 3.2 接入 RoMa v2(可选依赖): 设计 import 失败时的错误信息与 fallback 行为(不崩主流程)
- [ ] 3.3 实现 `KeypointIndexer`: 像素量化去重(例如 2px 网格),并保证索引稳定可复现
- [ ] 3.4 实现对极几何过滤: 基于固定 pose + K 计算对极误差,超过阈值直接丢弃 match
- [ ] 3.5 实现 match 限流: 每对图像最多保留 `max_matches_per_pair` 个匹配(按 certainty 或几何误差排序)
- [ ] 3.6 实现 COLMAP sqlite DB writer: 写入 cameras/images/keypoints/matches(零基 index),并支持 resume/跳过已写 pair

## 4. COLMAP Triangulation + MVS Runner(colmap-mvs-depth)

- [ ] 4.1 实现 `point_triangulator` runner(固定 pose,禁止 mapper),并校验 sparse 输出存在且 points3D 数量 > 0
- [ ] 4.2 实现 `image_undistorter` runner,输出 dense workspace
- [ ] 4.3 实现 `patch_match_stereo` runner,并提供 source images 数量限制(支持 30+ 帧可控)
- [ ] 4.4 实现 `stereo_fusion` runner,产出 `fused.ply`
- [ ] 4.5 失败诊断: 任何 colmap 命令失败必须带命令行与关键 stdout/stderr 片段,便于定位

## 5. Depth IO + Fusion(mvs-moge-depth-fusion)

- [ ] 5.1 实现 COLMAP depth map 读取(geometric),统一输出为 `Float32[H W]` 并标记有效像素
- [ ] 5.2 实现 `scale+shift` 对齐: 仅在(非 inpaint)且(MVS valid)且(MoGe valid)像素上拟合
- [ ] 5.3 实现融合: MVS valid 像素用 MVS depth,其余(含 inpaint 与 MVS hole)用对齐后的 MoGe depth
- [ ] 5.4 实现融合后的 confidence mask 语义,并在无 overlap/有效像素过少时安全回退到 MoGe-only

## 6. Pipeline Integration(singleimg-colmap-mvs-refine)

- [ ] 6.1 在 `SingleImagePipeline` 增加 `_colmap_mvs_refine_depth_and_rebuild_scene()` 阶段,仅在 `stage=coarse` 且 enable 时触发
- [ ] 6.2 实现重建: 用融合后的 depth 重建 frames,重新初始化 `Gaussian_Scene`,并再训练一轮(参数可配置)
- [ ] 6.3 实现严格回退: 任一步失败 -> 打印原因与建议 -> 保留原 scene -> 继续渲染/导出
- [ ] 6.4 输出 summary: 参与帧数,pairs 数,每帧 MVS coverage,融合 coverage,耗时,是否回退

## 7. Tests

- [ ] 7.1 增加单测: pairs 生成的数量/去重/排序稳定性(N=50,window/stride 组合)
- [ ] 7.2 增加单测: KeypointIndexer 去重与 index 稳定性(重复/近邻点)
- [ ] 7.3 增加单测: 对极几何过滤(合成 K+pose+匹配,验证外点被剔除)
- [ ] 7.4 增加单测: depth 融合策略(MVS 主干 + MoGe 补洞 + 无 overlap 回退)
- [ ] 7.5 (可选) 增加 smoke test: 若检测到 `colmap` CLI 存在,用极小数据跑通 `point_triangulator`(默认 skip)

## 8. Docs

- [ ] 8.1 更新 `docs/single_img_pipeline.md`: 增加离线精修的启用方式,依赖安装,运行耗时预期,以及常见失败与回退解释
