## ADDED Requirements

### Requirement: Fuse MVS depth (geometric) with MoGe depth for full coverage
系统 SHALL 将 MVS depth 与 MoGe depth 融合为最终 depth,用于重建/训练.
融合策略 MUST 满足:
- 在 MVS 有效像素上,以 MVS depth 为主干.
- 在 MVS 无效像素或 inpaint 区域,使用对齐后的 MoGe depth 补洞.

#### Scenario: MVS pixels dominate in valid regions
- **WHEN** 某像素在 MVS depth map 中有效(深度为有限正数)
- **THEN** 融合后的 depth MUST 等于 MVS depth(允许数值类型转换误差)

#### Scenario: MoGe fills holes where MVS is invalid
- **WHEN** 某像素在 MVS depth map 中无效(缺失/0/NaN/Inf)
- **THEN** 融合后的 depth MUST 使用 MoGe depth(在对齐后)

### Requirement: MoGe depth must be aligned to MVS depth via scale+shift on overlap
系统 SHALL 在 MVS 有效区域上拟合 MoGe depth 的 `scale,shift`,并用该对齐结果参与补洞.

#### Scenario: Alignment uses only overlapping valid pixels
- **WHEN** 计算 `scale,shift`
- **THEN** 系统 MUST 仅使用同时满足:
  - MVS depth 有效
  - mask 允许参与几何(非 inpaint)
  - MoGe depth 有效
  的像素点进行拟合

#### Scenario: No-overlap fallback is safe
- **WHEN** 某一帧的 MVS depth 完全无有效像素(或有效像素过少无法拟合)
- **THEN** 系统 MUST 对该帧回退到 MoGe-only depth

### Requirement: Output confidence mask must reflect fused depth provenance
系统 SHALL 输出一个与融合 depth 同尺寸的置信度 mask,供后续训练/统计使用.

#### Scenario: Confidence is true on MVS-valid pixels
- **WHEN** 某像素来自 MVS depth
- **THEN** 输出 confidence mask MUST 为 true

#### Scenario: Confidence on MoGe-filled pixels is conservative
- **WHEN** 某像素来自 MoGe 补洞(含 inpaint 区域)
- **THEN** 输出 confidence mask MUST 使用 MoGe 的 confidence(或更保守的 false),并保持语义一致
