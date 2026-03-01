# VistaDream Single Image Pipeline

This document explains the `single_img_pipeline` in `src/vistadream/api/single_img_pipeline.py`: what it does, how the data flows, and the roles of the main components it orchestrates.

## Overview
- Purpose: Turn a single input image into an expanded scene by outpainting, estimating depth, building a Gaussian Splat representation, and rendering a short camera trajectory.
- Core stages:
  - Outpainting via Flux to expand the input
  - Single-pass depth estimation on the outpainted image (reused for the input crop)
  - Initialize and iteratively grow a Gaussian scene with trainable splats
  - Render intermediate and final results, while logging to Rerun

## Stages (`--stage`)
The single-image CLI supports four stages. The key practical difference is whether coarse expansion runs and adds extra inpainted views into the final Gaussian scene (PLY).

- `no-outpaint` (default): Skip Flux. Only build the input frame (no extra border canvas). Exports a PLY built from a single supervised view.
- `outpaint`: Run Flux outpainting once to expand the border, then initialize the Gaussian scene from two views: `input` + `outpaint`. No coarse expansion.
- `fine`: Currently equivalent to `outpaint` in this repo (initialization only; no coarse expansion).
- `coarse`: Run `outpaint` initialization first, then iteratively:
  1) propose a camera motion pose, 2) render a hole mask, 3) pick a good pose, 4) RGBD inpaint, 5) add/train new splats.

If `--stage coarse` ends up selecting **zero** extra frames (see `[COARSE] Summary`), the exported PLY will look almost the same as `outpaint/fine` because the pipeline never added new views.

See also: [`docs/coarse_stage_guide.md`](coarse_stage_guide.md) (coarse frame selection diagnostics and tuning guide).

## Key Components
- `SingleImagePipeline`: Orchestration class coordinating the whole flow.
- `FluxInpainting`: Wrapper around Flux-Dev-Fill model for image inpainting.
- `RelativeDepthPrediction` + `MogeV1Predictor`: Predicts relative depth and intrinsic `K_33` from RGB.
- `Gaussian_Scene` and `Frame`: Scene data structure and per-view buffers for RGB/depth/masks/intrinsics/extrinsics.
- `Smooth_Connect_Tool`: Aligns predicted depth to rendered depth and smooths boundaries.
- `GS_Train_Tool`: Optimizer over Gaussian splats for newly added frames.
- `rerun` logging + blueprint: Visualize RGB/depth/masks, camera frusta, and the evolving scene.

## Data Contracts (Shapes/Dtypes)
- `rgb_hw3`: `UInt8[np.ndarray, "H W 3"]` in [0,255] for Flux and logging; frames store `float32` in [0,1].
- `dpt`: `Float[np.ndarray, "H W"]` real depth (meters).
- `inpaint`: `Bool[np.ndarray, "H W"]` supervision mask for training; `True` = supervise.
- `inpaint_wo_edge`: `Bool[np.ndarray, "H W"]` refined mask excluding depth edges and low-confidence areas.
- `dpt_conf_mask`: `Bool[np.ndarray, "H W"]` where depth is confident.
- `intrinsic`: `Float[np.ndarray, "3 3"]` pinhole intrinsics.
- `cam_T_world`: `Float[np.ndarray, "4 4"]` camera-to-world transform.

## High-Level Flow

```mermaid
flowchart TD
    A[Input image] --> B[process_image to max res]
    B --> C[add_border_and_mask
            expansion_percent → border]
    C --> D[FluxInpainting(outpaint_img, mask)
            → outpainted RGB]
    D --> E[Relative depth predictor
            → depth + K_33 + confidence]
    E --> F[Build outpaint Frame
            (rgb, dpt, masks, intr/extr)]
    E --> G[Build input Frame using
            outpaint dpt in original ROI
            (inpaint = all True)]
    F --> H[Log frames to Rerun]
    G --> H
    H --> I[Initialize Gaussian_Scene
            and train 100 iters]
    I --> J{run_coarse?}
    J -- yes --> K[Generate dense camera
                    trajectory]
    K --> L[Select poses with large but
             bounded inpaint area]
    L --> M[Render-for-inpaint → Frame,
             Flux inpaint, predict depth,
             align depth, refine masks]
    M --> N[Add frame, train 500 iters,
             log, repeat]
    J -- no --> O[Skip coarse]
    N --> P[Final render trajectory,
             log final point cloud + video]
    O --> P
```

## Initialization Details (`_initialize`)
- Resize input via `process_image` to fit `max_resolution` while respecting 32-multiple constraints.
- Compute outpainting canvas and binary mask with `add_border_and_mask`, using `expansion_percent` distributed per side.
- Run `FluxInpainting` once to fill the border; convert output to `rgb_hw3`.
- Run a single relative depth pass on the outpainted RGB to obtain:
  - `depth`: `H×W` float, NaN/Inf sanitized to 0
  - `confidence`: scalar per pixel → thresholded into `dpt_conf_mask`
  - `K_33`: used to build intrinsics for both frames
- Build outpaint `Frame`:
  - `inpaint` = the border mask; `inpaint_wo_edge` = mask ∧ ¬`depth_edges_mask(depth)` ∧ `dpt_conf_mask`
  - `intrinsic` and `PinholeParameters` from predictor `K_33` and image size
  - Extrinsics = identity (world origin)
- Build input `Frame` reusing the same depth pass:
  - Take the original image ROI from the outpainted depth and masks
  - Adjust `K_33` principal point to the input center
  - `inpaint` = all `True` to supervise all original pixels during training
  - `inpaint_wo_edge` = ¬edges ∧ `dpt_conf_mask` (restricted to ROI)
- Log both frames to Rerun and bootstrap the `Gaussian_Scene` by adding the two frames and running `GS_Train_Tool` for 100 iterations.

Note on training logic: `inpaint=True` means "supervise this pixel." For the input view we set all pixels to `True` to force perfect reproduction of original content. For the outpaint view, we supervise only the new regions.

## Coarse Expansion (`_coarse`)
- Generate a dense trajectory: `_generate_trajectory(scene, nframes = n_frames * coarse_dense_multiplier)` using `scene.traj_type` (default spiral).
- Score each candidate pose by rendering with `scene._render_for_inpaint(frame)` and measuring the fraction of `inpaint=True`.
- Selection policy (configurable; defaults shown):
  - `coarse_max_inpaint_ratio=0.25`: discard poses with inpaint ratio > max
  - `coarse_min_inpaint_ratio=0.05`: require inpaint ratio ≥ min
  - `coarse_adjacent_exclusion=1`: exclude poses within ±N of already-selected indices (avoid near-duplicates)
  - Pick the remaining pose with the largest inpaint ratio
  - Optional fallback: `coarse_fallback_mode=closest` can pick a "closest-to-band" pose even if no candidate falls into `[min,max]`
- For each selected pose:
  - Create a `Frame` via `pose_to_frame` (optionally expanded by `margin`)
  - Run `FluxInpainting` on its `inpaint` area
  - Predict depth for the inpainted RGB
  - Align predicted depth to the frame's original rendered depth using `Smooth_Connect_Tool._affine_dpt_to_GS` (coarse scale/shift + smooth refinement) and recompute masks:
    - `inpaint_wo_edge` = `inpaint` ∧ ¬`depth_edges_mask(aligned_dpt)` ∧ `dpt_conf_mask`
  - Log to Rerun, add to `Gaussian_Scene`, and run `GS_Train_Tool` for 500 iterations
  - Update the blueprint tab to "Coarse Scene"

### Coarse Summary and Selection Diagnostics
Coarse selection has two layers of "don't guess" logging:

1) A final summary (enabled by default via `coarse_print_summary=True`), which answers: "Did coarse actually select new frames, and how many splats were added?"

- `selected_extra_frames=X/Y`: how many extra views were successfully selected and added
- Per-frame stats:
  - `dense_pose_index`: where the pose comes from in the dense trajectory
  - `inpaint_ratio` / `inpaint_pixels`: hole size before filtering
  - `inpaint_wo_edge_pixels`: pixels that survive the edge + confidence filtering (this is what actually becomes new splats)
  - `added_splats`: how many splats this frame contributed after filtering
- Scene totals:
  - `scene_frames`, `gaussian_frames`, `total_splats`

2) A failure diagnostic (printed only when selection fails), prefixed with `[COARSE][DIAG]`, which tells you *why* no frame was selected:
- `inpaint_ratio(valid) stats: min/mean/max`
- `counts(valid): below_min / in_range / above_max (in_range_without_adj=...)`
- `topK(idx:ratio)=...` to show where the "most hole" part of the trajectory is

Use these numbers to decide whether you should change motion, thresholds, or adjacency exclusion (see tuning below).

### Tuning Guide (Make Coarse More "Aggressive")
If you see `selected_extra_frames=0` in the summary, coarse effectively degraded to initialization-only. Use the diagnostic hint to pick the right knob:

- Holes are too small (`max < coarse_min_inpaint_ratio`):
  - Increase camera motion: `--traj-forward-ratio`, `--traj-backward-ratio`, and/or widen `--traj-max-percentage`
  - Increase render margin: `--coarse-margin`
  - Or lower the minimum: `--coarse-min-inpaint-ratio`

- Holes are too large (`min > coarse_max_inpaint_ratio`):
  - Increase the maximum: `--coarse-max-inpaint-ratio`
  - Or reduce motion / reduce margin

- Blocked by adjacency exclusion (`in_range_without_adj > 0` but `in_range=0`):
  - Set `--coarse-adjacent-exclusion 0`
  - Or increase dense sampling: `--coarse-dense-multiplier` (slower, but increases the chance of landing in-band)

If you want a single "force it to pick something" switch for experimentation and PLY-difference amplification:
- Use `--coarse-fallback-mode closest`.
  - When strict selection finds no in-range candidate, it will pick a pose closest to the `[min,max]` band and continue.
  - The log line for the selected frame will include `(fallback_mode=closest)` if the chosen ratio is out of range.

Note: increasing `coarse_dense_multiplier` can significantly slow down coarse selection because it renders every candidate pose to measure the inpaint mask ratio.

### Frame Selection Logic
```mermaid
sequenceDiagram
    participant SC as Gaussian_Scene
    participant Traj as _generate_trajectory
    participant Sel as _next_frame
    participant Flux as FluxInpainting
    participant Pred as Depth Predictor
    SC->>Traj: Make dense cam_T_world traj
    Sel->>SC: pose_to_frame(cam_T_world, margin)
    Sel->>Sel: inpaint_ratio = mean(mask)
    Sel->>Sel: Filter: ratio <= coarse_max_inpaint_ratio, non-adjacent
    Sel->>Sel: Require ratio >= coarse_min_inpaint_ratio
    Sel->>Sel: Optional fallback: coarse_fallback_mode=closest
    Sel-->>SC: Return best next Frame
    SC->>Flux: Inpaint selected Frame RGB
    SC->>Pred: Predict depth on inpainted RGB
```

## Final Rendering (`_render_splats`)
- Build a final trajectory of up to 150 frames (scaled to `len(scene.frames)*25`).
- Log all splats as a single `Points3D` entity (positions from concatenated `gf.xyz`, colors from `sigmoid(gf.rgb)`).
- For each camera on the trajectory:
  - Render `RGB+Depth` with `scene._render_RGBD`
  - Log RGB and Depth along with the camera pinhole parameters (`Intrinsics`/`Extrinsics`)

## Rerun Blueprint (UI)
- Tabs: Initialization, Coarse Scene, Final Scene.
- 2D panels per camera: RGB / Depth / Confidence / Inpaint masks.
- 3D view excludes depth/mask tensors to keep the world view clean, shows camera frusta and point clouds.
- Orientation-aware layout: adjusts column shares for portrait vs. landscape input.

## Important Implementation Notes
- Depth reuse: Only one depth pass is computed (on the outpainted image). The input's depth and masks are cropped from that result, saving compute and ensuring consistent scaling.
- Intrinsics handling: Predictor `K_33` is reused but principal point is recentered for the cropped input frame.
- Mask semantics: `inpaint=True` means "supervise" in training, not "needs synthetic content now." This is why the input frame sets it to all `True`.
- Depth alignment: `Smooth_Connect_Tool` first finds a scale/shift to match rendered depth, then smooths differences at the inpaint boundary by iterative blurring, preventing seams.
- Image shape constraints: The pipeline relies on `process_image` to ensure sizes compatible with model/grid constraints (multiples-of-32 rule).

## Entrypoints and Usage
- Pipeline module: `src/vistadream/api/single_img_pipeline.py`
- CLI: `tools/run_single_img.py` (single-image pipeline) or `tools/run_vistadream.py` (full pipeline)
- Quick run example:
  - `pixi run example`
  - Or: `pixi run python tools/run_vistadream.py --image-path <img> --n-frames 10 --expansion-percent 0.2`

### Practical CLI Examples
- Initialization-only (no Flux outpaint):
  - `pixi run python tools/run_single_img.py --image-path data/office/IMG_4029.jpg --stage no-outpaint`

- Initialization-only with outpainting (same behavior for `outpaint` and `fine` in this repo):
  - `pixi run python tools/run_single_img.py --image-path data/office/IMG_4029.jpg --stage outpaint`
  - `pixi run python tools/run_single_img.py --image-path data/office/IMG_4029.jpg --stage fine`

- Coarse expansion (adds extra inpainted views when selection succeeds):
  - `pixi run python tools/run_single_img.py --image-path data/office/IMG_4029.jpg --stage coarse --export-gaussians-ply-path data/coarse`

- Coarse expansion with a "force pick something" fallback (more aggressive):
  - `pixi run python tools/run_single_img.py --image-path data/office/IMG_4029.jpg --stage coarse --coarse-fallback-mode closest --export-gaussians-ply-path data/coarse`

Note: if `--export-gaussians-ply-path` is a directory (no `.ply` suffix), the pipeline writes `gf.ply` inside that directory.

```mermaid
classDiagram
    class SingleImagePipeline {
      -config: SingleImageConfig
      -scene: Gaussian_Scene
      -flux_inpainter: FluxInpainting
      -predictor: BaseRelativePredictor
      -smooth_connector: Smooth_Connect_Tool
      -shared_intrinsics: Intrinsics
      -logged_cam_idx_list: list[int]
      +__call__()
      -_initialize()
      -_coarse()
      -_inpaint_next_frame(Frame) Frame
      -_render_splats()
      -_create_blueprint() Blueprint
    }
    class Gaussian_Scene {
      +_render_RGBD(Frame) (rgb, dpt, alpha)
      +_render_for_inpaint(Frame) Frame
      +_add_trainable_frame(Frame, require_grad)
      traj_type, traj_* params
    }
    class Frame {
      H, W
      rgb: float[H,W,3]
      dpt: float[H,W]
      inpaint, inpaint_wo_edge, dpt_conf_mask: bool[H,W]
      intrinsic: float[3,3]
      cam_T_world: float[4,4]
    }
    SingleImagePipeline --> Gaussian_Scene
    SingleImagePipeline --> FluxInpainting
    SingleImagePipeline --> Smooth_Connect_Tool
    SingleImagePipeline --> Frame
```

---
Suggested next additions (if needed): add a dedicated troubleshooting page with example `[COARSE] Summary` and `[COARSE][DIAG]` logs from real runs, plus recommended parameter presets for "small motion indoor" vs "large motion outdoor" scenes.
