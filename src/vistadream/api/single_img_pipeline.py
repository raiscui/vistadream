import gc
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import Literal

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from einops import rearrange
from icecream import ic
from jaxtyping import Bool, Float, UInt8
from monopriors.depth_utils import depth_edges_mask, depth_to_points
from monopriors.relative_depth_models import (
    RelativeDepthPrediction,
    get_relative_predictor,
)
from monopriors.relative_depth_models.base_relative_depth import BaseRelativePredictor
from PIL import Image
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.rerun_log_utils import log_pinhole

from vistadream.ops.connect import Smooth_Connect_Tool
from vistadream.ops.flux import FluxInpainting, FluxInpaintingConfig
from vistadream.ops.gs.basic import Frame, Gaussian_Scene, save_ply
from vistadream.ops.gs.train import GS_Train_Tool
from vistadream.ops.trajs import _generate_trajectory
from vistadream.rerun_setup import VistaRerunConfig, init_rerun_from_config, maybe_wait_after_run
from vistadream.resize_utils import add_border_and_mask, process_image


def log_frame(parent_log_path: Path, frame: Frame, cam_params: PinholeParameters, log_pcd: bool = False) -> None:
    cam_log_path: Path = parent_log_path / cam_params.name
    pinhole_log_path: Path = cam_log_path / "pinhole"
    # extract values from frame
    rgb_hw3: UInt8[np.ndarray, "H W 3"] = (deepcopy(frame.rgb) * 255).astype(np.uint8)
    depth_hw: Float[np.ndarray, "H W"] = deepcopy(frame.dpt)
    edges_mask: Bool[np.ndarray, "h w"] = depth_edges_mask(depth_hw, threshold=0.01)
    masked_depth_hw: Float[np.ndarray, "h w"] = depth_hw * ~edges_mask
    inpaint_mask: Bool[np.ndarray, "H W"] = deepcopy(frame.inpaint)
    inpaint_wo_edge_mask: Bool[np.ndarray, "H W"] = deepcopy(frame.inpaint_wo_edge)
    depth_conf_mask: Bool[np.ndarray, "H W"] = deepcopy(frame.dpt_conf_mask)
    # convert masked_depth to uint16 for depth image
    masked_depth_hw = (masked_depth_hw * 1000).astype(np.uint16)  # Convert to uint16 for depth image

    rr.log(f"{pinhole_log_path}/rgb", rr.Image(rgb_hw3, color_model=rr.ColorModel.RGB).compress())
    rr.log(f"{pinhole_log_path}/depth", rr.DepthImage(masked_depth_hw, meter=1000.0))
    rr.log(
        f"{pinhole_log_path}/dpt_conf_mask",
        rr.Image(depth_conf_mask.astype(np.uint8) * 255, color_model=rr.ColorModel.L).compress(),
    )
    rr.log(
        f"{pinhole_log_path}/inpaint_mask",
        rr.Image(inpaint_mask.astype(np.uint8) * 255, color_model=rr.ColorModel.L),
    )
    rr.log(
        f"{pinhole_log_path}/inpaint_wo_edges_mask",
        rr.Image(inpaint_wo_edge_mask.astype(np.uint8) * 255, color_model=rr.ColorModel.L),
    )
    log_pinhole(camera=cam_params, cam_log_path=cam_log_path, image_plane_distance=0.05)

    if log_pcd:
        # convert back to f32 for point cloud logging
        depth_1hw: Float[np.ndarray, "1 h w"] = rearrange(masked_depth_hw, "h w -> 1 h w").astype(np.float32) / 1000
        pts_3d: Float[np.ndarray, "h w 3"] = depth_to_points(
            depth_1hw, cam_params.intrinsics.k_matrix.astype(np.float32)
        )
        rr.log(
            f"{parent_log_path}/{cam_params.name}_point_cloud",
            rr.Points3D(
                positions=pts_3d.reshape(-1, 3),
                colors=rgb_hw3.reshape(-1, 3),
            ),
        )


@dataclass
class SingleImageConfig:
    """
    Configuration for Single Image Processing.
    """

    rr_config: VistaRerunConfig
    image_path: Path
    offload: bool = True
    num_steps: int = 25
    guidance: float = 30.0
    expansion_percent: float = 0.3
    n_frames: int = 10
    # coarse 阶段: 为了选帧更稳定,会对轨迹做更密集的采样,采样帧数 = n_frames * coarse_dense_multiplier.
    # 说明:
    # - 值越大,候选姿态越多,更容易找到“洞比例合适”的帧.
    # - 代价是 `_next_frame()` 会更慢,因为它会对每个候选姿态渲染一次 mask 做统计.
    coarse_dense_multiplier: int = 10
    # coarse 阶段: pose_to_frame 的额外像素 margin.
    # 说明:
    # - 值越大,相机位姿轻微变化也更容易产生洞,从而更容易触发 inpaint 生成新视角.
    # - 代价是每帧渲染更大,速度更慢,显存压力更高.
    coarse_margin: int = 32
    # coarse 选帧阈值: 允许的最大洞(inpaint)比例.
    # 说明:
    # - 默认 0.25 比较保守,避免一次 inpaint 过大导致质量下降/失败.
    # - 你想更激进时可以提高到 0.35/0.45,但需要接受更多的 inpaint 区域与更高的不确定性.
    coarse_max_inpaint_ratio: float = 0.25
    # coarse 选帧阈值: 允许的最小洞(inpaint)比例.
    # 说明:
    # - 洞太小,即使 inpaint 也不会给高斯带来明显增量,因此默认要求至少 5%.
    coarse_min_inpaint_ratio: float = 0.05
    # coarse 选帧策略: 剔除已选帧附近的相邻帧,避免连续帧视角过近.
    # 说明:
    # - 默认剔除 ±1 帧.
    # - 设为 0 可以取消相邻剔除(更激进,但更可能选到相似视角,增量可能更小).
    coarse_adjacent_exclusion: int = 1
    # coarse: 是否在结束时打印结构化摘要,用于确认 coarse 是否真的选到了新帧以及新增 splats 数量.
    coarse_print_summary: bool = True
    # coarse: 选帧失败时的兜底策略(默认关闭,避免无意中变得“更激进”).
    # 说明:
    # - none: 严格遵守 [min,max] 阈值,选不到就停.
    # - closest: 当严格模式选不到时,从剩余候选里选一个“最接近阈值区间”的帧继续推进.
    #   这更激进,常用于: (1) 验证 coarse 流程是否真的能补帧; (2) 希望补出更明显几何差异.
    coarse_fallback_mode: Literal["none", "closest"] = "none"
    # 相机轨迹参数(影响相机动作幅度,从而影响洞面积与 coarse 是否容易选到新帧).
    # 说明:
    # - 这几个参数会写入 `Gaussian_Scene` 并影响 `_generate_trajectory(...)`.
    # - 想让 coarse 更“动起来”,通常优先调大 `traj_forward_ratio`/`traj_backward_ratio`,
    #   或放宽 `traj_min_percentage`/`traj_max_percentage` 来增大 radius.
    traj_forward_ratio: float = 0.3
    traj_backward_ratio: float = 0.4
    traj_min_percentage: float = 5.0
    traj_max_percentage: float = 50.0
    # 说明:
    # - 用于控制输入图片最大边长,越小越省显存.
    # - 建议使用 32 的倍数,否则会在内部被向下对齐到 32 的倍数.
    max_resolution: int = 512
    # MoGe 深度模型的运行设备. 显存紧张时可以用 cpu 兜底(会显著变慢,但更稳).
    depth_device: Literal["cuda", "cpu"] = "cuda"
    stage: Literal["no-outpaint", "outpaint", "coarse", "fine"] = "no-outpaint"
    # 是否导出 3DGS 高斯点云(PLY). 默认开启,方便离线复用/调试.
    export_gaussians: bool = True
    # 导出 PLY 的目标路径.
    # 说明:
    # - 之前版本写死到 `data/test_dir/gf.ply`,用户不容易发现产物.
    # - 这里保留同样的默认值,但允许通过 CLI 覆盖.
    # - 如果你传的是目录(没有 .ply 后缀),会自动在目录下补上 `gf.ply`.
    export_gaussians_ply_path: Path = Path("data/test_dir/gf.ply")


@dataclass
class CoarseSelectedFrameStat:
    """
    coarse 阶段的选帧统计信息.

    说明:
    - dense_pose_index: 在 dense trajectory 里的姿态索引(用于定位是轨迹上的哪一帧被选中).
    - inpaint_ratio: 该姿态下,渲染得到的洞(inpaint mask)面积比例.
    - inpaint_pixels / inpaint_wo_edge_pixels: 洞区域像素数量,以及去除边缘+低置信后的像素数量.
    - added_splats: 本帧实际新增到场景里的 splats 数量(以 Gaussian_Frame 过滤后的数量为准).
    """

    dense_pose_index: int
    inpaint_ratio: float
    inpaint_pixels: int
    inpaint_wo_edge_pixels: int
    added_splats: int


def pose_to_frame(scene: Gaussian_Scene, cam_T_world: Float[np.ndarray, "4 4"], margin: int = 32) -> Frame:
    """
    Convert camera pose to a Frame object with optional margin expansion.

    Args:
        scene: Gaussian scene containing reference frames
        cam_T_world: Camera-to-world transformation matrix
        margin: Additional pixels to add to frame dimensions

    Returns:
        Frame object with rendered content for inpainting
    """
    # Calculate expanded dimensions
    base_height: int = scene.frames[0].H
    base_width: int = scene.frames[0].W
    expanded_height: int = base_height + margin
    expanded_width: int = base_width + margin

    # Create adjusted intrinsics for expanded frame
    base_intrinsic: Float[np.ndarray, "3 3"] = deepcopy(scene.frames[0].intrinsic)
    adjusted_intrinsic: Float[np.ndarray, "3 3"] = base_intrinsic.copy()
    adjusted_intrinsic[0, 2] = expanded_width / 2.0  # cx - principal point x
    adjusted_intrinsic[1, 2] = expanded_height / 2.0  # cy - principal point y

    # Create frame with expanded dimensions and adjusted intrinsics
    frame: Frame = Frame(
        H=expanded_height,
        W=expanded_width,
        intrinsic=adjusted_intrinsic,
        cam_T_world=cam_T_world,
    )

    # Render the frame for inpainting using the scene
    rendered_frame: Frame = scene._render_for_inpaint(frame)

    return rendered_frame


def _select_best_pose_index(
    inpaint_area_ratio_array: Float[np.ndarray, "n_frames"],
    selected_indices: list[int],
    *,
    min_inpaint_ratio: float,
    max_inpaint_ratio: float,
    adjacent_exclusion: int,
    fallback_mode: Literal["none", "closest"] = "none",
) -> tuple[int, float] | None:
    """
    在 dense trajectory 的候选姿态中,选择一个“洞比例合适”的最佳索引.

    说明:
    - 这是一个纯函数,用于把 coarse 选帧逻辑从渲染/估计流程中解耦出来,便于单测与调参.
    - 选帧策略: 过滤掉洞比例过大的候选(> max_inpaint_ratio),再剔除已选帧附近的相邻帧,
      最后在剩余候选里取洞比例最大的那一帧.
    """
    if inpaint_area_ratio_array.size == 0:
        return None

    # 复制一份用于过滤,避免调用方意外复用同一数组导致难以调试.
    ratios: Float[np.ndarray, "n_frames"] = inpaint_area_ratio_array.astype(np.float32).copy()
    n_frames: int = int(ratios.shape[0])

    # 先应用 adjacent_exclusion,得到仍可选的索引集合.
    # 说明:
    # - 这里用 mask 而不是直接把 ratio 写成 0,这样后续 diagnostics/fallback 更直观.
    valid_mask: Bool[np.ndarray, "n_frames"] = np.ones((n_frames,), dtype=np.bool_)
    if adjacent_exclusion < 0:
        adjacent_exclusion = 0
    for selected_idx in selected_indices:
        for offset in range(-adjacent_exclusion, adjacent_exclusion + 1):
            idx = selected_idx + offset
            if 0 <= idx < n_frames:
                valid_mask[idx] = False

    valid_indices: np.ndarray = np.nonzero(valid_mask)[0]
    if valid_indices.size == 0:
        return None

    valid_ratios: np.ndarray = ratios[valid_indices]

    # 严格模式: 只在 [min,max] 区间内选,并取 ratio 最大的那一帧.
    in_range_mask: np.ndarray = (valid_ratios >= float(min_inpaint_ratio)) & (valid_ratios <= float(max_inpaint_ratio))
    if np.any(in_range_mask):
        masked: np.ndarray = valid_ratios.copy()
        masked[~in_range_mask] = -np.inf
        best_local: int = int(np.argmax(masked))
        best_idx: int = int(valid_indices[best_local])
        return best_idx, float(ratios[best_idx])

    if fallback_mode == "none":
        return None

    # fallback 模式: 选一个“最接近阈值区间”的帧继续推进.
    # 说明:
    # - 如果所有候选都 > max,会选 ratio 最小的那一帧(洞相对最小,更稳).
    # - 如果所有候选都 < min,会选 ratio 最大的那一帧(洞相对最大,更激进).
    # - 如果两边都有,会选离 [min,max] 最近的那一帧.
    if fallback_mode != "closest":
        raise ValueError(f"Unknown fallback_mode={fallback_mode!r}")

    dist_to_band: np.ndarray = np.zeros_like(valid_ratios, dtype=np.float32)
    dist_to_band[valid_ratios < float(min_inpaint_ratio)] = float(min_inpaint_ratio) - valid_ratios[valid_ratios < float(min_inpaint_ratio)]
    dist_to_band[valid_ratios > float(max_inpaint_ratio)] = valid_ratios[valid_ratios > float(max_inpaint_ratio)] - float(max_inpaint_ratio)
    # ratio<=0 代表几乎没有洞,选它没有意义(会浪费一次 inpaint + 深度预测).
    dist_to_band[valid_ratios <= 0.0] = np.inf

    best_local: int = int(np.argmin(dist_to_band))
    if not np.isfinite(dist_to_band[best_local]):
        return None
    best_idx: int = int(valid_indices[best_local])
    return best_idx, float(ratios[best_idx])


def _print_coarse_frame_selection_diagnostics(
    inpaint_area_ratio_array: Float[np.ndarray, "n_frames"],
    selected_indices: list[int],
    *,
    min_inpaint_ratio: float,
    max_inpaint_ratio: float,
    adjacent_exclusion: int,
    topk: int = 10,
) -> None:
    """
    coarse 选帧失败时的诊断信息(只打印日志,不改变行为).

    为什么需要它:
    - 选不到帧通常只有三类原因:
      1) 洞太小: 所有 ratio < min
      2) 洞太大: 所有 ratio > max
      3) 洞在阈值内的帧被 adjacent_exclusion 全部剔除了
    - 没有分布信息时,用户很难判断应该调 motion 还是调阈值.
    """
    if inpaint_area_ratio_array.size == 0:
        print("[COARSE][DIAG] No candidate poses (empty dense trajectory).")
        return

    # NOTE: 复制一份,避免上游误复用同一数组导致调参时“看起来数值变了”.
    ratios: Float[np.ndarray, "n_frames"] = inpaint_area_ratio_array.astype(np.float32).copy()
    n_frames: int = int(ratios.shape[0])

    # 计算 adjacent_exclusion 后仍然“可用”的候选集合.
    valid_mask: Bool[np.ndarray, "n_frames"] = np.ones((n_frames,), dtype=np.bool_)
    if adjacent_exclusion < 0:
        adjacent_exclusion = 0
    for selected_idx in selected_indices:
        for offset in range(-adjacent_exclusion, adjacent_exclusion + 1):
            idx = selected_idx + offset
            if 0 <= idx < n_frames:
                valid_mask[idx] = False

    valid_indices: np.ndarray = np.nonzero(valid_mask)[0]
    excluded_by_adjacent: int = int(n_frames - valid_indices.shape[0])
    if valid_indices.size == 0:
        print(
            "[COARSE][DIAG] All candidates were excluded by adjacent_exclusion. "
            f"selected_indices={selected_indices}, adjacent_exclusion={adjacent_exclusion}."
        )
        return

    valid_ratios: np.ndarray = ratios[valid_indices]

    # --- 统计分布(用于判断是“太小”还是“太大”) ---
    min_ratio: float = float(np.min(valid_ratios))
    mean_ratio: float = float(np.mean(valid_ratios))
    max_ratio: float = float(np.max(valid_ratios))

    # --- 统计过滤原因(以 valid 候选集合为准) ---
    below_min_count: int = int(np.sum(valid_ratios < float(min_inpaint_ratio)))
    above_max_count: int = int(np.sum(valid_ratios > float(max_inpaint_ratio)))
    in_range_count: int = int(
        np.sum((valid_ratios >= float(min_inpaint_ratio)) & (valid_ratios <= float(max_inpaint_ratio)))
    )

    # 同时统计“忽略 adjacent_exclusion 时”的可用数量,用于判断是否是相邻剔除导致失败.
    in_range_count_without_adj: int = int(
        np.sum((ratios >= float(min_inpaint_ratio)) & (ratios <= float(max_inpaint_ratio)))
    )

    print(
        "[COARSE][DIAG] inpaint_ratio(valid) stats: "
        f"min={min_ratio:.3f}, mean={mean_ratio:.3f}, max={max_ratio:.3f} "
        f"(n_valid={valid_indices.size}, excluded_by_adjacent={excluded_by_adjacent})."
    )
    print(
        "[COARSE][DIAG] counts(valid): "
        f"below_min={below_min_count}, in_range={in_range_count}, above_max={above_max_count} "
        f"(in_range_without_adj={in_range_count_without_adj})."
    )

    # Top-K 候选,帮助快速定位“轨迹上哪一段最有洞”.
    k: int = int(min(int(topk), int(valid_indices.size)))
    if k > 0:
        order: np.ndarray = np.argsort(valid_ratios)[::-1]  # descending
        top_local: np.ndarray = order[:k]
        top_pairs = [(int(valid_indices[i]), float(valid_ratios[i])) for i in top_local]
        top_str: str = ", ".join([f"{idx}:{ratio:.3f}" for idx, ratio in top_pairs])
        print(f"[COARSE][DIAG] top{int(k)}(idx:ratio)={top_str}")

    # --- 调参建议(尽量给出可执行方向,避免误导) ---
    if in_range_count > 0:
        # 理论上这种情况下不该失败(应该能选到 max ratio),但仍保留提示,便于未来排查.
        print(
            "[COARSE][DIAG] Unexpected: there are in-range candidates but selection still failed. "
            "If this persists, please report with logs."
        )
        return

    if in_range_count_without_adj > 0 and excluded_by_adjacent > 0:
        print(
            "[COARSE][DIAG] Likely blocked by adjacent_exclusion. "
            "Try `--coarse-adjacent-exclusion 0` or increase dense sampling (coarse_dense_multiplier)."
        )
        return

    if max_ratio < float(min_inpaint_ratio):
        print(
            "[COARSE][DIAG] Holes are too small (< min). "
            "Try increasing motion (`--traj-forward-ratio/--traj-backward-ratio`, `--traj-max-percentage`), "
            "increasing `--coarse-margin`, or lowering `--coarse-min-inpaint-ratio`."
        )
        return

    if min_ratio > float(max_inpaint_ratio):
        print(
            "[COARSE][DIAG] Holes are too large (> max). "
            "Try increasing `--coarse-max-inpaint-ratio`, or reducing motion/margin."
        )
        return

    print(
        "[COARSE][DIAG] No candidate fell into the [min,max] band. "
        "Consider widening thresholds or adjusting motion so hole ratios land in-range."
    )


class SingleImagePipeline:
    """
    Pipeline for Flux Outpainting using VistaDream.
    """

    def __init__(self, config: SingleImageConfig):
        self.config: SingleImageConfig = config
        self.scene: Gaussian_Scene = Gaussian_Scene()
        # 将 CLI 暴露的轨迹参数写回 scene,确保 coarse 选帧与最终渲染使用一致的相机动作幅度.
        # 说明:
        # - 这些参数本质上会影响 trajectory 的半径/前后摆动幅度,从而影响洞(inpaint mask)大小与选帧成功率.
        self.scene.traj_forward_ratio = float(self.config.traj_forward_ratio)
        self.scene.traj_backward_ratio = float(self.config.traj_backward_ratio)
        self.scene.traj_min_percentage = float(self.config.traj_min_percentage)
        self.scene.traj_max_percentage = float(self.config.traj_max_percentage)
        if self.config.stage in ["outpaint", "coarse", "fine"]:
            self.flux_inpainter: FluxInpainting = FluxInpainting(FluxInpaintingConfig())
        # 注意:
        # - MoGe predictor 如果提前常驻在 GPU,可能导致 Flux 模型在 `to(cuda)` 时最后差一点显存就 OOM.
        # - 因此这里改为“延迟加载”: 只有真正需要预测深度时才创建 predictor.
        self.predictor: BaseRelativePredictor | None = None
        self.smooth_connector: Smooth_Connect_Tool = Smooth_Connect_Tool()
        # Initialize rerun with the provided configuration
        self.shared_intrinsics: Intrinsics | None = None
        self.image_plane_distance: float = 0.01
        self.logged_cam_idx_list: list[int] = []
        self.rerun_server_uri: str | None = None

        # Detect image orientation early for blueprint configuration
        input_image: Image.Image = Image.open(self.config.image_path).convert("RGB")
        self.orientation: Literal["landscape", "portrait"] = (
            "landscape" if input_image.width >= input_image.height else "portrait"
        )
        print(f"[INFO] Detected {self.orientation} image ({input_image.width}x{input_image.height})")

        ic("Pipeline initialized with configuration:", self.config)

    def __call__(self):
        # Initialize rerun with the provided configuration
        self.setup_rerun()
        # outpaint -> depth prediction -> scene generation
        self._initialize()
        # 对于非 coarse 的阶段,初始化完成后就不再需要 MoGe/Flux 相关大模型了.
        # 主动释放可以显著降低后续 GS 训练/渲染的峰值显存,减少 OOM 概率.
        if self.config.stage != "coarse":
            self._release_init_models()
        # generate the coarse scene
        if self.config.stage == "coarse":
            self._coarse()

        # 说明:
        # - `_render_splats()` 负责把最终结果记录到 Rerun(便于交互查看).
        # - PLY 导出是“离线产物”,很多用户更关心它落在哪里,因此这里显式打印路径.
        self._render_splats()

        if not self.config.export_gaussians:
            print("[INFO] export_gaussians=False, skip exporting Gaussian PLY.")
            return

        export_path: Path = self.config.export_gaussians_ply_path
        # 兼容用户传目录的情况: `--export-gaussians-ply-path tmp/out_dir`
        if export_path.suffix.lower() != ".ply":
            export_path = export_path / "gf.ply"

        export_path.parent.mkdir(exist_ok=True, parents=True)
        print(f"[INFO] Exporting Gaussian splats to: {export_path}")
        save_ply(self.scene, export_path)

        # 如果压缩工具可用,`save_ply` 会额外生成 `*.compressed.ply`.
        compressed_path: Path = export_path.parent / f"{export_path.stem}.compressed.ply"
        if compressed_path.exists():
            print(f"[INFO] Exported compressed Gaussian PLY to: {compressed_path}")

    def _release_init_models(self) -> None:
        """
        释放初始化阶段使用的大模型,降低后续阶段的显存峰值.

        说明:
        - `stage=coarse` 还会继续用到 predictor/flux,因此不能释放.
        - `stage=fine/outpaint/no-outpaint` 在 `_initialize` 之后就只用 GS 场景渲染/导出,可安全释放.
        """
        released_any: bool = False

        if getattr(self, "predictor", None) is not None:
            try:
                del self.predictor
                released_any = True
            except Exception:
                # 极端情况下 predictor 可能是第三方对象,del 失败也不应影响主流程.
                pass

        if hasattr(self, "flux_inpainter"):
            try:
                del self.flux_inpainter
                released_any = True
            except Exception:
                pass

        if released_any:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _get_predictor(self) -> BaseRelativePredictor:
        """
        延迟创建 MoGe predictor,避免和 Flux 模型争抢初始化阶段的峰值显存.
        """
        if self.predictor is not None:
            return self.predictor

        device: str = self.config.depth_device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        self.predictor = get_relative_predictor("MogeV1Predictor")(device=device)
        return self.predictor

    def _coarse(self):
        """
        Generate coarse scene by iteratively adding frames with good inpainting areas.
        """
        # Generate dense trajectory for frame selection
        dense_multiplier: int = int(self.config.coarse_dense_multiplier)
        if dense_multiplier <= 0:
            dense_multiplier = 10
        dense_nframes: int = int(self.config.n_frames) * dense_multiplier  # Dense trajectory for selection
        dense_cam_T_world_traj: Float[np.ndarray, "n_frames 4 4"] = _generate_trajectory(
            self.scene, nframes=dense_nframes
        )

        # Track selected frame indices to avoid adjacent selections
        select_frames: list[int] = []
        margin: int = int(self.config.coarse_margin)
        if margin < 0:
            margin = 0
        coarse_stats: list[CoarseSelectedFrameStat] = []

        print(f"[INFO] Generating coarse scene with up to {self.config.n_frames} frames...")

        # Iteratively add frames until we reach target count or no good frames found
        for frame_idx in range(self.config.n_frames - 2):  # -2 because we already have input+outpaint frames
            print(f"[INFO] Processing frame {frame_idx + 3}/{self.config.n_frames}...")

            # Find next best frame for inpainting
            next_frame_result = self._next_frame(
                dense_cam_T_world_traj=dense_cam_T_world_traj,
                select_frames=select_frames,
                margin=margin,
                min_inpaint_ratio=float(self.config.coarse_min_inpaint_ratio),
                max_inpaint_ratio=float(self.config.coarse_max_inpaint_ratio),
                adjacent_exclusion=int(self.config.coarse_adjacent_exclusion),
                fallback_mode=str(self.config.coarse_fallback_mode),
            )
            if next_frame_result is None:
                print("[INFO] No more suitable frames found for inpainting")
                break
            next_frame, dense_pose_index, inpaint_ratio = next_frame_result

            # Update logged camera list for blueprint
            cam_idx: int = len(self.scene.frames) + 2  # +2 for input and outpaint frames
            self.logged_cam_idx_list.append(cam_idx)

            # Inpaint the selected frame
            inpainted_frame: Frame = self._inpaint_next_frame(next_frame)
            cam_params: PinholeParameters = PinholeParameters(
                name=f"camera_{cam_idx}",
                intrinsics=self.shared_intrinsics,
                extrinsics=Extrinsics(
                    cam_R_world=inpainted_frame.cam_T_world[:3, :3],
                    cam_t_world=inpainted_frame.cam_T_world[:3, 3],
                ),
            )
            log_frame(self.parent_log_path, inpainted_frame, cam_params)

            # Add frame to scene and optimize
            gf = self.scene._add_trainable_frame(inpainted_frame, require_grad=True)
            # 记录本帧新增的统计信息,用于确认 coarse 是否真的“补进”了新视角.
            inpaint_pixels: int = int(np.sum(inpainted_frame.inpaint))
            inpaint_wo_edge_pixels: int = int(np.sum(inpainted_frame.inpaint_wo_edge))
            added_splats: int = int(gf.xyz.shape[0])
            coarse_stats.append(
                CoarseSelectedFrameStat(
                    dense_pose_index=dense_pose_index,
                    inpaint_ratio=float(inpaint_ratio),
                    inpaint_pixels=inpaint_pixels,
                    inpaint_wo_edge_pixels=inpaint_wo_edge_pixels,
                    added_splats=added_splats,
                )
            )
            self.scene: Gaussian_Scene = GS_Train_Tool(self.scene, iters=500)(self.scene.frames, log=False)
            rr.send_blueprint(blueprint=self._create_blueprint(tab_idx=1))

        # coarse 阶段结束后打印一份摘要,让用户不用肉眼猜“到底有没有选到新帧”.
        if self.config.coarse_print_summary:
            target_extra_frames: int = max(int(self.config.n_frames) - 2, 0)
            print("[COARSE] ==================== Summary ====================")
            print(
                "[COARSE] "
                f"selected_extra_frames={len(coarse_stats)}/{target_extra_frames}, "
                f"dense_nframes={dense_nframes}, "
                f"margin={margin}, "
                f"min_inpaint_ratio={self.config.coarse_min_inpaint_ratio}, "
                f"max_inpaint_ratio={self.config.coarse_max_inpaint_ratio}, "
                f"adjacent_exclusion={self.config.coarse_adjacent_exclusion}, "
                f"fallback_mode={self.config.coarse_fallback_mode}"
            )
            if not coarse_stats:
                print("[COARSE] No extra frames were selected. Consider loosening thresholds or increasing motion.")
            for i, stat in enumerate(coarse_stats, start=1):
                print(
                    "[COARSE] "
                    f"{i:02d}. dense_pose_index={stat.dense_pose_index}, "
                    f"inpaint_ratio={stat.inpaint_ratio:.3f}, "
                    f"inpaint_pixels={stat.inpaint_pixels}, "
                    f"inpaint_wo_edge_pixels={stat.inpaint_wo_edge_pixels}, "
                    f"added_splats={stat.added_splats}"
                )
            total_splats: int = int(sum(int(gf.xyz.shape[0]) for gf in self.scene.gaussian_frames))
            print(
                "[COARSE] "
                f"scene_frames={len(self.scene.frames)}, gaussian_frames={len(self.scene.gaussian_frames)}, total_splats={total_splats}"
            )
            print("[COARSE] =================================================")

    def _next_frame(
        self,
        dense_cam_T_world_traj: Float[np.ndarray, "n_frames 4 4"],
        select_frames: list[int],
        margin: int = 32,
        *,
        min_inpaint_ratio: float = 0.05,
        max_inpaint_ratio: float = 0.25,
        adjacent_exclusion: int = 1,
        fallback_mode: Literal["none", "closest"] = "none",
    ) -> tuple[Frame, int, float] | None:
        """
        Select the frame with largest inpaint holes while staying within ratio thresholds.

        Args:
            dense_cam_T_world_traj: Dense trajectory of camera poses
            select_frames: List of already selected frame indices
            margin: Margin for frame expansion
            min_inpaint_ratio: Minimum inpainting area ratio required to consider a frame suitable
            max_inpaint_ratio: Maximum allowed inpainting area ratio (too-large holes are filtered out)
            adjacent_exclusion: Exclude poses within ±N of already selected indices

        Returns:
            (Frame, dense_pose_index, inpaint_ratio), or None if no suitable frame found
        """
        # Calculate inpaint area ratio for each pose
        inpaint_area_ratios: list[float] = []

        for _pose_idx, cam_T_world in enumerate(dense_cam_T_world_traj):
            temp_frame: Frame = pose_to_frame(self.scene, cam_T_world, margin)
            inpaint_mask: Bool[np.ndarray, "H W"] = temp_frame.inpaint
            inpaint_ratio: float = float(np.mean(inpaint_mask.astype(np.float32)))
            inpaint_area_ratios.append(inpaint_ratio)

        inpaint_area_ratio_array: Float[np.ndarray, "n_frames"] = np.array(inpaint_area_ratios, dtype=np.float32)
        selected = _select_best_pose_index(
            inpaint_area_ratio_array,
            select_frames,
            min_inpaint_ratio=min_inpaint_ratio,
            max_inpaint_ratio=max_inpaint_ratio,
            adjacent_exclusion=adjacent_exclusion,
            fallback_mode=fallback_mode,
        )
        if selected is None:
            _print_coarse_frame_selection_diagnostics(
                inpaint_area_ratio_array,
                select_frames,
                min_inpaint_ratio=min_inpaint_ratio,
                max_inpaint_ratio=max_inpaint_ratio,
                adjacent_exclusion=adjacent_exclusion,
            )
            return None
        best_frame_idx, best_inpaint_ratio = selected

        # Add to selected frames and return the frame
        select_frames.append(best_frame_idx)
        selected_pose: Float[np.ndarray, "4 4"] = dense_cam_T_world_traj[best_frame_idx]
        selected_frame: Frame = pose_to_frame(self.scene, selected_pose, margin)

        extra_note: str = ""
        if fallback_mode != "none" and (
            best_inpaint_ratio < float(min_inpaint_ratio) or best_inpaint_ratio > float(max_inpaint_ratio)
        ):
            extra_note = f" (fallback_mode={fallback_mode})"
        print(f"[INFO] Selected frame {best_frame_idx} with inpaint ratio: {best_inpaint_ratio:.3f}{extra_note}")

        return selected_frame, best_frame_idx, best_inpaint_ratio

    def _inpaint_next_frame(self, frame: Frame) -> Frame:
        """
        Inpaint the given frame using Flux and update depth information.

        Args:
            frame: Frame object containing RGB, depth, and inpaint mask

        Returns:
            Inpainted frame with updated RGB, depth, and masks
        """
        # Convert frame RGB and mask for Flux inpainting
        frame_rgb_hw3: UInt8[np.ndarray, "H W 3"] = (frame.rgb * 255).astype(np.uint8)
        frame_mask: UInt8[np.ndarray, "H W"] = frame.inpaint.astype(np.uint8) * 255

        # Inpaint RGB using Flux
        inpainted_image: Image.Image = self.flux_inpainter(rgb_hw3=frame_rgb_hw3, mask=frame_mask)
        inpainted_rgb_hw3: UInt8[np.ndarray, "H W 3"] = np.array(inpainted_image.convert("RGB"))

        # Update frame with inpainted RGB
        frame.rgb = inpainted_rgb_hw3.astype(np.float32) / 255.0  # Convert to [0,1] range

        # Predict depth for the inpainted frame
        predictor: BaseRelativePredictor = self._get_predictor()
        depth_prediction: RelativeDepthPrediction = predictor.__call__(rgb=inpainted_rgb_hw3, K_33=frame.intrinsic)
        predicted_depth_hw: Float[np.ndarray, "H W"] = depth_prediction.depth

        # Remove any nans or infs and set them to 0
        predicted_depth_hw[np.isnan(predicted_depth_hw) | np.isinf(predicted_depth_hw)] = 0

        # Get the original rendered depth from the scene for alignment
        original_rendered_depth: Float[np.ndarray, "H W"] = frame.dpt

        # Align predicted depth with rendered depth using smooth connector
        aligned_depth_hw: Float[np.ndarray, "H W"] = self.smooth_connector._affine_dpt_to_GS(
            render_dpt=original_rendered_depth,
            inpaint_dpt=predicted_depth_hw,
            inpaint_msk=frame.inpaint,
        ).astype(np.float32)
        # aligned_depth_hw = predicted_depth_hw

        # Update frame depth with aligned depth
        frame.dpt = aligned_depth_hw

        # Calculate depth edges mask using aligned depth
        edges_mask: Bool[np.ndarray, "H W"] = depth_edges_mask(aligned_depth_hw, threshold=0.01)

        # Update inpaint mask without edges
        frame.inpaint_wo_edge = frame.inpaint & ~edges_mask

        # Get depth confidence mask from depth prediction confidence
        dpt_conf_mask: Float[np.ndarray, "H W"] = depth_prediction.confidence
        dpt_conf_mask_bool: Bool[np.ndarray, "H W"] = dpt_conf_mask > 0
        frame.dpt_conf_mask = dpt_conf_mask_bool

        # Further refine inpaint_wo_edge with depth confidence mask
        frame.inpaint_wo_edge = frame.inpaint_wo_edge & dpt_conf_mask_bool

        print(f"[INFO] Inpainted frame with {np.sum(frame.inpaint)}/{frame.inpaint.size} inpaint pixels")

        return frame

    def _render_splats(self):
        # render 5times frames
        nframes: int = min(len(self.scene.frames) * 25 if len(self.scene.frames) > 2 else 150, 150)
        cam_T_world_traj: Float[np.ndarray, "n_frames 4 4"] = _generate_trajectory(self.scene, nframes=nframes)
        # render
        print(f"[INFO] rendering final video with {nframes} frames...")
        rr.send_blueprint(blueprint=self._create_blueprint(tab_idx=2))
        # log the splats as point clouds
        g_xyz: Float[torch.Tensor, "n_splats 3"] = torch.cat(
            [gf.xyz.reshape(-1, 3) for gf in self.scene.gaussian_frames], dim=0
        )
        g_rgb: Float[torch.Tensor, "n_splats 3"] = torch.sigmoid(
            torch.cat([gf.rgb.reshape(-1, 3) for gf in self.scene.gaussian_frames], dim=0)
        )

        # log the gaussian scene
        rr.log(
            f"{self.final_log_path}/final_gaussian_scene",
            rr.Points3D(
                positions=g_xyz.clone().detach().float().cpu().numpy(),
                colors=g_rgb.clone().detach().float().cpu().numpy(),
            ),
            static=True,
        )

        for i, cam_T_world in enumerate(cam_T_world_traj, start=0):
            rr.set_time("time", sequence=i)
            frame = Frame(
                H=self.shared_intrinsics.height,
                W=self.shared_intrinsics.width,
                intrinsic=self.shared_intrinsics.k_matrix,
                cam_T_world=cam_T_world,
            )
            rgb, dpt, alpha = self.scene._render_RGBD(frame)
            rgb: Float[np.ndarray, "H W 3"] = rgb.detach().float().cpu().numpy()
            dpt: Float[np.ndarray, "H W"] = dpt.detach().float().cpu().numpy()

            rgb = (rgb * 255).astype(np.uint8)

            rr.set_time("time", sequence=i)
            cam_log_path: Path = self.final_log_path / "camera"
            pinhole_log_path: Path = cam_log_path / "pinhole"
            pinhole_param = PinholeParameters(
                name="camera",
                intrinsics=self.shared_intrinsics,
                extrinsics=Extrinsics(
                    cam_R_world=cam_T_world[:3, :3],
                    cam_t_world=cam_T_world[:3, 3],
                ),
            )
            rr.log(f"{pinhole_log_path}/rgb", rr.Image(rgb, color_model=rr.ColorModel.RGB).compress())
            rr.log(
                f"{pinhole_log_path}/depth",
                rr.DepthImage((deepcopy(dpt) * 1000).astype(np.uint16), meter=1000.0),
            )
            log_pinhole(
                camera=pinhole_param, cam_log_path=cam_log_path, image_plane_distance=self.image_plane_distance * 10
            )

        print(f"[INFO] DONE rendering {nframes} frames.")

    def _initialize(self) -> None:
        rr.set_time("time", sequence=0)

        input_image: Image.Image = Image.open(self.config.image_path).convert("RGB")
        # ensures image is correctly sized and processed
        # `process_image` 内部会处理缩放,但这里先把参数对齐到 32 的倍数,减少边界情况带来的显存抖动.
        max_dimension: int = int(self.config.max_resolution)
        if max_dimension <= 0:
            max_dimension = 512
        max_dimension = (max_dimension // 32) * 32
        max_dimension = max(32, max_dimension)
        input_image: Image.Image = process_image(input_image, max_dimension=max_dimension)

        if self.config.stage == "no-outpaint":
            # No outpainting, just use the input image directly
            outpaint_img: Image.Image = input_image
            outpaint_mask: Image.Image = Image.new("L", input_image.size, 0)
            # Just use the input image for depth prediction since there's no outpainting
            outpaint_rgb_hw3: UInt8[np.ndarray, "H W 3"] = np.array(outpaint_img.convert("RGB"))
            predictor: BaseRelativePredictor = self._get_predictor()
            outpaint_rel_depth: RelativeDepthPrediction = predictor.__call__(rgb=outpaint_rgb_hw3, K_33=None)
            dpt_conf_mask: Float[np.ndarray, "h w"] = outpaint_rel_depth.confidence
            # convert to boolean mask, depth confidence mask is a binary mask where values > 0 are considered confident
            dpt_conf_mask: Bool[np.ndarray, "H W"] = dpt_conf_mask > 0

            outpaint_depth_hw: Float[np.ndarray, "H W"] = outpaint_rel_depth.depth

            # convert to numpy arrays
            outpaint_mask: Bool[np.ndarray, "H W"] = np.array(outpaint_mask).astype(np.bool_)
            outpaint_edges_mask: Bool[np.ndarray, "H W"] = depth_edges_mask(outpaint_depth_hw, threshold=0.01)
            # inpaint/outpaint mask without edges (True where inpainting is applied, False near edges and where no inpainting)
            outpaint_wo_edges: Bool[np.ndarray, "H W"] = outpaint_mask & ~outpaint_edges_mask
            # final mask
            outpaint_wo_edges = outpaint_wo_edges & dpt_conf_mask

        elif self.config.stage != "no-outpaint":
            # Auto-generate outpainting setup: user-controlled border expansion
            border_percent: float = (
                self.config.expansion_percent / 2.0
            )  # Convert to fraction per side (divide by 2 for each side)
            border_output: tuple[Image.Image, Image.Image] = add_border_and_mask(
                input_image,
                zoom_all=1.0,
                zoom_left=border_percent,
                zoom_right=border_percent,
                zoom_up=border_percent,
                zoom_down=border_percent,
                overlap=0,
            )
            outpaint_img: Image.Image = border_output[0]
            outpaint_mask: Image.Image = border_output[1]
            # Create outpainted image using Flux Inpainting
            outpaint_img: Image.Image = self.flux_inpainter(
                rgb_hw3=np.array(outpaint_img), mask=np.array(outpaint_mask)
            )

            outpaint_rgb_hw3: UInt8[np.ndarray, "H W 3"] = np.array(outpaint_img.convert("RGB"))
            predictor: BaseRelativePredictor = self._get_predictor()
            outpaint_rel_depth: RelativeDepthPrediction = predictor.__call__(rgb=outpaint_rgb_hw3, K_33=None)
            dpt_conf_mask: Float[np.ndarray, "h w"] = outpaint_rel_depth.confidence
            # convert to boolean mask, depth confidence mask is a binary mask where values > 0 are considered confident
            dpt_conf_mask: Bool[np.ndarray, "H W"] = dpt_conf_mask > 0

            outpaint_depth_hw: Float[np.ndarray, "H W"] = outpaint_rel_depth.depth
            # remove any nans or infs and set them to 0
            outpaint_depth_hw[np.isnan(outpaint_depth_hw) | np.isinf(outpaint_depth_hw)] = 0
            # mask showing where outpainting (inpainting) is applied
            outpaint_mask: Bool[np.ndarray, "H W"] = np.array(outpaint_mask).astype(np.bool_)
            # depth edges, True near edges, False otherwise
            outpaint_edges_mask: Bool[np.ndarray, "H W"] = depth_edges_mask(outpaint_depth_hw, threshold=0.01)
            # inpaint/outpaint mask without edges (True where inpainting is applied, False near edges and where no inpainting)
            outpaint_wo_edges: Bool[np.ndarray, "H W"] = outpaint_mask & ~outpaint_edges_mask
            # final mask
            outpaint_wo_edges = outpaint_wo_edges & dpt_conf_mask

            outpaint_intri: Intrinsics = Intrinsics(
                camera_conventions="RDF",
                fl_x=outpaint_rel_depth.K_33[0, 0].item(),
                fl_y=outpaint_rel_depth.K_33[1, 1].item(),
                cx=outpaint_rel_depth.K_33[0, 2].item(),
                cy=outpaint_rel_depth.K_33[1, 2].item(),
                width=outpaint_rgb_hw3.shape[1],
                height=outpaint_rgb_hw3.shape[0],
            )
            outpaint_extri = Extrinsics(
                world_R_cam=np.eye(3, dtype=np.float32),
                world_t_cam=np.zeros(3, dtype=np.float32),
            )
            outpaint_pinhole: PinholeParameters = PinholeParameters(
                name="camera_outpaint",
                intrinsics=outpaint_intri,
                extrinsics=outpaint_extri,
            )

            outpaint_frame: Frame = Frame(
                H=outpaint_rgb_hw3.shape[0],
                W=outpaint_rgb_hw3.shape[1],
                rgb=outpaint_rgb_hw3.astype(np.float32) / 255.0,  # Convert to [0,1] range
                dpt=outpaint_depth_hw,
                intrinsic=outpaint_intri.k_matrix,
                cam_T_world=outpaint_extri.world_T_cam,  # Identity matrix for world coordinates
                inpaint=outpaint_mask,
                inpaint_wo_edge=outpaint_wo_edges,
                dpt_conf_mask=dpt_conf_mask,
            )

            log_frame(
                parent_log_path=self.parent_log_path, frame=outpaint_frame, cam_params=outpaint_pinhole, log_pcd=True
            )
            self.scene._add_trainable_frame(outpaint_frame, require_grad=True)

        input_rgb_hw3: UInt8[np.ndarray, "H W 3"] = np.array(input_image.convert("RGB"))
        # get input depth from outpaint depth, where outpaint mask is False.
        # This allows for getting the depth of the original image and only having to run the depth model once
        input_area = ~outpaint_mask
        input_depth_hw: Float[np.ndarray, "H W"] = outpaint_depth_hw[input_area].reshape(
            input_image.height, input_image.width
        )
        """
        ### Why Input Frame `inpaint` is Set to All `True`

        1.  **The Conceptual Issue**
            At first glance, this seems wrong because the input frame contains the original image data, and original pixels shouldn't need "inpainting". We'd expect `inpaint=False` for original content.

        2.  **The Training Logic Explanation**
            The `inpaint` mask in training determines which pixels should be supervised during optimization.
            - When `inpaint=True`: "Supervise this pixel - ensure the rendered result matches the target."
            - When `inpaint=False`: "Don't supervise this pixel - ignore it during training."

        3.  **Why All `True` Makes Sense for the Input Frame**
            For the input frame, setting `inpaint=True` everywhere means: "Train the Gaussians to perfectly reproduce the original image." All original pixels become supervision targets, forcing the 3D representation to learn to render the input view accurately.

        4.  **The Two-Frame Training Strategy**
            - **Input Frame**: Learns to reproduce original content. `input_frame.inpaint` is all `True`, so ALL pixels are supervised.
            - **Outpaint Frame**: Learns to reproduce only new content. `outpaint_frame.inpaint` is `True` only for outpainted areas.

        5.  **What This Achieves**
            - **Input Frame Training**: Every original pixel has `inpaint=True`, so the model learns to render them perfectly.
            - **Outpaint Frame Training**: Original pixels have `inpaint=False` (they are already learned, so we skip them), while new outpainted pixels have `inpaint=True` so the model learns to render the new content.

        6.  **The Alternative Would Be Problematic**
            If we set `input_frame.inpaint = False` everywhere, there would be no supervision on the original content. The Gaussians wouldn't learn to render the input view, leading to poor reconstruction quality for the reference image. Training would only learn the outpainted content.
        """
        input_mask: Bool[np.ndarray, "H W"] = np.full_like(input_depth_hw, True, dtype=np.bool_)
        input_edges_mask: Bool[np.ndarray, "H W"] = outpaint_edges_mask[input_area].reshape(
            input_image.height, input_image.width
        )
        input_mask_wo_edges: Bool[np.ndarray, "H W"] = ~input_edges_mask
        input_dpt_conf_mask: Bool[np.ndarray, "H W"] = dpt_conf_mask[input_area].reshape(
            input_image.height, input_image.width
        )
        input_k33: Float[np.ndarray, "3 3"] = outpaint_rel_depth.K_33.copy()
        # focal stays the same, but principal point is adjusted to center of input image
        input_k33[0, 2] = input_rgb_hw3.shape[1] / 2.0
        input_k33[1, 2] = input_rgb_hw3.shape[0] / 2.0

        input_intri: Intrinsics = Intrinsics(
            camera_conventions="RDF",
            fl_x=input_k33[0, 0].item(),
            fl_y=input_k33[1, 1].item(),
            cx=input_k33[0, 2].item(),
            cy=input_k33[1, 2].item(),
            width=input_rgb_hw3.shape[1],
            height=input_rgb_hw3.shape[0],
        )
        input_extri = Extrinsics(
            world_R_cam=np.eye(3, dtype=np.float32),
            world_t_cam=np.zeros(3, dtype=np.float32),
        )
        input_pinhole: PinholeParameters = PinholeParameters(
            name="camera_input",
            intrinsics=input_intri,
            extrinsics=input_extri,
        )
        self.shared_intrinsics = input_intri

        input_frame: Frame = Frame(
            H=input_rgb_hw3.shape[0],
            W=input_rgb_hw3.shape[1],
            rgb=input_rgb_hw3.astype(np.float32) / 255.0,  # Convert to [0,1] range
            dpt=input_depth_hw,
            intrinsic=input_k33,
            cam_T_world=input_extri.world_T_cam,  # Identity matrix for world coordinates
            inpaint=input_mask,
            inpaint_wo_edge=input_mask_wo_edges,
            dpt_conf_mask=input_dpt_conf_mask,
        )

        log_frame(parent_log_path=self.parent_log_path, frame=input_frame, cam_params=input_pinhole)
        self.scene._add_trainable_frame(input_frame, require_grad=True)

        self.scene: Gaussian_Scene = GS_Train_Tool(self.scene, iters=100)(self.scene.frames, log=False)

    def setup_rerun(self):
        # 显式初始化 rerun,避免隐式 init 在无 GUI 环境里触发 spawn viewer 导致 winit 报错.
        self.rerun_server_uri = init_rerun_from_config(self.config.rr_config)
        if self.rerun_server_uri is not None:
            print(f"[INFO] Rerun gRPC server: {self.rerun_server_uri}")
            print(f"[INFO] 你可以在本机(有 GUI 的那台机器)执行: rerun --connect {self.rerun_server_uri}")

        self.parent_log_path: Path = Path("/world")
        self.final_log_path: Path = Path("/final")

        rr.send_blueprint(blueprint=self._create_blueprint())
        rr.log("/", rr.ViewCoordinates.RDF, static=True)

    def _create_blueprint(self, tab_idx: int = 0, debug: bool = False) -> rrb.Blueprint:
        """
        Create a rerun blueprint for the pipeline.
        """

        def create_camera_views_panel(camera_ids: list[int | str]) -> rrb.Horizontal:
            """
            Creates a horizontal panel of vertical 2D views for a list of camera identifiers.
            """
            contents = []
            for i in camera_ids:
                name_suffix = i.title() if isinstance(i, str) else str(i)
                view = rrb.Vertical(
                    rrb.Vertical(
                        rrb.Spatial2DView(
                            origin=f"{self.parent_log_path}/camera_{i}/pinhole/rgb",
                            contents=["+ $origin/**"],
                            name=f"Camera {name_suffix} RGB",
                        ),
                        rrb.Spatial2DView(
                            origin=f"{self.parent_log_path}/camera_{i}/pinhole/depth",
                            contents=["+ $origin/**"],
                            name=f"Camera {name_suffix} Depth",
                        ),
                    ),
                    rrb.Horizontal(
                        rrb.Spatial2DView(
                            origin=f"{self.parent_log_path}/camera_{i}/pinhole/dpt_conf_mask",
                            contents=["+ $origin/**"],
                            name=f"Camera {name_suffix} Depth Conf Mask",
                        ),
                        rrb.Spatial2DView(
                            origin=f"{self.parent_log_path}/camera_{i}/pinhole/inpaint_mask",
                            contents=["+ $origin/**"],
                            name=f"Camera {name_suffix} Inpaint Mask",
                        ),
                        rrb.Spatial2DView(
                            origin=f"{self.parent_log_path}/camera_{i}/pinhole/inpaint_wo_edges_mask",
                            contents=["+ $origin/**"],
                            name=f"Camera {name_suffix} Inpaint w/o Edges",
                        ),
                    ),
                    row_shares=[9, 1],
                )
                contents.append(view)
            return rrb.Horizontal(*contents)

        # create tabs
        initial_cameras = ["input", "outpaint"]
        initialization_2d_views = create_camera_views_panel(initial_cameras)

        # Adjust column shares based on image orientation
        init_column_shares: list[int] = [1, 1] if self.orientation == "landscape" else [5, 2]

        # Create content list for initialization view
        initial_cameras = ["input", "outpaint"]
        init_content_3d = [
            "+ $origin/**",
            f"- {self.final_log_path}/camera/pinhole/depth",
            *[
                f"- {self.parent_log_path}/camera_{cam}/pinhole/depth"
                for cam in initial_cameras + self.logged_cam_idx_list
            ],
            *[
                f"- {self.parent_log_path}/camera_{cam}/pinhole/dpt_conf_mask"
                for cam in initial_cameras + self.logged_cam_idx_list
            ],
            *[
                f"- {self.parent_log_path}/camera_{cam}/pinhole/inpaint_mask"
                for cam in initial_cameras + self.logged_cam_idx_list
            ],
            *[
                f"- {self.parent_log_path}/camera_{cam}/pinhole/inpaint_wo_edges_mask"
                for cam in initial_cameras + self.logged_cam_idx_list
            ],
        ]

        initialization_view = rrb.Horizontal(
            rrb.Spatial3DView(
                origin=self.parent_log_path,
                contents=init_content_3d,
            ),
            initialization_2d_views,
            column_shares=init_column_shares,
            name="Initialization",
        )

        # only show at most 5 cameras, pick them distributed
        if len(self.logged_cam_idx_list) > 5:
            # Sample 5 indices evenly from the list (including endpoints)
            idxs = np.linspace(0, len(self.logged_cam_idx_list) - 1, 5, dtype=int)
            view_cam_list: list[int] = [self.logged_cam_idx_list[i] for i in idxs]
        else:
            view_cam_list: list[int] = self.logged_cam_idx_list

        grid_view = create_camera_views_panel(view_cam_list)

        # fmt: off
        content_3d = [
            "+ $origin/**",
            *[f"- {self.parent_log_path}/camera_{i}/pinhole/depth" for i in self.logged_cam_idx_list + initial_cameras],
            *[f"- {self.parent_log_path}/camera_{i}/pinhole/dpt_conf_mask" for i in self.logged_cam_idx_list + initial_cameras],
            *[f"- {self.parent_log_path}/camera_{i}/pinhole/inpaint_mask" for i in self.logged_cam_idx_list + initial_cameras],
            *[f"- {self.parent_log_path}/camera_{i}/pinhole/inpaint_wo_edges_mask" for i in self.logged_cam_idx_list + initial_cameras],
        ]
        # fmt: on

        # coarse scene view
        coarse_scene_view = rrb.Horizontal(
            rrb.Spatial3DView(origin=self.parent_log_path, contents=content_3d),
            grid_view,
            column_shares=[5, 3],
            name="Coarse Scene",
        )

        final_scene_view = rrb.Horizontal(
            rrb.Spatial3DView(
                origin=self.final_log_path,
                contents=[
                    "+ $origin/**",
                ],
            ),
            rrb.Vertical(
                rrb.Spatial2DView(
                    origin=f"{self.final_log_path}/camera/pinhole",
                    contents=[
                        "+ $origin/**",
                    ],
                    name="Final Rendering",
                ),
            ),
            column_shares=[9, 4],
            name="Final Scene",
        )

        blueprint = rrb.Blueprint(
            rrb.Tabs(
                initialization_view,
                coarse_scene_view,
                final_scene_view,
                active_tab=tab_idx,
            ),
            collapse_panels=True,
        )
        return blueprint


def main(config: SingleImageConfig) -> None:
    """
    Main function to run the Single Image Processing Outpainting/Depth/Splat.
    """
    start_time = timer()
    vd_pipeline = SingleImagePipeline(config)
    vd_pipeline()
    end_time: float = timer()
    # Format elapsed time
    elapsed: float = end_time - start_time
    minutes: int = int(elapsed // 60)
    seconds: float = elapsed % 60
    print(f"Processing time: {minutes}m {seconds:.2f}s")
    maybe_wait_after_run(config.rr_config, vd_pipeline.rerun_server_uri)
