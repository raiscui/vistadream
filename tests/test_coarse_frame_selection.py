import numpy as np
import pytest

from vistadream.api.single_img_pipeline import _select_best_pose_index


def test_select_best_pose_index_filters_by_max_ratio() -> None:
    # 说明: 洞比例超过 max_inpaint_ratio 的候选应被过滤(视为 0),因此不会被选中.
    ratios = np.array([0.10, 0.30, 0.20], dtype=np.float32)

    selected = _select_best_pose_index(
        ratios,
        selected_indices=[],
        min_inpaint_ratio=0.05,
        max_inpaint_ratio=0.25,
        adjacent_exclusion=1,
    )
    assert selected is not None
    idx, ratio = selected
    assert idx == 2
    assert ratio == pytest.approx(0.20)


def test_select_best_pose_index_respects_adjacent_exclusion() -> None:
    # 说明: 当 adjacent_exclusion=1 时,已选索引的 ±1 都会被剔除.
    ratios = np.array([0.10, 0.24, 0.23, 0.22], dtype=np.float32)

    selected = _select_best_pose_index(
        ratios,
        selected_indices=[1],
        min_inpaint_ratio=0.05,
        max_inpaint_ratio=0.25,
        adjacent_exclusion=1,
    )
    assert selected is not None
    idx, ratio = selected
    # 索引 0/1/2 会被剔除,因此只能选 3.
    assert idx == 3
    assert ratio == pytest.approx(0.22)


def test_select_best_pose_index_returns_none_when_too_small() -> None:
    # 说明: 最佳候选低于 min_inpaint_ratio 时,应返回 None 表示选不到合适的帧.
    ratios = np.array([0.01, 0.02, 0.03], dtype=np.float32)

    selected = _select_best_pose_index(
        ratios,
        selected_indices=[],
        min_inpaint_ratio=0.05,
        max_inpaint_ratio=0.25,
        adjacent_exclusion=1,
    )
    assert selected is None


def test_select_best_pose_index_negative_adjacent_exclusion_is_treated_as_zero() -> None:
    # 说明: adjacent_exclusion 传负数时,按 0 处理(仅剔除已选索引本身).
    ratios = np.array([0.10, 0.20, 0.15], dtype=np.float32)

    selected = _select_best_pose_index(
        ratios,
        selected_indices=[1],
        min_inpaint_ratio=0.05,
        max_inpaint_ratio=0.25,
        adjacent_exclusion=-3,
    )
    assert selected is not None
    idx, ratio = selected
    assert idx == 2
    assert ratio == pytest.approx(0.15)


def test_select_best_pose_index_empty_array_returns_none() -> None:
    # 说明: 没有候选帧时,直接返回 None.
    ratios = np.array([], dtype=np.float32)
    assert (
        _select_best_pose_index(
            ratios,
            selected_indices=[],
            min_inpaint_ratio=0.05,
            max_inpaint_ratio=0.25,
            adjacent_exclusion=1,
        )
        is None
    )


def test_select_best_pose_index_fallback_closest_picks_smallest_above_max() -> None:
    # 说明: 当所有候选都 > max 时,`fallback_mode=closest` 应选择“洞最小”的那一帧(最接近 max).
    ratios = np.array([0.60, 0.35, 0.40], dtype=np.float32)

    selected = _select_best_pose_index(
        ratios,
        selected_indices=[],
        min_inpaint_ratio=0.05,
        max_inpaint_ratio=0.25,
        adjacent_exclusion=1,
        fallback_mode="closest",
    )
    assert selected is not None
    idx, ratio = selected
    assert idx == 1
    assert ratio == pytest.approx(0.35)


def test_select_best_pose_index_fallback_closest_picks_largest_below_min() -> None:
    # 说明: 当所有候选都 < min 时,`fallback_mode=closest` 应选择“洞最大”的那一帧(最接近 min).
    ratios = np.array([0.01, 0.04, 0.03], dtype=np.float32)

    selected = _select_best_pose_index(
        ratios,
        selected_indices=[],
        min_inpaint_ratio=0.05,
        max_inpaint_ratio=0.25,
        adjacent_exclusion=1,
        fallback_mode="closest",
    )
    assert selected is not None
    idx, ratio = selected
    assert idx == 1
    assert ratio == pytest.approx(0.04)


def test_select_best_pose_index_fallback_closest_still_respects_adjacent_exclusion() -> None:
    # 说明: fallback 也必须遵守 adjacent_exclusion,如果全部被剔除则返回 None.
    ratios = np.array([0.60, 0.35, 0.40], dtype=np.float32)

    selected = _select_best_pose_index(
        ratios,
        selected_indices=[1],
        min_inpaint_ratio=0.05,
        max_inpaint_ratio=0.25,
        adjacent_exclusion=1,  # 会剔除 0/1/2
        fallback_mode="closest",
    )
    assert selected is None
