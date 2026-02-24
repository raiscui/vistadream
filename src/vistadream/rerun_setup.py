from __future__ import annotations

import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import rerun as rr

DEFAULT_GRPC_PORT: int = 9876


class RerunConfigLike(Protocol):
    """
    Rerun 配置的最小接口.

    说明:
    - 我们用 Protocol 做 duck typing,避免强依赖第三方 dataclass.
    - 只要对象具备这些字段,就可以被本模块使用.
    """

    application_id: str
    recording_id: str | uuid.UUID | None
    connect: bool
    save: Path | None
    serve: bool
    headless: bool


def _safe_application_id() -> str:
    """
    生成一个尽量稳定且可读的 application_id.

    设计目标:
    - 避免空字符串.
    - 尽量从 argv 推断出脚本名,便于多个脚本同时运行时区分.
    - 只保留 [a-zA-Z0-9_\\-],避免下游工具对特殊字符敏感.
    """
    argv0: str | None = sys.argv[0] if sys.argv else None
    if argv0:
        candidate = Path(argv0).stem
    else:
        candidate = "vistadream"

    candidate = candidate.strip() or "vistadream"
    candidate = re.sub(r"[^a-zA-Z0-9_\\-]+", "_", candidate)
    return candidate


@dataclass(frozen=True)
class VistaRerunConfig:
    """
    VistaDream 自己维护的 Rerun 配置.

    为什么不直接用 `simplecv.rerun_log_utils.RerunTyroConfig`?
    - 该类型在 `__post_init__` 中会产生副作用(例如默认 `rr.spawn()`).
    - 在 Remote-SSH/无 GUI 的 Linux 环境里,spawn 原生 viewer 会触发 winit 的 DISPLAY/WAYLAND 报错.
    - 更糟糕的是: 副作用发生在 tyro 构造配置对象时,会早于我们 pipeline 的任何代码,导致无法拦截.

    因此我们把 config 设计成纯数据,由 `init_rerun_from_config` 统一完成初始化与 sink 选择.
    """

    application_id: str = field(default_factory=_safe_application_id)
    recording_id: str | uuid.UUID | None = None
    connect: bool = False
    save: Path | None = None
    serve: bool = False
    headless: bool = False
    wait: bool = False
    wait_seconds: float | None = None


def _has_gui_display() -> bool:
    """
    判断当前进程是否有可用的 GUI 显示环境.

    说明:
    - 在 Remote-SSH / 无头 Linux 环境里,通常没有 DISPLAY/WAYLAND 相关变量.
    - 此时如果让 Rerun 尝试 spawn 原生 Viewer(基于 winit),会直接报错并退出.
    """
    return any(os.environ.get(key) for key in ("DISPLAY", "WAYLAND_DISPLAY", "WAYLAND_SOCKET"))


def _stringify_path(path: Path | None) -> str | None:
    # 统一把 Path 转成 str,避免下游 API 对类型过于敏感.
    if path is None:
        return None
    return str(path)


def init_rerun_from_config(rr_config: RerunConfigLike) -> str | None:
    """
    显式初始化 Rerun,并根据环境选择合适的 sink,避免无 GUI 环境触发 `spawn()` 带来的 winit 报错.

    设计目标:
    - 本机有 GUI 时,保持“默认 spawn viewer”的顺滑体验.
    - 无 GUI 时,自动切换到 `serve_grpc`,让本地 viewer 通过端口转发连接.
    - 用户显式传入 `--rr-config.connect/--rr-config.serve/--rr-config.save` 时,按用户意图优先.

    返回:
    - 如果启用了 `serve_grpc`,返回可用于 viewer 连接的 URI(例如 `rerun+http://127.0.0.1:9876/proxy`).
    - 否则返回 None.
    """
    rr.init(rr_config.application_id, recording_id=rr_config.recording_id, spawn=False)

    save_path: str | None = _stringify_path(rr_config.save)
    if save_path is not None:
        rr.save(save_path)

    if rr_config.connect:
        # 连接到已经启动的 viewer(通常是同机 9876 端口).
        rr.connect_grpc()
        return None

    def serve_grpc_with_fallback() -> str:
        """
        优先使用固定端口 9876,方便 ssh 端口转发.
        如果端口被占用,退化为让 Rerun 自己选择可用端口.
        """
        try:
            return rr.serve_grpc(grpc_port=DEFAULT_GRPC_PORT)
        except Exception as exc:  # noqa: BLE001 - 这里需要兜底,避免无头环境直接失败
            print(f"[WARN] Rerun gRPC 端口 {DEFAULT_GRPC_PORT} 启动失败,将自动选择端口. 原因: {exc}")
            return rr.serve_grpc()

    if rr_config.serve:
        return serve_grpc_with_fallback()

    # 自动模式: 无 GUI 环境就 serve,避免 winit 报 DISPLAY/WAYLAND 缺失.
    if rr_config.headless or not _has_gui_display():
        return serve_grpc_with_fallback()

    # 有 GUI 且用户没有指定 connect/serve,保持默认体验: spawn 原生 viewer.
    rr.spawn(port=DEFAULT_GRPC_PORT)
    return None


def maybe_wait_after_run(rr_config: RerunConfigLike, server_uri: str | None) -> None:
    """
    在任务执行结束后,可选地阻塞等待,避免 `serve_grpc` 模式下进程退出导致 viewer 断开.

    为什么需要这个:
    - `rr.serve_grpc(...)` 会立即返回,并不会阻塞当前线程.
    - gRPC server 依赖当前 Python 进程存活.
    - 如果 pipeline 很快跑完退出,本地 viewer 可能还没来得及连接.
    """
    if server_uri is None:
        return

    wait_seconds: float | None = getattr(rr_config, "wait_seconds", None)
    wait_flag: bool = bool(getattr(rr_config, "wait", False))

    if wait_seconds is None and not wait_flag:
        print("[INFO] Rerun gRPC server 将随进程退出而停止. 如需等待本地 viewer 连接,可加参数: --rr-config.wait")
        print("[INFO] 或者等待指定秒数后退出: --rr-config.wait-seconds 30")
        return

    if wait_seconds is not None:
        print(f"[INFO] 将等待 {wait_seconds:.1f}s 后退出,便于本地 viewer 连接检查: {server_uri}")
        time.sleep(max(0.0, wait_seconds))
        return

    # 默认等待回车.
    try:
        input(f"[INFO] 按回车退出(保持 gRPC server 在线): {server_uri}\n")
    except EOFError:
        # 在非交互环境 stdin 可能不可用,此时不应卡死.
        print("[WARN] stdin 不可用,无法等待回车,将直接退出.")
