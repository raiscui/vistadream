import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from safetensors.torch import load_file as load_sft

from vistadream.flux.model import Flux, FluxLoraWrapper, FluxParams
from vistadream.flux.modules.autoencoder import AutoEncoder, AutoEncoderParams, ConditionAutoEncoder
from vistadream.flux.modules.conditioner import HFEmbedder


@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    ckpt_path: str | None
    lora_path: str | None
    ae_path: str | None
    repo_id: str | None
    repo_flow: str | None
    repo_ae: str | None
    # 可选: ModelScope 镜像仓库 id,用于 HuggingFace 不可用时的下载回退
    # 例: "AI-ModelScope/FLUX.1-Fill-dev"
    modelscope_repo_id: str | None = None


configs = {
    "flux-dev": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV"),
        lora_path=None,
        params=FluxParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-schnell": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-schnell",
        repo_flow="flux1-schnell.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_SCHNELL"),
        lora_path=None,
        params=FluxParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-canny": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-Canny-dev",
        repo_flow="flux1-canny-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV_CANNY"),
        lora_path=None,
        params=FluxParams(
            in_channels=128,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-canny-lora": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV"),
        lora_path=os.getenv("FLUX_DEV_CANNY_LORA"),
        params=FluxParams(
            in_channels=128,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-depth": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-Depth-dev",
        repo_flow="flux1-depth-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV_DEPTH"),
        lora_path=None,
        params=FluxParams(
            in_channels=128,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-depth-lora": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV"),
        lora_path=os.getenv("FLUX_DEV_DEPTH_LORA"),
        params=FluxParams(
            in_channels=128,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-fill": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-Fill-dev",
        repo_flow="flux1-fill-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path="./ckpt/flux_fill/flux1-fill-dev.safetensors",
        lora_path=None,
        params=FluxParams(
            in_channels=384,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path="./ckpt/flux_fill/ae.safetensors",
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
        modelscope_repo_id="AI-ModelScope/FLUX.1-Fill-dev",
    ),
}


def _repo_root() -> Path:
    """
    返回仓库根目录.

    说明:
    - 本项目的权重通常放在 repo 内的 `ckpt/` 下.
    - 用 repo root 解析相对路径,可以避免用户在不同 cwd 运行时找不到文件.
    """

    # util.py 位于: src/vistadream/flux/util.py -> parents[3] 为仓库根目录
    return Path(__file__).resolve().parents[3]


def _resolve_existing_path(path: str | None) -> Path | None:
    """
    尝试把字符串路径解析为一个已存在的文件路径.

    - 先按原样判断(相对 cwd 或绝对路径).
    - 若不存在且是相对路径,再按 repo root 进行解析.
    """

    if path is None:
        return None

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None

    if candidate.exists():
        return candidate

    repo_candidate = _repo_root() / candidate
    return repo_candidate if repo_candidate.exists() else None


def _modelscope_resolve_url(model_id: str, revision: str, file_path: str) -> str:
    """
    构建 ModelScope 文件直链.

    直链格式(已验证可访问,且支持 Range 断点续传):
    https://modelscope.cn/models/{model_id}/resolve/{revision}/{file_path}
    """

    # 这里保留 "/" 作为路径分隔符,其余字符做 URL 编码
    model_id_q = quote(model_id, safe="/")
    revision_q = quote(revision, safe="")
    file_path_q = quote(file_path, safe="/")
    return f"https://modelscope.cn/models/{model_id_q}/resolve/{revision_q}/{file_path_q}"


def _sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    """计算文件 sha256(用于可选的离线校验)."""

    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _http_download_with_resume(
    url: str,
    dst_path: Path,
    *,
    chunk_bytes: int = 8 * 1024 * 1024,
    verify_sha256: bool = False,
    expected_sha256: str | None = None,
) -> None:
    """
    下载大文件到指定位置,支持断点续传.

    设计要点:
    - 使用 `.part` 临时文件,避免半成品覆盖目标文件.
    - 若 `.part` 存在,使用 `Range: bytes=<offset>-` 续传.
    - sha256 校验默认关闭(避免对 20GB+ 文件增加额外耗时). 如需可通过参数开启.
    """

    if dst_path.exists():
        return

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_suffix(dst_path.suffix + ".part")

    existing_size = tmp_path.stat().st_size if tmp_path.exists() else 0
    headers: dict[str, str] = {}
    mode = "wb"
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        mode = "ab"

    # 延迟 import: 避免 util.py 在纯离线/裁剪环境下因为缺少 httpx 而 import 失败
    import httpx

    timeout = httpx.Timeout(connect=30.0, read=300.0, write=300.0, pool=30.0)
    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        with client.stream("GET", url, headers=headers) as resp:
            resp.raise_for_status()

            # 如果我们尝试续传,但服务端返回 200(而不是 206),
            # 说明未接受 Range. 为避免拼接出损坏文件,直接删掉重下.
            if existing_size > 0 and resp.status_code == 200:
                tmp_path.unlink(missing_ok=True)
                return _http_download_with_resume(
                    url,
                    dst_path,
                    chunk_bytes=chunk_bytes,
                    verify_sha256=verify_sha256,
                    expected_sha256=expected_sha256,
                )

            with tmp_path.open(mode) as f:
                for chunk in resp.iter_bytes(chunk_size=chunk_bytes):
                    if chunk:
                        f.write(chunk)

    tmp_path.replace(dst_path)

    if verify_sha256 and expected_sha256 is not None:
        actual_sha256 = _sha256_file(dst_path)
        if actual_sha256.lower() != expected_sha256.lower():
            dst_path.unlink(missing_ok=True)
            raise ValueError(
                "下载完成但 sha256 校验失败. "
                f"expected={expected_sha256} actual={actual_sha256} url={url} path={dst_path}"
            )


def _ensure_checkpoint_file(
    *,
    name: str,
    kind: Literal["flow", "ae"],
    allow_download: bool,
    prefer_modelscope: bool,
) -> Path:
    """
    获取/下载指定模型的权重文件.

    下载优先级(在 allow_download=True 时):
    - 若 prefer_modelscope=True 且配置了 modelscope_repo_id,优先从 ModelScope 下载到本地 ckpt 目录.
    - 其次使用 HuggingFace Hub 下载(落在 huggingface cache 中).
    """

    spec = configs[name]

    if kind == "flow":
        local_path = spec.ckpt_path
        repo_file = spec.repo_flow
    else:
        local_path = spec.ae_path
        repo_file = spec.repo_ae

    resolved = _resolve_existing_path(local_path)
    if resolved is not None:
        return resolved

    if not allow_download:
        if name == "flux-dev-fill" and local_path is not None and repo_file is not None and spec.modelscope_repo_id is not None:
            url = _modelscope_resolve_url(spec.modelscope_repo_id, revision="master", file_path=repo_file)
            raise FileNotFoundError(
                f"缺少 {name} 的 {kind} 权重文件,且已禁用下载. local_path={local_path}. "
                f"可手动从 ModelScope 下载: curl -L -C - -o {local_path} \"{url}\""
            )
        raise FileNotFoundError(f"缺少 {name} 的 {kind} 权重文件,且已禁用下载. local_path={local_path}")

    # 1) ModelScope: 只在配置了镜像仓库 id 且能确定要下载的文件名时启用
    if prefer_modelscope and spec.modelscope_repo_id is not None and local_path is not None and repo_file is not None:
        dst_path = Path(local_path)
        if not dst_path.is_absolute():
            dst_path = _repo_root() / dst_path

        url = _modelscope_resolve_url(spec.modelscope_repo_id, revision="master", file_path=repo_file)
        print(f"[INFO] Downloading {name}/{kind} from ModelScope: {url}")
        try:
            _http_download_with_resume(url, dst_path)
            return dst_path
        except Exception as e:
            # 不中断: 允许回退到 HuggingFace(例如 ModelScope 访问受限/断网)
            print(f"[WARN] ModelScope download failed, fallback to HuggingFace. error={e}")

    # 2) HuggingFace Hub 回退
    if spec.repo_id is not None and repo_file is not None:
        print(f"[INFO] Downloading {name}/{kind} from HuggingFace: {spec.repo_id}/{repo_file}")
        try:
            return Path(hf_hub_download(spec.repo_id, repo_file))
        except Exception as e:
            # 对于 flux-dev-fill,给出更明确的可执行恢复路径
            if (
                name == "flux-dev-fill"
                and local_path is not None
                and repo_file is not None
                and spec.modelscope_repo_id is not None
            ):
                url = _modelscope_resolve_url(spec.modelscope_repo_id, revision="master", file_path=repo_file)
                raise RuntimeError(
                    f"从 HuggingFace 下载 {name}/{kind} 失败. "
                    f"建议改用 ModelScope 直链手动下载: curl -L -C - -o {local_path} \"{url}\""
                ) from e
            raise

    if name == "flux-dev-fill" and local_path is not None and repo_file is not None and spec.modelscope_repo_id is not None:
        url = _modelscope_resolve_url(spec.modelscope_repo_id, revision="master", file_path=repo_file)
        raise FileNotFoundError(
            f"无法确定 {name} 的 {kind} 权重来源. local_path={local_path} repo_id={spec.repo_id}. "
            f"可手动从 ModelScope 下载: curl -L -C - -o {local_path} \"{url}\""
        )
    raise FileNotFoundError(f"无法确定 {name} 的 {kind} 权重来源. local_path={local_path} repo_id={spec.repo_id}")


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


def load_flow_model(
    name: str, device: str | torch.device = "cuda", hf_download: bool = True, verbose: bool = False
) -> Flux:
    # Loading Flux
    print("Init model")
    prefer_modelscope = name == "flux-dev-fill"
    ckpt_path = _ensure_checkpoint_file(name=name, kind="flow", allow_download=hf_download, prefer_modelscope=prefer_modelscope)
    lora_path = configs[name].lora_path

    with torch.device("meta"):
        if lora_path is not None:
            model = FluxLoraWrapper(params=configs[name].params).to(torch.bfloat16)
        else:
            model = Flux(configs[name].params).to(torch.bfloat16)

    print("Loading checkpoint")
    # load_sft doesn't support torch.device
    sd = load_sft(os.fspath(ckpt_path), device=str(device))
    sd = optionally_expand_state_dict(model, sd)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if verbose:
        print_load_warning(missing, unexpected)

    if configs[name].lora_path is not None:
        print("Loading LoRA")
        lora_sd = load_sft(configs[name].lora_path, device=str(device))
        # loading the lora params + overwriting scale values in the norms
        missing, unexpected = model.load_state_dict(lora_sd, strict=False, assign=True)
        if verbose:
            print_load_warning(missing, unexpected)
    return model


def load_t5(device: str | torch.device = "cuda", max_length: int = 512) -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    return HFEmbedder("./ckpt/google/t5-v1_1-xxl", max_length=max_length, torch_dtype=torch.bfloat16).to(device)


def load_clip(device: str | torch.device = "cuda") -> HFEmbedder:
    return HFEmbedder(
        "./ckpt/openai/clip-vit-large-patch14", max_length=77, torch_dtype=torch.bfloat16, is_clip=True
    ).to(device)


def load_ae(name: str, device: str | torch.device = "cuda", hf_download: bool = True) -> AutoEncoder:
    prefer_modelscope = name == "flux-dev-fill"
    ckpt_path = _ensure_checkpoint_file(name=name, kind="ae", allow_download=hf_download, prefer_modelscope=prefer_modelscope)

    # Loading the autoencoder
    print("Init AE")
    with torch.device("meta"):
        ae = AutoEncoder(configs[name].ae_params)

    sd = load_sft(os.fspath(ckpt_path), device=str(device))
    missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
    print_load_warning(missing, unexpected)
    return ae


def load_cond_ae(name: str, device: str | torch.device = "cuda", hf_download: bool = True) -> AutoEncoder:
    prefer_modelscope = name == "flux-dev-fill"
    ckpt_path = _ensure_checkpoint_file(name=name, kind="ae", allow_download=hf_download, prefer_modelscope=prefer_modelscope)

    # Loading the autoencoder
    print("Init AE")
    # with torch.device("meta" if ckpt_path is not None else device):
    ddconfig = OmegaConf.load("configs/condition_decoder.yaml")
    ae = ConditionAutoEncoder(configs[name].ae_params, ddconfig)

    sd = load_sft(os.fspath(ckpt_path), device=str(device))
    missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
    print_load_warning(missing, unexpected)
    return ae


def optionally_expand_state_dict(model: torch.nn.Module, state_dict: dict) -> dict:
    """
    Optionally expand the state dict to match the model's parameters shapes.
    """
    for name, param in model.named_parameters():
        if name in state_dict and state_dict[name].shape != param.shape:
            print(
                f"Expanding '{name}' with shape {state_dict[name].shape} to model parameter with shape {param.shape}."
            )
            # expand with zeros:
            expanded_state_dict_weight = torch.zeros_like(param, device=state_dict[name].device)
            slices = tuple(slice(0, dim) for dim in state_dict[name].shape)
            expanded_state_dict_weight[slices] = state_dict[name]
            state_dict[name] = expanded_state_dict_weight

    return state_dict
