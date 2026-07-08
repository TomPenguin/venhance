"""Model weight registry: download to ~/.cache/venhance/models with sha256 check."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from rich.progress import Progress

CACHE_DIR = Path(
    os.environ.get("VENHANCE_CACHE", Path.home() / ".cache" / "venhance")
) / "models"

_RELEASE_BASE = (
    "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/"
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    filename: str
    url: str
    sha256: str
    arch_ver: str  # IFNet architecture version string
    scale_list: tuple[int, ...]


MODELS: dict[str, ModelSpec] = {
    "rife-v4.7": ModelSpec(
        name="rife-v4.7",
        filename="rife47.pth",
        url=_RELEASE_BASE + "rife47.pth",
        sha256="6a8a825ab2750558bdd20dcced386fd82b7222c7ba58c11d3b611d9c44f1be63",
        arch_ver="4.7",
        scale_list=(8, 4, 2, 1),
    ),
    "rife-v4.26": ModelSpec(
        name="rife-v4.26",
        filename="rife426.pth",
        url=_RELEASE_BASE + "rife426.pth",
        sha256="606421fe2148a9fdeca14e58d94dc339e87b87e0ebcc5dec84a50d6f488cfe7b",
        arch_ver="4.26",
        scale_list=(16, 8, 4, 2, 1),
    ),
}

DEFAULT_MODEL = "rife-v4.7"

_ESRGAN_BASE = "https://github.com/xinntao/Real-ESRGAN/releases/download/"


@dataclass(frozen=True)
class SrModelSpec:
    name: str
    filename: str
    url: str
    sha256: str
    arch: str  # "compact" (SRVGGNetCompact) | "rrdb" (RRDBNet)
    scale: int  # native upscale factor
    state_key: str  # key holding the weights inside the checkpoint
    num_feat: int = 64
    num_conv: int = 16  # compact only
    num_block: int = 23  # rrdb only


SR_MODELS: dict[str, SrModelSpec] = {
    "realesr-general": SrModelSpec(
        name="realesr-general",
        filename="realesr-general-x4v3.pth",
        url=_ESRGAN_BASE + "v0.2.5.0/realesr-general-x4v3.pth",
        sha256="8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
        arch="compact",
        scale=4,
        state_key="params",
        num_conv=32,
    ),
    "realesr-anime": SrModelSpec(
        name="realesr-anime",
        filename="realesr-animevideov3.pth",
        url=_ESRGAN_BASE + "v0.2.5.0/realesr-animevideov3.pth",
        sha256="b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d",
        arch="compact",
        scale=4,
        state_key="params",
        num_conv=16,
    ),
    "realesrgan-x4plus": SrModelSpec(
        name="realesrgan-x4plus",
        filename="RealESRGAN_x4plus.pth",
        url=_ESRGAN_BASE + "v0.1.0/RealESRGAN_x4plus.pth",
        sha256="4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
        arch="rrdb",
        scale=4,
        state_key="params_ema",
    ),
}

DEFAULT_SR_MODEL = "realesr-general"

ALL_MODELS: dict[str, ModelSpec | SrModelSpec] = {**MODELS, **SR_MODELS}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def model_path(name: str) -> Path:
    return CACHE_DIR / ALL_MODELS[name].filename


def is_downloaded(name: str) -> bool:
    return model_path(name).exists()


def ensure_model(name: str, quiet: bool = False) -> Path:
    if name not in ALL_MODELS:
        raise ValueError(
            f"unknown model: {name} (available: {', '.join(ALL_MODELS)})"
        )
    spec = ALL_MODELS[name]
    dest = model_path(name)
    if dest.exists():
        return dest

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(spec.url, headers={"User-Agent": "venhance"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        with Progress(disable=quiet) as progress:
            task = progress.add_task(f"download {spec.filename}", total=total or None)
            with tmp.open("wb") as f:
                while chunk := resp.read(1 << 18):
                    f.write(chunk)
                    progress.update(task, advance=len(chunk))

    digest = _sha256(tmp)
    if digest != spec.sha256:
        tmp.unlink()
        raise RuntimeError(
            f"sha256 mismatch for {spec.filename} (expected {spec.sha256}, got {digest}). "
            "The download may be corrupted, or the upstream file may have changed."
        )
    tmp.rename(dest)
    return dest
