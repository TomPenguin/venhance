"""ffprobe wrapper: extract the video metadata the pipeline needs."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


class FfmpegNotFoundError(RuntimeError):
    pass


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FfmpegNotFoundError(
            f"{name} not found. Install it with `brew install ffmpeg`."
        )
    return path


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: Fraction  # rate used for CFR decode
    duration: float | None  # seconds
    nb_frames: int | None  # source frame count (estimated if missing)
    codec: str
    pix_fmt: str
    has_audio: bool
    is_vfr: bool
    # ffprobe values with unknown/unspecified normalized to None
    color_space: str | None = None
    color_primaries: str | None = None
    color_trc: str | None = None
    color_range: str | None = None

    @property
    def fps_float(self) -> float:
        return float(self.fps)


def _color_tag(stream: dict, key: str) -> str | None:
    val = stream.get(key)
    return None if val in (None, "unknown", "unspecified", "N/A") else val


def _parse_rate(s: str | None) -> Fraction | None:
    if not s or s in ("0/0", "N/A"):
        return None
    try:
        num, _, den = s.partition("/")
        f = Fraction(int(num), int(den)) if den else Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None
    return f if f > 0 else None


def probe(path: Path) -> VideoInfo:
    require_tool("ffprobe")
    if not path.exists():
        raise FileNotFoundError(path)
    cmd = [
        "ffprobe", "-v", "error",
        "-show_streams", "-show_format",
        "-of", "json", str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {out.stderr.strip()}")
    data = json.loads(out.stdout)

    vstreams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    if not vstreams:
        raise ValueError(f"no video stream: {path}")
    v = vstreams[0]
    has_audio = any(s.get("codec_type") == "audio" for s in data.get("streams", []))

    r_rate = _parse_rate(v.get("r_frame_rate"))
    avg_rate = _parse_rate(v.get("avg_frame_rate"))
    is_vfr = bool(r_rate and avg_rate and r_rate != avg_rate)
    # For VFR sources avg_frame_rate is the sensible CFR target; otherwise they agree.
    fps = avg_rate or r_rate
    if fps is None:
        raise ValueError(f"cannot determine frame rate: {path}")

    duration = None
    for src in (v.get("duration"), data.get("format", {}).get("duration")):
        if src not in (None, "N/A"):
            duration = float(src)
            break

    nb_frames = None
    if v.get("nb_frames", "N/A") not in (None, "N/A"):
        nb_frames = int(v["nb_frames"])
    elif duration is not None:
        nb_frames = round(duration * fps)

    return VideoInfo(
        path=path,
        width=int(v["width"]),
        height=int(v["height"]),
        fps=fps,
        duration=duration,
        nb_frames=nb_frames,
        codec=v.get("codec_name", "?"),
        pix_fmt=v.get("pix_fmt", "?"),
        has_audio=has_audio,
        is_vfr=is_vfr,
        color_space=_color_tag(v, "color_space"),
        color_primaries=_color_tag(v, "color_primaries"),
        color_trc=_color_tag(v, "color_transfer"),
        color_range=_color_tag(v, "color_range"),
    )
