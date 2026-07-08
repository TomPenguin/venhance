"""Frame interpolation pipeline: decode -> RIFE -> encode, streaming."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from . import timemap
from .models import DEFAULT_MODEL
from .probe import VideoInfo, probe
from .rife import Rife
from .video_io import FrameReader, FrameWriter

console = Console()

# Mean abs diff (0..1) between downsampled consecutive frames above which we
# treat the boundary as a scene cut and duplicate instead of interpolating.
SCENE_THRESHOLD_DEFAULT = 0.15


@dataclass
class InterpOptions:
    fps: Fraction | None = None
    factor: Fraction | None = None
    model: str = DEFAULT_MODEL
    codec: str = "hevc"
    quality: int = 65
    scene_threshold: float = SCENE_THRESHOLD_DEFAULT
    fp16: bool | None = None  # None = auto (fp16 on mps/cuda)
    device: str = "auto"


@dataclass
class InterpStats:
    n_src: int = 0
    n_out: int = 0
    n_cuts: int = 0


def resolve_target_fps(src_fps: Fraction, opts: InterpOptions) -> Fraction:
    if (opts.fps is None) == (opts.factor is None):
        raise ValueError("--fps か --factor のどちらか一方を指定してください。")
    return opts.fps if opts.fps is not None else src_fps * opts.factor


def is_scene_cut(f0: np.ndarray, f1: np.ndarray, threshold: float) -> bool:
    if threshold <= 0:
        return False
    a = f0[::8, ::8].astype(np.int16)
    b = f1[::8, ::8].astype(np.int16)
    diff = float(np.abs(a - b).mean()) / 255.0
    return diff > threshold


def default_output_path(input_path: Path, dst_fps: Fraction) -> Path:
    label = f"{float(dst_fps):g}fps"
    return input_path.with_name(f"{input_path.stem}_{label}.mp4")


def interp_stream(
    reader: FrameReader,
    rife: Rife,
    src_fps: Fraction,
    dst_fps: Fraction,
    scene_threshold: float,
    stats: InterpStats,
):
    """Yield output frames at dst_fps; progress is observable via stats."""
    frames = iter(reader)
    prev = next(frames, None)
    if prev is None:
        raise RuntimeError("フレームをデコードできませんでした。")
    stats.n_src = 1
    prev_t = None  # lazy: tensor conversion only when a pair needs inference

    for cur in frames:
        pair = stats.n_src - 1
        stats.n_src += 1
        cut = is_scene_cut(prev, cur, scene_threshold)
        if cut:
            stats.n_cuts += 1
        cur_t = None
        for of in timemap.plan_pair(pair, stats.n_out, src_fps, dst_fps):
            if not of.needs_interp:
                yield prev if of.pair == pair else cur
            elif cut:
                yield prev if of.t < Fraction(1, 2) else cur
            else:
                if prev_t is None:
                    prev_t = rife.to_tensor(prev)
                if cur_t is None:
                    cur_t = rife.to_tensor(cur)
                yield rife.interpolate(prev_t, cur_t, float(of.t))
            stats.n_out += 1
        prev, prev_t = cur, cur_t

    for _ in timemap.plan_tail(stats.n_src - 1, stats.n_out, src_fps, dst_fps):
        yield prev
        stats.n_out += 1


def run_interp(input_path: Path, output_path: Path | None, opts: InterpOptions) -> Path:
    info: VideoInfo = probe(input_path)
    src_fps = info.fps
    dst_fps = resolve_target_fps(src_fps, opts)
    out = output_path or default_output_path(input_path, dst_fps)
    if out.resolve() == input_path.resolve():
        raise ValueError("出力パスが入力と同じです。")

    if dst_fps <= src_fps:
        raise ValueError(
            f"目標fps ({float(dst_fps):g}) が入力fps ({float(src_fps):g}) 以下です。"
        )
    if info.is_vfr:
        console.print(
            "[yellow]入力はVFR（可変フレームレート）です。"
            f"平均 {float(src_fps):.3f}fps のCFRに変換してから補間します。[/yellow]"
        )

    console.print(
        f"[bold]{input_path.name}[/bold] {info.width}x{info.height} "
        f"{float(src_fps):g}fps -> [bold]{float(dst_fps):g}fps[/bold] "
        f"(model={opts.model}, codec={opts.codec})"
    )

    rife = Rife(opts.model, device=opts.device, fp16=opts.fp16)
    console.print(f"device: {rife.device.type} ({rife.precision})")

    total = (
        timemap.total_output_frames(info.nb_frames, src_fps, dst_fps)
        if info.nb_frames
        else None
    )

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    stats = InterpStats()
    with (
        FrameReader(info) as reader,
        FrameWriter(out, info, dst_fps, codec=opts.codec, quality=opts.quality) as writer,
        progress,
    ):
        task = progress.add_task("interpolate", total=total)
        for frame in interp_stream(
            reader, rife, src_fps, dst_fps, opts.scene_threshold, stats
        ):
            writer.write(frame)
            progress.update(task, completed=stats.n_out)

    console.print(
        f"[green]完了[/green] {out} — 入力 {stats.n_src} フレーム -> 出力 {stats.n_out} フレーム"
        + (f"（シーンカット検出 {stats.n_cuts} 箇所）" if stats.n_cuts else "")
    )
    return out
