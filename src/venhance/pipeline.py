"""Combined pipeline: decode -> RIFE -> Real-ESRGAN -> encode in one stream.

Interpolation runs first, at the low source resolution (cheaper), and every
output frame then goes through super-resolution (DESIGN.md §3).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

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
from .interpolate import (
    SCENE_THRESHOLD_DEFAULT,
    InterpStats,
    interp_stream,
    resolve_target_fps,
)
from .models import DEFAULT_MODEL, DEFAULT_SR_MODEL, SR_MODELS
from .probe import VideoInfo, probe
from .rife import Rife
from .sr import Upscaler
from .upscale import output_dimensions
from .video_io import FrameReader, FrameWriter

console = Console()


@dataclass
class RunOptions:
    fps: Fraction | None = None
    factor: Fraction | None = None
    scale: float = 2.0
    interp_model: str = DEFAULT_MODEL
    sr_model: str = DEFAULT_SR_MODEL
    codec: str = "hevc"
    quality: int = 65
    scene_threshold: float = SCENE_THRESHOLD_DEFAULT
    tile: int | None = None
    fp16: bool | None = None
    device: str = "auto"


def default_output_path(input_path: Path, dst_fps: Fraction, scale: float) -> Path:
    return input_path.with_name(
        f"{input_path.stem}_{float(dst_fps):g}fps_{scale:g}x.mp4"
    )


def run_pipeline(input_path: Path, output_path: Path | None, opts: RunOptions) -> Path:
    if opts.sr_model not in SR_MODELS:
        raise ValueError(
            f"unknown super-resolution model: {opts.sr_model} (available: {', '.join(SR_MODELS)})"
        )
    native = SR_MODELS[opts.sr_model].scale
    if not 1.0 < opts.scale <= native:
        raise ValueError(
            f"--scale must be greater than 1 and at most {native}: {opts.scale:g}"
        )

    info: VideoInfo = probe(input_path)
    src_fps = info.fps
    dst_fps = resolve_target_fps(src_fps, opts)
    if dst_fps <= src_fps:
        raise ValueError(
            f"target fps ({float(dst_fps):g}) is not greater than input fps ({float(src_fps):g})."
        )
    out = output_path or default_output_path(input_path, dst_fps, opts.scale)
    if out.resolve() == input_path.resolve():
        raise ValueError("output path is the same as the input.")
    out_w, out_h = output_dimensions(info, opts.scale)

    if info.is_vfr:
        console.print(
            "[yellow]Input is VFR (variable frame rate); "
            f"converting to CFR at its average {float(src_fps):.3f}fps before processing.[/yellow]"
        )
    console.print(
        f"[bold]{input_path.name}[/bold] {info.width}x{info.height} "
        f"{float(src_fps):g}fps -> [bold]{out_w}x{out_h} {float(dst_fps):g}fps[/bold] "
        f"(interp={opts.interp_model}, sr={opts.sr_model}, codec={opts.codec})"
    )

    rife = Rife(opts.interp_model, device=opts.device, fp16=opts.fp16)
    upscaler = Upscaler(
        opts.sr_model, device=opts.device, tile=opts.tile, fp16=opts.fp16
    )
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
        FrameWriter(
            out, info, dst_fps, codec=opts.codec, quality=opts.quality,
            width=out_w, height=out_h,
        ) as writer,
        progress,
    ):
        task = progress.add_task("interp+upscale", total=total)
        for frame in interp_stream(
            reader, rife, src_fps, dst_fps, opts.scene_threshold, stats
        ):
            writer.write(upscaler.upscale(frame, (out_h, out_w)))
            progress.update(task, completed=stats.n_out)

    console.print(
        f"[green]done[/green] {out} — {stats.n_src} input frames -> {stats.n_out} output frames, "
        f"{info.width}x{info.height} -> {out_w}x{out_h}"
        + (f" ({stats.n_cuts} scene cuts detected)" if stats.n_cuts else "")
    )
    return out
