"""Super-resolution pipeline: decode -> Real-ESRGAN -> encode, streaming."""

from __future__ import annotations

from dataclasses import dataclass
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

from .models import DEFAULT_SR_MODEL, SR_MODELS
from .probe import VideoInfo, probe
from .sr import Upscaler
from .video_io import FrameReader, FrameWriter

console = Console()


@dataclass
class UpscaleOptions:
    scale: float = 2.0
    model: str = DEFAULT_SR_MODEL
    codec: str = "hevc"
    quality: int = 65
    tile: int | None = None  # None = auto
    fp16: bool | None = None  # None = auto (fp16 on mps/cuda)
    device: str = "auto"


def output_dimensions(info: VideoInfo, scale: float) -> tuple[int, int]:
    """Target (width, height), rounded down to even for yuv420p."""
    w = int(round(info.width * scale)) // 2 * 2
    h = int(round(info.height * scale)) // 2 * 2
    return w, h


def default_output_path(input_path: Path, scale: float) -> Path:
    return input_path.with_name(f"{input_path.stem}_{scale:g}x.mp4")


def run_upscale(input_path: Path, output_path: Path | None, opts: UpscaleOptions) -> Path:
    if opts.model not in SR_MODELS:
        raise ValueError(
            f"unknown super-resolution model: {opts.model} (available: {', '.join(SR_MODELS)})"
        )
    native = SR_MODELS[opts.model].scale
    if not 1.0 < opts.scale <= native:
        raise ValueError(
            f"--scale must be greater than 1 and at most {native}: {opts.scale:g}"
        )

    info: VideoInfo = probe(input_path)
    out = output_path or default_output_path(input_path, opts.scale)
    if out.resolve() == input_path.resolve():
        raise ValueError("output path is the same as the input.")
    out_w, out_h = output_dimensions(info, opts.scale)

    console.print(
        f"[bold]{input_path.name}[/bold] {info.width}x{info.height} -> "
        f"[bold]{out_w}x{out_h}[/bold] ({opts.scale:g}x, model={opts.model}, "
        f"codec={opts.codec})"
    )

    upscaler = Upscaler(opts.model, device=opts.device, tile=opts.tile, fp16=opts.fp16)
    console.print(f"device: {upscaler.device.type} ({upscaler.precision})")

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    n = 0
    with (
        FrameReader(info) as reader,
        FrameWriter(
            out, info, info.fps, codec=opts.codec, quality=opts.quality,
            width=out_w, height=out_h,
        ) as writer,
        progress,
    ):
        task = progress.add_task("upscale", total=info.nb_frames)
        for frame in reader:
            writer.write(upscaler.upscale(frame, (out_h, out_w)))
            n += 1
            progress.update(task, completed=n)

    if n == 0:
        raise RuntimeError("could not decode any frames.")
    console.print(
        f"[green]done[/green] {out} — {n} frames, {info.width}x{info.height} -> {out_w}x{out_h}"
    )
    return out
