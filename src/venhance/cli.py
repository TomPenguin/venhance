"""venhance CLI entry point."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from . import __version__, models
from .probe import FfmpegNotFoundError

app = typer.Typer(
    name="venhance",
    help="ローカルで動作するAI動画エンハンサー（フレーム補間・超解像）",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
models_app = typer.Typer(help="モデル重みの管理", no_args_is_help=True)
app.add_typer(models_app, name="models")

console = Console()
err_console = Console(stderr=True)


def _parse_rate(value: str, opt: str) -> Fraction:
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            f = Fraction(int(num), int(den))
        else:
            f = Fraction(value)
    except (ValueError, ZeroDivisionError):
        raise typer.BadParameter(f"{opt} の値を解釈できません: {value}")
    if f <= 0:
        raise typer.BadParameter(f"{opt} は正の値で指定してください: {value}")
    return f


def _fail(e: Exception) -> None:
    err_console.print(f"[red]エラー:[/red] {e}")
    raise typer.Exit(code=1)


@app.callback(invoke_without_command=True)
def _version_callback(
    version: Annotated[bool, typer.Option("--version", help="バージョン表示")] = False,
) -> None:
    if version:
        console.print(f"venhance {__version__}")
        raise typer.Exit()


@app.command()
def probe(
    input: Annotated[Path, typer.Argument(help="入力動画", exists=True, dir_okay=False)],
) -> None:
    """入力動画の情報（解像度・fps・コーデックなど）を表示する。"""
    from .probe import probe as do_probe

    try:
        info = do_probe(input)
    except (FfmpegNotFoundError, RuntimeError, ValueError) as e:
        _fail(e)

    table = Table(show_header=False)
    table.add_row("path", str(info.path))
    table.add_row("codec", f"{info.codec} ({info.pix_fmt})")
    table.add_row("resolution", f"{info.width}x{info.height}")
    fps_note = " [yellow](VFR: 平均値)[/yellow]" if info.is_vfr else ""
    table.add_row("fps", f"{float(info.fps):.3f} ({info.fps}){fps_note}")
    table.add_row("duration", f"{info.duration:.2f}s" if info.duration else "?")
    table.add_row("frames", str(info.nb_frames) if info.nb_frames else "?")
    table.add_row("audio", "あり" if info.has_audio else "なし")
    console.print(table)


@app.command()
def interp(
    input: Annotated[Path, typer.Argument(help="入力動画", exists=True, dir_okay=False)],
    output: Annotated[
        Optional[Path],
        typer.Option("-o", "--output", help="出力パス（省略時: 入力名_60fps.mp4）"),
    ] = None,
    fps: Annotated[
        Optional[str],
        typer.Option("--fps", help="目標fps（例: 60, 59.94, 60000/1001）"),
    ] = None,
    factor: Annotated[
        Optional[str],
        typer.Option("--factor", help="fps倍率（例: 2, 2.5）"),
    ] = None,
    model: Annotated[
        str, typer.Option("--model", help=f"補間モデル（{', '.join(models.MODELS)}）")
    ] = models.DEFAULT_MODEL,
    codec: Annotated[
        str, typer.Option("--codec", help="出力コーデック: hevc / h264 / prores")
    ] = "hevc",
    quality: Annotated[
        int, typer.Option("--quality", min=1, max=100, help="品質 1-100（videotoolbox -q:v）")
    ] = 65,
    scene_threshold: Annotated[
        float,
        typer.Option(
            "--scene-threshold",
            help="シーンカット検出しきい値（0で無効。カット境界は補間せず複製）",
        ),
    ] = 0.15,
    fp32: Annotated[
        bool, typer.Option("--fp32", help="fp16推論を無効化する（デフォルト: MPS/CUDAでfp16）")
    ] = False,
    device: Annotated[
        str, typer.Option("--device", help="推論デバイス: auto / mps / cpu")
    ] = "auto",
) -> None:
    """AIフレーム補間でfpsを上げる（例: 30fps -> 60fps）。"""
    from .interpolate import InterpOptions, run_interp

    opts = InterpOptions(
        fps=_parse_rate(fps, "--fps") if fps else None,
        factor=_parse_rate(factor, "--factor") if factor else None,
        model=model,
        codec=codec,
        quality=quality,
        scene_threshold=scene_threshold,
        fp16=False if fp32 else None,
        device=device,
    )
    try:
        run_interp(input, output, opts)
    except (FfmpegNotFoundError, RuntimeError, ValueError) as e:
        _fail(e)
    except KeyboardInterrupt:
        err_console.print("\n[yellow]中断しました。出力ファイルは不完全です。[/yellow]")
        raise typer.Exit(code=130)


@app.command()
def run(
    input: Annotated[Path, typer.Argument(help="入力動画", exists=True, dir_okay=False)],
    output: Annotated[
        Optional[Path],
        typer.Option("-o", "--output", help="出力パス（省略時: 入力名_60fps_2x.mp4）"),
    ] = None,
    fps: Annotated[
        Optional[str],
        typer.Option("--fps", help="目標fps（例: 60, 59.94, 60000/1001）"),
    ] = None,
    factor: Annotated[
        Optional[str],
        typer.Option("--factor", help="fps倍率（例: 2, 2.5）"),
    ] = None,
    scale: Annotated[
        float, typer.Option("--scale", help="出力倍率（1より大きく4以下、小数可）")
    ] = 2.0,
    interp_model: Annotated[
        str,
        typer.Option("--interp-model", help=f"補間モデル（{', '.join(models.MODELS)}）"),
    ] = models.DEFAULT_MODEL,
    sr_model: Annotated[
        str,
        typer.Option("--sr-model", help=f"超解像モデル（{', '.join(models.SR_MODELS)}）"),
    ] = models.DEFAULT_SR_MODEL,
    codec: Annotated[
        str, typer.Option("--codec", help="出力コーデック: hevc / h264 / prores")
    ] = "hevc",
    quality: Annotated[
        int, typer.Option("--quality", min=1, max=100, help="品質 1-100（videotoolbox -q:v）")
    ] = 65,
    scene_threshold: Annotated[
        float,
        typer.Option(
            "--scene-threshold",
            help="シーンカット検出しきい値（0で無効。カット境界は補間せず複製）",
        ),
    ] = 0.15,
    tile: Annotated[
        Optional[int],
        typer.Option("--tile", help="タイルサイズpx（0で分割なし。省略時: モデルに応じ自動）"),
    ] = None,
    fp32: Annotated[
        bool, typer.Option("--fp32", help="fp16推論を無効化する（デフォルト: MPS/CUDAでfp16）")
    ] = False,
    device: Annotated[
        str, typer.Option("--device", help="推論デバイス: auto / mps / cpu")
    ] = "auto",
) -> None:
    """フレーム補間と超解像を1パスで両方行う（例: 30fps 720p -> 60fps 1440p）。"""
    from .pipeline import RunOptions, run_pipeline

    opts = RunOptions(
        fps=_parse_rate(fps, "--fps") if fps else None,
        factor=_parse_rate(factor, "--factor") if factor else None,
        scale=scale,
        interp_model=interp_model,
        sr_model=sr_model,
        codec=codec,
        quality=quality,
        scene_threshold=scene_threshold,
        tile=tile,
        fp16=False if fp32 else None,
        device=device,
    )
    try:
        run_pipeline(input, output, opts)
    except (FfmpegNotFoundError, RuntimeError, ValueError) as e:
        _fail(e)
    except KeyboardInterrupt:
        err_console.print("\n[yellow]中断しました。出力ファイルは不完全です。[/yellow]")
        raise typer.Exit(code=130)


@app.command()
def upscale(
    input: Annotated[Path, typer.Argument(help="入力動画", exists=True, dir_okay=False)],
    output: Annotated[
        Optional[Path],
        typer.Option("-o", "--output", help="出力パス（省略時: 入力名_2x.mp4）"),
    ] = None,
    scale: Annotated[
        float, typer.Option("--scale", help="出力倍率（1より大きく4以下、小数可）")
    ] = 2.0,
    model: Annotated[
        str,
        typer.Option("--model", help=f"超解像モデル（{', '.join(models.SR_MODELS)}）"),
    ] = models.DEFAULT_SR_MODEL,
    codec: Annotated[
        str, typer.Option("--codec", help="出力コーデック: hevc / h264 / prores")
    ] = "hevc",
    quality: Annotated[
        int, typer.Option("--quality", min=1, max=100, help="品質 1-100（videotoolbox -q:v）")
    ] = 65,
    tile: Annotated[
        Optional[int],
        typer.Option("--tile", help="タイルサイズpx（0で分割なし。省略時: モデルに応じ自動）"),
    ] = None,
    fp32: Annotated[
        bool, typer.Option("--fp32", help="fp16推論を無効化する（デフォルト: MPS/CUDAでfp16）")
    ] = False,
    device: Annotated[
        str, typer.Option("--device", help="推論デバイス: auto / mps / cpu")
    ] = "auto",
) -> None:
    """AI超解像で解像度を上げる（例: 720p -> 1440p）。"""
    from .upscale import UpscaleOptions, run_upscale

    opts = UpscaleOptions(
        scale=scale,
        model=model,
        codec=codec,
        quality=quality,
        tile=tile,
        fp16=False if fp32 else None,
        device=device,
    )
    try:
        run_upscale(input, output, opts)
    except (FfmpegNotFoundError, RuntimeError, ValueError) as e:
        _fail(e)
    except KeyboardInterrupt:
        err_console.print("\n[yellow]中断しました。出力ファイルは不完全です。[/yellow]")
        raise typer.Exit(code=130)


@models_app.command("list")
def models_list() -> None:
    """利用可能なモデルとダウンロード状態を表示する。"""
    table = Table()
    table.add_column("name")
    table.add_column("kind")
    table.add_column("file")
    table.add_column("downloaded")
    for name, spec in models.ALL_MODELS.items():
        mark = "[green]✓[/green]" if models.is_downloaded(name) else "-"
        kind = "interp" if name in models.MODELS else "upscale"
        is_default = name in (models.DEFAULT_MODEL, models.DEFAULT_SR_MODEL)
        default = " (default)" if is_default else ""
        table.add_row(name + default, kind, spec.filename, mark)
    console.print(table)


@models_app.command("download")
def models_download(
    name: Annotated[str, typer.Argument(help="モデル名（省略時: デフォルト）")] = models.DEFAULT_MODEL,
) -> None:
    """モデル重みを事前ダウンロードする。"""
    try:
        path = models.ensure_model(name)
    except (ValueError, RuntimeError) as e:
        _fail(e)
    console.print(f"[green]OK[/green] {path}")


if __name__ == "__main__":
    app()
