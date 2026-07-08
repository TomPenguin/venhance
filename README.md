# venhance

AI video enhancement CLI that runs **natively on Apple Silicon Macs**.
Frame interpolation (e.g. 30fps → 60fps, RIFE) and super-resolution
(e.g. 720p → 1440p, Real-ESRGAN), with PyTorch MPS inference and
VideoToolbox hardware encoding.

[日本語版 README はこちら](README.ja.md)

## Why venhance?

Free AI video enhancement tools exist, but on macOS they all come with
caveats: [Video2X](https://github.com/k4yt3x/video2x) has no native macOS
build (it depends on Vulkan), the `*-ncnn-vulkan` CLIs work on image
sequences and leave the video/audio plumbing to you, and most other options
are Windows-centric GUIs. venhance fills that gap:

- **Native Apple Silicon inference** — PyTorch MPS with fp16, no
  MoltenVK/Vulkan translation layer
- **Video in, video out** — a single command; no exporting frames to PNG
- **Streaming pipeline, no intermediate files** — raw frames are piped
  through ffmpeg; a 10-minute 1080p video never touches your disk as
  tens of GB of images
- **VideoToolbox hardware encoding** — HEVC / H.264 / ProRes output
- **Interpolation + super-resolution in one pass** — frames are
  interpolated while still low-resolution, then upscaled, so the expensive
  SR model runs on every output frame exactly once
- **Audio and color metadata are preserved** — audio streams are copied
  as-is; color primaries/transfer/matrix tags are carried over from the
  source

## Requirements

- macOS (Apple Silicon)
- ffmpeg: `brew install ffmpeg`
- [uv](https://docs.astral.sh/uv/): `brew install uv`

## Installation

Install as a global command (recommended):

```sh
uv tool install .        # run at the repository root
uv tool update-shell     # only if ~/.local/bin is not on your PATH (restart your shell)
```

After that, run `venhance ...` from anywhere. After updating the code,
reinstall with `uv tool install . --force`. Uninstall with
`uv tool uninstall venhance`.

For development, run directly inside the repository:

```sh
uv sync
uv run venhance ...
```

## Usage

```sh
# Show input video info
venhance probe input.mp4

# 30fps -> 60fps (model weights are downloaded automatically on first run)
venhance interp input.mp4 --fps 60

# Specify a factor and an output path
venhance interp input.mp4 --factor 2 -o out.mp4

# Super-resolution 720p -> 1440p
venhance upscale input.mp4 --scale 2

# Interpolation + super-resolution in one pass (30fps 720p -> 60fps 1440p)
venhance run input.mp4 --fps 60 --scale 2

# Model management
venhance models list
venhance models download rife-v4.26
```

Main options (`interp`):

| Option | Description |
|---|---|
| `--fps` / `--factor` | Target fps (`60`, `59.94`, `60000/1001`) or a multiplier — one or the other |
| `--model` | `rife-v4.7` (default, fast) / `rife-v4.26` (higher quality) |
| `--codec` | `hevc` (default) / `h264` / `prores` (VideoToolbox hardware encoding) |
| `--quality` | 1-100 (default 65) |
| `--scene-threshold` | Scene-cut detection threshold. Frames at cut boundaries are duplicated instead of interpolated (0 to disable) |
| `--device` | `auto` / `mps` / `cpu` |

Main options (`upscale`):

| Option | Description |
|---|---|
| `--scale` | Output scale factor, >1 and ≤4, fractional values allowed (default 2) |
| `--model` | `realesr-general` (default, live-action) / `realesr-anime` (anime, fastest) / `realesrgan-x4plus` (best quality, slow) |
| `--tile` | Tile size in px. Auto-selected per model when omitted (compact models: no tiling, x4plus: 512) |
| `--fp32` | Disable fp16 inference (fp16 is the default on MPS/CUDA) |
| `--codec` / `--quality` / `--device` | Same as `interp` |

`run` performs both `interp` and `upscale` in a single streaming pass
(interpolate while still low-resolution → upscale every frame). Its options
are the union of the two; models are selected individually with
`--interp-model` / `--sr-model`.

Inference uses fp16 by default (on MPS/CUDA). Compared to fp32 the output
stays visually identical (PSNR > 47 dB) while frame interpolation runs about
1.4× faster. Disable with `--fp32`.

## Performance

Measured on an Apple M5, 720p input, fp16 on MPS:

| Task | Model | Speed |
|---|---|---|
| Interpolation | rife-v4.7 | ~51 ms/frame |
| Super-resolution 2x | realesr-anime | ~152 ms/frame |
| Super-resolution 2x | realesr-general | ~276 ms/frame |
| Super-resolution 4x | realesrgan-x4plus | ~8.5 s/frame |

## Design

See [DESIGN.md](DESIGN.md) (Japanese).

## License

MIT — see [LICENSE](LICENSE).

This project vendors code from other open-source projects:

- `src/venhance/vendor/rife_arch.py` is derived from
  [Fannovel16/ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation) (MIT),
  based on [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) (MIT).
  Model weights are downloaded from that repository's GitHub Releases.
- `src/venhance/vendor/realesrgan_arch.py` is derived from
  [BasicSR](https://github.com/XPixelGroup/BasicSR) (Apache-2.0), the
  architecture used by [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
  (BSD-3-Clause). Model weights are downloaded from Real-ESRGAN's GitHub
  Releases.
