# venhance — Local Video Enhancer CLI Design

A video enhancement CLI that runs entirely locally (macOS / Apple Silicon).
It provides two features:

- **Frame interpolation (interpolate)**: raise fps by generating frames with AI, e.g. 30fps → 60fps
- **Super-resolution (upscale)**: raise resolution with AI, e.g. 720p → 1440p/4K

Command name: `venhance`.

---

## 1. Guiding principles

- **Use pre-trained models**; no training of our own.
  - Frame interpolation: **RIFE v4.x** (the de-facto standard for its balance of practical speed and quality)
  - Super-resolution: **Real-ESRGAN** (live-action) / **Real-ESRGAN anime** (anime)
- **Leave all video decode/encode/audio to ffmpeg.**
  What we write is just the plumbing: pull a frame out → run it through the model → write it back.
- **Streaming processing with no intermediate files** by default
  (raw video is piped through; expanding a 10-minute 1080p video to PNGs would be tens of GB).

### On difficulty

For both super-resolution and frame interpolation, the core is just "feed frames into
a pre-trained model and get frames out." The real effort is in the shared pipeline:

| Part | Difficulty | Notes |
|---|---|---|
| ffmpeg I/O pipes | Medium | Write once, shared by both features |
| Super-resolution (applying Real-ESRGAN) | Low | Each frame is independent; just loop |
| Frame interpolation (applying RIFE) | Medium | Managing adjacent-frame pairs, scene-change handling |
| Audio / timestamp preservation | Medium | Covered by ffmpeg know-how |

---

## 2. Tech stack

| Item | Choice | Rationale |
|---|---|---|
| Language | Python 3.12+ | The model ecosystem assumes Python |
| Package management | uv | Fast; `uv tool install` makes CLI distribution easy |
| CLI framework | Typer | Subcommands + type-hint based |
| Inference | PyTorch (MPS backend) | Uses the Apple GPU via `device="mps"` |
| Video I/O | ffmpeg (subprocess + pipes) | Installed via Homebrew; bundling `imageio-ffmpeg` also considered |
| Progress display | rich | Progress bar, ETA |

**Alternative (Plan B)**: orchestrate the pre-built `rife-ncnn-vulkan` /
`realesrgan-ncnn-vulkan` binaries. This drastically reduces the Python-side
implementation, but goes through image files and is disk-I/O heavy. Kept as a
fallback in case the PyTorch/MPS implementation hits a wall. For that reason the
inference part is abstracted behind a Backend interface (§5).

---

## 3. CLI design

```
venhance interp  INPUT [-o OUTPUT] --fps 60          # interpolation only
venhance interp  INPUT --factor 2                    # by multiplier (30→60)
venhance upscale INPUT [-o OUTPUT] --scale 2         # super-resolution only (2x/4x)
venhance run     INPUT --fps 60 --scale 2            # both (interpolate → upscale)
venhance probe   INPUT                               # show input info (fps/resolution/codec)
venhance models  [download|list]                     # manage model weights
```

Common options:

```
-o, --output PATH        Default: auto-named like input_60fps.mp4 / input_2x.mp4
--model NAME             e.g. rife-v4.6 / realesrgan-x4 / realesrgan-anime
--codec {h264,hevc,prores}   Output codec (default: hevc + videotoolbox)
--crf / --bitrate        Quality control
--chunk-secs N           Chunk-processing unit (default 30s)
--resume                 Resume an interrupted job
--dry-run                Only print the execution plan and ffmpeg commands
```

### Processing order (when `run` specifies both)

Do **interpolate → upscale**. Interpolating while still low-resolution is cheaper,
and super-resolution then only needs one pass over every generated frame. As a
single streaming pass, `decode → RIFE → Real-ESRGAN → encode` is wired directly
with no intermediate video.

---

## 4. Pipeline architecture

```
                    ┌─────────────────────────────────────────┐
 input.mp4 ──────►  │ ffmpeg (decode)                          │
                    │  -i in.mp4 -vf fps=to-CFR -pix_fmt rgb24 │
                    │  -f rawvideo pipe:1                      │
                    └───────────────┬─────────────────────────┘
                                    │ raw RGB frames (stdout)
                                    ▼
                    ┌─────────────────────────────────────────┐
                    │ venhance core (Python)                   │
                    │  FrameSource → [Interpolator] →          │
                    │  [Upscaler] → FrameSink                  │
                    │  (each stage chained as a generator)     │
                    └───────────────┬─────────────────────────┘
                                    │ raw RGB frames (stdin)
                                    ▼
                    ┌─────────────────────────────────────────┐
                    │ ffmpeg (encode)                          │
                    │  -f rawvideo pipe:0 ... -c:v hevc_       │
                    │  videotoolbox  + copy audio from source  │
                    └───────────────┬─────────────────────────┘
                                    ▼
                                output.mp4
```

- Each stage is a generator taking `Iterator[Frame]` and returning `Iterator[Frame]`.
  Enabling/disabling interpolation and super-resolution is expressed by inserting stages.
- **Interpolator**: holds the previous frame, passes the pair (t, t+1) to RIFE to
  generate intermediate frames. It performs scene-change detection (frame diff, or
  RIFE's built-in flag) and, at cut boundaries, duplicates the previous frame instead
  of interpolating (to avoid morphing artifacts across cuts).
- **Upscaler**: runs Real-ESRGAN frame by frame. To save VRAM it can tile if needed
  (512px tiles + overlap).

### Audio and metadata

- Audio is copied from the source without re-encoding via `-map 1:a -c:a copy`.
- If the input is VFR (variable frame rate, common in phone footage), it is converted
  to CFR with the `fps` filter at decode time before interpolating. `probe` warns when
  it detects VFR.
- Color space (BT.709, etc.) and HDR metadata: v1 deliberately supports only SDR/BT.709.
  HDR inputs are detected and warned about.

### Chunk processing and resume

For long videos, process in `--chunk-secs` (default 30s) units, writing a segment file
to a temp directory (`~/.cache/venhance/jobs/<hash>/`) as each chunk completes. After all
chunks finish, join them with ffmpeg's concat demuxer. `--resume` skips already-completed
segments. Job identity is determined by "hash of the input file + options."

---

## 5. Module structure

```
venhance/
├── pyproject.toml
├── DESIGN.md
├── src/venhance/
│   ├── cli.py               # Typer entry point
│   ├── probe.py             # ffprobe wrapper (fps/resolution/VFR/HDR detection)
│   ├── pipeline.py          # stage chaining, chunk/resume control
│   ├── io/
│   │   ├── decoder.py       # ffmpeg decode → frame iterator
│   │   └── encoder.py       # frame iterator → ffmpeg encode
│   ├── stages/
│   │   ├── base.py          # Stage protocol definition
│   │   ├── interpolate.py   # RIFE stage (pair management + scene detection)
│   │   └── upscale.py       # Real-ESRGAN stage (with tiling)
│   ├── backends/
│   │   ├── base.py          # InferenceBackend protocol
│   │   ├── torch_mps.py     # PyTorch + MPS implementation (default)
│   │   └── ncnn.py          # ncnn-binary implementation (Plan B, can defer)
│   └── models/
│       └── registry.py      # model weight download/cache/verify (sha256)
└── tests/
    ├── fixtures/            # a few-second test clips (with a generation script)
    └── ...
```

Model weights are downloaded on first run from GitHub Releases etc. into
`~/.cache/venhance/models/` and verified by sha256 (they can also be fetched ahead of
time with `venhance models download`).

> Note: the current implementation is flatter than the tree above (e.g. `interpolate.py`,
> `upscale.py`, `rife.py`, `sr.py`, `video_io.py`, `models.py` live directly under
> `src/venhance/`). The tree here reflects the original target structure.

---

## 6. Performance estimates (assuming Apple M5)

| Task | Estimated speed | For a 5-min 1080p/30fps video |
|---|---|---|
| RIFE interpolation (1080p, 30→60) | measured ~7 fps processed (fp32/MPS, M5) | ~25 min (room to shorten with fp16) |
| Real-ESRGAN 2x (1080p→4K) | 2–6 fps processed | 25–75 min |
| Both | dominated by super-resolution | ~1.5 hours |

Super-resolution is overwhelmingly heavy, so when `--scale` is given, show an estimated
duration before starting and prompt for confirmation. Future optimization candidates:
Core ML conversion (leveraging the ANE), fp16 inference, batch inference.

### Optimization measurements (M5, 720p input, MPS)

| Measure | Result | Decision |
|---|---|---|
| fp16 inference (RIFE) | 1.36× faster, PSNR 56dB vs fp32 | ✅ Adopted (default on MPS/CUDA, disable with `--fp32`) |
| fp16 inference (Real-ESRGAN) | adopted (from the start) | ✅ |
| Batch inference (SR, batch=2/4) | no benefit (about 5% worse; GPU already saturated at 720p) | ❌ Rejected |
| channels_last | no benefit | ❌ Rejected |
| Double buffering (overlap transfer and inference) | no benefit (already hidden by MPS's async queue) | ❌ Rejected |
| Decode prefetch thread | unnecessary (decode is 0.7ms/frame, not the bottleneck) | ❌ Rejected |

The only remaining large headroom is Core ML / ANE conversion.

---

## 7. Milestones

1. **M1: skeleton** — `probe` + decode→(no-op)→encode passthrough works, including audio
   copy and chunk joining. Solidify pipeline correctness here. ✅ Implemented (chunk
   splitting not implemented; single-pass streaming instead)
2. **M2: interp** — integrate RIFE (torch/MPS) and complete `interp`, including scene
   detection. ✅ Implemented (rife-v4.7 / v4.26, arbitrary fps ratios)
3. **M3: upscale** — complete `upscale` with Real-ESRGAN + tiling. ✅ Implemented
   (realesr-general / realesr-anime / realesrgan-x4plus, fp16 inference, tiling.
   Measured: 720p input at 2x — general ~3.6fps, anime ~6.6fps, x4plus ~0.12fps @ MPS fp16)
4. **M4: run/resume** — composite pipeline, resume, progress ETA, model management.
   🔶 `run` (interp→upscale single-pass composite), progress ETA, and model management are
   implemented. Chunk splitting and resume are not.
5. **M5 and beyond (optional)** — ncnn backend, Core ML optimization, anime models,
   batch processing (whole-directory), HDR support.

## 8. Testing approach

- Generate a few-second fixture video with `ffmpeg -f lavfi -i testsrc` (not committed to the repo).
- Verification items: output frame count (2× ±1 for interpolation), matching duration,
  resolution, audio-stream preservation.
- Keep image quality to SSIM/PSNR regression checks only (no absolute-quality testing).
- On pipe breakage or interruption (Ctrl-C), no temp files should be left behind, and resume should work.
