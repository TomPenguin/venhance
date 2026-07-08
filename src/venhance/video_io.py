"""ffmpeg subprocess pipes: raw RGB frame reader / writer.

Decode: ffmpeg -i input -vf fps=CFR,format=rgb24 -f rawvideo pipe:1
Encode: raw frames on stdin + original file as second input for audio copy.
"""

from __future__ import annotations

import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from types import TracebackType

import numpy as np

from .probe import VideoInfo, require_tool

# ffprobe color_space -> libswscale matrix name (scale filter in/out_color_matrix)
_SWS_MATRIX = {
    "bt709": "bt709",
    "smpte170m": "smpte170m",
    "bt470bg": "bt601",
    "smpte240m": "smpte240m",
    "bt2020nc": "bt2020",
    "fcc": "fcc",
}


def _sws_color_opts(source: VideoInfo, direction: str) -> str:
    """scale filter options pinning the YUV<->RGB conversion to the source's
    tags (e.g. "in_color_matrix=bt709:in_range=tv"). Untagged properties are
    left to swscale defaults — symmetric across read/write, so the round trip
    preserves the original YUV either way."""
    opts = []
    matrix = _SWS_MATRIX.get(source.color_space or "")
    if matrix:
        opts.append(f"{direction}_color_matrix={matrix}")
    if source.color_range in ("tv", "pc"):
        opts.append(f"{direction}_range={source.color_range}")
    return ":".join(opts)


class FfmpegProcess:
    """Shared plumbing: stderr goes to a temp file, surfaced on failure."""

    def __init__(self, cmd: list[str], **popen_kw) -> None:
        require_tool("ffmpeg")
        self._stderr_file = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(cmd, stderr=self._stderr_file, **popen_kw)

    def _stderr_tail(self, max_bytes: int = 4000) -> str:
        self._stderr_file.seek(0)
        data = self._stderr_file.read()
        return data[-max_bytes:].decode(errors="replace")

    def check_exit(self) -> None:
        code = self.proc.wait()
        stderr = self._stderr_tail()
        self._stderr_file.close()
        if code != 0:
            raise RuntimeError(f"ffmpeg がエラー終了しました (code {code}):\n{stderr}")

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        self._stderr_file.close()


class FrameReader(FfmpegProcess):
    """Yields HxWx3 uint8 RGB frames, CFR-normalized to info.fps."""

    def __init__(self, info: VideoInfo) -> None:
        self.info = info
        self.frame_bytes = info.width * info.height * 3
        sws = _sws_color_opts(info, "in")
        to_rgb = f"scale={sws},format=rgb24" if sws else "format=rgb24"
        cmd = [
            "ffmpeg", "-v", "error", "-nostdin",
            "-i", str(info.path),
            "-map", "0:v:0",
            "-vf", f"fps={info.fps},{to_rgb}",
            "-f", "rawvideo",
            "pipe:1",
        ]
        super().__init__(cmd, stdout=subprocess.PIPE)

    def __iter__(self):
        h, w = self.info.height, self.info.width
        assert self.proc.stdout is not None
        while True:
            buf = self.proc.stdout.read(self.frame_bytes)
            if not buf:
                break
            if len(buf) < self.frame_bytes:
                # pipe truncated mid-frame; check_exit will report the cause
                break
            # bytearray copy makes the array writable (torch.from_numpy needs it)
            yield np.frombuffer(bytearray(buf), dtype=np.uint8).reshape(h, w, 3)

    def __enter__(self) -> "FrameReader":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.proc.stdout.close()
            self.check_exit()
        else:
            self.kill()


_CODEC_ARGS = {
    "hevc": ["-c:v", "hevc_videotoolbox", "-tag:v", "hvc1"],
    "h264": ["-c:v", "h264_videotoolbox"],
    "prores": ["-c:v", "prores_videotoolbox", "-profile:v", "2"],
}


class FrameWriter(FfmpegProcess):
    """Consumes HxWx3 uint8 RGB frames; muxes audio copied from the source."""

    def __init__(
        self,
        output: Path,
        source: VideoInfo,
        fps: Fraction,
        codec: str = "hevc",
        quality: int = 65,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        if codec not in _CODEC_ARGS:
            raise ValueError(f"未対応のコーデックです: {codec}")
        width = width or source.width
        height = height or source.height
        codec_args = list(_CODEC_ARGS[codec])
        if codec != "prores":
            codec_args += ["-q:v", str(quality)]
        # Propagate the source's color tags; an untagged source stays untagged
        # so players interpret input and output identically.
        tag_args: list[str] = []
        for flag, value in [
            ("-colorspace", source.color_space),
            ("-color_primaries", source.color_primaries),
            ("-color_trc", source.color_trc),
            ("-color_range", source.color_range),
        ]:
            if value is not None:
                tag_args += [flag, value]
        sws = _sws_color_opts(source, "out")
        to_yuv = f"scale={sws},format=yuv420p" if sws else "format=yuv420p"
        cmd = [
            "ffmpeg", "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "pipe:0",
            "-i", str(source.path),
            "-map", "0:v:0", "-map", "1:a?",
            "-c:a", "copy",
            *codec_args,
            "-vf", to_yuv,
            *tag_args,
            "-movflags", "+faststart",
            str(output),
        ]
        super().__init__(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(frame.tobytes())

    def __enter__(self) -> "FrameWriter":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.proc.stdin.close()
            self.check_exit()
        else:
            self.kill()
