"""Pure time-mapping logic for frame interpolation.

Output frame k (at k/dst_fps seconds) maps to source position
pos = k * src_fps / dst_fps, i.e. between source frames floor(pos) and
floor(pos)+1 with blend weight t = pos - floor(pos). Works for any fps
ratio (2x, 2.5x, 24 -> 60, 29.97 -> 60, ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator

# Blend weights this close to a source frame just copy it (avoids wasted
# inference and float noise around exact ratios like 2x).
EPS = Fraction(1, 1000)


@dataclass(frozen=True)
class OutputFrame:
    index: int  # output frame number
    pair: int  # source pair p: between source frames p and p+1
    t: Fraction  # 0 <= t < 1; 0 means "copy source frame p"

    @property
    def needs_interp(self) -> bool:
        return EPS < self.t < 1 - EPS


def source_position(index: int, src_fps: Fraction, dst_fps: Fraction) -> Fraction:
    return Fraction(index) * src_fps / dst_fps


def plan_pair(
    pair: int, next_index: int, src_fps: Fraction, dst_fps: Fraction
) -> Iterator[OutputFrame]:
    """All output frames whose source position falls in [pair, pair+1)."""
    index = next_index
    while True:
        pos = source_position(index, src_fps, dst_fps)
        if pos >= pair + 1:
            return
        t = pos - pair
        if t >= 1 - EPS:
            # Snap to the next source frame; emit as t=0 of the next pair.
            yield OutputFrame(index=index, pair=pair + 1, t=Fraction(0))
        elif t <= EPS:
            yield OutputFrame(index=index, pair=pair, t=Fraction(0))
        else:
            yield OutputFrame(index=index, pair=pair, t=t)
        index += 1


def plan_tail(
    last_frame: int, next_index: int, src_fps: Fraction, dst_fps: Fraction
) -> Iterator[OutputFrame]:
    """Output frames at/after the last source frame: duplicate it.

    Keeps output duration ~= source duration (n_src / src_fps).
    """
    index = next_index
    n_src = last_frame + 1
    while source_position(index, src_fps, dst_fps) < n_src:
        yield OutputFrame(index=index, pair=last_frame, t=Fraction(0))
        index += 1


def total_output_frames(n_src: int, src_fps: Fraction, dst_fps: Fraction) -> int:
    """ceil(n_src * dst / src): count of k with k*src/dst < n_src."""
    ratio = Fraction(n_src) * dst_fps / src_fps
    return int(ratio) if ratio.denominator == 1 else int(ratio) + 1
