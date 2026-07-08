from fractions import Fraction

from venhance import timemap


def collect(n_src: int, src: Fraction, dst: Fraction):
    out = []
    for pair in range(n_src - 1):
        out.extend(timemap.plan_pair(pair, len(out), src, dst))
    out.extend(timemap.plan_tail(n_src - 1, len(out), src, dst))
    return out


def test_exact_2x():
    frames = collect(4, Fraction(30), Fraction(60))
    # 4 src frames -> 8 out frames: copy,mid,copy,mid,copy,mid,copy(last),dup(last)
    assert len(frames) == 8
    assert [f.index for f in frames] == list(range(8))
    assert [(f.pair, f.t) for f in frames] == [
        (0, 0), (0, Fraction(1, 2)),
        (1, 0), (1, Fraction(1, 2)),
        (2, 0), (2, Fraction(1, 2)),
        (3, 0),  # snapped to source frame 3 by plan_pair(pair=2)
        (3, 0),  # tail duplicate
    ]
    assert sum(f.needs_interp for f in frames) == 3


def test_total_matches_plan_for_various_ratios():
    cases = [
        (10, Fraction(30), Fraction(60)),
        (10, Fraction(24), Fraction(60)),  # 2.5x
        (7, Fraction(30000, 1001), Fraction(60)),  # 29.97 -> 60
        (5, Fraction(25), Fraction(30)),
    ]
    for n_src, src, dst in cases:
        frames = collect(n_src, src, dst)
        assert len(frames) == timemap.total_output_frames(n_src, src, dst)
        assert [f.index for f in frames] == list(range(len(frames)))
        # duration preserved within one output frame
        assert abs(len(frames) / dst - n_src / src) < 1 / dst


def test_copies_reference_valid_source_frames():
    frames = collect(10, Fraction(24), Fraction(60))
    for f in frames:
        assert 0 <= f.pair <= 9
        assert 0 <= f.t < 1
        if not f.needs_interp:
            assert f.t <= timemap.EPS


def test_no_upsample_needed_below_ratio_two():
    # 25 -> 30: most output frames are interpolated at fractional t
    frames = collect(6, Fraction(25), Fraction(30))
    ts = {f.t for f in frames if f.needs_interp}
    assert ts == {Fraction(1, 6), Fraction(2, 6), Fraction(3, 6), Fraction(4, 6), Fraction(5, 6)}
