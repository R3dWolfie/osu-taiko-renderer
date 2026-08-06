"""Regression test for taiko reconcile miss PLACEMENT.

Real corpus replay (Jayceko on mentholjockey - @h ive _ #flow [strain], nomod):
the .osr header counts GREAT 1045 / OK 135 / MISS 29, max combo 826. The honest
per-key sweep only finds 3 pressless misses (the player's inputs look like
near-hits on almost every note), so _reconcile must FABRICATE the other 26 to
reach the header total. It used to dump those fabricated misses onto the very
first notes — notes the player cleanly hit — because (a) the protected-run
tiebreak shifted the clean 826-run off a slightly-loose-but-hit opening and
(b) the top-up clustered extras next to the resulting structural boundary trim.

The fix keeps the counts + max combo HEADER-EXACT (results-screen numbers never
change) but places the fabricated misses in the actually-failed dense section
(this play collapses ~140s in) and NEVER on the confident opening. Perfectly
reconstructing WHICH 29 notes were missed is not recoverable from the replay's
press stream; this test pins the placement invariants we CAN guarantee.

Runnable: pytest tests/test_taiko_miss_placement.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from osu_taiko_renderer.beatmap.beatmap import parse_beatmap
from osu_taiko_renderer.beatmap.models import RenderConfig
from osu_taiko_renderer.beatmap.replay import parse_replay
from osu_taiko_renderer.render.scene import GREAT, MISS, OK, TaikoSim

HERE = Path(__file__).resolve().parent
OSR = HERE / "fixtures" / "jayceko_strain.osr"
OSU = HERE / "fixtures" / "jayceko_strain.osu"

# .osr header ground truth (authoritative — the results-screen numbers).
HDR_GREAT, HDR_OK, HDR_MISS, HDR_COMBO = 1045, 135, 29, 826


def _judge():
    frames, meta = parse_replay(OSR)
    bm = parse_beatmap(OSU, mods=meta.mods)
    sim = TaikoSim(bm, frames, RenderConfig(), skin=None, has_bg=False, meta=meta)
    # final per-note verdict, in time order
    order = sorted(range(len(sim.notes)), key=lambda i: sim.notes[i].time_ms)
    verdict = [sim.note_hit[id(sim.notes[i])][1] for i in order]
    times = [sim.notes[i].time_ms for i in order]
    return verdict, times


def _longest_run(verdict):
    run = mx = 0
    for v in verdict:
        if v == MISS:
            run = 0
        else:
            run += 1
            mx = max(mx, run)
    return mx


def test_counts_stay_header_exact():
    """The displayed tier totals + max combo must equal the .osr header exactly."""
    verdict, _ = _judge()
    assert verdict.count(GREAT) == HDR_GREAT
    assert verdict.count(OK) == HDR_OK
    assert verdict.count(MISS) == HDR_MISS
    assert _longest_run(verdict) == HDR_COMBO


def test_no_miss_on_the_confident_opening():
    """The reported bug: fabricated misses landed on the first notes the player
    cleanly hit. The clean opening must be miss-free."""
    verdict, times = _judge()
    # first 20 notes (time order) — the easy intro, all cleanly hit
    assert MISS not in verdict[:20]
    # no miss anywhere in the first 6 seconds of gameplay
    first_t = times[0]
    early = [v for v, t in zip(verdict, times) if t - first_t <= 6000]
    assert MISS not in early


def test_misses_land_in_the_failed_section():
    """This play holds a long clean run and only collapses in the dense ending;
    every fabricated/real miss should sit in that late failed region, not sprayed
    across a section the player comboed through."""
    verdict, times = _judge()
    miss_times = [t for v, t in zip(verdict, times) if v == MISS]
    assert len(miss_times) == HDR_MISS
    # the real collapse starts ~140s in; nothing before it should be a miss
    assert min(miss_times) >= 120_000


if __name__ == "__main__":
    for fn in (test_counts_stay_header_exact,
               test_no_miss_on_the_confident_opening,
               test_misses_land_in_the_failed_section):
        fn()
        print("ok:", fn.__name__)
    print("all passed")
