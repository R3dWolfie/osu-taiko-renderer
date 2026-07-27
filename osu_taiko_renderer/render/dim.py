"""Background dim envelope — the std engine's DimEnvelope ported to taiko
(osu-std osu_std_renderer/render/background.py, via the catch port in
osu-catch osu_catch_renderer/dim.py; kept glide-for-glide so all engines
fade identically — cross-engine consistency is the whole point).

Semantics (std §4.10, R3D preset keys bg_dim_intro/game/breaks 0-100%):

    intro dim   HELD until the FIRST object's approach begins (startTime -
                preempt); the glide INTO gameplay dim STARTS there and runs
                over GLIDE_MS, so the background is still bright as the first
                note scrolls in and dims underneath it (danser)
    game dim    through gameplay
    breaks dim  during [Events] breaks: the brighten glide starts at the
                break start, holds bright across the break, and the re-dim
                glide STARTS at the resume anchor (min(break end, next
                object's approach start)) — the background stays bright until
                gameplay resumes and dims during the first post-break
                approach
    breaks too short to fit both glides are skipped (dim stays at game)

Taiko's breaks are plain ``(start_ms, end_ms)`` tuples (beatmap._parse_breaks)
— the same shape catch uses, so build_dim_envelope is catch's verbatim.
`preempt` here is the FIRST note's on-screen travel time
(geo.scroll_time / its scroll_vel — the same value TaikoSim.first_spawn_ms is
derived from); per-note SVs vary, but the envelope keeps std/catch's
single-preempt signature so the port stays glide-for-glide identical.
Pure math, no GL.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass

GLIDE_MS = 900.0     # std render/background.py GLIDE_MS — danser's measured
                     # break/intro dim fades (~0.85-0.9 s); keep IDENTICAL to
                     # std/catch so a multi-mode render set fades in lockstep.


def smoothstep(p: float) -> float:
    """3p² - 2p³ on [0,1] (clamped) — the glide ease (std's, verbatim)."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    return p * p * (3.0 - 2.0 * p)


@dataclass(frozen=True)
class _Glide:
    t0: float
    t1: float
    v0: float
    v1: float


class DimEnvelope:
    """Piecewise dim level: holds between glides, smoothsteps inside them.

    glides — (t0, t1, target_level) time-sorted; a glide overlapping the
    previous one (or zero/negative length) is DROPPED, keeping the level
    it would have started from (the envelope never jumps).
    """

    def __init__(self, initial: float,
                 glides: list[tuple[float, float, float]] = ()):
        self.initial = float(initial)
        self._glides: list[_Glide] = []
        level = self.initial
        last_end = -float("inf")
        for t0, t1, target in glides:
            if t0 < last_end or t1 <= t0:
                continue  # overlap/degenerate → dropped, level unchanged
            self._glides.append(_Glide(t0, t1, level, float(target)))
            level = float(target)
            last_end = t1
        self._starts = [g.t0 for g in self._glides]

    def level(self, t: float) -> float:
        """Dim in [0,1] at map time t (ms)."""
        idx = bisect.bisect_right(self._starts, t) - 1
        if idx < 0:
            return self.initial
        g = self._glides[idx]
        if t >= g.t1:
            return g.v1
        return g.v0 + (g.v1 - g.v0) * smoothstep((t - g.t0) / (g.t1 - g.t0))


def build_dim_envelope(intro: float, normal: float, breaks: float,
                       object_starts: list[float], preempt: float,
                       break_periods, glide_ms: float = GLIDE_MS,
                       ) -> DimEnvelope:
    """std build_dim_envelope with (start, end) break tuples (catch's exact
    signature). Levels are fractions 0..1 (callers divide the 0-100 preset
    values)."""
    starts = sorted(object_starts)
    if not starts:
        return DimEnvelope(normal)
    glides: list[tuple[float, float, float]] = []
    first_spawn = starts[0] - preempt
    # HOLD Dim.Intro until the first object's approach begins, then fade to
    # gameplay dim over the glide (danser dims DURING that first approach).
    glides.append((first_spawn, first_spawn + glide_ms, normal))
    last_end = first_spawn + glide_ms
    for b0, b1 in sorted(break_periods):
        nxt = next((s for s in starts if s >= b1), None)
        anchor = b1 if nxt is None else min(b1, nxt - preempt)
        # Enter the break BRIGHT at break start, hold, then begin the re-dim
        # AT the resume anchor (break end / next approach) — the background
        # stays bright until gameplay actually resumes and dims DURING the
        # first post-break approach. Both glides must fit: enter
        # [start, start+g], a bright plateau to the anchor, exit
        # [anchor, anchor+g]; too-short/out-of-order breaks are skipped
        # (dim stays at the gameplay level — std's exact rule).
        if b0 < last_end or b0 + glide_ms > anchor:
            continue
        glides.append((b0, b0 + glide_ms, breaks))
        glides.append((anchor, anchor + glide_ms, normal))
        last_end = anchor + glide_ms
    return DimEnvelope(intro, glides)
