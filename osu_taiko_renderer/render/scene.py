"""Taiko simulation + per-frame scene builder.

Judges the replay against the map with a faithful in-order taiko sim (each
key-press consumes the FRONTMOST still-hittable same-colour note inside the OD
hit window — notelock, no press-stealing; a note whose OK window passes unhit
is a MISS), giving GREAT/OK/MISS placement that lands combo breaks on the notes
the player actually missed. The .osr header stays the count-authority: a
reconcile pass snaps the great/ok/miss TOTALS to the header (fewest placement
changes) and a max-combo pass makes the longest unbroken run equal the header's
max_combo, so counts/acc/score are header-exact while WHICH notes broke combo
is now correct. Then for any frame time it builds the right->left scrolling
scene: hit target, don/kat notes, drumroll bodies, swells, and hit explosions.
"""
from __future__ import annotations

import bisect
import math

from osu_taiko_renderer.argon import _const as AC
from osu_taiko_renderer.argon import geometry as ag_geom
from osu_taiko_renderer.render.dim import build_dim_envelope
from osu_taiko_renderer.skin.taiko_skin import TaikoSkin
from osu_taiko_renderer.beatmap.models import (
    SceneState,
    Sprite,
    TaikoType,
    od_to_hit_windows_ms,
)
from osu_taiko_renderer.beatmap.replay import hit_events

GREAT, OK, MISS = "great", "ok", "miss"

# Always render the judge’s HONEST judgments instead of fabricating misses to
# match the .osr header miss total / max_combo (Red 2026-07-27, #25). Flip to
# False to restore the header-count reconcile. Score still anchors to the .osr.
_TAIKO_ALWAYS_HONEST = True

# osu!lazer ScoreV3 total-score mod multipliers — ppy/osu#37967 (mode-agnostic;
# identical table to the std/catch/mania engines). Rate mods (DT/HT) at the
# standard rate; unlisted mods -> 1.0. Keyed by the osu! mod bit.
_MOD_SCORE_MULT = {
    1 << 1: 0.80, 1 << 3: 1.04, 1 << 4: 1.09, 1 << 6: 1.23,
    1 << 8: 0.55, 1 << 9: 1.23, 1 << 10: 1.20, 1 << 12: 0.95,
}


def mods_score_multiplier(mods: int) -> float:
    mods = int(mods or 0)
    if mods & (1 << 9):        # NC stored as DT|NC — count the speed mult once
        mods &= ~(1 << 6)
    m = 1.0
    for bit, mult in _MOD_SCORE_MULT.items():
        if mods & bit:
            m *= mult
    return m


# osu!(lazer) taiko hit windows. Stable uses od_to_hit_windows_ms (models.py:
# GREAT=50-3·OD, OK=120-8·OD). Lazer replays (game_version >= LAZER_GAME_VERSION,
# the same < 30000000 stable / else lazer split the std engine uses) use
# osu.Game.Rulesets.Taiko/Scoring/TaikoHitWindows.cs: DifficultyRange-interpolated
# Great 50/35/20, Ok 120/80/50 (ms half-widths at OD 0/5/10). The Miss range
# (135/95/70) only bounds CanBeHit; we treat the OK window edge as the hittable
# boundary (a note past OK+ with no matching press is a miss), matching stable.
LAZER_GAME_VERSION = 30000000


def _difficulty_range(diff: float, min_v: float, mid_v: float, max_v: float) -> float:
    """IBeatmapDifficultyInfo.DifficultyRange — linear interpolation across
    OD 0/5/10 (ppy/osu osu.Game/Beatmaps/IBeatmapDifficultyInfo.cs)."""
    if diff > 5:
        return mid_v + (max_v - mid_v) * (diff - 5.0) / 5.0
    if diff < 5:
        return mid_v - (mid_v - min_v) * (5.0 - diff) / 5.0
    return mid_v


def _lazer_taiko_windows(od: float) -> tuple[float, float]:
    """(GREAT, OK) half-widths in ms from lazer's TaikoHitWindows."""
    great = _difficulty_range(od, 50.0, 35.0, 20.0)
    ok = _difficulty_range(od, 120.0, 80.0, 50.0)
    return great, ok


def _key_edges(frames, attr) -> list:
    """Rising-edge press times of one physical taiko key (cl/cr/rl/rr) — a
    per-key stream, unlike hit_events' colour-collapsed edges."""
    out = []
    prev = False
    for f in frames:
        cur = getattr(f, attr)
        if cur and not prev:
            out.append(f.time_ms)
        prev = cur
    return out


def _nearest_abs(sorted_times, t) -> float:
    """|delta| to the nearest value in a time-sorted list (1e9 if empty)."""
    if not sorted_times:
        return 1e9
    j = bisect.bisect_left(sorted_times, t)
    best = 1e9
    for k in (j - 1, j):
        if 0 <= k < len(sorted_times):
            best = min(best, abs(sorted_times[k] - t))
    return best


def _longest_run(res_in_time_order) -> int:
    """Longest run of non-MISS results = the displayed max combo."""
    run = mx = 0
    for r in res_in_time_order:
        if r == MISS:
            run = 0
        else:
            run += 1
            if run > mx:
                mx = run
    return mx


class _JudgeState:
    """Mutable per-note judgment arrays shared by the reconcile passes (all
    indexed by note index; `order_t` gives time order)."""
    __slots__ = ("order_t", "col", "res", "hit_t", "signed", "err_mag",
                 "real_press", "miss_prox", "don_press", "kat_press",
                 "great_w", "ok_w")

    def __init__(self, order_t, col, res, hit_t, signed, err_mag, real_press,
                 miss_prox, don_press, kat_press, great_w, ok_w):
        self.order_t = order_t
        self.col = col
        self.res = res
        self.hit_t = hit_t
        self.signed = signed
        self.err_mag = err_mag
        self.real_press = real_press
        self.miss_prox = miss_prox
        self.don_press = don_press
        self.kat_press = kat_press
        self.great_w = great_w
        self.ok_w = ok_w


class TaikoSim:
    def __init__(self, bm, frames, cfg, *, skin=None, has_bg=False, meta=None):
        self.bm = bm
        self.cfg = cfg
        self.skin = skin
        self.has_bg = has_bg
        self.meta = meta
        # Hidden (HD, mod bit 8): notes fade out as they approach the drum.
        self.hidden = bool(int(getattr(meta, "mods", 0) or 0) & 8)
        self.w, self.h = cfg.resolution
        self.geo = ag_geom.compute(self.w, self.h)
        self.pp = 0.0
        self._final_pp = 0.0
        # user skin element presence (drum/hit-target/barline/drumroll come from
        # the skin when it provides them, else Argon).
        self.skin = TaikoSkin(cfg.skin_dir)
        self.sk_drum = self.skin.has("taiko-bar-left")
        self.sk_drum_in = self.skin.has("taiko-drum-inner")
        self.sk_drum_out = self.skin.has("taiko-drum-outer")
        self.sk_barline = self.skin.has("taiko-barline")
        self.sk_roll = self.skin.has("taiko-roll-middle")
        self.sk_lane = self.skin.has("taiko-bar-right")
        # mascot (pippidon): count frames per state; aspect for sizing
        self.mascot_counts = {}
        for _st in ("idle", "kiai", "clear", "fail"):
            _n = 0
            while self.skin.has(f"pippidon{_st}{_n}"):
                _n += 1
            if _n:
                self.mascot_counts[_st] = _n
        self.sk_mascot = bool(self.mascot_counts)
        self._mascot_aspect = 1.0
        self._mascot_head, self._mascot_foot = 0.0, 1.0     # visible-body y fractions
        if self.sk_mascot:
            _s0 = "idle" if "idle" in self.mascot_counts else next(iter(self.mascot_counts))
            _im = self.skin.load(f"pippidon{_s0}0")
            if _im is not None and _im.shape[0]:
                self._mascot_aspect = _im.shape[1] / _im.shape[0]
                _rows = (_im[..., 3] > 8).any(axis=1).nonzero()[0]
                if len(_rows):
                    self._mascot_head = _rows[0] / _im.shape[0]
                    self._mascot_foot = (_rows[-1] + 1) / _im.shape[0]

        def _aspect(name, default):
            img = self.skin.load(name)
            return (img.shape[1] / img.shape[0]) if img is not None else default
        self._barline_aspect = _aspect("taiko-barline", 0.05)
        self._drum_aspect = _aspect("taiko-bar-left", 1.0)
        # drum-inner/outer are half-width left-side graphics; ratio vs bar-left.
        _bl = self.skin.load("taiko-bar-left")
        _in = self.skin.load("taiko-drum-inner")
        self._drum_inner_ratio = (_in.shape[1] / _bl.shape[1]
                                  if (_in is not None and _bl is not None) else 0.49)

        # Per-quadrant press timestamps for the input-drum flash. Each list is
        # already time-sorted (frames are sorted in parse_replay).
        self._zpress = {
            "cl": [f.time_ms for f in frames if f.cl],
            "cr": [f.time_ms for f in frames if f.cr],
            "rl": [f.time_ms for f in frames if f.rl],
            "rr": [f.time_ms for f in frames if f.rr],
        }

        # Rising-edge press counts per key, for the Argon key counter (B1–B4).
        self._zedges = {}
        for zone in ("cl", "cr", "rl", "rr"):
            edges, prev = [], False
            for f in frames:
                cur = getattr(f, zone)
                if cur and not prev:
                    edges.append(f.time_ms)
                prev = cur
            self._zedges[zone] = edges

        # VISUAL hit windows: the shipped stable formula, UNCONDITIONALLY — these
        # drive the miss-judgement/explosion display time (note.time + ok_w) and
        # the HUD, so keeping them exactly as before makes clean plays render
        # byte-identically. The game-version-aware windows (lazer TaikoHitWindows
        # for lazer replays) are used ONLY inside the placement sweep in _judge.
        self.great_w, self.ok_w = od_to_hit_windows_ms(bm.od)
        _gv = int(getattr(meta, "game_version", 0) or 0)
        if _gv >= LAZER_GAME_VERSION:
            self.sweep_great_w, self.sweep_ok_w = _lazer_taiko_windows(bm.od)
        else:
            self.sweep_great_w, self.sweep_ok_w = self.great_w, self.ok_w
        self.notes = [o for o in bm.objects
                      if o.kind in (TaikoType.DON, TaikoType.KAT)]
        self.rolls = [o for o in bm.objects
                      if o.kind in (TaikoType.DRUMROLL, TaikoType.SWELL)]
        self.bar_lines = getattr(bm, "bar_lines", [])
        self.kiai = getattr(bm, "kiai_ranges", [])
        self.timing = getattr(bm, "timing", None)
        # R3D intro splash (show_logo): render.py sets logo_start_ms when the
        # flag is on; the splash fades out exactly as the first note begins
        # its scroll-in (first object time - its on-screen travel time, i.e.
        # the same "first approach start" the std/catch splashes fade out at).
        self.logo_start_ms: float | None = None
        if bm.objects:
            _first = min(bm.objects, key=lambda o: o.time_ms)
            _fsv = max(getattr(_first, "scroll_vel", 1.0) or 1.0, 0.1)
            self.first_spawn_ms = _first.time_ms - self.geo.scroll_time / _fsv
            _preempt = _first.time_ms - self.first_spawn_ms
        else:
            self.first_spawn_ms = 0.0
            _preempt = 0.0
        # Background dim envelope (std's DimEnvelope, ported in dim.py): the
        # dim GLIDES intro→game as the first note begins its scroll-in,
        # brightens into [Events] breaks and re-dims at the resume anchor —
        # smoothstep over the same 900 ms std/catch use, replacing the old
        # constant bg_dim_game level (intro and breaks previously rendered at
        # gameplay dim; bg_dim_intro/bg_dim_breaks were parsed but unused).
        self._dim_env = build_dim_envelope(
            cfg.bg_dim_intro / 100.0, cfg.bg_dim_game / 100.0,
            cfg.bg_dim_breaks / 100.0,
            [o.time_ms for o in bm.objects], _preempt,
            getattr(bm, "breaks", []) or [])
        self.note_hit: dict[int, tuple[int, str]] = {}
        self._judge(frames)

    # --- judgment -------------------------------------------------------------

    def _judge(self, frames):
        """Hybrid judgment: keep the shipped timing/great-ok layer for clean
        plays byte-identical, fix ONLY which notes broke combo.

        (A) VISUAL/TIMING LAYER (unchanged): the original greedy nearest-match
        (collapsed-edge, ±200 ms) gives each note its jump-off/explosion time and
        the great/ok split ranks notes by that timing error. A clean full-combo
        therefore renders exactly as before — no placement to get wrong.

        (B) MISS PLACEMENT (the fix): a faithful in-order per-key sweep — each
        rising-edge press consumes the FRONTMOST still-hittable same-colour note
        whose OK window covers it (taiko notelock, no press-stealing, no slop; a
        note whose OK window passes unhit is a MISS) — decides WHICH notes the
        player actually missed. Ported from osu.Game.Rulesets.Taiko/Objects/
        Drawables/DrawableHit.cs (CheckForResult / HitWindows.ResultFor|CanBeHit)
        + the DrawableTaikoHitObject input path (a press judges the earliest
        unjudged matching-colour note; a big/finisher note needs only one
        correct-colour hit for the base result). Crucially it uses PER-KEY press
        edges (cl|cr = don, rl|rr = kat): hit_events collapses cl|cr into one
        `don` boolean and only fires on the combined rising edge, losing the
        overlapping/alternating same-colour taps a taiko stream is full of.

        (C) RECONCILE (header = count-authority): _reconcile snaps the miss TOTAL
        to the .osr header AND makes the longest unbroken run equal the header's
        max_combo, keeping the honest pressless misses and adding the rest on the
        most miss-like notes outside the protected clean run — the std renderer's
        reconcile-after-sim + max-combo pass. Counts/acc/score stay header-exact.
        """
        hits = hit_events(frames)
        htimes = [hh[0] for hh in hits]
        # Press times (sorted) — reused by the swell to derive mash progress
        # (completion = hits-in-window / required_hits). Kept as the collapsed
        # stream so swell visuals are unchanged.
        self._hit_times = htimes

        notes = self.notes
        n = len(notes)
        ok_w = self.ok_w                              # VISUAL window (unchanged)
        # PLACEMENT windows for the sweep: lazer TaikoHitWindows for lazer
        # replays, stable formula for stable (auto-detected from game_version).
        sweep_great_w, sweep_ok_w = self.sweep_great_w, self.sweep_ok_w

        # --- (A) shipped greedy nearest-match: VISUAL timing + great/ok layer.
        # Verbatim from the original _judge so clean plays render byte-identical.
        old_err = [1e9] * n         # timing error per note (1e9 = no nearby hit)
        old_hit_t = [0] * n         # matched hit time (jump-off / explosion)
        used = [False] * len(hits)
        for i, note in enumerate(notes):
            want = "don" if note.kind is TaikoType.DON else "kat"
            j = bisect.bisect_left(htimes, note.time_ms)
            best, bd = -1, 201.0
            for k in range(max(0, j - 8), min(len(hits), j + 9)):
                if used[k] or hits[k][1] != want:
                    continue
                d = abs(hits[k][0] - note.time_ms)
                if d < bd:
                    bd, best = d, k
            if best >= 0:
                used[best] = True
                old_err[i] = bd
                old_hit_t[i] = hits[best][0]
            else:
                old_hit_t[i] = int(note.time_ms + ok_w)

        # --- (B) per-key in-order sweep: honest MISS PLACEMENT. A don note is hit
        # by EITHER centre key (cl/cr), a kat by EITHER rim key (rl/rr) — a
        # colour's presses are the UNION of its two keys' rising edges.
        don_press = sorted(_key_edges(frames, "cl") + _key_edges(frames, "cr"))
        kat_press = sorted(_key_edges(frames, "rl") + _key_edges(frames, "rr"))
        order_t = sorted(range(n), key=lambda i: notes[i].time_ms)
        col = ["don" if notes[i].kind is TaikoType.DON else "kat"
               for i in range(n)]
        sres = [MISS] * n           # honest sweep result (hit vs MISS)
        s_err = [1e9] * n           # sweep timing error (hit quality)
        real_press = [False] * n    # judged by a genuine in-window press
        for want, presses in (("don", don_press), ("kat", kat_press)):
            seq = [i for i in order_t if col[i] == want]
            ni = 0
            for p in presses:
                # notes whose OK window has fully passed can no longer be hit
                while ni < len(seq) and notes[seq[ni]].time_ms + sweep_ok_w < p:
                    ni += 1
                if ni >= len(seq):
                    break
                i = seq[ni]
                d = p - notes[i].time_ms
                if -sweep_ok_w <= d <= sweep_ok_w:         # window covers press
                    sres[i] = GREAT if abs(d) <= sweep_great_w else OK
                    s_err[i] = abs(d)
                    real_press[i] = True
                    ni += 1
                # else: press earlier than this note's window — wasted (no steal)

        # nearest matching-colour press |delta| for each honest MISS (reconcile
        # ranking: near press = ambiguous, un-missed first; far/absent = a real
        # combo break that must stay a miss).
        miss_prox = [1e9] * n
        for i in range(n):
            if sres[i] == MISS:
                stream = don_press if col[i] == "don" else kat_press
                miss_prox[i] = _nearest_abs(stream, notes[i].time_ms)

        st = _JudgeState(order_t, col, sres, old_hit_t, [0.0] * n, s_err,
                         real_press, miss_prox, don_press, kat_press,
                         sweep_great_w, sweep_ok_w)
        # --- (C) choose the final miss set: header miss total + max_combo ---
        is_miss = self._reconcile(st)

        # --- (D) assemble results: misses at their window-close time; hits keep
        # the shipped greedy timing, split into the header's GREAT/OK by that
        # (shipped) timing error so a clean play is byte-identical.
        m = self.meta
        if _TAIKO_ALWAYS_HONEST:
            hg = sum(1 for r in st.res if r == GREAT)   # honest great/ok split
        elif m is not None:
            hg = int(getattr(m, "count_300", 0) or 0)
        else:
            hg = sum(1 for i in range(n) if not is_miss[i])
        hit_idx = [i for i in range(n) if not is_miss[i]]
        hit_idx.sort(key=lambda i: old_err[i])       # stable: ties keep note order
        results: list[tuple[int, str]] = [(0, MISS)] * n
        for rank, i in enumerate(hit_idx):
            results[i] = (old_hit_t[i], GREAT if rank < hg else OK)
        for i in range(n):
            if is_miss[i]:
                results[i] = (int(notes[i].time_ms + ok_w), MISS)
            self.note_hit[id(notes[i])] = results[i]

        order = sorted(range(len(results)), key=lambda i: results[i][0])
        self._rt = [results[i][0] for i in order]
        # signed hit errors for the hit-error/UR bar: only notes with a genuine
        # matched press (old_err<1e8) and a hit result — same set the shipped
        # renderer drew.
        self._hit_errors = sorted(
            (results[i][0], old_hit_t[i] - notes[i].time_ms, results[i][1])
            for i in range(n)
            if results[i][1] != MISS and old_err[i] < 1e8)
        self._he_times = [e[0] for e in self._hit_errors]
        self._he_csum = [0.0]
        self._he_csq = [0.0]
        for _, e, _r in self._hit_errors:
            self._he_csum.append(self._he_csum[-1] + e)
            self._he_csq.append(self._he_csq[-1] + e * e)
        combo = great = ok = miss = score = 0
        hp = 1.0
        self._cum: list[tuple] = []
        for i in order:
            r = results[i][1]
            if r == GREAT:
                great += 1; combo += 1; score += 300; hp = min(1.0, hp + 0.02)
            elif r == OK:
                ok += 1; combo += 1; score += 100; hp = min(1.0, hp + 0.005)
            else:
                miss += 1; combo = 0; hp = max(0.0, hp - 0.05)
            self._cum.append((combo, great, ok, miss, score, hp))
        self._build_scorev2(order, results)
        self._setup_health(order, results)

    # --- reconcile (header = count-authority; std's reconcile-after-sim) -------

    def _reconcile(self, st):
        """Choose the FINAL miss set. Snap the miss TOTAL to the .osr header AND
        make the longest unbroken run equal the header max_combo — the std
        renderer's reconcile-after-sim + max-combo reposition, specialised to
        taiko's linear note stream. Returns is_miss[] (bool, by note index); the
        caller supplies the hit timing + great/ok split from the shipped
        greedy-match layer, so a clean play is byte-identical.

        The honest per-key sweep already lands the clean sections perfectly (its
        longest run typically already equals the real max_combo), but the .osr
        miss count is higher: in busy sections the greedy sweep 'saves' notes with
        stray mash presses that the client counted as misses. We (1) PROTECT the
        honest peak run (the max_combo-length clean stretch) so the displayed max
        combo is preserved, and (2) place the header's miss TOTAL on the most
        miss-like notes OUTSIDE that stretch — keeping every honest pressless miss
        (a real combo break) and never breaking the protected clean run."""
        m = self.meta
        n = len(st.res)
        seq = st.order_t
        sim_g = sum(1 for r in st.res if r == GREAT)
        sim_o = sum(1 for r in st.res if r == OK)
        sim_m = sum(1 for r in st.res if r == MISS)
        self._sim_counts = (sim_g, sim_o, sim_m)
        honest_longest = _longest_run([st.res[i] for i in seq])
        self._maxcombo_before = honest_longest
        is_miss = [st.res[i] == MISS for i in range(n)]
        if m is None or n == 0:
            self._reconcile_note = "taiko: no meta — honest sim kept"
            self._maxcombo_target = 0
            self._maxcombo_after = honest_longest
            self._maxcombo_note = f"taiko: max combo {honest_longest} (honest)"
            return is_miss
        if _TAIKO_ALWAYS_HONEST:
            # Honest judgments: do NOT fabricate misses to hit the header.
            hg0 = int(getattr(m, "count_300", 0) or 0)
            ho0 = int(getattr(m, "count_100", 0) or 0)
            hm0 = int(getattr(m, "count_miss", 0) or 0)
            T0 = int(getattr(m, "max_combo", 0) or 0)
            self._maxcombo_target = T0
            self._maxcombo_after = honest_longest
            self._reconcile_note = (
                f"taiko: HONEST kept {sim_g}/{sim_o}/{sim_m} "
                f"(header {hg0}/{ho0}/{hm0} NOT forced)")
            self._maxcombo_note = (
                f"taiko: max combo {honest_longest} (honest; header {T0})")
            return is_miss
        hg = int(getattr(m, "count_300", 0) or 0)
        ho = int(getattr(m, "count_100", 0) or 0)
        hm = int(getattr(m, "count_miss", 0) or 0)
        T = int(getattr(m, "max_combo", 0) or 0)
        self._maxcombo_target = T
        if hg + ho + hm != n:
            # totals fold in drumroll/swell tick scoring or a note-parse
            # off-by-one — keep the honest per-note-window sim; score anchors to
            # the .osr in _build_scorev2.
            self._reconcile_note = (
                f"taiko: header totals {hg}/{ho}/{hm} sum != {n} notes — honest "
                f"sim kept ({sim_g}/{sim_o}/{sim_m})")
            self._maxcombo_after = honest_longest
            self._maxcombo_note = (
                f"taiko: max combo {honest_longest} (honest; header {T})")
            return is_miss

        M = hm
        # per-POSITION hit-likeness (immutable snapshot of the honest sweep):
        # small = a clean hit; large = a miss-like note. Real hits rank by timing
        # error; misses rank AFTER every hit by how far the nearest matching press
        # sits (a pressless miss = worst = a definite combo break).
        qpos = [0.0] * n
        hmiss = [0] * n
        for p in range(n):
            i = seq[p]
            if st.res[i] == MISS:
                qpos[p] = st.ok_w + 1.0 + st.miss_prox[i]
                hmiss[p] = 1
            else:
                qpos[p] = st.err_mag[i]

        chosen = self._choose_miss_positions(n, M, T, qpos, hmiss)
        is_miss = [False] * n
        for p in chosen:
            is_miss[seq[p]] = True

        final_longest = _longest_run(
            [MISS if is_miss[i] else GREAT for i in seq])
        self._maxcombo_after = final_longest
        changed = sum(1 for i in range(n) if (st.res[i] == MISS) != is_miss[i])
        self._reconcile_note = (
            f"taiko: honest {sim_g}/{sim_o}/{sim_m} -> header {hg}/{ho}/{hm} "
            f"({changed} miss-set changes)")
        note = (f"taiko: max combo honest {honest_longest} -> {final_longest} "
                f"(target {T})")
        if T > 0 and final_longest != T:
            note = "!!! " + note + " — MISMATCH"
            print("[taiko-renderer] " + note, file=__import__("sys").stderr)
        self._maxcombo_note = note
        return is_miss

    def _choose_miss_positions(self, n, M, T, qpos, hmiss):
        """Pick exactly M note POSITIONS (time order) to be misses so the longest
        run == T, preferring the most miss-like notes and keeping the honest
        pressless misses. Returns a set of positions in [0, n)."""
        if M <= 0:
            return set()
        if M >= n:
            return set(range(n))
        if T <= 0:                                  # no combo target — pure quality
            return set(sorted(range(n), key=lambda p: (-qpos[p], p))[:M])

        # (a) protected peak = the T-length window with the FEWEST honest misses
        # inside (tie: most hit-like = lowest total q), so forcing it to a clean
        # run disturbs the honest sim least. For a faithful sim this is exactly
        # the honest peak run.
        Tw = min(T, n)
        pre_hm = [0] * (n + 1)
        pre_q = [0.0] * (n + 1)
        for p in range(n):
            pre_hm[p + 1] = pre_hm[p] + hmiss[p]
            pre_q[p + 1] = pre_q[p] + qpos[p]
        a, best = 0, None
        for s in range(0, n - Tw + 1):
            e = s + Tw
            key = (pre_hm[e] - pre_hm[s], pre_q[e] - pre_q[s])
            if best is None or key < best:
                best, a = key, s
        b = a + Tw - 1
        protect = set(range(a, b + 1))

        chosen = set()
        # (b) bound the protected run at exactly T: the notes just outside it are
        # breaks (free when they were already honest misses; a count-preserving
        # trim otherwise).
        if a - 1 >= 0:
            chosen.add(a - 1)
        if b + 1 <= n - 1:
            chosen.add(b + 1)
        # (c) keep every honest miss OUTSIDE the protected run (real combo breaks)
        for p in range(n):
            if hmiss[p] and p not in protect:
                chosen.add(p)
        # (d) top up to exactly M with the most miss-like notes OUTSIDE the
        # protected run (never break the clean stretch)
        if len(chosen) < M:
            pool = sorted((p for p in range(n)
                           if p not in chosen and p not in protect),
                          key=lambda p: (-qpos[p], p))
            for p in pool:
                if len(chosen) >= M:
                    break
                chosen.add(p)
        # (e) still short only if the busy region can't hold M (rare): allow
        # breaking the protected run at its most miss-like notes
        if len(chosen) < M:
            pool = sorted((p for p in protect if p not in chosen),
                          key=lambda p: (-qpos[p], p))
            for p in pool:
                if len(chosen) >= M:
                    break
                chosen.add(p)
        # (f) too many (boundary trims overshot a tiny M): drop the most hit-like
        # non-honest-miss break
        while len(chosen) > M:
            drop = min(chosen, key=lambda p: (hmiss[p], -qpos[p], -p))
            chosen.discard(drop)
        return chosen

    def _setup_health(self, order, results):
        """HP source + fail detection. Ground-truth is the .osr life-bar graph;
        when the replay carries none we fall back to the cumulative model.

        Fail/end-on-death is gated STRICTLY on the life-bar: we only cut the video
        short when the recorded HP actually reaches ~0 and the graph stops well
        before the last object. Without a life bar we never cut (the replay-frame
        end-time is not a reliable fail signal — trailing frames / modded
        time-axis differences false-positive it and truncate good renders)."""
        m = self.meta
        lb = tuple(getattr(m, "life_bar", ()) or ())
        self._lb_t = [p[0] for p in lb]
        self._lb_v = [p[1] for p in lb]
        last_obj = max((o.end_ms or o.time_ms for o in self.bm.objects), default=0)
        no_fail = bool(int(getattr(m, "mods", 0) or 0) & (1 << 0))
        self.failed = False
        self.fail_time_ms = 0
        if not no_fail and self._lb_t:
            end_t, end_v = self._lb_t[-1], self._lb_v[-1]
            if end_v <= 0.02 and end_t < last_obj - 3000:
                self.failed = True
                self.fail_time_ms = end_t
        # A genuine pass should never visibly "die"; floor the fallback model so a
        # rough patch doesn't read as death when the player actually survived.
        self._hp_floor = 0.0 if (self.failed or self._lb_t) else 0.15

    def _hp_at(self, t, model_hp):
        """HP at map-time t: interpolate the life-bar graph when present, else the
        (floored) cumulative model."""
        lt = self._lb_t
        if lt:
            if t <= lt[0]:
                return self._lb_v[0]
            if t >= lt[-1]:
                return self._lb_v[-1]
            j = bisect.bisect_right(lt, t) - 1
            t0, t1 = lt[j], lt[j + 1]
            v0, v1 = self._lb_v[j], self._lb_v[j + 1]
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return v0 + (v1 - v0) * f
        return max(self._hp_floor, model_hp)

    def _build_scorev2(self, order, results):
        """osu! standardised (score v2) cumulative curve, per ScoreProcessor:
        total = 500000·Acc·comboProgress + 500000·Acc^5·accProgress (+bonus),
        with Acc = curBase/maxBase, comboPortion += 300·√(comboAfter),
        comboProgress = comboPortion/maxComboPortion, accProgress = judged/total.
        Taiko base scores: GREAT=300, OK=150, MISS=0. Normalised so the final
        equals the .osr's authoritative score (absorbs the mod multiplier + tick
        bonus we don't simulate)."""
        n = len(order)
        max_combo_portion = sum(300.0 * ((j + 1) ** 0.5) for j in range(n))
        cur_base = max_base = combo_portion = 0.0
        combo = judged = 0
        raw: list[float] = []
        for idx, i in enumerate(order):
            res = results[i][1]
            base = 300.0 if res == GREAT else (150.0 if res == OK else 0.0)
            max_base += 300.0
            cur_base += base
            combo = 0 if res == MISS else combo + 1
            combo_portion += 300.0 * (combo ** 0.5)
            judged += 1
            acc = cur_base / max_base if max_base else 1.0
            cprog = combo_portion / max_combo_portion if max_combo_portion else 1.0
            aprog = judged / n if n else 1.0
            raw.append(500000.0 * acc * cprog + 500000.0 * (acc ** 5) * aprog)
        # ScoreV3 × mod multiplier (ppy/osu#37967). Previously the standardised
        # curve was scaled to the .osr's authoritative score, which carried
        # osu's OLD mod multiplier and made taiko inconsistent with the other
        # modes. Now the curve is scaled by the NEW mod multiplier — unified
        # across engines. (Trade-off: drops the drumroll/swell tick bonus the
        # .osr scaling used to absorb; same class of simplification std makes.)
        _mm = mods_score_multiplier(int(getattr(self.meta, "mods", 0) or 0))
        _m = self.meta
        _tot = ((getattr(_m, "count_300", 0) or 0) + (getattr(_m, "count_100", 0) or 0)
                + (getattr(_m, "count_miss", 0) or 0)) if _m is not None else 0
        if raw and _m is not None and _tot != len(order) and getattr(_m, "score", 0) and raw[-1] > 0:
            # Judgment reconciliation FELL BACK: replay hit-count (_tot) != sim
            # note count (an off-by-one in note parsing, ~2.5% of plays). The
            # per-note judging is then UNRELIABLE, so raw*mult renders a wrong
            # score (e.g. 351k for a 99.3% play). Anchor the curve to the
            # replay's authoritative score instead (the pre-ScoreV3 behavior) so
            # these plays are no worse than before the unification.
            _k = _m.score / raw[-1]
            self._scorev2 = [int(round(r * _k)) for r in raw]
        elif raw:
            self._scorev2 = [int(round(r * _mm)) for r in raw]
        else:
            self._scorev2 = [c[4] for c in self._cum]   # fallback: internal sum

    def _state_at(self, t):
        i = bisect.bisect_right(self._rt, t) - 1
        if i < 0:
            return 0, 0, 0, 0, 0, 1.0, self._hp_at(t, 1.0)
        combo, great, ok, miss, _sum, hp = self._cum[i]
        score = self._scorev2[i] if i < len(self._scorev2) else _sum
        tot = great + ok + miss
        acc = (great + ok * 0.5) / tot if tot else 1.0
        return combo, great, ok, miss, score, acc, self._hp_at(t, hp)

    def compute_pp_curve(self, osu_path, mods):
        """Final taiko pp via rosu-pp (if available in the venv). The live HUD
        counter scales it by play progress in build_scene; the end value matches
        the play's actual pp. Fails soft to 0 if rosu-pp is missing."""
        self.pp = 0.0
        self._final_pp = 0.0
        try:
            import rosu_pp_py as rosu
        except Exception:  # noqa: BLE001
            return
        try:
            bm = rosu.Beatmap(path=str(osu_path))
            try:
                bm.convert(rosu.GameMode.Taiko, int(mods))
            except Exception:  # noqa: BLE001
                try:
                    bm.convert(rosu.GameMode.Taiko)
                except Exception:  # noqa: BLE001
                    pass
            m = self.meta
            perf = rosu.Performance(
                mods=int(mods),
                n300=int(getattr(m, "count_300", 0) or 0),
                n100=int(getattr(m, "count_100", 0) or 0),
                misses=int(getattr(m, "count_miss", 0) or 0),
                combo=int(getattr(m, "max_combo", 0) or 0),
            )
            self._final_pp = float(perf.calculate(bm).pp)
        except Exception:  # noqa: BLE001
            self._final_pp = 0.0

    # --- scene ----------------------------------------------------------------

    def recent_errors(self, t, window=3500):
        """Recent hit timing errors for the hit-error bar: (err_ms, res, age_ms)
        for hits judged within the last `window` ms."""
        lo = bisect.bisect_left(self._he_times, t - window)
        hi = bisect.bisect_right(self._he_times, t)
        return [(e[1], e[2], t - e[0]) for e in self._hit_errors[lo:hi]]

    def ur_at(self, t):
        """Unstable rate = 10 × stddev of all hit errors up to t."""
        hi = bisect.bisect_right(self._he_times, t)
        if hi < 2:
            return 0.0
        mean = self._he_csum[hi] / hi
        var = self._he_csq[hi] / hi - mean * mean
        return 10.0 * (max(0.0, var) ** 0.5)

    def kiai_pulse(self, t):
        """Beat flash intensity during kiai (ArgonCirclePiece.OnNewBeat): a
        0.15α flash on each beat, fading over ~0.75 of the beat (OutSine).
        0 outside kiai."""
        if not self.kiai or self.timing is None:
            return 0.0
        if not any(s <= t < e for s, e in self.kiai):
            return 0.0
        bl = self.timing.beat_length(t)
        if bl <= 0:
            return 0.0
        ptime = 0.0
        for pt, _b, _m in getattr(self.timing, "uninherited", []):
            if pt <= t:
                ptime = pt
            else:
                break
        phase = ((t - ptime) % bl) / bl
        if phase < 0.75:
            import math
            return 0.15 * math.cos(phase / 0.75 * math.pi / 2)   # OutSine fade
        return 0.0

    def _x_at(self, time_ms, scroll_vel, t):
        g = self.geo
        visible = g.scroll_time / max(scroll_vel, 0.1)
        progress = (time_ms - t) / visible          # 1=spawn, 0=target
        spawn_x = self.w + g.note_d
        return g.target_x + progress * (spawn_x - g.target_x), progress

    def _drum_flash(self, zone: str, t: int) -> float:
        """Input-drum quadrant flash intensity (ArgonInputDrum): on press alpha
        jumps to ~0.5 then fades out over 200ms with an OutQuint curve."""
        times = self._zpress[zone]
        i = bisect.bisect_right(times, t) - 1
        if i < 0:
            return 0.0
        age = t - times[i]
        if age < 0 or age > AC.DRUM_PRESS_UP_MS:
            return 0.0
        f = 1.0 - age / AC.DRUM_PRESS_UP_MS
        return AC.DRUM_PRESS_ALPHA * (f ** 5)        # OutQuint-ish

    def _mascot_sprites(self, t):
        """pippidon mascot at the playfield left, animating by state (kiai during
        kiai, fail on low HP, else idle) at the skin's 60fps. Drawn behind the
        notes; feet near the lane top, facing right."""
        g = self.geo
        counts = self.mascot_counts
        hp = self._state_at(t)[6]
        in_kiai = bool(self.kiai) and any(s <= t < e for s, e in self.kiai)
        if in_kiai and "kiai" in counts:
            state = "kiai"
        elif hp < 0.5 and "fail" in counts:
            state = "fail"
        elif "idle" in counts:
            state = "idle"
        else:
            state = next(iter(counts))
        frame = int(t * 0.06) % counts[state]          # AnimationFramerate 60
        pf_top = g.center_y - g.pf_h / 2.0
        vis = max(0.25, self._mascot_foot - self._mascot_head)
        mh = (g.pf_h * 1.25) / vis                      # visible body ~1.25x lane height (lazer-ish)
        mw = mh * self._mascot_aspect
        feet_y = pf_top + g.pf_h * 0.42                 # feet into the lane so legs sit BEHIND the bar
        cy = feet_y - (self._mascot_foot - 0.5) * mh    # place by real feet position
        cx = g.drum_x + g.drum_d * 0.45                 # over the input drum
        return [Sprite(cx, cy, mw, mh, f"pippidon_{state}_{frame}", (1, 1, 1, 1))]

    def build_scene(self, t: int) -> SceneState:
        g = self.geo
        s = SceneState(time_ms=t)
        sp = s.sprites
        cy = g.center_y
        w, h = self.w, self.h
        kp = self.kiai_pulse(t)            # kiai beat-flash intensity (0 if none)

        # --- background: dimmed beatmap image, then the Argon playfield strip.
        # lazer: ArgonPlayfieldBackgroundLeft (input-drum area) = solid black;
        # ArgonPlayfieldBackgroundRight (the note lane) = black @0.7 so the bg
        # shows through dimly. Boundary = the input-drum edge, not the target. ---
        if self.has_bg:
            # preset bg dim via the DimEnvelope (intro/game/breaks levels with
            # std's smoothstep glides): % dim (higher=darker) → brightness
            v = max(0.0, 1.0 - self._dim_env.level(t))
            sp.append(Sprite(w / 2, h / 2, w, h, "bg", (v, v, v, 1.0)))
        strip_h = g.pf_h * 1.18
        if self.sk_mascot:
            sp.extend(self._mascot_sprites(t))
        if self.sk_lane:
            # legacy lane background (taiko-bar-right), stretched across the
            # playfield height; the drum draws on top of its left portion.
            sp.append(Sprite(w / 2, cy, w, g.pf_h, "skin_hit_target", (1, 1, 1, 1)))
        else:
            drum_edge = 2.0 * g.drum_x            # = INPUT_DRUM_WIDTH * scale
            sp.append(Sprite(drum_edge / 2, cy, drum_edge, strip_h, None,
                             (0, 0, 0, 1.0)))                   # input-drum: solid black
            rw = w - drum_edge
            sp.append(Sprite(drum_edge + rw / 2, cy, rw, strip_h, None,
                             (0, 0, 0, 0.7)))                   # note lane: black @0.7

        # --- input drum: skin (taiko-bar-left + drum-inner/outer) or Argon ---
        if self.sk_drum:
            dd = g.pf_h
            dw = dd * self._drum_aspect
            # lazer LegacyInputDrum: taiko-bar-left is drawn LEFT-aligned at the
            # playfield left (Position 0,0); the two halves mirror about the
            # input-drum centre (drum_x = INPUT_DRUM_WIDTH/2). Centring the bar on
            # drum_x instead misaligns the press halves with the skin's own drum
            # circle/divider — the artist places it ~INPUT_DRUM_WIDTH/2 from the
            # LEFT, not at the texture's geometric centre — and the right half ends
            # up offset.
            sp.append(Sprite(dw / 2.0, cy, dw, dd, "skin_drum_idle", (1, 1, 1, 1)))
            iw = dw * self._drum_inner_ratio        # half-width press graphic
            # Left presses sit flush at the bar's left edge; right presses are the
            # left ones mirrored about drum_x. Rim is flipped opposite the Centre
            # (lazer gives the Rim sprite Scale(-1,1)): left kat flipped, right not.
            cx_l = iw / 2.0                          # left edge at 0
            cx_r = 2.0 * g.drum_x - iw / 2.0         # mirror of [0,iw] about drum_x
            for zone, key, cx, ok in (
                    ("cl", "skin_drum_inner", cx_l, self.sk_drum_in),
                    ("cr", "skin_drum_inner_r", cx_r, self.sk_drum_in),
                    ("rl", "skin_drum_outer_r", cx_l, self.sk_drum_out),
                    ("rr", "skin_drum_outer", cx_r, self.sk_drum_out)):
                a = self._drum_flash(zone, t)
                if ok and a > 0.01:
                    sp.append(Sprite(cx, cy, iw, dd, key,
                                     (1, 1, 1, min(1.0, a * 2.0))))
        else:
            sp.append(Sprite(g.drum_x, cy, g.drum_d, g.drum_d, "argon_drum_idle",
                             (1, 1, 1, 1)))

        # --- hit target: Argon double circle + white bars. A skin lane
        # (taiko-bar-right / taiko-bar-left) carries its own target mark, so
        # skip the Argon overlay when the skin provides the lane. ---
        if not self.sk_lane:
            sp.append(Sprite(g.target_x, cy, g.note_d, g.note_d,
                             "argon_hit_target", (1, 1, 1, 1)))
            bar_w = max(2.0, AC.HIT_TARGET_BORDER * g.scale)
            bar_h = (1.0 - AC.DEFAULT_STRONG_SIZE) * g.pf_h
            top_y = cy - g.pf_h / 2.0
            bot_y = cy + g.pf_h / 2.0
            sp.append(Sprite(g.target_x, top_y + bar_h / 2.0, bar_w, bar_h, None,
                             (1, 1, 1, 1)))
            sp.append(Sprite(g.target_x, bot_y - bar_h / 2.0, bar_w, bar_h, None,
                             (1, 1, 1, 1)))
        # kiai glow at the hit target (TaikoPlayfield KiaiGlow), pulsing
        if kp > 0.001:
            gd = g.pf_h * 1.3
            sp.append(Sprite(g.target_x, cy, gd, gd, "argon_note_flash",
                             (1.0, 0.86, 0.55, kp * 1.25)))

        # --- bar lines (measure lines, scroll with notes; major brighter) ---
        blw = max(2, int(w * 0.0013))
        for (btime, bsv, major) in self.bar_lines:
            bx, bp = self._x_at(btime, bsv, t)
            if bp < -0.03 or bp > 1.1:
                continue
            a = 1.0 if major else 0.5
            if self.sk_barline:
                bw_img = max(2.0, g.pf_h * self._barline_aspect)
                sp.append(Sprite(bx, cy, bw_img, g.pf_h, "skin_barline",
                                 (1, 1, 1, a)))
                continue
            sp.append(Sprite(bx, cy, blw, g.pf_h, None, (1, 1, 1, a)))
            if major:                       # faint white gradient anchors OUTSIDE
                ext = AC.BARLINE_MAJOR_EXT * g.scale        # major_extension = 10
                aa = a * 0.3                                # mainLine.Alpha * 0.3
                sp.append(Sprite(bx, cy - g.pf_h / 2 - ext / 2, blw, ext,
                                 "argon_barline_anchor", (1, 1, 1, aa)))   # above
                sp.append(Sprite(bx, cy + g.pf_h / 2 + ext / 2, blw, ext,
                                 "argon_barline_anchor_f", (1, 1, 1, aa)))  # below

        # --- drumroll bodies / swells (under the notes) ---
        for o in self.rolls:
            end_ms = o.end_ms or o.time_ms
            d = g.big_d if o.big else g.note_d
            if o.kind is TaikoType.DRUMROLL:
                x0, p0 = self._x_at(o.time_ms, o.scroll_vel, t)
                xe, pe = self._x_at(end_ms, o.scroll_vel, t)
                if pe > 1.1 or p0 < -0.2:
                    continue
                head, tail = min(x0, xe), max(x0, xe)   # head = leading (left) end
                length = max(d, tail - head)
                if self.sk_roll:
                    # skin: stretched taiko-roll-middle body + taiko-roll-end cap;
                    # head uses the gold drumroll note (argon_drumroll is the
                    # skin's gold-tinted note when a skin is present).
                    sp.append(Sprite((head + tail) / 2, cy, length, d,
                                     "skin_roll_mid", (1, 1, 1, 1)))
                    sp.append(Sprite(tail, cy, d, d, "skin_roll_end", (1, 1, 1, 1)))
                    sp.append(Sprite(head, cy, d, d, "argon_drumroll", (1, 1, 1, 1)))
                else:
                    # capsule body, rounded gold caps, chevron ticks (no glow —
                    # ArgonElongatedCirclePiece has no glow layer)
                    sp.append(Sprite((head + tail) / 2, cy, length, d,
                                     "argon_drumroll_body", (1, 1, 1, 1)))
                    sp.append(Sprite(head, cy, d, d, "argon_drumroll", (1, 1, 1, 1)))
                    sp.append(Sprite(tail, cy, d, d, "argon_drumroll", (1, 1, 1, 1)))
                    step = max(d * 0.9, length / max(1, round(length / (d * 0.9))))
                    x = head
                    while x <= tail + 1:
                        sp.append(Sprite(x, cy, d, d, "argon_tick", (1, 1, 1, 1)))
                        x += step
            else:  # SWELL — faithful osu!lazer Argon DefaultSwell port:
                # scroll in -> lock at target -> target ring expands 5x -> the
                # asterisk spins + a yellow ring brightens with mash progress ->
                # body fades + scales out on clear.
                TARGET_SCALE = 5.0          # DefaultSwell.target_ring_scale
                # lazer DefaultSwell: the target ring + centre circle are BOTH
                # RelativeSizeAxes=Both (start coincident at the swell-circle size)
                # and the ring does ScaleTo(5, OutQuint). Base it on the swell
                # circle (big note) so it expands to ~5x and overflows the lane
                # like lazer — the old `pf_h*0.95` cap made the "5x" ring barely
                # note-sized, so the swell never read as a swell. Base on the
                # note size (→ full ring ≈ 5×note ≈ 2.4× lane height: dramatic
                # but stays on-screen; 5×big_d overflowed into the HUD).
                base_ring = g.note_d
                dur = max(1.0, end_ms - o.time_ms)

                # clear/disappear (lazer: bodyContainer FadeOut 300ms OutQuad +
                # ScaleTo 1.4 at HitStateUpdateTime ~= end).
                end_age = t - end_ms
                if end_age > 300:
                    continue
                if end_age > 0:
                    ct = end_age / 300.0
                    body_alpha = (1.0 - ct) ** 2          # OutQuad fade-out
                    body_scale = 1.0 + 0.4 * ct           # ScaleTo 1.4
                else:
                    body_alpha, body_scale = 1.0, 1.0

                # position: scroll in until start, then sit at the hit target.
                if t < o.time_ms:
                    sx, sp_ = self._x_at(o.time_ms, o.scroll_vel, t)
                    if sp_ > 1.1:
                        continue
                else:
                    sx = g.target_x

                # mash completion = hits inside [start, min(t,end)] / required.
                req = max(1, int(getattr(o, "required_hits", 0) or 0))
                if self._hit_times:
                    lo = bisect.bisect_left(self._hit_times, o.time_ms)
                    hi = bisect.bisect_right(self._hit_times, min(t, end_ms))
                    completion = min(1.0, max(0, hi - lo) / req)
                else:                                     # no inputs -> time proxy
                    completion = min(1.0, max(0.0, (t - o.time_ms) / dur))

                # target ring: base size while scrolling in, expands base->5x
                # over [start+100, start+500] with OutQuint.
                if t < o.time_ms:
                    ring_scale = 1.0
                else:
                    rt = min(1.0, max(0.0, (t - o.time_ms - 100.0) / 400.0))
                    oq = 1.0 - (1.0 - rt) ** 5            # Easing.OutQuint
                    ring_scale = 1.0 + (TARGET_SCALE - 1.0) * oq
                ring_d = base_ring * ring_scale * body_scale

                # expanding yellow ring (lazer additive; approximated with a
                # bright yellow glow on the dark playfield): scales 1->5 and
                # brightens with completion.
                if t >= o.time_ms:
                    exp_scale = 1.0 + (TARGET_SCALE - 1.0) * min(1.0, completion * 1.3)
                    exp_d = base_ring * exp_scale * body_scale
                    exp_a = min(0.55, 0.12 + completion * 0.6) * body_alpha
                    if exp_a > 0.01:
                        sp.append(Sprite(sx, cy, exp_d, exp_d, "argon_swell_glow",
                                         (1.0, 0.92, 0.32, exp_a)))

                # thin target ring on top of the glow — DefaultSwell tints it
                # YellowDark(eeaa00)@0.25 additive; on our dark playfield we draw
                # that gold crisp ring at a readable alpha (was a hard white ring,
                # which broke the swell's all-gold Argon theme).
                sp.append(Sprite(sx, cy, ring_d, ring_d, "argon_swell_ring",
                                 (0.95, 0.71, 0.09, 0.8 * body_alpha)))

                # centre asterisk: spins by completion * Duration / 8 (degrees).
                cd = g.big_d * body_scale
                cs = Sprite(sx, cy, cd, cd, "argon_swell", (1, 1, 1, body_alpha))
                cs.rotation = math.radians(completion * dur / 8.0)
                sp.append(cs)

        # --- don/kat notes (earliest on top: draw reversed) ---
        for o in reversed(self.notes):
            rt, res = self.note_hit.get(id(o), (0, MISS))
            d = g.big_d if o.big else g.note_d
            if o.kind is TaikoType.DON:
                key = "argon_don"
            else:
                key = "argon_kat"
            if o.big:
                key = key + "_big"          # skin taikobigcircle (or scaled note)
            if res != MISS and t >= rt:
                # hit: gravity "jump off" (DrawableHit, SnapJudgementLocation
                # off) — the note KEEPS scrolling left while it arcs up 200u
                # (OutQuad 300ms) then falls 400u (InQuad), scale→0.8, fade 800ms.
                age = t - rt
                if age > 800:
                    continue
                nx, _ = self._x_at(o.time_ms, o.scroll_vel, t)   # continues left
                if age <= 300:
                    yoff = -(1.0 - (1.0 - age / 300.0) ** 2) * g.pf_h     # up, OutQuad
                else:
                    q = (age - 300.0) / 600.0
                    yoff = (-1.0 + (q * q) * 2.0) * g.pf_h                # fall, InQuad
                scale = 1.0 - 0.2 * min(1.0, age / 600.0)
                a = max(0.0, 1.0 - age / 800.0)
                ny, dd = cy + yoff, d * scale
                # (ArgonCirclePiece has no glow layer — note is core+rings+icon)
                sp.append(Sprite(nx, ny, dd, dd, key, (1, 1, 1, a)))
                continue
            x, p = self._x_at(o.time_ms, o.scroll_vel, t)
            if p > 1.1:
                continue
            # osu!taiko Hidden (TaikoModHidden): a note starts fading the
            # moment it spawns and is fully invisible after 37.5% of its
            # travel, so it vanishes well before the drum and you hit blind.
            # `p` is 1 at spawn → 0 at the target; fade_out_start_time=1,
            # fade_out_duration=0.375 → alpha = (p - 0.625) / 0.375.
            na = 1.0
            if self.hidden:
                na = (p - 0.625) / 0.375
                if na <= 0.0:
                    continue          # fully hidden — skip
                if na > 1.0:
                    na = 1.0
            if res == MISS and t >= rt:
                _ma = t - rt
                if _ma > 100:
                    continue          # lazer FadeOut(100) done -> despawned
                na *= max(0.0, 1.0 - _ma / 100.0)
            elif p < -0.15:
                continue              # non-miss unhit backstop
            sp.append(Sprite(x, cy, d, d, key, (1, 1, 1, na)))
            if kp > 0.001:        # kiai beat flash on the note (additive white)
                sp.append(Sprite(x, cy, d, d, "argon_note_flash", (1, 1, 1, kp * na)))

        # R3D intro splash -- topmost intro element, over the idle scene
        if self.logo_start_ms is not None:
            sp.extend(self._logo_sprites(t))

        combo, great, ok, miss, score, acc, hp = self._state_at(t)
        s.combo, s.score, s.accuracy, s.hp = combo, score, acc, hp
        s.counts = (great, ok, miss)
        # live pp: scale final pp by the score-v2 progress (combo-weighted, so
        # it accelerates with the play) rather than a flat object count.
        hdr = self._scorev2[-1] if self._scorev2 else 0
        s.pp = (self._final_pp * (score / hdr)) if hdr > 0 else 0.0
        return s

    def _logo_sprites(self, t: int) -> list[Sprite]:
        """The R3D 'R' tile intro splash (show_logo), fading out exactly as
        the first note begins its scroll-in -- ported from the std renderer's
        _draw_logo (via the catch port) so the splash is identical across
        modes. std/catch draw the halo additively; taiko's sprite pass is
        straight-alpha only, so the same radial glow texture is blended
        normally -- visually equivalent over the dark intro playfield (the
        same approximation the swell glow already uses)."""
        from osu_taiko_renderer.render.effects import logo_alpha, logo_scale, LOGO_UI_SIZE
        la = logo_alpha(t, self.logo_start_ms, self.first_spawn_ms)
        if la is None:
            return []
        k_ui = self.h / 1080.0
        d = LOGO_UI_SIZE * k_ui * logo_scale(t, self.logo_start_ms)
        cx = self.w / 2.0
        cy = self.h * 0.44
        return [
            Sprite(cx, cy, d * 1.9, d * 1.9, texture_key="logo_glow",
                   color=(0.95, 0.28, 0.30, 0.45 * la)),
            Sprite(cx, cy, d, d, texture_key="logo_tile",
                   color=(1.0, 1.0, 1.0, la)),
        ]

    def key_counts(self, t: int):
        """(B1,B2,B3,B4) press counts up to t — left→right: rim-L, centre-L,
        centre-R, rim-R."""
        return tuple(bisect.bisect_right(self._zedges[z], t)
                     for z in ("rl", "cl", "cr", "rr"))

    def drum_flashes(self, t: int):
        """Active input-drum press flashes: (is_rim, left, intensity 0..1) for
        each pressed quadrant, composited additively at the drum."""
        out = []
        if self.sk_drum:           # skin drum handles its own press in build_scene
            return out
        for zone, is_rim, left in (("cl", False, True), ("cr", False, False),
                                   ("rl", True, True), ("rr", True, False)):
            a = self._drum_flash(zone, t)
            if a > 0.01:
                out.append((is_rim, left, a))
        return out

    def active_effects(self, t: int):
        """Effects to composite additively over the readback frame at time t:
        (explosions, judgements). explosions: (is_rim, age_ms, big, result).
        judgements: (result, age_ms). Both keyed off each note's judged time."""
        exps = []
        judges = []
        for o in self.notes:
            rt, res = self.note_hit.get(id(o), (0, MISS))
            age = t - rt
            if res != MISS:
                dur = (AC.EXPLOSION_GREAT_OUT_MS if res == GREAT
                       else AC.EXPLOSION_OK_OUT_MS)
                if 0 <= age <= dur + AC.EXPLOSION_GREAT_IN_MS:
                    exps.append((o.kind is TaikoType.KAT, age, o.big, res))
            if 0 <= age <= AC.JUDGE_MOVE_MS:
                judges.append((res, age, rt, o.big))
        return exps, judges
