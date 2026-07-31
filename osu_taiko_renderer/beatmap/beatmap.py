"""Parse a .osu file into the osu!taiko object stream.

Circle  -> DON (red centre) or KAT (blue rim), by hit-sound; finish -> big note
Slider  -> DRUMROLL (real taiko maps, or long/slow converts) OR a stream of
           DON/KAT hits (short sliders on std->taiko converts), per lazer's
           TaikoBeatmapConverter
Spinner -> SWELL (denden) over [time, end], with a required hit count

Drumroll duration uses the active timing point's beat length and slider-velocity
multiplier (same as the game). Each note also carries an effective scroll
multiplier (BPM x SV) so notes under different timing scroll at the right speed.
"""
from __future__ import annotations

from pathlib import Path

from osu_taiko_renderer.beatmap.models import (TaikoBeatmap, TaikoObject,
                                               TaikoType, HitSample, SamplePoint)

# HitObject type bitfield
_TYPE_CIRCLE = 1 << 0
_TYPE_SLIDER = 1 << 1
_TYPE_NEW_COMBO = 1 << 2
_TYPE_SPINNER = 1 << 3

# HitSound bitfield
_HS_WHISTLE = 1 << 1
_HS_FINISH = 1 << 2
_HS_CLAP = 1 << 3


class BeatmapParseError(RuntimeError):
    pass


def parse_beatmap(path: Path, *, mods: int = 0, lazer: bool = False) -> TaikoBeatmap:
    text = path.read_text(encoding="utf-8", errors="replace")
    sections = _split_sections(text)

    diff = _kv(sections.get("Difficulty", ""))
    meta = _kv(sections.get("Metadata", ""))
    general = _kv(sections.get("General", ""))

    cs = _f(diff.get("CircleSize"), 5.0)
    ar = _f(diff.get("ApproachRate"), _f(diff.get("OverallDifficulty"), 5.0))
    od = _f(diff.get("OverallDifficulty"), 5.0)
    hp = _f(diff.get("HPDrainRate"), 5.0)
    slider_mult = _f(diff.get("SliderMultiplier"), 1.4)
    slider_tick_rate = _f(diff.get("SliderTickRate"), 1.0)
    source_mode = int(_f(general.get("Mode"), 0.0))   # 1 = real taiko (no convert)

    ez = bool(mods & (1 << 1))
    hr = bool(mods & (1 << 4))
    if ez:
        od *= 0.5; hp *= 0.5
        # TaikoModEasy (lazer) only halves OD/HP(/CS/AR) — it does NOT touch
        # SliderMultiplier, so EZ leaves taiko scroll velocity unchanged.
    if hr:
        od = min(10.0, od * 1.4); hp = min(10.0, hp * 1.4)
    # Object times stay on the map-time axis (replay frames share it); rate only
    # compresses the output video + atempos audio in render.py.
    dt = bool(mods & (1 << 6)) or bool(mods & (1 << 9))
    ht = bool(mods & (1 << 8))
    rate = 1.5 if dt else (0.75 if ht else 1.0)

    timing = _parse_timing(sections.get("TimingPoints", ""))
    # SliderMultiplier IS lazer's taiko Velocity — it drives the scroll speed
    # (scroll_mult below). HardRock multiplies it ON TOP of the map value:
    # TaikoModHardRock.ApplyToDifficulty does `SliderMultiplier *= 1.4 * 4/3`
    # (=1.86667) in addition to the base OD/HP*1.4, so HR notes scroll ~1.87x
    # faster/wider than nomod. lazer applies difficulty mods AFTER the
    # beatmap->taiko conversion, so this bump affects ONLY the scroll velocity;
    # the slider->object conversion below keeps the ORIGINAL `slider_mult`
    # (passed to _parse_hit_objects unchanged). This is a SEPARATE factor from
    # the hidden _VELOCITY_MULTIPLIER=1.4 conversion constant — HR stacks on top.
    scroll_slider_mult = slider_mult * (1.4 * 4.0 / 3.0) if hr else slider_mult
    timing.slider_mult = scroll_slider_mult
    objects = _parse_hit_objects(
        sections.get("HitObjects", ""),
        timing=timing, slider_mult=slider_mult, od=od,
        slider_tick_rate=slider_tick_rate, is_for_taiko=(source_mode == 1),
    )
    objects.sort(key=lambda o: o.time_ms)
    first_t = objects[0].time_ms if objects else 0
    last_t = max((o.end_ms or o.time_ms for o in objects), default=0)
    bar_lines = _generate_bar_lines(timing, first_t, last_t)
    kiai = _parse_kiai(sections.get("TimingPoints", ""), last_t)

    return TaikoBeatmap(
        objects=objects,
        bar_lines=bar_lines,
        kiai_ranges=kiai,
        timing=timing,
        cs=cs, ar=ar, od=od, hp=hp, rate=rate,
        audio_filename=general.get("AudioFilename"),
        background=_parse_background(sections.get("Events", "")),
        breaks=_parse_breaks(sections.get("Events", "")),
        title=meta.get("Title", ""),
        artist=meta.get("Artist", ""),
        version=meta.get("Version", ""),
        sample_points=list(getattr(timing, "sample_points", []) or []),
        default_sample_set=(general.get("SampleSet", "Normal") or "Normal").lower(),
    )


# --- timing -------------------------------------------------------------------

class _Timing:
    """Resolves beat length and SV multiplier at any time."""

    def __init__(self, points: list[tuple[float, float, bool]]):
        self.points = points
        # base (first uninherited) beat length — the reference BPM for scroll.
        self.base_beat = next((v for _, v, u in points if u and v > 0), 500.0)
        self.slider_mult = 1.0      # set to the map's SliderMultiplier (Velocity)

    def beat_length(self, t: float) -> float:
        bl = self.base_beat
        for time, val, uninh in self.points:
            if time > t:
                break
            if uninh and val > 0:
                bl = val
        return bl

    def sv_mult(self, t: float) -> float:
        mult = 1.0
        for time, val, uninh in self.points:
            if time > t:
                break
            if not uninh and val < 0:
                mult = 100.0 / -val
            elif uninh:
                mult = 1.0
        return mult

    def scroll_mult(self, t: float) -> float:
        """lazer MultiplierControlPoint.Multiplier for taiko (BaseBeatLength is
        the fixed 60-BPM DEFAULT_BEAT_LENGTH=1000, NOT the map's base BPM):
          Multiplier = SliderMultiplier(Velocity) x SV(ScrollSpeed) x 1000/BeatLength.
        Visible time = TimeRange / Multiplier."""
        bl = self.beat_length(t)
        bpm_factor = 1000.0 / bl if bl > 0 else 1.0
        return self.slider_mult * self.sv_mult(t) * bpm_factor


def _parse_timing(block: str) -> _Timing:
    pts: list[tuple[float, float, bool]] = []
    uninh: list[tuple[float, float, int]] = []
    samps: list = []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        time = _f(parts[0], 0.0)
        beat = _f(parts[1], 500.0)
        uninherited = True if len(parts) < 7 else parts[6].strip() == "1"
        meter = int(_f(parts[2], 4.0)) if len(parts) > 2 else 4
        pts.append((time, beat, uninherited))
        s_set = int(_f(parts[3], 0.0)) if len(parts) > 3 else 0
        s_idx = int(_f(parts[4], 0.0)) if len(parts) > 4 else 0
        s_vol = int(_f(parts[5], 100.0)) if len(parts) > 5 else 100
        samps.append(SamplePoint(int(time), s_set, s_idx, s_vol or 100))
        if uninherited and beat > 0:
            uninh.append((time, beat, meter if meter > 0 else 4))
    pts.sort(key=lambda p: p[0])
    uninh.sort(key=lambda p: p[0])
    tm = _Timing(pts)
    tm.uninherited = uninh
    tm.sample_points = samps
    return tm


def _parse_kiai(block: str, last_t: float):
    """Kiai ranges from timing points (effects bit 0). A point with kiai on
    starts a section; it ends at the next point that turns it off."""
    pts = []
    for line in block.splitlines():
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        time = _f(parts[0], 0.0)
        effects = int(_f(parts[7], 0.0)) if len(parts) > 7 else 0
        pts.append((time, bool(effects & 1)))
    pts.sort(key=lambda p: p[0])
    ranges = []
    start = None
    for time, kiai in pts:
        if kiai and start is None:
            start = time
        elif not kiai and start is not None:
            ranges.append((start, time))
            start = None
    if start is not None:
        ranges.append((start, last_t + 1))
    return ranges


def _generate_bar_lines(timing: "_Timing", first_hit: float, last_hit: float):
    """Measure bar lines (BarLineGenerator): one per measure
    (BeatLength × meter) from each uninherited timing point; Major every
    meter-th measure. Returns [(time, scroll_vel, major)]."""
    import math
    pts = getattr(timing, "uninherited", [])
    if not pts:
        return []
    gen_start = min(0.0, first_hit)
    end_all = last_hit + 1.0
    out = []
    for idx, (ptime, beat, meter) in enumerate(pts):
        bar_len = beat * meter
        if bar_len <= 0:
            continue
        end = pts[idx + 1][0] if idx < len(pts) - 1 else end_all + bar_len
        if ptime > gen_start:
            start = ptime
        else:
            n = math.ceil((gen_start - ptime) / bar_len)
            start = ptime + n * bar_len
        beat_i = 0
        t = start
        while t < end + 1e-3:
            out.append((round(t), timing.scroll_mult(t), beat_i % meter == 0))
            t += bar_len
            beat_i += 1
    return out


# --- hit objects --------------------------------------------------------------

def _parse_hit_sample(field: str) -> HitSample:
    """Parse `normalSet:additionSet:index:volume:filename` (any part optional)."""
    p = (field or "").split(":")
    def gi(i):
        try:
            return int(p[i]) if i < len(p) and p[i] != "" else 0
        except ValueError:
            return 0
    return HitSample(gi(0), gi(1), gi(2), gi(3), p[4] if len(p) > 4 else "")



def _parse_hit_objects(block: str, *, timing, slider_mult, od,
                       slider_tick_rate: float = 1.0,
                       is_for_taiko: bool = False) -> list[TaikoObject]:
    out: list[TaikoObject] = []
    started = False
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        f = line.split(",")
        if len(f) < 5:
            continue
        time = int(float(f[2]))
        typ = int(f[3])
        hs = int(f[4])
        is_new = bool(typ & _TYPE_NEW_COMBO) or not started
        started = True
        big = bool(hs & _HS_FINISH)
        scroll = timing.scroll_mult(time)

        if typ & _TYPE_CIRCLE:
            kind = TaikoType.KAT if (hs & (_HS_WHISTLE | _HS_CLAP)) else TaikoType.DON
            _hsamp = _parse_hit_sample(f[5]) if len(f) > 5 else None
            out.append(TaikoObject(time, kind, big=big, scroll_vel=scroll,
                                   new_combo=is_new, hit_sound=hs, hit_sample=_hsamp))
        elif typ & _TYPE_SLIDER:
            out.extend(_convert_slider(
                f, time, hs, big, scroll, is_new,
                timing=timing, slider_mult=slider_mult,
                slider_tick_rate=slider_tick_rate, is_for_taiko=is_for_taiko))
        elif typ & _TYPE_SPINNER:
            end = int(float(f[5])) if len(f) > 5 else time + 1000
            hits = _swell_hits(end - time, od)
            out.append(TaikoObject(time, TaikoType.SWELL, end_ms=end,
                                   required_hits=hits, scroll_vel=scroll,
                                   new_combo=is_new))
    return out


# osu!std -> taiko slider conversion (ported from lazer TaikoBeatmapConverter).
_VELOCITY_MULTIPLIER = 1.4               # hidden global taiko speed factor
_OSU_BASE_SCORING_DISTANCE = 100.0


def _convert_slider(f, time, hs, big, scroll, is_new, *,
                    timing, slider_mult, slider_tick_rate, is_for_taiko):
    """Convert a slider hit-object the way osu!lazer's TaikoBeatmapConverter does:

      * real taiko maps (is_for_taiko)        -> DRUMROLL (a taiko slider IS a roll)
      * converted std map, long/slow slider   -> DRUMROLL
      * converted std map, short/fast slider  -> a stream of DON/KAT hits spaced
        `tick_spacing` apart.

    The last case is what was missing: every slider used to become a drumroll, so
    converted (Mode 0) maps rendered the wrong objects and the replay's individual
    hits matched nothing -> phantom misses. Counts/timings now mirror lazer so the
    .osr's authoritative great/ok/miss totals line up with the object stream.
    """
    spans = 1
    if len(f) > 6:
        try:
            spans = max(1, int(f[6]))
        except ValueError:
            spans = 1
    pixel_length = _f(f[7], 0.0) if len(f) > 7 else 0.0

    beat = timing.beat_length(time)
    sv = timing.sv_mult(time)
    tick_rate = max(0.1, slider_tick_rate)

    # lazer keeps these two multiplies separate for 1:1 float compatibility.
    distance = pixel_length * _VELOCITY_MULTIPLIER * spans
    taiko_velocity = _OSU_BASE_SCORING_DISTANCE * slider_mult * _VELOCITY_MULTIPLIER
    if taiko_velocity <= 0:
        taiko_velocity = 1.0
    # GetPrecisionAdjustedBeatLength: slider SV shortens the effective beat.
    adj_beat = beat / sv if sv > 0 else beat
    taiko_duration = int(distance / taiko_velocity * adj_beat)

    def _drumroll():
        return [TaikoObject(time, TaikoType.DRUMROLL, big=big,
                            end_ms=int(time + max(0, taiko_duration)),
                            scroll_vel=scroll, new_combo=is_new)]

    if is_for_taiko:
        return _drumroll()

    osu_velocity = taiko_velocity * (1000.0 / adj_beat) if adj_beat > 0 else 0.0
    # beatmap version >= 8 (all modern maps): tick spacing off the UNADJUSTED beat.
    tick_spacing = min(beat / tick_rate, taiko_duration / spans) if spans else 0.0
    if not (tick_spacing > 0 and osu_velocity > 0
            and distance / osu_velocity * 1000.0 < 2.0 * beat):
        return _drumroll()

    node_hs = _edge_sounds(f, hs, spans)
    out: list[TaikoObject] = []
    j = float(time)
    end = time + taiko_duration + tick_spacing / 8.0
    i = 0
    while j <= end + 1e-6:
        nhs = node_hs[i % len(node_hs)]
        kind = TaikoType.KAT if (nhs & (_HS_WHISTLE | _HS_CLAP)) else TaikoType.DON
        out.append(TaikoObject(int(round(j)), kind, big=bool(nhs & _HS_FINISH),
                               scroll_vel=scroll, new_combo=(is_new and i == 0)))
        i += 1
        if tick_spacing <= 1e-9:
            break
        j += tick_spacing
    return out or _drumroll()


def _edge_sounds(f, base_hs, spans):
    """Per-node hit-sounds (slider edge sounds f[8], cycled across the hit stream
    like lazer's NodeSamples). Falls back to the slider's base hit-sound."""
    if len(f) > 8 and "|" in f[8]:
        vals = []
        for tok in f[8].split("|"):
            try:
                vals.append(int(tok))
            except ValueError:
                vals.append(base_hs)
        if vals:
            return vals
    return [base_hs]


def _swell_hits(duration: float, od: float) -> int:
    """Required alternating hits to clear a swell. Approximates osu!taiko's
    duration- and OD-scaled hit count (refine vs lazer once validated)."""
    rate = 3.0 + od * 0.4          # hits/sec, rises with OD
    return max(1, int(duration / 1000.0 * rate))


# --- shared parsing helpers ---------------------------------------------------

def _split_sections(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    cur = None
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith("[") and line.rstrip().endswith("]"):
            if cur is not None:
                out[cur] = "\n".join(buf)
            cur = line.strip()[1:-1]
            buf = []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        out[cur] = "\n".join(buf)
    return out


def _kv(block: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in block.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def _parse_breaks(events: str) -> list:
    out = []
    for line in events.splitlines():
        f = line.split(",")
        if len(f) >= 3 and f[0].strip() in ("2", "Break"):
            try:
                out.append((int(float(f[1])), int(float(f[2]))))
            except ValueError:
                continue
    return out


def _parse_background(events: str) -> str | None:
    for line in events.splitlines():
        f = line.split(",")
        if len(f) >= 3 and f[0].strip() in ("0", "Background"):
            return f[2].strip().strip('"')
    return None


def _f(s, default: float) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return default
