"""Authoritative lazer-standardised total score for a taiko replay (#155/#115).

R3D shows ONE score scale everywhere — osu!lazer's standardised (ScoreV3/
ScoreV2, max ~1,000,000 × mod multiplier) — no matter which client set the
play. The number the in-video counter ends on, the results screen, and the
website card must all be the SAME standardised total, and it must be the
number the player recognises from lazer / the osu! website.

The .osr legacy header `score` field means different things per source:

  • true osu!stable play ............ the stable ScoreV1 total (e.g. 3,847,220)
  • lazer play downloaded from the
    osu! website (legacy .osr export,
    stable-style game_version) ...... the CLASSIC-converted display total
                                      (lazer ScoreInfoExtensions.GetDisplayScore
                                      → convertStandardisedToClassic, case 1)
  • lazer client local export ....... ScoreInfo.TotalScore = the standardised
                                      total itself (LegacyScoreEncoder.Encode
                                      writes `(int)score.ScoreInfo.TotalScore`)

We convert the header under every interpretation and pick the candidate
closest to our own standardised simulation (TaikoSim's live ScoreV3 curve
endpoint), which is client-agnostic and accurate to a few percent — while the
wrong interpretation is typically off by ~10× (a ScoreV1 total read as
standardised, or a classic total read as ScoreV1).

Exact conversion sources (ppy/osu master, 2026-07):

  A. stable ScoreV1 → standardised (this is the important case: it turns a
     stable ScoreV1 total in the millions into the ~1,000,000-scale number the
     player recognises):
       osu.Game/Database/StandardisedScoreMigrationTools.cs
         convertFromLegacyTotalScore, setup + ruleset 1 (taiko):
           withoutMods = round(250000·comboProportion
                               + 750000·Accuracy^3.6
                               + bonusProportion)
           total       = round(withoutMods · modMultiplier)   # ScoreV3 mult
       osu.Game.Rulesets.Taiko/Difficulty/TaikoLegacyScoreSimulator.cs
         Simulate + simulateHit + GetLegacyScoreMultiplier   (ScoreV1 attrs)
       osu.Game/Rulesets/Objects/Legacy/LegacyRulesetExtensions.cs
         CalculateDifficultyPeppyStars   (decimal, banker's rounding; taiko
         forces CircleSize = 2 before this call)
     This is the SAME math osu-web/osu-queue-score-statistics runs server-side.

  B. classic ↔ standardised (lazer display conversion, exact & linear):
       osu.Game/Scoring/Legacy/ScoreInfoExtensions.cs
         convertStandardisedToClassic case 1 (taiko):
           classic = round((objectCount·1109 + 100000) · std / 1_000_000)
       objectCount = the map's maximum basic-judgement count = the number of
       Hit objects (= attributes.MaxCombo; combo only increments on Hits).
     Linear and monotonic, so the inverse is exact up to ±0.5 header rounding.
"""
from __future__ import annotations

import math
import re
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path
from struct import pack, unpack

from osu_taiko_renderer.beatmap.models import TaikoType

MAX_SCORE = 1_000_000.0

# .osr game_version boundary: stable writes 8-digit YYYYMMDD (~2025xxxx), lazer
# local exports write >= 30_000_000. Mirrors scene.py's LAZER_GAME_VERSION.
LAZER_GAME_VERSION_BOUNDARY = 30_000_000

# osu! legacy mod bit flags.
_NF = 1 << 0
_EZ = 1 << 1
_HD = 1 << 3
_HR = 1 << 4
_DT = 1 << 6
_RELAX = 1 << 7
_HT = 1 << 8       # HalfTime (Daycore shares this legacy bit)
_NC = 1 << 9       # Nightcore is stored DT|NC — count the rate once
_FL = 1 << 10
_SCORE_V2 = 1 << 29

# lazer bonus base scores (osu.Game/Rulesets/Judgements/Judgement.cs).
_SMALL_BONUS = 10   # HitResult.SmallBonus  (drum-roll tick)
_LARGE_BONUS = 50   # HitResult.LargeBonus  (completed swell)
# SwellTick is HitResult.IgnoreHit → base score 0.


# ── taiko legacy (ScoreV1) mod multipliers ─────────────────────────────────
# TaikoLegacyScoreSimulator.GetLegacyScoreMultiplier (ppy/osu master). NB these
# are the TAIKO-specific multipliers and differ from catch's table (taiko:
# HD/HR = 1.06, DT/NC/FL = 1.12).

def legacy_mod_multiplier(mods: int) -> float:
    mods = int(mods or 0)
    if mods & _RELAX:            # TaikoModRelax disqualifies → 0
        return 0.0
    if mods & _NC:              # NC stored as DT|NC — count the rate once
        mods &= ~_DT
    v2 = bool(mods & _SCORE_V2)
    m = 1.0
    if mods & _NF:
        m *= 1.0 if v2 else 0.5
    if mods & _EZ:
        m *= 0.5
    if mods & _HD:
        m *= 1.06
    if mods & _HR:
        m *= 1.06
    if mods & _DT:
        m *= 1.12
    if mods & _HT:              # HT / Daycore
        m *= 0.3
    if mods & _NC:
        m *= 1.12
    if mods & _FL:
        m *= 1.12
    return m


# ── CalculateDifficultyPeppyStars (LegacyRulesetExtensions) ────────────────

def _f32(v: float) -> float:
    """C# BeatmapDifficulty stores float32; mirror (decimal)(double)(float)v."""
    return unpack("f", pack("f", float(v)))[0]


def difficulty_peppy_stars(hp: float, od: float, cs: float,
                           object_count: int, drain_length_s: int) -> int:
    if drain_length_s != 0:
        otd = Decimal(object_count) / Decimal(drain_length_s) * 8
        otd = max(Decimal(0), min(Decimal(16), otd))
    else:
        otd = Decimal(16)
    total = (Decimal(_f32(hp)) + Decimal(_f32(od)) + Decimal(_f32(cs)) + otd)
    return int((total / Decimal(38) * Decimal(5))
               .quantize(Decimal(1), rounding=ROUND_HALF_EVEN))


def _round_even(x: float) -> int:
    return int(Decimal(x).quantize(Decimal(1), rounding=ROUND_HALF_EVEN))


# ── raw .osu facts for the peppy-stars inputs + the drum-roll tick delay ────
# The legacy sim uses the BASE beatmap (no mods): raw HP/OD, CircleSize forced
# to 2 (taiko), the raw hit-object count, drain seconds, plus SliderTickRate +
# the beatmap format version (both drive getSliderTaikoMinHitDelay).

def parse_base_osu_facts(osu_path: Path) -> dict:
    text = Path(osu_path).read_text(encoding="utf-8", errors="replace")
    version = 14
    m = re.search(r"osu file format v(\d+)", text)
    if m:
        version = int(m.group(1))
    section = None
    hp = 5.0
    od = 5.0
    slider_tick_rate = 1.0
    n_objects = 0
    first_t: float | None = None
    last_t: float | None = None
    break_ms = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        mm = re.match(r"^\[(.+)\]$", line)
        if mm:
            section = mm.group(1).lower()
            continue
        if section == "difficulty":
            k, _, v = line.partition(":")
            k = k.strip().lower()
            try:
                fv = float(v.strip())
            except ValueError:
                continue
            if k == "hpdrainrate":
                hp = fv
            elif k == "overalldifficulty":
                od = fv
            elif k == "slidertickrate":
                slider_tick_rate = fv
        elif section == "events":
            parts = line.split(",")
            if len(parts) >= 3 and parts[0].strip() in ("2", "Break"):
                try:
                    bs, be = float(parts[1]), float(parts[2])
                except ValueError:
                    continue
                break_ms += _round_even(be) - _round_even(bs)
        elif section == "hitobjects":
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                t = float(parts[2])
            except ValueError:
                continue
            n_objects += 1
            if first_t is None:
                first_t = t
            last_t = t
    drain_s = 0
    if n_objects and first_t is not None and last_t is not None:
        drain_s = (_round_even(last_t) - _round_even(first_t) - break_ms) // 1000
    return {"hp": hp, "od": od, "object_count": n_objects,
            "drain_length_s": int(drain_s), "slider_tick_rate": slider_tick_rate,
            "beatmap_version": version}


# ── TaikoLegacyScoreSimulator.Simulate / simulateHit (ScoreV1 attributes) ──

def _slider_taiko_min_hit_delay(beat_length: float, tick_rate: float,
                                version: int) -> float:
    """getSliderTaikoMinHitDelay — the ms spacing the legacy engine ticks a
    drum-roll at (osu-stable SliderTaiko). Uses the UNINHERITED beat length at
    the roll's start; the SliderTickRate 3/6/1.5 cases tick on 1/6 of the beat,
    else 1/8, then the rate is doubled/halved into the [60, 120] ms band."""
    if beat_length <= 0:
        beat_length = 1000.0
    if version >= 8 and (abs(tick_rate - 3.0) < 1e-9 or abs(tick_rate - 6.0) < 1e-9
                         or abs(tick_rate - 1.5) < 1e-9):
        max_rate = beat_length / 6.0
    else:
        max_rate = beat_length / 8.0
    while max_rate < 60.0:
        max_rate *= 2.0
    while max_rate > 120.0:
        max_rate /= 2.0
    return max_rate


def legacy_attributes(objects, facts: dict, timing, kiai_ranges) -> dict:
    """Walk the CONVERTED taiko objects (engine parse order == lazer playable
    order) accumulating stable ScoreV1 attributes, generating the drum-roll and
    swell ticks the legacy engine scores (they are not in `objects`). Returns
    the LegacyScoreAttributes equivalent:
      accuracy_score, combo_score, bonus_score, bonus_ratio, max_combo.

    Faithful port of TaikoLegacyScoreSimulator.simulateHit — all combo-multiplier
    arithmetic is INTEGER (C# int division) as in the source."""
    peppy = difficulty_peppy_stars(
        facts["hp"], facts["od"], 2.0,                 # taiko forces CS = 2
        facts["object_count"], facts["drain_length_s"])
    tick_rate = facts["slider_tick_rate"]
    version = facts["beatmap_version"]

    kiai = tuple(kiai_ranges or ())

    def kiai_at(t) -> bool:
        return any(s <= t < e for s, e in kiai)

    def beat_at(t) -> float:
        try:
            return float(timing.beat_length(t)) if timing is not None else 0.0
        except Exception:  # noqa: BLE001
            return 0.0

    acc_score = 0
    combo_score = 0
    legacy_bonus = 0
    standardised_bonus = 0
    combo = 0

    objs = sorted(objects, key=lambda o: o.time_ms)
    n = len(objs)
    for idx, o in enumerate(objs):
        nxt = objs[idx + 1] if idx + 1 < n else None

        if o.kind in (TaikoType.DON, TaikoType.KAT):
            # --- Hit -----------------------------------------------------
            score_increase = 300
            old = score_increase
            score_increase += (score_increase // 35) * 2 * (peppy + 1) \
                * (min(100, combo) // 10)
            if kiai_at(o.time_ms):
                score_increase = int(score_increase * 1.2)
            combo_inc = score_increase - old
            if o.big:                                  # strong (finish) note
                score_increase *= 2
                combo_inc *= 2
            score_increase -= combo_inc
            combo_score += combo_inc
            acc_score += score_increase                # always 300 (or 600)
            combo += 1

        elif o.kind is TaikoType.DRUMROLL:
            # --- DrumRoll → DrumRollTick(s) (bonus) ----------------------
            mhd = _slider_taiko_min_hit_delay(beat_at(o.time_ms), tick_rate, version)
            end = o.end_ms if o.end_ms is not None else o.time_ms
            imhd = int(mhd)
            if nxt is not None and nxt.kind is TaikoType.DRUMROLL:
                nxt_hittable = nxt.time_ms - _slider_taiko_min_hit_delay(
                    beat_at(nxt.time_ms), tick_rate, version)
            else:
                nxt_hittable = nxt.time_ms if nxt is not None else None
            if nxt_hittable is None:
                endpoint_hittable = True
            else:
                endpoint_hittable = (nxt_hittable - (end + imhd)) > imhd
            hittable_end = (end + imhd) if endpoint_hittable else end
            tick_kiai = kiai_at(o.time_ms)             # kiai at Parent.StartTime
            i = float(o.time_ms)
            while i < hittable_end:
                si = 300
                if tick_kiai:
                    si = int(si * 1.2)
                if o.big:                              # strong tick: +20%
                    si += si // 5
                legacy_bonus += si
                standardised_bonus += _SMALL_BONUS
                if mhd <= 0:
                    break
                i += mhd

        elif o.kind is TaikoType.SWELL:
            # --- Swell → SwellTick(s) + the swell (combo + bonus) --------
            dur = (o.end_ms if o.end_ms is not None else o.time_ms) - o.time_ms
            half = int(dur / 1000.0 * 7.5)             # minimum_rotations_per_second
            half = int(max(1, half * 1.65))
            half = max(1, int(half * 1.5))
            for _ in range(half + 1):                  # for i in 0..=half
                legacy_bonus += 300                    # SwellTick: IgnoreHit → +0 std
            # the swell object itself
            score_increase = 300
            old = score_increase
            score_increase += (score_increase // 35) * 2 * (peppy + 1) \
                * (min(100, combo) // 10)
            end_t = o.end_ms if o.end_ms is not None else o.time_ms
            if kiai_at(end_t):                         # kiai at GetEndTime()
                score_increase = int(score_increase * 1.2)
            combo_inc = score_increase - old
            score_increase *= 2                        # swell is always doubled
            combo_inc *= 2
            score_increase -= combo_inc
            combo_score += combo_inc
            legacy_bonus += score_increase
            standardised_bonus += _LARGE_BONUS
            # increaseCombo == false for a swell

    return {
        "accuracy_score": acc_score,
        "combo_score": combo_score,
        "bonus_score": legacy_bonus,
        "bonus_ratio": (standardised_bonus / legacy_bonus) if legacy_bonus else 0.0,
        "max_combo": combo,
        "peppy_stars": int(peppy),
    }


# ── play accuracy (osu!taiko: OK weighted 0.5, identical stable & lazer) ────

def taiko_accuracy(meta) -> float:
    great = int(getattr(meta, "count_300", 0) or 0)
    ok = int(getattr(meta, "count_100", 0) or 0)
    miss = int(getattr(meta, "count_miss", 0) or 0)
    total = great + ok + miss
    if total <= 0:
        return 1.0
    return (great + 0.5 * ok) / total


# ── stable ScoreV1 total → standardised (convertFromLegacyTotalScore case 1) ─

def stable_to_standardised(meta, attrs: dict, new_mod_multiplier: float) -> int:
    mods = int(getattr(meta, "mods", 0) or 0)
    legacy_total = int(meta.score)
    if mods & _SCORE_V2:
        # ScoreV2-mod stable scores are already 1M-standardised — TotalScore
        # stays as-is (matches osu-catch's score_fidelity).
        return legacy_total

    legacy_mult = legacy_mod_multiplier(mods)
    max_acc = int(attrs["accuracy_score"])
    max_combo_score = int(round(attrs["combo_score"] * legacy_mult))
    max_bonus = int(attrs["bonus_score"])
    bonus_ratio = float(attrs["bonus_ratio"])

    acc = max(0.0, min(1.0, taiko_accuracy(meta)))
    legacy_acc_score = max_acc * acc

    denom = max_combo_score + max_bonus
    if denom > 0:
        combo_proportion = max(legacy_total - legacy_acc_score, 0.0) / denom
    else:
        combo_proportion = 0.0 if legacy_mult == 0 else 1.0

    max_base = max_acc + max_combo_score
    bonus_proportion = max(0.0, (legacy_total - max_base) * bonus_ratio)

    without_mods = round(250000.0 * combo_proportion
                         + 750000.0 * (acc ** 3.6)
                         + bonus_proportion)
    return int(round(without_mods * new_mod_multiplier))


# ── classic ↔ standardised (exact lazer display conversion, taiko) ──────────

def standardised_to_classic(std: int, object_count: int) -> int:
    return int(round((object_count * 1109 + 100000) * std / MAX_SCORE))


def classic_to_standardised(classic: int, object_count: int) -> int:
    """Inverse of the linear taiko classic conversion, snapped to the exact
    preimage when one exists (classic headers are rounded)."""
    classic = int(classic)
    if classic <= 0:
        return 0
    factor = (object_count * 1109 + 100000) / MAX_SCORE
    s = classic / factor if factor > 0 else float(classic)
    best = int(round(s))
    for cand in range(max(0, best - 3), best + 4):
        if standardised_to_classic(cand, object_count) == classic:
            return cand
    return best


# ── candidate resolution (mirrors osu-catch score_fidelity) ────────────────

def compute_candidates(meta, bm, osu_path, new_mod_multiplier: float) -> dict:
    """All defensible standardised interpretations of the header score."""
    facts = parse_base_osu_facts(osu_path)
    attrs = legacy_attributes(bm.objects, facts, getattr(bm, "timing", None),
                              getattr(bm, "kiai_ranges", ()))
    # objectCount for the classic conversion = maximum basic judgements = the
    # Hit count (combo only increments on Hits) = attributes.MaxCombo.
    object_count = int(attrs["max_combo"])
    gv = int(getattr(meta, "game_version", 0) or 0)
    header = int(meta.score)

    cands: dict[str, int | None] = {
        "stable_v1": stable_to_standardised(meta, attrs, new_mod_multiplier),
        "lazer_classic": classic_to_standardised(header, object_count),
        "lazer_direct": header if gv >= LAZER_GAME_VERSION_BOUNDARY else None,
    }
    return {
        "candidates": cands,
        "object_count": object_count,
        "beatmap_max_combo": int(attrs["max_combo"]),
        "legacy_attrs": attrs,
        "osu_facts": facts,
        "game_version": gv,
        "header_score": header,
    }


def resolve_authoritative(fid: dict, sim_final: int) -> tuple[int, str]:
    """Pick the authoritative standardised total. Returns (score, source_tag).

    game_version < 30M (stable-format .osr): the header is a legacy-space
    (ScoreV1) total — a true stable play stores its real ScoreV1 total, and an
    osu-web download of a lazer play stores the server's synthesised
    legacy_total_score, both of which round-trip through the same taiko
    convertFromLegacyTotalScore math back to the standardised number the player
    saw. So stable_v1 is the correct decode; the sim is only a sanity guard
    (near-FC plays legitimately sim a few % off the worst-case-combo estimate).

    game_version >= 30M (lazer client export): current encoders write the
    standardised total itself; older ones wrote the classic display total. Pick
    whichever interpretation is closest to the sim — they differ by ~10× so the
    choice is unambiguous.

    NB: for the stable branch this trusts stable_v1 MORE broadly than the catch
    engine's ±50% guard, because the taiko honest-judge sim is a less reliable
    proximity reference than catch's CatchSim (its combo reconciliation can be
    well off on messy replays). Since there is no competing interpretation for a
    sub-30M header (it is always legacy-space), stable_v1 — lazer's own
    server-side conversion — is authoritative; the sim only rejects it on a
    >5× disagreement, i.e. a wrong-interpretation-class (~10×) blow-up, not the
    few-percent noise of an imperfect sim."""
    gv = int(fid.get("game_version", 0) or 0)
    ref = max(1, int(sim_final))
    if gv < LAZER_GAME_VERSION_BOUNDARY:
        val = fid["candidates"].get("stable_v1")
        if val is not None and val > 0:
            if sim_final > 0 and (val / ref > 5.0 or ref / max(1, val) > 5.0):
                return int(sim_final), "sim"     # stable_v1 is garbage → sim
            return int(val), "stable_v1"
        return int(sim_final), "sim"
    best: tuple[float, int, str] | None = None
    for tag in ("lazer_direct", "lazer_classic"):
        val = fid["candidates"].get(tag)
        if val is None or val < 0:
            continue
        err = abs(int(val) - ref) / ref
        if best is None or err < best[0]:
            best = (err, int(val), tag)
    if best is None or best[0] > 0.35:
        return int(sim_final), "sim"
    return best[1], best[2]
