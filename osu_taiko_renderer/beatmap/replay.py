"""Decode an osu!taiko .osr into per-frame don/kat key state.

Taiko replays store the standard .osr key bitfield. Empirically verified
against lazer taiko replays (player 'nichijou', a clean 154-note FC):
  bit 1 (M1) = DON (centre), bit 2 (M2) = KAT (rim) — every don press hit a
  don note, every kat press a kat note.
The second key pair (K1=4, K2=8) is treated symmetrically (K1 don, K2 kat) —
the common osu!taiko config; pending a multi-config replay to nail it exactly,
this is correct for single-pair players and a sane default otherwise. A big
note (finish) is hit by pressing both centre or both rim keys at once.
"""
from __future__ import annotations

import json
import lzma
import struct
from pathlib import Path

from osrparse import Replay

from osu_taiko_renderer.beatmap.models import ReplayMeta, TaikoFrame

_SEED_DELTA = -12345
_DON_BITS = 1 | 4   # M1, K1 (centre)
_KAT_BITS = 2 | 8   # M2, K2 (rim)

# Cutoff between stable's date-based game_version (~20251128, 8 digits) and
# lazer's identifier-style version (~30000016, 9 digits). At/above this is a
# lazer-format replay; below is stable. (Mirrors mania_ordr's
# LAZER_GAME_VERSION_BOUNDARY and osu-taiko's scene.LAZER_GAME_VERSION.)
LAZER_GAME_VERSION = 30_000_000


class ReplayParseError(RuntimeError):
    pass


def _recover_leadin_offset(path: Path) -> int:
    """Recover the replay-clock lead-in that osrparse silently discards.

    osu!stable replays begin with up to two placeholder frames at the
    off-screen sentinel position (256, -500). osu!'s own LegacyScoreDecoder
    ACCUMULATES each frame's time delta into the running clock *before* it
    drops those placeholders (``lastTime += diff`` precedes the
    ``i < 2 && (256,-500)`` skip), so the second placeholder's delta carries
    the audio lead-in / intro-skip offset. osrparse instead ``continue``s past
    these frames (osrparse/replay.py: ``if i < 2 and float(x) == 256 and
    float(y) == -500: continue``), throwing their deltas away — the lead-in
    never reaches ``Replay.replay_data``. Accumulating that stream from 0 then
    starts the clock too early by the whole lead-in, so every object is
    sampled before the player reached it (mass over-miss on stable replays
    whose intro-skip is not cancelled by a <-5000 ms first frame).

    Reproduce osu!'s accumulation: read the raw replay-data blob and sum the
    deltas of exactly the leading placeholder frames osrparse strips. Returns 0
    when there are none (lazer replays carry no placeholder frames; a clean
    stable play carries a ~0 ms lead-in), so already-aligned replays are left
    byte-identical. Fail-soft: any decode problem returns 0 (the pre-fix
    behaviour) rather than raising.
    """
    try:
        data = Path(path).read_bytes()
        off = 0

        def _skip_string() -> None:
            nonlocal off
            tag = data[off]
            off += 1
            if tag == 0x00:
                return
            if tag != 0x0b:
                raise ValueError(f"bad string tag {tag}")
            length = shift = 0
            while True:
                byte = data[off]
                off += 1
                length |= (byte & 0x7F) << shift
                if not (byte & 0x80):
                    break
                shift += 7
            off += length

        off += 1                       # mode (byte)
        off += 4                       # game version (int32)
        _skip_string()                 # beatmap md5
        _skip_string()                 # player name
        _skip_string()                 # replay md5
        off += 2 * 6                   # 300/100/50/geki/katu/miss (6 shorts)
        off += 4                       # score (int32)
        off += 2                       # max combo (short)
        off += 1                       # perfect (byte)
        off += 4                       # mods (int32)
        _skip_string()                 # life-bar graph
        off += 8                       # timestamp (int64)
        rlen = struct.unpack_from("<i", data, off)[0]
        off += 4                       # replay-data length (int32)
        raw = lzma.decompress(data[off:off + rlen],
                              format=lzma.FORMAT_AUTO).decode("ascii", "replace")

        lead = 0
        for i, group in enumerate(raw.rstrip(",").split(",")):
            if not group:
                continue
            fields = group.split("|")
            delta = int(fields[0])
            if delta == _SEED_DELTA:   # RNG seed (never a leading frame) — stop
                break
            # osrparse strips only the first two frames, and only when they sit
            # at the (256, -500) sentinel; mirror that set exactly.
            if i < 2 and float(fields[1]) == 256.0 and float(fields[2]) == -500.0:
                lead += delta
                continue
            break                      # first real frame: nothing more to strip
        return lead
    except Exception:  # noqa: BLE001 - never let a header quirk break parsing
        return 0


def _uleb128(buf: bytes, pos: int) -> tuple[int, int]:
    val = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            return val, pos
        shift += 7


def _skip_osr_string(buf: bytes, pos: int) -> int:
    marker = buf[pos]
    pos += 1
    if marker == 0x00:
        return pos
    if marker != 0x0b:
        raise ValueError(f"unknown string marker {marker:#x}")
    length, pos = _uleb128(buf, pos)
    return pos + length


def _detect_classic(path: Path, game_version: int) -> bool:
    """True when a lazer-format .osr carries the Classic mod (CL).

    Lazer appends an LZMA-compressed ScoreInfo JSON blob after the legacy
    replay body (even for Classic-mod plays, so stable-format scores round-trip
    through lazer losslessly). Its ``mods`` field is a list of
    ``{"acronym": "..."}`` dicts; CL marks a stable-mechanics / stable-origin
    play. The legacy ``mods`` int can't encode CL, so this blob is the only
    place it survives. Self-contained parse (mirrors
    mania_ordr.lazer_extension.parse_lazer_score_info) so osu-taiko stays a
    standalone package. Fail-soft: any problem -> False (treat as not-classic;
    the caller then leans on game_version, defaulting ambiguous to stable).

    Only meaningful for lazer-format replays; stable .osr files have no trailing
    blob, so we skip the walk entirely below the lazer version boundary.
    """
    if game_version < LAZER_GAME_VERSION:
        return False
    try:
        buf = path.read_bytes()
    except OSError:
        return False
    if len(buf) < 32:
        return False
    pos = 0
    try:
        pos += 1                       # mode (byte)
        pos += 4                       # game_version (int)
        pos = _skip_osr_string(buf, pos)   # beatmap hash
        pos = _skip_osr_string(buf, pos)   # username
        pos = _skip_osr_string(buf, pos)   # replay hash
        pos += 6 * 2                   # 6 judgement counts (shorts)
        pos += 4                       # total score (int)
        pos += 2                       # max_combo (short)
        pos += 1                       # perfect (byte)
        pos += 4                       # mods bitfield (int)
        pos = _skip_osr_string(buf, pos)   # life-bar graph
        pos += 8                       # timestamp (long)
        (replay_len,) = struct.unpack_from("<i", buf, pos)
        pos += 4
        if replay_len < 0 or pos + replay_len > len(buf):
            return False
        pos += replay_len              # opaque LZMA replay data
        pos += 8                       # score_id / online_score_id (long)
        # Trailing ScoreInfo blob: 4-byte LE length prefix (BinaryWriter.Write
        # (int)), NOT the ULEB128 the rest of the .osr uses. Stable files end
        # here (EOF) -> not a lazer replay -> not classic.
        if pos + 4 > len(buf):
            return False
        (length,) = struct.unpack_from("<i", buf, pos)
        pos += 4
        if length <= 0 or pos + length > len(buf):
            return False
        compressed = buf[pos:pos + length]
        try:
            raw = lzma.decompress(compressed, format=lzma.FORMAT_ALONE)
        except lzma.LZMAError:
            raw = lzma.decompress(compressed, format=lzma.FORMAT_AUTO)
        info = json.loads(raw.decode("utf-8"))
        mods = info.get("mods") or []
        return any(
            (m.get("acronym") if isinstance(m, dict) else str(m)) == "CL"
            for m in mods
        )
    except (struct.error, ValueError, lzma.LZMAError, json.JSONDecodeError,
            UnicodeDecodeError, IndexError, KeyError, TypeError):
        return False


def parse_replay(path: Path) -> tuple[list[TaikoFrame], ReplayMeta]:
    if not path.exists():
        raise ReplayParseError(f"replay not found: {path}")
    try:
        r = Replay.from_path(path)
    except Exception as e:  # noqa: BLE001 - osrparse raises bare exceptions
        raise ReplayParseError(f"osrparse failed: {e}") from e

    frames: list[TaikoFrame] = []
    # Seed the clock with the lead-in osrparse discarded (see
    # _recover_leadin_offset) so the key timeline matches osu! exactly; 0 for
    # lazer / already-aligned stable replays.
    t = _recover_leadin_offset(path)
    for ev in r.replay_data or []:
        delta = int(getattr(ev, "time_delta", 0))
        if delta == _SEED_DELTA:
            continue
        t += delta
        k = getattr(ev, "keys", 0)
        k = int(getattr(k, "value", k))
        cl, cr = bool(k & 1), bool(k & 4)   # centre-left / centre-right (don)
        rl, rr = bool(k & 2), bool(k & 8)   # rim-left / rim-right (kat)
        don = cl or cr
        kat = rl or rr
        big = (cl and cr) or (rl and rr)
        frames.append(TaikoFrame(time_ms=max(t, 0), don=don, kat=kat, big=big,
                                 cl=cl, cr=cr, rl=rl, rr=rr))
    frames.sort(key=lambda f: f.time_ms)
    replay_end_ms = frames[-1].time_ms if frames else 0

    # Life-bar graph: osu! records the player's actual HP over time. When present
    # it's the ground-truth HP bar (exactly true-to-game). osrparse exposes it as
    # a list of LifeBarState(time, life); it's often empty for lazer/API replays.
    lb = getattr(r, "life_bar_graph", None) or []
    life_bar = tuple((int(getattr(s, "time", 0)), float(getattr(s, "life", 0.0)))
                     for s in lb)

    total = r.count_300 + r.count_100 + r.count_miss
    if total > 0:
        acc = (r.count_300 + r.count_100 * 0.5) / total
    else:
        acc = 1.0
    meta = ReplayMeta(
        mode=int(r.mode.value if hasattr(r.mode, "value") else r.mode),
        beatmap_md5=str(getattr(r, "beatmap_hash", "") or ""),
        player_name=r.username,
        mods=int(r.mods),
        score=int(r.score),
        max_combo=int(r.max_combo),
        count_300=int(r.count_300),
        count_100=int(r.count_100),
        count_50=int(r.count_50),
        count_katu=int(r.count_katu),
        count_miss=int(r.count_miss),
        accuracy=round(acc * 100, 2),
        grade=_grade(acc, r),
        game_version=int(getattr(r, "game_version", 0) or 0),
        is_classic=_detect_classic(path, int(getattr(r, "game_version", 0) or 0)),
        life_bar=life_bar,
        replay_end_ms=replay_end_ms,
    )
    return frames, meta


def hit_events(frames: list[TaikoFrame]) -> list[tuple[int, str, bool]]:
    """Rising-edge hits: (time_ms, 'don'|'kat', is_big). A don and kat rising
    on the same frame both count (a big hit / simultaneous)."""
    out: list[tuple[int, str, bool]] = []
    pd = pk = False
    for f in frames:
        if f.don and not pd:
            out.append((f.time_ms, "don", f.big))
        if f.kat and not pk:
            out.append((f.time_ms, "kat", f.big))
        pd, pk = f.don, f.kat
    return out


def _grade(acc: float, r) -> str:
    """osu!lazer rank (ScoreProcessor.RankFromScore): accuracy-only thresholds
    SS=100%, S>=95%, A>=90%, B>=80%, C>=70%, else D. (Silver SS/S for HD/FL is
    applied at draw time by colour.)"""
    if acc >= 1.0:
        return "SS"
    if acc >= 0.95:
        return "S"
    if acc >= 0.90:
        return "A"
    if acc >= 0.80:
        return "B"
    if acc >= 0.70:
        return "C"
    return "D"
