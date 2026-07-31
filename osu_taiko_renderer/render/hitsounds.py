"""Per-note hitsound track for the R3D taiko renderer.

Mixes each NON-MISS note's resolved osu! hitsounds (normal + whistle/finish/clap
additions, honouring the beatmap's per-note sample overrides + timing-point
sample set/index/volume + the [General] default set) into one stereo WAV aligned
to the FINAL video timeline, plus a combobreak on big combo breaks. The WAV is
amixed with the song in render.py.

Samples resolve first-match across [beatmap dir, user skin, default skin,
bundled fallback] so maps that ship no custom samples still play the skin's
defaults. Decode/encode go through ffmpeg + a struct-built RIFF (this venv has
no soundfile) — same shape as the catch renderer's hitsound path.
"""
from __future__ import annotations

import bisect
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np

from osu_taiko_renderer.beatmap.models import TaikoType

MISS = "miss"
SAMPLE_RATE = 44100
CHANNELS = 2
DEFAULT_HIT_GAIN = 0.55       # ceiling on top of per-note volume
COMBO_BREAK_THRESHOLD = 20    # stable: combobreak only on a combo >= 20 break
COMBO_BREAK_GAIN = 0.65
# Last-resort sample source (has every set x type + combobreak).
_FALLBACK_SKIN = Path("/home/foof/r3drender/osu-catch/night05-skin")
# Bundled osu! DEFAULT nightcore drums (ppy/osu-resources Legacy skin) — the
# final fallback for the NC-mod overlay when the skin OMITS a nightcore sample
# (osu! falls back to its default skin). A skin's SILENT nightcore file resolves
# first and wins, so this never overrides a deliberately-silenced sample.
_DEFAULT_NC_DIR = Path(__file__).resolve().parent.parent / "assets" / "default_nightcore"

_SET_NAMES = {1: "normal", 2: "soft", 3: "drum"}     # 0 => beatmap default
_ADDITIONS = ((2, "whistle"), (4, "finish"), (8, "clap"))


def _decode_pcm(path: Path) -> np.ndarray | None:
    """Decode a sample file -> float32 stereo 44.1k (N,2) via ffmpeg. Zero-byte
    or header-only files are deliberate SILENCE; undecodable -> None."""
    try:
        if path.stat().st_size == 0:
            return np.zeros((1, CHANNELS), dtype=np.float32)
    except OSError:
        return None
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path),
           "-f", "f32le", "-acodec", "pcm_f32le",
           "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS), "pipe:1"]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False, timeout=30)
    except Exception:  # noqa: BLE001 — a bad sample never kills a render
        return None
    if proc.returncode != 0:
        return None
    pcm = np.frombuffer(proc.stdout, dtype=np.float32)
    n = (len(pcm) // CHANNELS) * CHANNELS
    if n == 0:
        return np.zeros((1, CHANNELS), dtype=np.float32)
    return pcm[:n].reshape(-1, CHANNELS).copy()


def _write_wav_f32(path: Path, buf: np.ndarray) -> Path:
    """float32 WAV writer (no soundfile in this venv; ffmpeg reads it natively)."""
    data = buf.astype("<f4").tobytes()
    with open(path, "wb") as fh:
        byte_rate = SAMPLE_RATE * CHANNELS * 4
        fh.write(b"RIFF")
        fh.write(struct.pack("<I", 36 + len(data)))
        fh.write(b"WAVEfmt ")
        fh.write(struct.pack("<IHHIIHH", 16, 3, CHANNELS, SAMPLE_RATE,
                             byte_rate, CHANNELS * 4, 32))
        fh.write(b"data")
        fh.write(struct.pack("<I", len(data)))
        fh.write(data)
    return path


class _SampleCache:
    """Decode each unique file once (via ffmpeg) -> float32 stereo 44.1k;
    remember misses so we don't re-probe."""

    def __init__(self) -> None:
        self._cache: dict[str, np.ndarray | None] = {}
        self._dir_index: dict[str, dict] = {}

    def _index(self, d: Path) -> dict:
        di = self._dir_index.get(str(d))
        if di is None:
            di = {}
            try:
                for p in Path(d).iterdir():
                    if p.is_file():
                        di.setdefault(p.name.lower(), p)
            except OSError:
                pass
            self._dir_index[str(d)] = di
        return di

    def find_in(self, d: Path, name: str) -> np.ndarray | None:
        # osu! skins are case-INSENSITIVE; the host FS is not. Match by lowered
        # filename so e.g. normal-hitNormal.wav resolves.
        p = self._index(d).get(name.lower())
        return self.get(p) if p is not None else None

    def get(self, path: Path) -> np.ndarray | None:
        key = str(path)
        if key in self._cache:
            return self._cache[key]
        arr = _decode_pcm(path) if path.is_file() else None
        self._cache[key] = arr
        return arr


def _candidate_names(set_name: str, type_name: str, index: int) -> list[str]:
    # osu!taiko checks the taiko-prefixed sample first, then the generic one.
    out = []
    for prefix in ("taiko-", ""):
        base = f"{prefix}{set_name}-hit{type_name}"
        if index > 1:
            out += [f"{base}{index}.wav", f"{base}{index}.ogg"]
        out += [f"{base}.wav", f"{base}.ogg"]
    return out


def _find(dirs, cache, names) -> np.ndarray | None:
    for d in dirs:
        for n in names:
            arr = cache.find_in(d, n)
            if arr is not None:
                return arr
    return None


def _active_sample_point(points, t: int):
    if not points:
        return None
    times = [p.time_ms for p in points]
    i = bisect.bisect_right(times, t) - 1
    return points[max(0, i)]


def _resolve_samples(o, beatmap, cache, dirs) -> list[tuple[np.ndarray, float]]:
    """(sample_array, gain) list to mix at note `o`'s time."""
    hs = getattr(o, "hit_sample", None)
    tp = _active_sample_point(getattr(beatmap, "sample_points", ()), o.time_ms)
    tp_set = tp.sample_set if tp else 0
    tp_index = tp.custom_index if tp else 0
    tp_volume = tp.volume if tp else 100

    eff_set = (hs.normal_set if hs and hs.normal_set else tp_set)
    eff_index = (hs.index if hs and hs.index else tp_index)
    set_name = _SET_NAMES.get(eff_set) or (beatmap.default_sample_set or "normal")

    vol_pct = (hs.volume if hs and hs.volume else 0) or tp_volume or 100
    gain = (vol_pct / 100.0) * DEFAULT_HIT_GAIN

    out: list[tuple[np.ndarray, float]] = []
    fn = (hs.filename if hs else "") or ""
    if fn:
        arr = _find(dirs, cache, [fn])
        if arr is not None:
            return [(arr, gain)]
    arr = _find(dirs, cache, _candidate_names(set_name, "normal", eff_index))
    if arr is not None:
        out.append((arr, gain))
    add_set = (hs.addition_set if hs and hs.addition_set else eff_set)
    add_name = _SET_NAMES.get(add_set) or set_name
    hbits = getattr(o, "hit_sound", 0) or 0
    for bit, tname in _ADDITIONS:
        if hbits & bit:
            arr = _find(dirs, cache, _candidate_names(add_name, tname, eff_index))
            if arr is not None:
                out.append((arr, gain))
    return out


def _nearest_dist(sorted_times, t) -> float:
    """|delta| to the nearest value in a time-sorted list (huge if empty)."""
    if not sorted_times:
        return 1e18
    i = bisect.bisect_left(sorted_times, t)
    best = 1e18
    for k in (i - 1, i):
        if 0 <= k < len(sorted_times):
            best = min(best, abs(sorted_times[k] - t))
    return best


def _is_color(o, color: str) -> bool:
    return (o.kind is TaikoType.DON) if color == "don" else (o.kind is TaikoType.KAT)


def _tap_samples(type_name, beatmap, cache, dirs, t) -> list[tuple[np.ndarray, float]]:
    """Empty drum-tap sample at press time `t`: a centre press plays the normal
    hitnormal ('normal'), a rim press the hitclap ('clap') — osu!lazer
    DrumSampleTriggerSource (Centre→HIT_NORMAL, Rim→HIT_CLAP). Bank / custom
    index / volume come from the active sample (timing) point, like a note's own
    normal sample."""
    tp = _active_sample_point(getattr(beatmap, "sample_points", ()), t)
    tp_set = tp.sample_set if tp else 0
    tp_index = tp.custom_index if tp else 0
    tp_volume = tp.volume if tp else 100
    set_name = _SET_NAMES.get(tp_set) or (beatmap.default_sample_set or "normal")
    gain = ((tp_volume or 100) / 100.0) * DEFAULT_HIT_GAIN
    arr = _find(dirs, cache, _candidate_names(set_name, type_name, tp_index))
    return [(arr, gain)] if arr is not None else []


def _mix(track: np.ndarray, arr: np.ndarray, at_ms: float, gain: float):
    start = int(at_ms / 1000.0 * SAMPLE_RATE)
    if start < 0:
        arr = arr[-start:] if -start < arr.shape[0] else arr[:0]
        start = 0
    end = min(start + arr.shape[0], track.shape[0])
    if end > start:
        track[start:end] += arr[:end - start] * gain


def build_taiko_hitsound_track(
    *, notes, note_hit, beatmap, sample_dirs, output_wav: Path,
    video_ms: float, start_ms: int, rate: float,
    miss_hitsound: bool = True,
    nightcore: bool = False,
    nc_mod: bool = False,
    press_edges: dict | None = None, ok_window_ms: float = 0.0,
) -> Path | None:
    """Build the stereo hitsound WAV at `output_wav`, aligned to the final video
    (a note at map time T lands at video time (T - start_ms)/rate). Returns the
    path, or None if nothing was mixed.

    `press_edges` = {'don': [ms...], 'kat': [ms...]} rising-edge replay key
    presses (from TaikoSim.drum_press_edges): every press that does NOT land on a
    judged note gets a plain drum tap sample so empty taps click too (#100).
    `ok_window_ms` is the hit window used to tell an on-note press (already
    covered by that note's hitsound — no double-play) from an empty tap."""
    dirs = [Path(d) for d in sample_dirs if d and Path(d).is_dir()]
    if _FALLBACK_SKIN.is_dir() and _FALLBACK_SKIN not in dirs:
        dirs.append(_FALLBACK_SKIN)

    cache = _SampleCache()
    total = max(1, int(video_ms / 1000.0 * SAMPLE_RATE))
    track = np.zeros((total, CHANNELS), dtype=np.float32)

    cb = _find(dirs, cache, ["combobreak.wav", "combobreak.ogg"]) if miss_hitsound else None
    placed = combo = 0
    for o in sorted(notes, key=lambda o: o.time_ms):
        rt, res = note_hit.get(id(o), (0, MISS))
        vt = (o.time_ms - start_ms) / rate
        if res != MISS:
            for arr, gain in _resolve_samples(o, beatmap, cache, dirs):
                _mix(track, arr, vt, gain)
                placed += 1
            combo += 1
        else:
            if cb is not None and combo >= COMBO_BREAK_THRESHOLD:
                _mix(track, cb, vt, COMBO_BREAK_GAIN)
            combo = 0

    # Empty drum taps (#100): osu! plays a drum sample on EVERY key press, even
    # one that lands on no note. For each replay press that did NOT judge a note
    # (a press within ok_window_ms of a judged-hit note of its colour is already
    # covered by that note's hitsound above — no double-play), add the plain drum
    # tap: centre press -> hitnormal, rim press -> hitclap.
    if press_edges:
        for color, type_name in (("don", "normal"), ("kat", "clap")):
            edges = press_edges.get(color) or ()
            hit_times = sorted(
                o.time_ms for o in notes
                if _is_color(o, color) and note_hit.get(id(o), (0, MISS))[1] != MISS)
            for p in edges:
                if _nearest_dist(hit_times, p) <= ok_window_ms:
                    continue                       # on-note press -> already voiced
                for arr, gain in _tap_samples(type_name, beatmap, cache, dirs, p):
                    _mix(track, arr, (p - start_ms) / rate, gain)
                    placed += 1

    # The general metronome is SUPPRESSED while NC is active (osu! only plays
    # the NC drum overlay then) — never both on one render.
    nc_beats = 0
    if nightcore and not nc_mod:
        nc_beats = _layer_metronome(track, beatmap, cache, dirs, start_ms, rate)

    # ModNightcore beat overlay — AUTOMATIC when the Nightcore mod is active,
    # independent of the `nightcore` metronome toggle above (both may lay).
    # Taiko has no readily-available SliderTickRate on the beatmap, so hats
    # play unconditionally (osu gates them on SliderTickRate%2==0).
    nc_mod_beats = 0
    if nc_mod:
        nc_mod_beats = _layer_nightcore_mod(track, beatmap, cache, dirs,
                                            start_ms, rate, play_hats=True)

    if placed == 0 and nc_beats == 0 and nc_mod_beats == 0:
        return None
    np.clip(track, -1.0, 1.0, out=track)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    return _write_wav_f32(output_wav, track)


_METRONOME_GAIN = 0.35        # beat-overlay click sits under the per-note hits


def _layer_metronome(track, beatmap, cache, dirs, start_ms, rate) -> int:
    """Beat-overlay metronome (site 'Beat overlay (metronome)' toggle): a clap
    on every beat + a finish on every downbeat (beat 1 of the measure), across
    the whole song, mixed into the hitsound track. Beats come from the map's
    uninherited (red) timing points; the point's meter sets the measure length
    (taiko already carries the (time, beat, meter) uninherited list it uses for
    bar lines). Placed in VIDEO time ((t_map - start_ms)/rate) so DT/NC/HT stay
    beat-aligned. Mod-INDEPENDENT — a general metronome, not gated on the NC
    mod (mirrors the std/mania v2 overlay). Returns beats laid."""
    timing = getattr(beatmap, "timing", None)
    pts = list(getattr(timing, "uninherited", []) or []) if timing else []
    if not pts:
        return 0
    # map-time horizon = the last sample the track can hold
    horizon = start_ms + (track.shape[0] / SAMPLE_RATE * 1000.0) * (rate or 1.0)
    default_set = getattr(beatmap, "default_sample_set", "normal") or "normal"

    def _click(sound, t):
        # resolve through the active sample point's set/index, then fall back
        # to the default set and finally hitnormal so a beat never goes silent.
        tp = _active_sample_point(getattr(beatmap, "sample_points", ()), int(t))
        s_set = _SET_NAMES.get(tp.sample_set if tp else 0) or default_set
        idx = (tp.custom_index if tp else 0) or 0
        for names in (_candidate_names(s_set, sound, idx),
                      _candidate_names(default_set, sound, idx),
                      _candidate_names(s_set, "normal", idx)):
            arr = _find(dirs, cache, names)
            if arr is not None:
                return arr
        return None

    laid = 0
    for i, (ptime, beat, meter) in enumerate(pts):
        beat = max(60.0, float(beat))          # cap <60ms (>1000 BPM) sanity
        meter = int(meter) if meter and int(meter) > 0 else 4
        seg_end = pts[i + 1][0] if i + 1 < len(pts) else horizon
        seg_end = min(seg_end, horizon)
        k = 0
        t = float(ptime)
        while t < seg_end:
            downbeat = (k % meter == 0)
            arr = _click("finish" if downbeat else "clap", t)
            if arr is not None:
                _mix(track, arr, (t - start_ms) / (rate or 1.0), _METRONOME_GAIN)
                laid += 1
            k += 1
            t = ptime + k * beat
    return laid


# --- ModNightcore beat overlay (NC-mod-gated, distinct from the metronome) -----

_NC_MOD_GAIN = 0.5        # nightcore-kick/clap/hat/finish drums


def _nc_pattern(k: int, seg_len: int, mod: int, clap_pos: int, play_hats: bool):
    """One half-beat of osu! ModNightcore.NightcoreBeatContainer.OnNewBeat.
    `k` = half-beat index from the red timing point (firstBeat==0 per segment,
    BeatSyncedContainer Divisor=2). Yields the sound name(s) to play this step:
    kick on segment beat 0 mod `mod`, clap on `clap_pos`, else hat (if enabled);
    finish additionally at each segment start."""
    bseg = k % seg_len
    r = bseg % mod
    if r == 0:
        yield "kick"
    elif r == clap_pos:
        yield "clap"
    elif play_hats:
        yield "hat"
    if bseg == 0:
        yield "finish"


def _nc_find(dirs, cache, base: str) -> np.ndarray | None:
    names = [f"{base}.wav", f"{base}.ogg", f"{base}.mp3"]
    arr = _find(dirs, cache, names)
    # Bundled osu! default — FINAL fallback, only when the sample is ABSENT from
    # the skin chain (a skin's SILENT file already resolved above and wins).
    if arr is None and _DEFAULT_NC_DIR.is_dir():
        arr = _find([_DEFAULT_NC_DIR], cache, names)
    return arr


def _layer_nightcore_mod(track, beatmap, cache, dirs, start_ms, rate,
                         *, play_hats: bool = True) -> int:
    """osu! ModNightcore beat overlay — the drum pattern osu! plays on each
    beat AUTOMATICALLY while the Nightcore mod is active. NOT the general
    'Beat overlay (metronome)' (_layer_metronome) above — both can lay onto
    the same track. Half-beat grid (Divisor=2): within a 4-bar segment, kick
    on the downbeat of each `mod`-cycle, clap on the backbeat, hat on the
    off-beats, plus a finish cymbal at each segment start — resolved from the
    SKIN's nightcore-kick/-clap/-hat/-finish samples. A skin that ships SILENT
    nightcore samples correctly plays (near-)nothing; a skin that omits them
    plays nothing here (no synth fallback). Beats come from the map's
    uninherited (red) timing points; placed in VIDEO time ((t-start)/rate) so
    they ride the sped-up track. Returns samples laid.

    Port of osu.Game/Rulesets/Mods/ModNightcore.NightcoreBeatContainer."""
    timing = getattr(beatmap, "timing", None)
    pts = list(getattr(timing, "uninherited", []) or []) if timing else []
    if not pts:
        return 0
    horizon = start_ms + (track.shape[0] / SAMPLE_RATE * 1000.0) * (rate or 1.0)
    samples = {name: _nc_find(dirs, cache, f"nightcore-{name}")
               for name in ("kick", "clap", "hat", "finish")}
    if not any(v is not None for v in samples.values()):
        return 0
    laid = 0
    for i, (ptime, beat, meter) in enumerate(pts):
        beat = max(60.0, float(beat))          # cap <60ms (>1000 BPM) sanity
        half = beat / 2.0
        sig = int(meter) if meter and int(meter) > 0 else 4
        seg_len = sig * 8                      # beatsPerBar * Divisor(2) * 4 bars
        triplet = (sig % 3 == 0)
        mod = 6 if triplet else 4
        clap_pos = 3 if triplet else 2
        seg_end = pts[i + 1][0] if i + 1 < len(pts) else horizon
        seg_end = min(seg_end, horizon)
        k = 0
        t = float(ptime)
        while t < seg_end:
            for name in _nc_pattern(k, seg_len, mod, clap_pos, play_hats):
                arr = samples.get(name)
                if arr is not None:
                    _mix(track, arr, (t - start_ms) / (rate or 1.0), _NC_MOD_GAIN)
                    laid += 1
            k += 1
            t = ptime + k * half
    return laid
