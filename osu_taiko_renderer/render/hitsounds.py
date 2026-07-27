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

MISS = "miss"
SAMPLE_RATE = 44100
CHANNELS = 2
DEFAULT_HIT_GAIN = 0.55       # ceiling on top of per-note volume
COMBO_BREAK_THRESHOLD = 20    # stable: combobreak only on a combo >= 20 break
COMBO_BREAK_GAIN = 0.65
# Last-resort sample source (has every set x type + combobreak).
_FALLBACK_SKIN = Path("/home/foof/r3drender/osu-catch/night05-skin")

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
) -> Path | None:
    """Build the stereo hitsound WAV at `output_wav`, aligned to the final video
    (a note at map time T lands at video time (T - start_ms)/rate). Returns the
    path, or None if nothing was mixed."""
    dirs = [Path(d) for d in sample_dirs if d and Path(d).is_dir()]
    if _FALLBACK_SKIN.is_dir() and _FALLBACK_SKIN not in dirs:
        dirs.append(_FALLBACK_SKIN)

    cache = _SampleCache()
    total = max(1, int(video_ms / 1000.0 * SAMPLE_RATE))
    track = np.zeros((total, CHANNELS), dtype=np.float32)

    cb = _find(dirs, cache, ["combobreak.wav", "combobreak.ogg"])
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
    if placed == 0:
        return None
    np.clip(track, -1.0, 1.0, out=track)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    return _write_wav_f32(output_wav, track)
