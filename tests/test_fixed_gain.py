"""Song loudness by one measured gain (render/render.py, the switch
R3D_TAIKO_FIXED_GAIN / R3D_FIXED_GAIN).

What must hold:
  * with nothing set the cache builds what it always built, under the key it
    always used, and hands back the chain it always did;
  * the engine's own switch wins over the node-wide one, and the stock switch
    wins over both;
  * with the switch on, the entry has its own key, is the same chain's output
    times ONE constant, lands on the target loudness and stays under the
    ceiling; with a start trim too;
  * a second call is a hit (nothing is rebuilt);
  * an ffmpeg whose summary cannot be read gives the stock entry.

Runnable two ways:  pytest tests/test_fixed_gain.py   OR   python tests/test_fixed_gain.py
"""
from __future__ import annotations

import importlib
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from osu_taiko_renderer.render import envflag as sw   # noqa: E402
from osu_taiko_renderer.render import render as R     # noqa: E402

_ENV = ("R3D_TAIKO_FIXED_GAIN", "R3D_FIXED_GAIN", "R3D_TAIKO_STOCK")


def _switch(**kw):
    """envflag.FIXED_GAIN as a fresh process would resolve it under `kw`."""
    old = {k: os.environ.get(k) for k in _ENV}
    try:
        for k in _ENV:
            os.environ.pop(k, None)
        for k, v in kw.items():
            os.environ[k] = str(v)
        return importlib.reload(sw).FIXED_GAIN
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(sw)


def test_the_switch():
    assert _switch() is False
    assert _switch(R3D_FIXED_GAIN=1) is True
    assert _switch(R3D_TAIKO_FIXED_GAIN=1) is True
    assert _switch(R3D_FIXED_GAIN=1, R3D_TAIKO_FIXED_GAIN=0) is False
    assert _switch(R3D_FIXED_GAIN=0, R3D_TAIKO_FIXED_GAIN=1) is True
    assert _switch(R3D_FIXED_GAIN=1, R3D_TAIKO_STOCK=1) is False
    assert _switch(R3D_TAIKO_FIXED_GAIN=1, R3D_TAIKO_STOCK=1) is False
    # it is a sound choice, not a speedup: it must not put a render on the
    # "run again on the stock path" list
    old = {k: os.environ.get(k) for k in _ENV}
    try:
        for k in _ENV:
            os.environ.pop(k, None)
        os.environ["R3D_TAIKO_STOCK"] = "0"
        base = importlib.reload(sw).ANY_FAST
        os.environ["R3D_TAIKO_FIXED_GAIN"] = "1"
        assert importlib.reload(sw).ANY_FAST == base
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(sw)


def test_gain_arithmetic_and_the_summary():
    f = R._fg_gain_db
    assert abs(f(-7.6, 1.0) - (-10.4)) < 1e-9
    assert abs(f(-30.0, 0.1) - 12.0) < 1e-9
    want = R._FG_PEAK_CEILING_DB - 20 * math.log10(0.9)
    assert abs(f(-30.0, 0.9) - want) < 1e-9 and f(-30.0, 0.9) < 12.0
    assert f(None, 0.5) == 0.0 and f(-70.0, 0.5) == 0.0 and f(-18.0, 0.0) == 0.0
    p = R._fg_parse_lufs
    s = ("  Integrated loudness:\n    I:          -7.6 LUFS\n"
         "    Threshold: -17.8 LUFS\n\n  Loudness range:\n"
         "    LRA:         2.7 LU\n    LRA low:    -9.2 LUFS\n")
    assert p(s) == -7.6 and p("") is None
    assert p(s.replace("    I:          -7.6 LUFS\n", "")) is None
    assert R._FG_PARAM == "fixedgain:I=-18:P=-1.5"        # the other engines' name


def test_the_measurement_stands_where_loudnorm_stood():
    for kw in ({}, {"rate": 1.5}, {"rate": 1.5, "is_nc": True}, {"start_ms": 2500},
               {"start_ms": -800}):
        a = dict(start_ms=0, rate=1.0, total_dur_s=30.0)
        a.update(kw)
        pre, _post = R._audio_parts(**a)
        m = R._fg_pre(pre)
        # loudnorm's own rate is pinned in its place, then the measurement
        assert m == pre.replace(R._LOUDNORM_FILTER,
                                "aformat=sample_rates=192000," + R._FG_EBUR128)
        assert pre.count(R._LOUDNORM_FILTER) == 1 and "loudnorm" not in m
        assert m.split(",")[:-2] == pre.split(",")[:-1]
    assert R._fg_pre("volume=0.5") is None


def _wav(path, seconds, level, rate=48000):
    """A tone with a slow swell (something a gain rider would ride)."""
    n = int(seconds * rate)
    t = np.arange(n) / rate
    x = level * (0.35 + 0.65 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.4 * t))) \
        * np.sin(2 * np.pi * 440.0 * t)
    data = np.repeat(x[:, None], 2, axis=1).astype("<f4").tobytes()
    with open(path, "wb") as fh:
        fh.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt ")
        fh.write(struct.pack("<IHHIIHH", 16, 3, 2, rate, rate * 8, 8, 32))
        fh.write(b"data" + struct.pack("<I", len(data)) + data)
    return Path(path)


def _samples(path):
    off, n = R._wav_data_span(path)
    return np.fromfile(path, dtype="<f8", offset=off, count=n // 8)


def _lufs(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "info",
                        "-i", str(path), "-af", R._FG_EBUR128, "-f", "null", "-"],
                       capture_output=True)
    return R._fg_parse_lufs(r.stderr.decode(errors="replace"))


class _Cache:
    def __init__(self, d):
        self.d = Path(d)

    def __enter__(self):
        self.old = R._LOUDNORM_CACHE_DIR
        R._LOUDNORM_CACHE_DIR = str(self.d)
        os.environ.pop("R3D_TAIKO_LOUDNORM_CACHE", None)

    def __exit__(self, *a):
        R._LOUDNORM_CACHE_DIR = self.old


def test_the_samples_are_found_in_a_wav():
    d = Path(tempfile.mkdtemp(prefix="r3d-gain-"))
    try:
        off, n = R._wav_data_span(_wav(d / "a.wav", 1.0, 0.5))
        assert (off, n) == (44, 48000 * 8)
        (d / "bad.wav").write_bytes(b"x" * 64)
        try:
            R._wav_data_span(d / "bad.wav")
        except ValueError:
            pass
        else:
            raise AssertionError("a non-wav was accepted")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_nothing_set_builds_what_it_always_built():
    if not shutil.which("ffmpeg"):
        return
    d = Path(tempfile.mkdtemp(prefix="r3d-gain-"))
    try:
        src = _wav(d / "song.wav", 6.0, 0.6)
        pre, post = R._audio_parts(0, 1.0, 10.0)
        with _Cache(d / "cache"):
            got, chain = R._resolve_music_audio(src, pre, post, fixed_gain=False)
            assert (got, chain) == R._resolve_music_audio(src, pre, post)   # the default
            want = d / "want.wav"
            assert R._build_loudnorm_cache(src, pre, want)
            assert got == d / "cache" / (R._loudnorm_cache_key(src, pre) + ".wav")
            assert got.read_bytes() == want.read_bytes() and chain == post
            assert [p.name for p in (d / "cache").iterdir()] == [got.name]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_switch_on_one_gain_on_the_target():
    if not shutil.which("ffmpeg"):
        return
    d = Path(tempfile.mkdtemp(prefix="r3d-gain-"))
    try:
        with _Cache(d / "cache"):
            for level, name, start in ((0.9, "loud", 0), (0.02, "quiet", 0),
                                       (0.5, "trimmed", 1500)):
                src = _wav(d / f"{name}.wav", 8.0, level)
                pre, post = R._audio_parts(start, 1.0, 10.0)
                got, chain = R._resolve_music_audio(src, pre, post, fixed_gain=True)
                assert chain == post and got.parent == d / "cache"
                assert got.name != R._loudnorm_cache_key(src, pre) + ".wav"
                # the same chain without any loudness step, to compare against
                plain = d / f"{name}.plain.wav"
                bare = ",".join(pre.split(",")[:-1] + [R._FG_PIN_192K])
                subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                                "-i", str(src), "-af", bare, "-c:a", "pcm_f64le",
                                "-f", "wav", str(plain)], check=True)
                a, b = _samples(got), _samples(plain)
                assert a.shape == b.shape and a.size > 0, (name, a.shape, b.shape)
                k = float(np.abs(a).max() / np.abs(b).max())
                assert np.allclose(a, b * k, rtol=0, atol=1e-9), name
                assert (k < 1.0) == (name != "quiet"), (name, k)
                lufs = _lufs(got)
                assert abs(lufs - R._FG_TARGET_LUFS) <= 0.15, (name, lufs)
                assert float(np.abs(a).max()) <= 10 ** (R._FG_PEAK_CEILING_DB / 20) + 1e-9
                # a second call is a hit: the file is not written again
                before = got.stat().st_mtime_ns
                assert R._resolve_music_audio(src, pre, post, fixed_gain=True)[0] == got
                assert got.stat().st_mtime_ns == before
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _noise(path, seconds, level=0.25, rate=48000, seed=7):
    """Steady band-limited noise: nothing periodic, so two time-stretches of it
    only line up if they are the same stretch."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(seconds * rate))
    x = np.convolve(x, np.ones(8) / 8.0, mode="same") * level * 2.0
    data = np.repeat(x[:, None], 2, axis=1).astype("<f4").tobytes()
    with open(path, "wb") as fh:
        fh.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt ")
        fh.write(struct.pack("<IHHIIHH", 16, 3, 2, rate, rate * 8, 8, 32))
        fh.write(b"data" + struct.pack("<I", len(data)) + data)
    return Path(path)


def _as_aac(wav):
    """The same sound as an .m4a. Songs are mp3 or ogg, whose decoders hand
    ffmpeg PLANAR samples; that is the case in which the stock chain stretches
    at 192 kHz (a float wav does not trigger it). AAC decodes planar too, and
    every ffmpeg can encode it."""
    out = Path(str(wav)[:-4] + ".m4a")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i",
                    str(wav), "-c:a", "aac", "-b:a", "192k", str(out)], check=True)
    return out


def _corr0(a, b):
    """Correlation at lag 0 of two equally long signals."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))


def test_a_speed_changed_song_is_stretched_exactly_as_stock():
    if not shutil.which("ffmpeg"):
        return
    d = Path(tempfile.mkdtemp(prefix="r3d-gain-"))
    try:
        src = _as_aac(_noise(d / "noise.wav", 6.0))
        with _Cache(d / "cache"):
            for rate, start in ((1.5, 0), (0.75, 0), (1.5, 1200), (1.0, 0)):
                pre, post = R._audio_parts(start, rate, 10.0)
                stock, _c = R._resolve_music_audio(src, pre, post, fixed_gain=False)
                fixed, _c = R._resolve_music_audio(src, pre, post, fixed_gain=True)
                a, b = _samples(fixed), _samples(stock)
                # the same stretch and the same kind of file: the same length
                # to the sample, lined up, at stock's sample rate
                assert a.shape == b.shape, (rate, start, a.shape, b.shape)
                assert _corr0(a, b) >= 0.97, (rate, start, _corr0(a, b))
                assert _fmt(fixed) == _fmt(stock)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _fmt(path):
    """The wav's `fmt ` chunk: sample format, channels, sample rate."""
    with open(path, "rb") as f:
        head = f.read(60)
    assert head[12:16] == b"fmt "
    return head[12:60]


def test_an_unreadable_summary_gives_the_stock_entry():
    if not shutil.which("ffmpeg"):
        return
    d = Path(tempfile.mkdtemp(prefix="r3d-gain-"))
    real = R._fg_parse_lufs
    try:
        src = _wav(d / "song.wav", 5.0, 0.5)
        pre, post = R._audio_parts(0, 1.0, 10.0)
        R._fg_parse_lufs = lambda text: None
        with _Cache(d / "cache"):
            got, chain = R._resolve_music_audio(src, pre, post, fixed_gain=True)
        assert got.name == R._loudnorm_cache_key(src, pre) + ".wav" and chain == post
        assert [p.name for p in (d / "cache").iterdir()] == [got.name]
    finally:
        R._fg_parse_lufs = real
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    for _n, _f in sorted(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f()
            print("ok  ", _n)
