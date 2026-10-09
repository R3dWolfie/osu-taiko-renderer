"""The automatic encoder pick tries a hardware encoder before trusting it
(render.py _probe_encoder + _encoder_starts; the check is the mania engine's,
osu-mania-renderer #31).

What must hold:
  * an explicit encoder passes through untouched, and nothing is run for it;
  * with no hardware encoder listed (every Mac) only the listing is read;
  * a listed hardware encoder that starts is chosen, as before;
  * a listed one that does NOT start is skipped: the next one is tried, and
    libx264 in the end;
  * a hardware encoder that hangs is given up on at the timeout;
  * R3D_ENCODER_PROBE=0 brings back "the listing decides".

A stand-in `ffmpeg` on PATH plays the machine.

Runnable two ways:  pytest tests/test_encoder_probe.py   OR   python tests/test_encoder_probe.py
"""
from __future__ import annotations

import os
import shutil
import stat
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from osu_taiko_renderer.render import render as _R   # noqa: E402

COPIES = (_R,)

FAKE = r'''#!/bin/sh
echo "$*" >> "$FAKE_LOG"
case " $* " in *" -encoders "*)
  for e in $FAKE_LIST; do echo " V....D $e   fake"; done; exit 0;;
esac
enc=""; prev=""
for a in "$@"; do [ "$prev" = "-c:v" ] && enc="$a"; prev="$a"; done
cat > /dev/null
for b in $FAKE_HANG; do [ "$b" = "$enc" ] && exec sleep 30; done
for b in $FAKE_BROKEN; do [ "$b" = "$enc" ] && { echo "[$enc] Cannot load the encoder here" >&2; exit 1; }; done
exit 0
'''


class _Machine:
    def __init__(self, listed, broken="", hang="", **env):
        self.env = dict(FAKE_LIST=listed, FAKE_BROKEN=broken, FAKE_HANG=hang, **env)

    def __enter__(self):
        self.d = tempfile.mkdtemp(prefix="r3d-enc-")
        f = os.path.join(self.d, "ffmpeg")
        with open(f, "w") as fh:
            fh.write(FAKE)
        os.chmod(f, os.stat(f).st_mode | stat.S_IEXEC)
        self.log = os.path.join(self.d, "calls.log")
        self.dev = os.path.join(self.d, "renderD129")      # a "device" that exists
        open(self.dev, "w").close()
        self.old = {k: os.environ.get(k) for k in
                    ("PATH", "FAKE_LIST", "FAKE_BROKEN", "FAKE_HANG", "FAKE_LOG",
                     "R3D_ENCODER_PROBE", "R3D_MAC_HW_ENCODE")}
        os.environ.pop("R3D_ENCODER_PROBE", None)
        os.environ.pop("R3D_MAC_HW_ENCODE", None)
        os.environ.update(self.env, FAKE_LOG=self.log,
                          PATH=self.d + os.pathsep + "/bin" + os.pathsep + "/usr/bin")
        return self

    def cfg(self, encoder="auto", device=True):
        return SimpleNamespace(encoder=encoder,
                               encoder_device=self.dev if device else None)

    def tried(self):
        out = []
        for line in open(self.log).read().splitlines() if os.path.exists(self.log) else []:
            a = line.split()
            if "-c:v" in a:
                out.append(a[a.index("-c:v") + 1])
        return out

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.d, ignore_errors=True)


def test_an_explicit_encoder_passes_through_and_runs_nothing():
    for R in COPIES:
        with _Machine("h264_nvenc h264_vaapi libx264", broken="h264_nvenc h264_vaapi") as m:
            assert R._probe_encoder(m.cfg("libx264")) == ("libx264", m.dev)
            assert R._probe_encoder(m.cfg("h264_nvenc")) == ("h264_nvenc", m.dev)
            assert R._probe_encoder(m.cfg("h264_vaapi")) == ("h264_vaapi", m.dev)
            assert R._probe_encoder(m.cfg("h264_vaapi", device=False)) == \
                ("h264_vaapi", "/dev/dri/renderD128")
            assert not os.path.exists(m.log)


def test_no_hardware_encoder_listed_means_only_the_listing_is_read():
    for R in COPIES:
        with _Machine("libx264 h264_videotoolbox") as m:
            assert R._probe_encoder(m.cfg()) == ("libx264", None)
            assert m.tried() == []


def test_a_listed_encoder_that_starts_is_chosen():
    for R in COPIES:
        with _Machine("h264_nvenc h264_vaapi libx264") as m:
            assert R._probe_encoder(m.cfg()) == ("h264_nvenc", None)
            assert m.tried() == ["h264_nvenc"]
        with _Machine("h264_vaapi libx264") as m:
            assert R._probe_encoder(m.cfg()) == ("h264_vaapi", m.dev)
            assert m.tried() == ["h264_vaapi"]
            assert f"-vaapi_device {m.dev}" in open(m.log).read()


def test_a_listed_encoder_that_does_not_start_is_skipped():
    for R in COPIES:
        with _Machine("h264_nvenc h264_vaapi libx264", broken="h264_nvenc") as m:
            assert R._probe_encoder(m.cfg()) == ("h264_vaapi", m.dev)
            assert m.tried() == ["h264_nvenc", "h264_vaapi"]
        with _Machine("h264_nvenc h264_vaapi libx264",
                      broken="h264_nvenc h264_vaapi") as m:
            assert R._probe_encoder(m.cfg()) == ("libx264", None)
            assert m.tried() == ["h264_nvenc", "h264_vaapi"]   # libx264 is not checked


def test_an_encoder_that_hangs_is_given_up_on():
    for R in COPIES:
        old = R._ENCODER_PROBE_TIMEOUT_S
        R._ENCODER_PROBE_TIMEOUT_S = 0.5
        try:
            with _Machine("h264_nvenc libx264", hang="h264_nvenc") as m:
                t = time.monotonic()
                assert R._probe_encoder(m.cfg()) == ("libx264", None)
                assert time.monotonic() - t < 5.0
        finally:
            R._ENCODER_PROBE_TIMEOUT_S = old


def test_the_switch_brings_back_the_listing_alone():
    for R in COPIES:
        with _Machine("h264_nvenc libx264", broken="h264_nvenc",
                      R3D_ENCODER_PROBE="0") as m:
            assert R._probe_encoder(m.cfg()) == ("h264_nvenc", None)
            assert m.tried() == []


if __name__ == "__main__":
    for _n, _f in sorted(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f()
            print("ok  ", _n)
