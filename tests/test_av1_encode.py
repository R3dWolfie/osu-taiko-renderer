"""AV1 master encoding (Leon's AV1 migration, 2026-10-23).

What must hold:
  * each AV1 backend gets its OWN controls — NOT a mechanical h264_* rename:
    av1_vaapi uses `-rc_mode CQP -global_quality` (never `-qp`, which VAAPI
    ignores), av1_nvenc the VBR ladder, av1_qsv ICQ, av1_amf VBR; all emit
    8-bit 4:2:0 Main with a ~2 s GOP and +faststart;
  * the auto `av1` probe selects a backend ONLY when a real 128x128 encode
    produces ffprobe codec_name=av1, and FAILS CLOSED (raises, no H.264 / CPU /
    software-AV1 fallback) when none does;
  * an AV1 master never changes the H.264 Discord embed/compact/preview copies;
  * the H.264 `auto` path is unchanged;
  * (opt-in, R3D_AV1_HW_TEST=1) a real av1_vaapi encode on this box IS av1.

Mirrors std's tests/test_encode.py (probe mocking) and taiko's
tests/test_preview_hw.py (argv capture). The real-hardware case is opt-in so a
plain `pytest` run stays CPU-only and light.

Runnable two ways:  pytest tests/test_av1_encode.py   OR   python tests/test_av1_encode.py
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from osu_taiko_renderer.beatmap.models import RenderConfig  # noqa: E402
from osu_taiko_renderer.render import render as R  # noqa: E402


class _Captured(Exception):
    pass


def _cmd(encoder, *, audio=None, total=60.0, preview=None, compact=None,
         res=(1280, 720), out="m.mp4"):
    """The ffmpeg command taiko would start for `encoder`, without starting it."""
    saved_popen = R.subprocess.Popen
    saved_cache = os.environ.get("R3D_TAIKO_LOUDNORM_CACHE")
    got = []

    def fake_popen(cmd, **kw):
        got.append(list(cmd))
        raise _Captured()

    R.subprocess.Popen = fake_popen
    os.environ["R3D_TAIKO_LOUDNORM_CACHE"] = "0"   # never build a cache in a test
    try:
        cfg = RenderConfig(resolution=res, fps=60, encoder=encoder)
        try:
            R._spawn_ffmpeg(cfg, Path(out), Path(audio) if audio else None, 0, 1.0,
                            total, preview_path=Path(preview) if preview else None,
                            compact_path=Path(compact) if compact else None)
        except _Captured:
            pass
        assert len(got) == 1, "ffmpeg was not started exactly once"
        return got[0]
    finally:
        R.subprocess.Popen = saved_popen
        if saved_cache is None:
            os.environ.pop("R3D_TAIKO_LOUDNORM_CACHE", None)
        else:
            os.environ["R3D_TAIKO_LOUDNORM_CACHE"] = saved_cache


def _after(cmd, token):
    return cmd[cmd.index(token) + 1]


def _codec_after(cmd, token):
    # the rawvideo INPUT also carries a -pix_fmt, so look only at the codec args
    ci = cmd.index("-c:v")
    return cmd[ci + cmd[ci:].index(token) + 1]


# ---- arg builder: backend-specific controls ---------------------------------

def test_av1_vaapi_uses_cqp_global_quality_never_qp():
    cmd = _cmd("av1_vaapi")
    assert "-c:v" in cmd and _after(cmd, "-c:v") == "av1_vaapi"
    # the VAAPI trap: -qp is ignored on VAAPI, so it must NOT be used
    assert "-qp" not in cmd, cmd
    assert _after(cmd, "-rc_mode") == "CQP"
    assert _after(cmd, "-global_quality") == "90"
    # 8-bit 4:2:0 Main, uploaded to a GPU surface
    assert _after(cmd, "-vf") == "format=nv12,hwupload"
    assert _after(cmd, "-profile:v") == "main"
    # ~2 s GOP at 60 fps, web-streamable master
    assert _after(cmd, "-g") == str(R._av1_gop(60)) == "120"
    assert "-movflags" in cmd and _after(cmd, "-movflags") == "+faststart"
    # the render device is defaulted for VAAPI
    assert _after(cmd, "-vaapi_device") == "/dev/dri/renderD128"


def test_av1_nvenc_uses_the_vbr_ladder():
    cmd = _cmd("av1_nvenc", res=(1920, 1080))
    assert _after(cmd, "-c:v") == "av1_nvenc"
    assert _after(cmd, "-rc") == "vbr"
    tgt = R.nvenc_target_bps(1920, 1080, 60)
    assert _after(cmd, "-b:v") == str(tgt)
    assert _after(cmd, "-maxrate") == str(int(tgt * 1.5))
    assert _codec_after(cmd, "-pix_fmt") == "yuv420p"
    assert _after(cmd, "-profile:v") == "main"
    assert _after(cmd, "-g") == "120"
    assert "-vaapi_device" not in cmd and "-global_quality" not in cmd


def test_av1_qsv_uses_icq_and_amf_uses_vbr():
    qsv = _cmd("av1_qsv")
    assert _after(qsv, "-c:v") == "av1_qsv"
    assert _after(qsv, "-global_quality") == "90"
    assert _codec_after(qsv, "-pix_fmt") == "nv12"
    assert _after(qsv, "-profile:v") == "main" and _after(qsv, "-g") == "120"

    amf = _cmd("av1_amf", res=(1920, 1080))
    assert _after(amf, "-c:v") == "av1_amf"
    assert _after(amf, "-rc") == "vbr_latency"
    assert _after(amf, "-b:v") == str(R.nvenc_target_bps(1920, 1080, 60))
    assert _codec_after(amf, "-pix_fmt") == "yuv420p" and _after(amf, "-profile:v") == "main"


def test_av1_video_args_vaapi_switches_to_vbr_when_a_bitrate_is_given():
    # CQP by default, explicit bitrate -> VBR (Leon tested both recipes)
    cqp = R._av1_video_args("av1_vaapi", 1280, 720, 60, None)
    assert "-rc_mode" in cqp and cqp[cqp.index("-rc_mode") + 1] == "CQP"
    vbr = R._av1_video_args("av1_vaapi", 1280, 720, 60, 4_000_000)
    assert vbr[vbr.index("-rc_mode") + 1] == "VBR"
    assert vbr[vbr.index("-b:v") + 1] == "4000000"
    assert "-global_quality" not in vbr and "-qp" not in vbr


def test_unknown_av1_encoder_is_rejected():
    try:
        R._av1_video_args("av1_bogus", 1280, 720, 60, None)
    except R.TaikoRenderError as exc:
        assert "av1_bogus" in str(exc)
    else:
        raise AssertionError("an unknown AV1 encoder must raise")


# ---- the AV1 master does not touch the H.264 Discord/preview copies ---------

def test_av1_master_keeps_discord_and_preview_on_h264():
    cmd = _cmd("av1_vaapi", preview="m.embed.mp4", compact="m-embed-sm.mp4")
    # the master output is av1_vaapi...
    master_end = cmd.index("m.mp4")
    assert "av1_vaapi" in cmd[:master_end]
    # ...and its nv12 upload moved onto the master branch of the graph,
    # so the master's own -vf is not duplicated
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "[vm0]format=nv12,hwupload[vm]" in graph, graph
    assert "-rc_mode" in cmd[:master_end]
    # the preview and the Discord copy STAY H.264 — Discord cannot play AV1
    tail = cmd[master_end + 1:]
    assert "av1_vaapi" not in tail and "av1_nvenc" not in tail
    assert tail.count("libx264") == 2, tail            # preview + Discord copy


def test_av1_single_output_has_no_second_vf_or_preview():
    cmd = _cmd("av1_nvenc")
    assert "-filter_complex" not in cmd
    assert cmd.count("av1_nvenc") == 1 and "libx264" not in cmd


# ---- auto `av1` probe: real encode, fail-closed -----------------------------

def _auto_av1(outcomes, encoder_device=None, device_exists=True,
              has=("av1_nvenc", "av1_vaapi", "av1_amf", "av1_qsv")):
    log = io.StringIO()
    cfg = RenderConfig(encoder="av1", encoder_device=encoder_device)
    with patch.object(R, "_ffmpeg_has", side_effect=lambda n: n in has), \
            patch.object(R.Path, "exists", return_value=device_exists), \
            patch.object(R, "_encoder_usable_av1", side_effect=outcomes) as probe, \
            redirect_stderr(log):
        try:
            result = R._probe_encoder(cfg)
            return result, probe.call_args_list, log.getvalue(), None
        except R.TaikoRenderError as exc:
            return None, probe.call_args_list, log.getvalue(), exc


def test_auto_av1_selects_first_backend_that_really_produces_av1():
    # nvenc is tried first and fails to produce av1; vaapi does
    result, calls, log, err = _auto_av1([(False, "no NVENC driver"), (True, "")])
    assert err is None and result == ("av1_vaapi", "/dev/dri/renderD128")
    assert [c.args[0] for c in calls] == ["av1_nvenc", "av1_vaapi"]
    assert "auto selected av1_vaapi" in log and "no NVENC driver" in log


def test_auto_av1_fails_closed_with_no_h264_or_cpu_fallback():
    result, calls, _log, err = _auto_av1([
        (False, "NVENC: no driver"), (False, "VAAPI: not supported"),
        (False, "AMF: windows only"), (False, "QSV: no device")])
    assert result is None and err is not None
    msg = str(err)
    assert "no usable hardware AV1 encoder" in msg
    assert all(r in msg for r in ("NVENC", "VAAPI", "AMF", "QSV"))
    # fail-closed: it must NOT silently hand back an H.264 or CPU encoder
    assert "libx264" not in msg and "h264" not in msg
    assert len(calls) == 4


def test_auto_av1_skips_vaapi_when_no_render_device():
    result, calls, _log, err = _auto_av1(
        [(False, "no NVENC"), (True, "")], encoder_device=None,
        device_exists=False, has=("av1_nvenc", "av1_amf", "av1_qsv"))
    assert err is None and result[0] == "av1_amf"
    assert "av1_vaapi" not in [c.args[0] for c in calls]


def test_explicit_av1_backend_is_not_probed():
    # explicit stays explicit; the render encode itself is the proof
    with patch.object(R, "_ffmpeg_has") as has, \
            patch.object(R, "_encoder_usable_av1") as probe:
        assert R._probe_encoder(RenderConfig(encoder="av1_vaapi")) == (
            "av1_vaapi", "/dev/dri/renderD128")     # device defaulted
        assert R._probe_encoder(RenderConfig(encoder="av1_nvenc")) == ("av1_nvenc", None)
        has.assert_not_called()
        probe.assert_not_called()


def test_h264_auto_is_unchanged():
    # the existing H.264 auto path must still resolve to nvenc-first H.264
    with patch.object(R, "_ffmpeg_has", side_effect=lambda n: n == "h264_nvenc"):
        assert R._probe_encoder(RenderConfig(encoder="auto")) == ("h264_nvenc", None)
    with patch.object(R, "_ffmpeg_has", side_effect=lambda n: n == "h264_vaapi"), \
            patch.object(R.Path, "exists", return_value=True):
        assert R._probe_encoder(RenderConfig(encoder="auto")) == (
            "h264_vaapi", "/dev/dri/renderD128")


# ---- the real-encode probe gate: codec_name=av1 is mandatory ----------------

def test_encoder_usable_av1_demands_codec_name_av1():
    # the encode succeeds but the output is NOT av1 -> unusable
    def runs(ok_codec):
        def _run(cmd, **kw):
            if cmd[0] == "ffprobe":
                return SimpleNamespace(returncode=0, stdout=ok_codec, stderr="")
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        return _run

    with patch.object(R.subprocess, "run", side_effect=runs("h264\n")):
        ok, why = R._encoder_usable_av1("av1_vaapi", "/dev/dri/renderD128")
        assert not ok and "not av1" in why and "h264" in why
    with patch.object(R.subprocess, "run", side_effect=runs("av1\n")):
        ok, why = R._encoder_usable_av1("av1_vaapi", "/dev/dri/renderD128")
        assert ok and why == ""


def test_encoder_usable_av1_retains_driver_error_and_bounds_startup():
    bad = SimpleNamespace(returncode=1, stdout=b"",
                          stderr=b"[av1_vaapi] not supported\nmore detail")
    with patch.object(R.subprocess, "run", return_value=bad):
        ok, why = R._encoder_usable_av1("av1_vaapi", "/dev/dri/renderD128")
        assert not ok and "not supported" in why
    for err in (OSError("ffmpeg gone"), subprocess.TimeoutExpired("ffmpeg", 20)):
        with patch.object(R.subprocess, "run", side_effect=err):
            ok, why = R._encoder_usable_av1("av1_nvenc", None)
            assert not ok and why


def test_encoder_usable_av1_probe_command_shape():
    seen = {}

    def _run(cmd, **kw):
        if cmd[0] == "ffmpeg":
            seen["cmd"] = cmd
            seen["input"] = kw.get("input")
            seen["timeout"] = kw.get("timeout")
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        return SimpleNamespace(returncode=0, stdout="av1\n", stderr="")

    with patch.object(R.subprocess, "run", side_effect=_run):
        assert R._encoder_usable_av1("av1_vaapi", "/dev/dri/renderD129")[0]
    cmd = seen["cmd"]
    assert cmd[cmd.index("-vaapi_device") + 1] == "/dev/dri/renderD129"
    assert cmd[cmd.index("-s") + 1] == "128x128"            # a tiny frame
    assert len(seen["input"]) == 128 * 128 * 3
    assert seen["timeout"] == 20 and cmd[cmd.index("-c:v") + 1] == "av1_vaapi"


# ---- opt-in real hardware encode (R3D_AV1_HW_TEST=1) ------------------------

def test_real_av1_vaapi_encode_is_actually_av1():
    """One short 128x128 av1_vaapi encode on this machine, verified AV1. OPT-IN
    (R3D_AV1_HW_TEST=1) so a plain pytest run does not load the GPU while other
    agents run."""
    if os.environ.get("R3D_AV1_HW_TEST") != "1":
        print("SKIP real av1_vaapi encode (set R3D_AV1_HW_TEST=1 to run)")
        return
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("SKIP (no ffmpeg/ffprobe)")
        return
    dev = os.environ.get("R3D_AV1_DEVICE", "/dev/dri/renderD128")
    if not Path(dev).exists() or not R._ffmpeg_has("av1_vaapi"):
        print(f"SKIP (no av1_vaapi / {dev})")
        return
    ok, why = R._encoder_usable_av1("av1_vaapi", dev)
    assert ok, f"av1_vaapi did not produce AV1 on {dev}: {why}"
    print(f"ok   real av1_vaapi encode on {dev} is av1")


if __name__ == "__main__":
    for _n, _f in sorted(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f()
            print("ok  ", _n)
    print("all taiko AV1 tests PASSED")
