"""R3D_TAIKO_STREAM_READ: the readback ring's buffers declared GL_STREAM_READ
(render/gl.py _readback_ring).

What must hold:
  * on by default on a Mac, off with =0, off on the stock path, never on off a Mac;
  * with it on every ring buffer is GL_STREAM_READ and still the size asked for;
    with it off the ring is what it always was (GL_STATIC_DRAW);
  * the frames that come back are the same bytes on and off.

The GL tests skip where there is no GL context.
Runnable two ways:  pytest tests/test_stream_read.py   OR   python tests/test_stream_read.py
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from osu_taiko_renderer.render import gl

_GL_BUFFER_SIZE, _GL_BUFFER_USAGE = 0x8764, 0x8765
_GL_STATIC_DRAW, _GL_STREAM_READ = 0x88E4, 0x88E1


def _resolved(env: dict) -> bool:
    code = ("import json, osu_taiko_renderer.render.envflag as s;"
            "print(json.dumps([s.STREAM_READ, s.ANY_FAST]))")
    e = {k: v for k, v in os.environ.items() if not k.startswith("R3D_")}
    e.update(env)
    out = subprocess.run([sys.executable, "-c", code], env=e, capture_output=True, text=True,
                         cwd=str(Path(__file__).resolve().parents[1]), check=True).stdout
    return json.loads(out)


def test_the_switch():
    mac = sys.platform == "darwin"
    assert _resolved({})[0] is mac                                   # the default
    assert _resolved({"R3D_TAIKO_STREAM_READ": "0"})[0] is False
    assert _resolved({"R3D_TAIKO_STREAM_READ": "1"})[0] is mac       # Apple's GL only
    assert _resolved({"R3D_TAIKO_STREAM_READ": "1", "R3D_TAIKO_STOCK": "1"}) == [False, False]
    if mac:
        # by itself it counts as "not the stock path", so a failed render is run again on stock
        off = {k: "0" for k in ("R3D_TAIKO_GPU_FX", "R3D_TAIKO_ROUND", "R3D_MAP_READBACK",
                                "R3D_MAC_SOCKET_PIPE", "R3D_TAIKO_RESULTS_AHEAD")}
        assert _resolved(dict(off, R3D_TAIKO_STREAM_READ="1")) == [True, True]
        assert _resolved(dict(off, R3D_TAIKO_STREAM_READ="0")) == [False, False]


def _usage_and_size(buf):
    g = gl._load_gl_c()
    g.glGetBufferParameteriv.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_int)]
    g.glGetBufferParameteriv.restype = None
    out = []
    g.glBindBuffer(gl._GL_PIXEL_PACK_BUFFER, buf.glo)
    for what in (_GL_BUFFER_USAGE, _GL_BUFFER_SIZE):
        v = ctypes.c_int(0)
        g.glGetBufferParameteriv(gl._GL_PIXEL_PACK_BUFFER, what, ctypes.byref(v))
        out.append(v.value)
    g.glBindBuffer(gl._GL_PIXEL_PACK_BUFFER, 0)
    return tuple(out)


def _renderer(w=64, h=32):
    if sys.platform != "darwin":
        print("SKIP (the ring's GL handle is Apple's OpenGL)")
        return None
    try:
        return gl.SpriteRenderer(w, h)
    except Exception:  # noqa: BLE001 -- no GL device on this box
        print("SKIP (no GL context)")
        return None


def test_the_ring_is_declared_as_asked():
    spr = _renderer()
    if spr is None:
        return
    old = gl._STREAM_READ
    try:
        for on, usage in ((True, _GL_STREAM_READ), (False, _GL_STATIC_DRAW)):
            gl._STREAM_READ = on
            ring = spr._readback_ring(64 * 32 * 3, 5)
            assert [_usage_and_size(b) for b in ring] == [(usage, 64 * 32 * 3)] * 5, on
            for b in ring:
                b.release()
    finally:
        gl._STREAM_READ = old
        spr.release()


def _frames(on: bool, pictures):
    """Each picture put in the scene and read back through the async ring."""
    old = gl._STREAM_READ
    gl._STREAM_READ = on
    spr = gl.SpriteRenderer(64, 32)
    try:
        out = []
        for picture in pictures:
            spr._scene_tex.write(np.ascontiguousarray(picture[::-1]).tobytes())
            f = spr.read_rgb_async()
            if f is not None:
                out.append(np.array(f[..., :3], copy=True))
        while True:
            tail = spr.read_rgb_drain() if hasattr(spr, "read_rgb_drain") else None
            if tail is None or len(tail) == 0:
                break
            if isinstance(tail, list):
                out += [np.array(t[..., :3], copy=True) for t in tail]
                break
            out.append(np.array(tail[..., :3], copy=True))
        return out
    finally:
        gl._STREAM_READ = old
        spr.release()


def test_the_same_frames_on_and_off():
    probe = _renderer()
    if probe is None:
        return
    probe.release()
    rng = np.random.default_rng(11)
    pictures = [rng.integers(0, 256, (32, 64, 4), dtype=np.uint8) for _ in range(30)]   # the ring comes round twice
    on, off = _frames(True, pictures), _frames(False, pictures)
    assert len(on) == len(off) and len(on) >= len(pictures) - gl._PBO_LAT
    for i, (a, b) in enumerate(zip(on, off)):
        assert (a == b).all(), f"frame {i}"
        assert (a == pictures[i][..., :3]).all(), f"frame {i} is not the picture put in"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
