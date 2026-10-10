"""Minimal moderngl sprite batch for the catch renderer.

Owns a standalone EGL context and an offscreen RGBA framebuffer. Draws
textured/solid quads with straight-alpha blending in painter's order, then
reads back tightly-packed RGB24 for the ffmpeg pipe. Deliberately tiny and
self-contained so it can be discarded when the VRender branch takes over.
"""
from __future__ import annotations

import os
import sys

import numpy as np

try:
    import moderngl
except Exception as e:  # noqa: BLE001
    raise RuntimeError("moderngl is required for the catch renderer") from e

from osu_taiko_renderer.beatmap.models import Sprite
import osu_taiko_renderer.render.envflag as _sw
from osu_taiko_renderer.render.envflag import envflag

_VERT = """
#version 330
in vec2 in_pos;      // unit quad corner [-0.5,0.5]
in vec2 in_uv;
uniform vec2 u_screen;   // (w, h) in px
uniform vec2 u_center;   // sprite center in px (origin top-left)
uniform vec2 u_size;     // sprite w,h in px
uniform float u_rot;     // radians
uniform vec2 u_uv_off;   // texture UV offset (storyboard flip mirroring)
uniform vec2 u_uv_scale; // texture UV scale  (default (1,1); -1 mirrors an axis)
out vec2 v_uv;
void main() {
    vec2 p = in_pos * u_size;
    float c = cos(u_rot), s = sin(u_rot);
    p = vec2(p.x * c - p.y * s, p.x * s + p.y * c);
    vec2 px = u_center + p;
    // px -> clip, with y flipped (top-left origin)
    vec2 ndc = vec2(px.x / u_screen.x * 2.0 - 1.0,
                    1.0 - px.y / u_screen.y * 2.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
    // identity default ((0,0)/(1,1)) == `in_uv`, so non-storyboard sprites are
    // sampled bit-identically; a storyboard flip passes off=1,scale=-1 per axis.
    v_uv = in_uv * u_uv_scale + u_uv_off;
}
"""

_FRAG = """
#version 330
in vec2 v_uv;
uniform sampler2D u_tex;
uniform vec4 u_color;
out vec4 f_color;
void main() {
    vec4 t = texture(u_tex, v_uv);
    f_color = t * u_color;
}
"""


# Mapped readback. glGetBufferSubData is specified to synchronise; catch measured
# it at 0.663 ms standalone for 8.29 MB, vs 0.316 ms for glMapBufferRange(READ|WRITE)
# — which also REMOVES the host copy entirely, since the mapped pointer is wrapped
# as a numpy view and handed straight to the compositors and the writer.
# moderngl exposes no mapping API, so the two missing calls come via ctypes off the
# already-loaded framework. No PyOpenGL, no new dependency.
_DRAW_COUNT = envflag("R3D_TAIKO_DRAWCOUNT")
_NC = 3      # bytes per pixel read back and piped (rgb24)
_DC: dict = {}
if _DRAW_COUNT:
    import atexit as _dc_atx

    @_dc_atx.register
    def _dc_dump():
        import sys as _s
        f = max(1, _DC.get("frames", 1))
        print(f"[draw-count] per frame over {int(f)} frames:", file=_s.stderr)
        keys = ["sprites", "runs", "texsw", "nruns"] + \
               [k for k in sorted(_DC) if k.startswith("runlen_")]
        for k in keys:
            print(f"[draw-count]   {k:<10} {_DC.get(k,0)/f:8.2f}", file=_s.stderr)


_MAP_READBACK = _sw.MAP_READBACK
_STREAM_READ = _sw.STREAM_READ
# R3D_TAIKO_GPU_YUV: convert RGB -> yuv420p ON THE GPU and read back 1.5 bytes/px
# instead of 3, feeding ffmpeg `-pix_fmt yuv420p` so swscale does no conversion at all.
#
# WHY: measured, the encoder alone runs 649.8 fps fed rgb24 and 817.8 fps fed yuv420p
# (+26.4%), and production is renderer/encoder CONTENTION, not a slow encoder. This
# cuts both sides: half the bytes through readback and the pipe, and swscale deleted.
#
# THE ARITHMETIC IS ffmpeg's OWN for rgb24 -> yuv420p (libswscale's general path:
# an eight-row chroma filter), taken from std, where it is held equal to ffmpeg on
# every sample by a test. The first version here copied swscale's OTHER routine
# (the bgr24 one: a 2x2 box average), which is exact against THAT routine and wrong
# against what this engine's rgb24 pipe actually gets: chroma off by up to 23.
# It is exact for the ffmpeg build it was recovered from (8.1, arm64); another
# build can differ by one level in chroma, so `ffmpeg_matches_twin()` checks the
# local ffmpeg and the caller leaves the conversion to ffmpeg where they differ.
# NOTE it is a LOSS with a null sink (-3.8% gameplay) because its entire win is
# relieving encoder back-pressure: benchmark against a real encoder.
_GPU_YUV = False          # decided below, once rgb_to_yuv420p exists
_RGB2YUV_SHIFT = 15


def _yuv_coef(k, scale):
    """swscale's own coefficient derivation (utils.c:694). int() TRUNCATES after the
    +0.5 -- it is not a round, and a 1-LSB error here shows up as thousands of
    differing pixels that look like a rounding bug."""
    return int(k * scale / 255.0 * (1 << _RGB2YUV_SHIFT) + 0.5)


# ffmpeg's rgb24 -> yuv420p (libswscale's general path, default flags), recovered
# from its own output and exact on every sample tested: 21 M samples over real
# std frames (the routine is ffmpeg's, not an engine's), noise at three sizes and a one-pixel stripe pattern (ffmpeg 8.1).
#   Y   = ((((RY*R + GY*G + BY*B) + (0x801 << 8)) >> 9) + 32) >> 6
#   row = ((CU*R2 + CG*G2 + CB*B2) + (0x4001 << 9)) >> 10      R2 = the 2 pixels of
#                                                              a column pair, summed
#   C   = clamp((sum(_CHROMA_TAPS[k] * row[2j - 3 + k]) + (1 << 18)) >> 19, 0, 255)
# with source rows clamped at the top and bottom edge. The eight weights sum to
# 8192; rounding them to 8 bits (-4 -11 31 112 ...) matches only 92% on noise,
# and a 2x2 box average only 7%: fit a filter on NOISE, a gradient hides it.
_CHROMA_TAPS = (-116, -344, 984, 3572, 3572, 984, -344, -116)


def _yuv_matrix():
    RY, GY, BY = (_yuv_coef(k, 219) for k in (0.299, 0.587, 0.114))
    cu = (-_yuv_coef(0.169, 224), -_yuv_coef(0.331, 224), _yuv_coef(0.500, 224))
    cv = (_yuv_coef(0.500, 224), -_yuv_coef(0.419, 224), -_yuv_coef(0.081, 224))
    return (RY, GY, BY), cu, cv


def rgb_to_yuv420p(rgb):
    """CPU twin of the GPU conversion: ffmpeg's own rgb24 -> yuv420p, exactly.

    Not on the render path -- it is the ORACLE: tests/test_gpu_yuv.py holds it
    equal to ffmpeg itself, and the shader pair is the same integer arithmetic.

    `rgb` is (h, w, 3) uint8, top-down. Returns a flat uint8 yuv420p buffer
    (Y | U | V). int64 throughout: the shifts must be arithmetic on a signed
    type, and numpy's uint promotion rules make that easy to get subtly wrong."""
    h, w = rgb.shape[:2]
    r = rgb[..., 0].astype(np.int64)
    g = rgb[..., 1].astype(np.int64)
    b = rgb[..., 2].astype(np.int64)
    (RY, GY, BY), cu, cv = _yuv_matrix()
    y = ((((RY * r + GY * g + BY * b) + (0x801 << 8)) >> 9) + 32) >> 6
    r2, g2, b2 = (c[:, 0::2] + c[:, 1::2] for c in (r, g, b))

    def chroma(c):
        row = ((c[0] * r2 + c[1] * g2 + c[2] * b2) + (0x4001 << 9)) >> 10
        acc = np.zeros((h // 2, w // 2), np.int64)
        j2 = np.arange(h // 2) * 2
        for k, tap in enumerate(_CHROMA_TAPS):
            acc += tap * row[np.clip(j2 - 3 + k, 0, h - 1)]
        return np.clip((acc + (1 << 18)) >> 19, 0, 255)

    u, v = chroma(cu), chroma(cv)
    out = np.empty(w * h * 3 // 2, np.uint8)
    out[:w * h] = y.ravel()
    out[w * h:w * h + u.size] = u.ravel()
    out[w * h + u.size:] = v.ravel()
    return out

_ffmpeg_match: "bool | None" = None


def ffmpeg_matches_twin() -> bool:
    """Does the ffmpeg on THIS machine convert rgb24 -> yuv420p to exactly the
    bytes rgb_to_yuv420p gives? One small noise frame through it, once per
    ffmpeg binary (the answer is cached beside the other r3d caches, keyed on
    the binary's path, size and mtime). Any trouble counts as "no"."""
    global _ffmpeg_match
    if _ffmpeg_match is not None:
        return _ffmpeg_match
    import hashlib
    import shutil
    import subprocess
    import tempfile
    ok = False
    try:
        exe = shutil.which("ffmpeg")
        if exe:
            st = os.stat(os.path.realpath(exe))
            key = hashlib.sha1(f"{os.path.realpath(exe)}|{st.st_size}|{st.st_mtime_ns}|v1"
                               .encode()).hexdigest()[:16]
            base = (os.path.expanduser("~/Library/Caches/r3d") if sys.platform == "darwin"
                    else os.path.join(tempfile.gettempdir(), "r3d-cache"))
            mark = os.path.join(base, f"taiko-gpu-yuv-probe-{key}")
            try:
                with open(mark) as fh:
                    cached = fh.read().strip()
            except OSError:
                cached = ""
            if cached in ("1", "0"):
                ok = cached == "1"
            else:
                w, h = 128, 96
                rgb = np.random.default_rng(20261006).integers(
                    0, 256, (h, w, 3), dtype=np.uint8)
                p = subprocess.run(
                    [exe, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                     "-s", f"{w}x{h}", "-i", "pipe:0", "-pix_fmt", "yuv420p",
                     "-f", "rawvideo", "pipe:1"],
                    input=rgb.tobytes(), capture_output=True, timeout=15)
                ok = (p.returncode == 0
                      and p.stdout == rgb_to_yuv420p(rgb).tobytes())
                try:
                    os.makedirs(base, exist_ok=True)
                    with open(mark, "w") as fh:
                        fh.write("1" if ok else "0")
                except OSError:
                    pass
    except Exception:  # noqa: BLE001 - a probe must never fail a render
        ok = False
    _ffmpeg_match = ok
    return ok


# Asked for (R3D_TAIKO_GPU_YUV, see render/envflag.py), it is on only where the
# local ffmpeg agrees with the twin; where it does not, ffmpeg converts.
if _sw.GPU_YUV:
    _GPU_YUV = ffmpeg_matches_twin()
    if not _GPU_YUV:
        print("[taiko-renderer] R3D_TAIKO_GPU_YUV: this machine's ffmpeg converts "
              "rgb24 -> yuv420p differently from the GPU routine; leaving the "
              "conversion to ffmpeg", file=sys.stderr)


# R3D_TAIKO_PBO_LAT: readback latency in frames (default 3, the historical value).
# Raising it gives the GPU more time to finish a readback before we map it, at the
# cost of more frames buffered. The PBO pool depth is derived from this, so the two
# cannot drift apart -- see read_rgb_async.
_PBO_LAT = max(1, int(os.environ.get("R3D_TAIKO_PBO_LAT", "3")))
# MUST match render.py's FrameWriter._QUEUE_FRAMES. Not imported, because render.py
# imports this module and the cycle would break the build; kept as a constant with
# this note instead. If that queue depth changes, change this too.
_WRITER_QUEUE_FRAMES = 4
# Slack above the holder budget. Catch showed a pool that is merely "big enough"
# corrupts output silently when anything holds a frame a beat longer than assumed.
_PBO_MARGIN = 3

_GL_PIXEL_PACK_BUFFER = 0x88EB
_GL_STREAM_READ = 0x88E1
_GL_MAP_READ_BIT = 0x0001
_GL_MAP_WRITE_BIT = 0x0002
_gl_c = None


def _load_gl_c():
    """ctypes handle on the already-loaded GL, for the calls moderngl lacks."""
    global _gl_c
    if _gl_c is None:
        import ctypes
        lib = ctypes.CDLL("/System/Library/Frameworks/OpenGL.framework/OpenGL")
        lib.glBindBuffer.argtypes = [ctypes.c_uint, ctypes.c_uint]
        lib.glBindBuffer.restype = None
        lib.glMapBufferRange.argtypes = [ctypes.c_uint, ctypes.c_ssize_t,
                                         ctypes.c_ssize_t, ctypes.c_uint]
        lib.glMapBufferRange.restype = ctypes.c_void_p
        lib.glUnmapBuffer.argtypes = [ctypes.c_uint]
        lib.glUnmapBuffer.restype = ctypes.c_ubyte
        lib.glBufferData.argtypes = [ctypes.c_uint, ctypes.c_ssize_t,
                                     ctypes.c_void_p, ctypes.c_uint]
        lib.glBufferData.restype = None
        _gl_c = lib
    return _gl_c


# --- instanced sprite path (R3D_TAIKO_INSTANCED) ------------------------------
# One draw call per blend run instead of one per sprite, with the per-sprite data
# as vertex attributes rather than uniforms. Ported from catch.
#
# Measured on taiko AFTER the HUD moved to the GPU (116 sprites/frame): ablating the
# per-sprite uniform writes and render() calls recovered 0.526 ms of a 1.350 ms
# gl_draw, i.e. 448 -> 645 fps. At the ORIGINAL 60 sprites/frame the same ablation
# recovered only 0.306 ms and this port was correctly rejected as poor value — moving
# the HUD onto the GPU is what made it worth doing.
#
# A per-instance sampler INDEX is not dynamically uniform in GLSL 330, so indexing a
# sampler array with it is illegal. Hence 16 discrete samplers and an if/else chain.
_I_VERT = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
in vec2 i_center;
in vec2 i_size;
in float i_rot;
in vec4 i_color;
in vec2 i_uv_off;
in vec2 i_uv_scale;
in float i_tex;
uniform vec2 u_screen;
out vec2 v_uv;
out vec4 v_color;
flat out int v_tex;
void main() {
    vec2 p = in_pos * i_size;
    float c = cos(i_rot), s = sin(i_rot);
    p = vec2(p.x * c - p.y * s, p.x * s + p.y * c);
    vec2 px = i_center + p;
    vec2 ndc = vec2(px.x / u_screen.x * 2.0 - 1.0,
                    1.0 - px.y / u_screen.y * 2.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_uv = in_uv * i_uv_scale + i_uv_off;
    v_color = i_color;
    v_tex = int(i_tex);
}
"""

_I_FRAG = """
#version 330
in vec2 v_uv;
in vec4 v_color;
flat in int v_tex;
out vec4 f_color;
uniform sampler2D u_tex0;
uniform sampler2D u_tex1;
uniform sampler2D u_tex2;
uniform sampler2D u_tex3;
uniform sampler2D u_tex4;
uniform sampler2D u_tex5;
uniform sampler2D u_tex6;
uniform sampler2D u_tex7;
uniform sampler2D u_tex8;
uniform sampler2D u_tex9;
uniform sampler2D u_tex10;
uniform sampler2D u_tex11;
uniform sampler2D u_tex12;
uniform sampler2D u_tex13;
uniform sampler2D u_tex14;
uniform sampler2D u_tex15;
void main() {
    int t = v_tex;
    vec4 tc = vec4(0.0);
    if (t == 0) tc = texture(u_tex0, v_uv);
    else if (t == 1) tc = texture(u_tex1, v_uv);
    else if (t == 2) tc = texture(u_tex2, v_uv);
    else if (t == 3) tc = texture(u_tex3, v_uv);
    else if (t == 4) tc = texture(u_tex4, v_uv);
    else if (t == 5) tc = texture(u_tex5, v_uv);
    else if (t == 6) tc = texture(u_tex6, v_uv);
    else if (t == 7) tc = texture(u_tex7, v_uv);
    else if (t == 8) tc = texture(u_tex8, v_uv);
    else if (t == 9) tc = texture(u_tex9, v_uv);
    else if (t == 10) tc = texture(u_tex10, v_uv);
    else if (t == 11) tc = texture(u_tex11, v_uv);
    else if (t == 12) tc = texture(u_tex12, v_uv);
    else if (t == 13) tc = texture(u_tex13, v_uv);
    else if (t == 14) tc = texture(u_tex14, v_uv);
    else if (t == 15) tc = texture(u_tex15, v_uv);
    f_color = tc * v_color;
}
"""

_I_STRIDE = 14          # center2 size2 rot1 color4 uv_off2 uv_scale2 tex1
_I_MAX_TEX = 16
_INSTANCED = _sw.INSTANCED


class SpriteRenderer:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        # Honor R3D_EGL_DEVICE_INDEX so renders pin to the right GPU (pool
        # isolation: e.g. 1070=index 1 for Pool B). EGL ignores
        # CUDA_VISIBLE_DEVICES, so the device must be selected explicitly.
        if sys.platform in ("win32", "darwin"):
            # Windows contributors: glcontext ships no EGL backend (ImportError
            # cannot import name egl), so use the default WGL standalone context.
            # R3D_EGL_DEVICE_INDEX is an EGL-only GPU pin (Linux pools).
            self.ctx = moderngl.create_context(standalone=True)
        else:
            dev = os.environ.get("R3D_EGL_DEVICE_INDEX", "").strip()
            if dev.isdigit():
                self.ctx = moderngl.create_context(
                    standalone=True, backend="egl", device_index=int(dev))
            else:
                self.ctx = moderngl.create_context(standalone=True, backend="egl")
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        self.prog = self.ctx.program(vertex_shader=_VERT, fragment_shader=_FRAG)
        # unit quad centered at origin, uv 0..1
        # in_pos.y=-0.5 renders at screen-top -> texture-top (v=0); in_pos.y=+0.5
        # renders at screen-bottom -> texture-bottom (v=1). (Matters for
        # vertically-asymmetric sprites like the catcher.)
        quad = np.array([
            -0.5, -0.5, 0.0, 0.0,
             0.5, -0.5, 1.0, 0.0,
            -0.5,  0.5, 0.0, 1.0,
             0.5,  0.5, 1.0, 1.0,
        ], dtype="f4")
        self.vbo = self.ctx.buffer(quad.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog, [(self.vbo, "2f 2f", "in_pos", "in_uv")],
        )
        self.prog["u_screen"].value = (float(width), float(height))
        # perf: cache the uniform objects once (prog["..."] is a dict lookup +
        # object construction per access — it was per-sprite in draw()) and
        # bind the sampler slot a single time.
        self.prog["u_tex"].value = 0
        self._u_color = self.prog["u_color"]
        self._u_center = self.prog["u_center"]
        self._u_size = self.prog["u_size"]
        self._u_rot = self.prog["u_rot"]
        # UV flip uniforms (storyboard mirroring). Bound to identity so the
        # very first draw with default sprites is unchanged; draw() only
        # re-binds them when a sprite's uv differs from the last one drawn.
        self._u_uv_off = self.prog["u_uv_off"]
        self._u_uv_scale = self.prog["u_uv_scale"]
        self._u_uv_off.value = (0.0, 0.0)
        self._u_uv_scale.value = (1.0, 1.0)

        # Scene target is a TEXTURE (was a renderbuffer) so the flip pass can
        # sample it; RGBA8 rasterization into either is identical.
        self._scene_tex = self.ctx.texture((width, height), 4)
        self._scene_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.fbo = self.ctx.framebuffer(color_attachments=[self._scene_tex])
        # perf: y-flip on the GPU. The CPU used to hand ffmpeg a np.flipud view
        # whose flip copy ran per frame (writer-thread tobytes). Instead an
        # exact texelFetch pass mirrors the scene into a second FBO, so the
        # PBO readback is already top-left origin and fully contiguous —
        # written to the pipe zero-copy. texelFetch is an integer texel copy
        # (no filtering/blending): bytes are identical to the CPU flip.
        self._flip_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                void main() { gl_Position = vec4(in_pos * 2.0, 0.0, 1.0); }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D u_tex;
                out vec4 f_color;
                void main() {
                    ivec2 sz = textureSize(u_tex, 0);
                    f_color = texelFetch(u_tex,
                        ivec2(int(gl_FragCoord.x),
                              sz.y - 1 - int(gl_FragCoord.y)), 0);
                }
            """,
        )
        self._flip_prog["u_tex"].value = 0
        self._flip_vao = self.ctx.vertex_array(
            self._flip_prog, [(self.vbo, "2f 2x4", "in_pos")],
        )
        rb_flip = self.ctx.renderbuffer((width, height))
        self._flip_fbo = self.ctx.framebuffer(color_attachments=[rb_flip])
        self._textures: dict[str, moderngl.Texture] = {}
        # persistent textures for per-frame content (see dyn_texture)
        self._dyn: dict[str, tuple] = {}
        self._pbo_size = 0
        self._lat = 3
        self._mapped: list = []
        self._fl_prog = None
        self._fl_vao = None
        self._white = self._make_texture_rgba(np.full((1, 1, 4), 255, dtype="u1"))

        # async PBO ring state (read_rgb_async / read_drain) — ported from
        # the std renderer's proven pipeline (osu_std_renderer/render/gl.py)
        self._pbos: list["moderngl.Buffer"] | None = None
        self._pbo_head = 0
        self._pbo_tail = 0

    # --- texture management ---------------------------------------------------

    def upload_texture(self, key: str, rgba: np.ndarray,
                       clamp: bool = False, mipmaps: bool = True) -> None:
        """rgba: HxWx4 uint8 array (top-left origin). clamp=True sets
        clamp-to-edge wrapping (the storyboard samples with a flipped UV that
        can graze the edge texel; repeat wrap would wrap the far edge in).
        mipmaps default True — existing callers pass neither and are unchanged
        (mipmapped LINEAR, repeat wrap, exactly as before)."""
        if rgba.dtype != np.uint8:
            rgba = rgba.astype("u1")
        if rgba.shape[2] == 3:
            a = np.full(rgba.shape[:2] + (1,), 255, dtype="u1")
            rgba = np.concatenate([rgba, a], axis=2)
        tex = self._make_texture_rgba(rgba, mipmaps=mipmaps)
        if clamp:
            tex.repeat_x = False
            tex.repeat_y = False
        self._textures[key] = tex

    def draw_flashlight(self, cx: float, cy: float, ri: int, core: float) -> None:
        """Darken the scene to the flashlight spotlight with ONE full-screen
        multiply-blend quad, replacing `TaikoFlashlight.composite`'s CPU pass
        (measured 2.57 ms/frame — larger than the HUD, gl_draw and readback
        combined).

        GL blend computes `src*sf + dst*df`; with (ZERO, SRC_COLOR) that is
        `dst * keep`, which is exactly the CPU's `frame * (1 - alpha)`.

        Two details are load-bearing:
          * The CPU builds a 2ri x 2ri ramp whose centre sits at index ri-0.5 and
            places it at `round(c) - ri`, so the effective centre is
            `(round(cx) - 0.5, round(cy) - 0.5)` — not `(cx, cy)`.
          * Outside the radius the ramp saturates to alpha=1 (black), so the
            CPU's "zeros everywhere, disc bbox only" is identical to evaluating
            the falloff over the whole frame. No bbox needed.

        Residual delta vs the CPU: it does `astype(np.uint8)` (TRUNCATES) while an
        RGBA8 multiply blend rounds to nearest — i.e. the same ~1 LSB floor as the
        rest of the GPU work.
        """
        if self._fl_prog is None:
            self._fl_prog = self.ctx.program(
                vertex_shader="""
                    #version 330
                    in vec2 in_pos;
                    void main() { gl_Position = vec4(in_pos * 2.0, 0.0, 1.0); }
                """,
                fragment_shader="""
                    #version 330
                    uniform vec2 u_c;        // (round(cx)-0.5, round(cy)-0.5)
                    uniform float u_ri;      // disc radius in px
                    uniform float u_core;    // fully-lit fraction of the radius
                    uniform float u_h;       // frame height, for the y flip
                    out vec4 f_color;
                    void main() {
                        // gl_FragCoord is bottom-left origin and pixel-centred;
                        // the scene FBO is top-left origin in screen terms.
                        float col = gl_FragCoord.x - 0.5;
                        float row = u_h - 0.5 - gl_FragCoord.y;
                        float dn = length(vec2(col, row) - u_c) / u_ri;
                        float x = clamp((dn - u_core) / (1.0 - u_core), 0.0, 1.0);
                        float a = x * x * (3.0 - 2.0 * x);   // smoothstep
                        f_color = vec4(vec3(1.0 - a), 1.0);  // keep factor
                    }
                """,
            )
            self._fl_vao = self.ctx.vertex_array(
                self._fl_prog, [(self.vbo, "2f 2x4", "in_pos")])
        self._fl_prog["u_c"].value = (round(cx) - 0.5, round(cy) - 0.5)
        self._fl_prog["u_ri"].value = float(ri)
        self._fl_prog["u_core"].value = float(core)
        self._fl_prog["u_h"].value = float(self.height)
        self.ctx.blend_func = (moderngl.ZERO, moderngl.SRC_COLOR)
        self._fl_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

    def draw_flashlight_exact(self, x0: int, y0: int, n: int, keep) -> None:
        """BYTE-EXACT flashlight: sample the CPU's own `1 - alpha` ramp as an
        R32F texture instead of recomputing the falloff in GLSL.

        `draw_flashlight` evaluates the smoothstep in the shader, which is
        mathematically identical to numpy's but rounds differently at a handful of
        float32 boundaries -- measured at ~25 differing channel values per frame,
        all +-1. Three cheaper explanations were tested and refuted first
        (rounding ties, 8-bit quantisation of keep, and length() vs
        sqrt(dx*dx+dy*dy)); see draw_flashlight's docstring.

        Sampling removes the arithmetic entirely: `keep` comes from the array
        composite() itself multiplies by, so the only remaining step is
        `dst * keep` in the blend and the 8-bit store, which both sides round to
        nearest. NEAREST filtering + texelFetch means no interpolation.

        Outside the ramp's bbox composite() writes 0 (`out = np.zeros_like`), so
        the shader returns keep = 0 there -- one full-screen quad covers both
        cases with no second pass and no scissor.
        """
        import numpy as _np
        if getattr(self, "_flx_prog", None) is None:
            self._flx_prog = self.ctx.program(
                vertex_shader="""
                    #version 330
                    in vec2 in_pos;
                    void main() { gl_Position = vec4(in_pos * 2.0, 0.0, 1.0); }
                """,
                fragment_shader="""
                    #version 330
                    uniform sampler2D u_keep;   // R32F, n x n, = 1 - alpha
                    uniform int   u_x0;
                    uniform int   u_y0;
                    uniform int   u_n;
                    uniform float u_h;
                    out vec4 f_color;
                    void main() {
                        int col = int(gl_FragCoord.x - 0.5);
                        int row = int(u_h - 0.5 - gl_FragCoord.y);
                        int ix = col - u_x0;
                        int iy = row - u_y0;
                        float k = 0.0;
                        if (ix >= 0 && iy >= 0 && ix < u_n && iy < u_n)
                            k = texelFetch(u_keep, ivec2(ix, iy), 0).r;
                        f_color = vec4(vec3(k), 1.0);
                    }
                """,
            )
            self._flx_vao = self.ctx.vertex_array(
                self._flx_prog, [(self.vbo, "2f 2x4", "in_pos")])
            self._flx_tex = None
            self._flx_key = None
        # (re)upload only when the ramp actually changes -- identity is enough,
        # the caller caches by integer radius.
        if self._flx_key is not keep:
            arr = _np.ascontiguousarray(keep, dtype="f4")
            if self._flx_tex is None or self._flx_tex.size != (n, n):
                if self._flx_tex is not None:
                    self._flx_tex.release()
                self._flx_tex = self.ctx.texture((n, n), 1, arr.tobytes(),
                                                 dtype="f4")
                self._flx_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            else:
                self._flx_tex.write(arr.tobytes())
            self._flx_key = keep
        self._flx_tex.use(0)
        self._flx_prog["u_keep"].value = 0
        self._flx_prog["u_x0"].value = int(x0)
        self._flx_prog["u_y0"].value = int(y0)
        self._flx_prog["u_n"].value = int(n)
        self._flx_prog["u_h"].value = float(self.height)
        self.ctx.blend_func = (moderngl.ZERO, moderngl.SRC_COLOR)
        self._flx_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

    def dyn_texture(self, key: str, max_w: int, max_h: int) -> None:
        """Allocate a persistent texture for content that changes EVERY frame
        (the HUD score/combo images). Allocated once and then written with
        write_dyn, i.e. glTexSubImage2D — never reallocated and never
        re-mipmapped, unlike upload_texture, which builds a fresh texture and a
        fresh mipmap chain per call (fine once at startup, ruinous per frame;
        it also drops the previous texture without releasing it).

        No mipmaps and clamp wrapping: the quad is drawn 1:1 at the image's own
        size, so only mip 0 is ever sampled and a repeat wrap could only pull in
        the far edge."""
        ent = self._dyn.get(key)
        if ent is not None:
            if ent[1] >= int(max_w) and ent[2] >= int(max_h):
                return                      # existing reservation is big enough
            # GROW: an element got larger than its first reservation. Without this
            # the early-return kept the small texture, write_dyn refused the write,
            # and the caller fell back to the CPU path for EVERY frame — which for
            # the HUD meant composing it twice and running ~20% slower than not
            # moving it at all.
            try:
                ent[0].release()
            except Exception:  # noqa: BLE001 — context may be tearing down
                pass
            del self._dyn[key]
            self._textures.pop(key, None)
        tex = self.ctx.texture((int(max_w), int(max_h)), 4)
        # NEAREST, not LINEAR: these sprites are drawn 1:1 at integer positions, so
        # there is nothing to interpolate — and LINEAR will blend between adjacent
        # texels wherever the uv mapping across the quad does not land exactly on
        # texel centres, which shows up as a 1-2 LSB difference against the CPU
        # composite. NEAREST makes the sample an exact texel fetch.
        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        tex.repeat_x = False
        tex.repeat_y = False
        self._dyn[key] = (tex, int(max_w), int(max_h))
        self._textures[key] = tex

    def write_dyn(self, key: str, rgba: np.ndarray):
        """Write `rgba` (HxWx4 uint8) into the top-left of the pre-allocated
        texture. Returns (w, h, uv_scale_x, uv_scale_y) — the caller sets the
        sprite's uv_scale so it samples only the written sub-rect, since the
        texture is larger than the image."""
        ent = self._dyn.get(key)
        if ent is None:
            return None
        tex, mw, mh = ent
        h, w = rgba.shape[:2]
        if w > mw or h > mh:
            return None                      # caller falls back to the CPU blit
        if rgba.dtype != np.uint8:
            rgba = rgba.astype("u1")
        if not rgba.flags.c_contiguous:
            rgba = np.ascontiguousarray(rgba)
        tex.write(rgba.tobytes(), viewport=(0, 0, w, h))
        return (w, h, w / mw, h / mh)

    def has_texture(self, key: str) -> bool:
        return key in self._textures

    def release_texture(self, key: str) -> None:
        """Free a cached texture by key (storyboard LRU eviction). No-op if
        the key is absent."""
        tex = self._textures.pop(key, None)
        if tex is not None:
            try:
                tex.release()
            except Exception:  # noqa: BLE001 - context may be tearing down
                pass

    def _make_texture_rgba(self, rgba: np.ndarray,
                           mipmaps: bool = True) -> "moderngl.Texture":
        h, w = rgba.shape[:2]
        tex = self.ctx.texture((w, h), 4, rgba.tobytes())
        if mipmaps:
            tex.build_mipmaps()
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        else:
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return tex

    # --- drawing --------------------------------------------------------------

    def begin(self, clear=(0.04, 0.04, 0.06)) -> None:
        if _DRAW_COUNT:
            _DC["frames"] = _DC.get("frames", 0) + 1
        self.fbo.use()
        self.ctx.clear(*clear)

    def draw(self, sprites: list[Sprite]) -> None:
        # Fast path — no additive sprite in the list: straight-alpha painter's
        # order, exactly as before. Every existing (gameplay/HUD) sprite is
        # additive=False, so this branch is taken for all live renders and the
        # per-sprite GL state is identical to the old loop (the uv uniforms
        # resolve to `in_uv`), so output is byte-identical.
        if not any(sp.additive for sp in sprites):
            self._draw_run(sprites)
            return
        # Storyboard / glow: draw the non-additive sprites first (straight
        # alpha), then the additive ones (SRC_ALPHA, ONE) — the same two-phase
        # order the std renderer uses. The storyboard renderer already splits
        # its sprite list into maximal same-blend runs and calls draw() per
        # run, so each run lands entirely in one phase (the other is empty) and
        # back-to-front z within a blend is preserved.
        normal = [sp for sp in sprites if not sp.additive]
        additive = [sp for sp in sprites if sp.additive]
        self._draw_run(normal)
        if additive:
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
            self._draw_run(additive)
            self.ctx.blend_func = (moderngl.SRC_ALPHA,
                                   moderngl.ONE_MINUS_SRC_ALPHA)

    def draw_ordered(self, sprites: list[Sprite]) -> None:
        """Draw `sprites` in STRICT list order, flushing at every blend-mode
        change. draw() above is two-phase (every normal sprite, then every
        additive one), which is what the playfield relies on: the Argon drum
        press flashes are additive sprites in the middle of the scene list and
        must land on top of all of it. The effects the GPU path moves into
        this pass need the other thing, painter's order ACROSS blend modes
        (explosions additive, then the judgement popup in straight alpha on
        top, per lazer's z-order), so they come through here in their own
        call and the playfield's order is left exactly as production has it."""
        if not sprites:
            return
        add = (moderngl.SRC_ALPHA, moderngl.ONE)
        straight = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        run_start = 0
        cur = sprites[0].additive
        for i in range(1, len(sprites) + 1):
            if i < len(sprites) and sprites[i].additive == cur:
                continue
            if cur:
                self.ctx.blend_func = add
            if _DRAW_COUNT:
                _DC["nruns"] = _DC.get("nruns", 0) + 1
                _DC[f"runlen_{min(i - run_start, 9)}"] = \
                    _DC.get(f"runlen_{min(i - run_start, 9)}", 0) + 1
            self._draw_run(sprites[run_start:i])
            if cur:
                self.ctx.blend_func = straight
            if i < len(sprites):
                run_start = i
                cur = sprites[i].additive

    def _ensure_instanced(self):
        if getattr(self, "_i_prog", None) is not None:
            return
        self._i_prog = self.ctx.program(vertex_shader=_I_VERT,
                                        fragment_shader=_I_FRAG)
        self._i_prog["u_screen"].value = (float(self.width), float(self.height))
        for u in range(_I_MAX_TEX):
            self._i_prog[f"u_tex{u}"].value = u
        self._i_cap = 512
        self._i_alloc()

    def _i_alloc(self):
        self._i_buf = self.ctx.buffer(reserve=self._i_cap * _I_STRIDE * 4)
        self._i_vao = self.ctx.vertex_array(self._i_prog, [
            (self.vbo, "2f 2f", "in_pos", "in_uv"),
            (self._i_buf, "2f 2f 1f 4f 2f 2f 1f/i", "i_center", "i_size",
             "i_rot", "i_color", "i_uv_off", "i_uv_scale", "i_tex"),
        ])
        self._i_scratch = np.empty((self._i_cap, _I_STRIDE), dtype="f4")

    def _draw_instanced(self, sprites) -> None:
        """One draw call per blend run, order preserved within the run.

        Flushes when a 17th distinct texture appears so the 16 sampler units never
        overflow — flushing rather than reordering is what keeps painter's order.
        Per-sprite data is accumulated into a flat Python list and converted once:
        catch measured that 14 numpy scalar stores per sprite WAS the draw cost."""
        n = len(sprites)
        if n == 0:
            return
        self._ensure_instanced()
        if n > self._i_cap:
            self._i_cap = 1 << (n - 1).bit_length()
            self._i_alloc()
        arr = self._i_scratch
        flat = arr.reshape(-1)
        vals: list = []
        units: dict = {}

        def flush():
            c = len(vals) // _I_STRIDE
            if not c:
                return
            flat[:c * _I_STRIDE] = vals
            # arr[:c] is C-contiguous, so moderngl consumes it via the buffer
            # protocol — .tobytes() would allocate and copy every frame.
            self._i_buf.write(arr[:c])
            self._i_vao.render(moderngl.TRIANGLE_STRIP, instances=c)
            del vals[:]
            units.clear()

        textures = self._textures
        white = self._white
        extend = vals.extend
        for sp in sprites:
            tex = textures.get(sp.texture_key) if sp.texture_key else white
            if tex is None:
                tex = white
            u = units.get(id(tex))
            if u is None:
                if len(units) >= _I_MAX_TEX:      # 17th texture: flush, keep order
                    flush()
                u = len(units)
                units[id(tex)] = u
                tex.use(location=u)
            c0, c1, c2, c3 = sp.color
            o0, o1 = sp.uv_off
            s0, s1 = sp.uv_scale
            extend((sp.x, sp.y, sp.w, sp.h, sp.rotation,
                    c0, c1, c2, c3, o0, o1, s0, s1, u))
        flush()

    def _draw_run(self, sprites: list[Sprite]) -> None:
        # perf: hoisted locals + cached uniform objects + redundant-bind skip.
        if not sprites:
            return
        textures = self._textures
        white = self._white
        u_color, u_center = self._u_color, self._u_center
        u_size, u_rot = self._u_size, self._u_rot
        u_uv_off, u_uv_scale = self._u_uv_off, self._u_uv_scale
        render = self.vao.render
        if _INSTANCED:
            if _DRAW_COUNT:
                _DC["runs"] = _DC.get("runs", 0) + 1
                _DC["sprites"] = _DC.get("sprites", 0) + len(sprites)
            self._draw_instanced(sprites)
            return
        strip = moderngl.TRIANGLE_STRIP
        prev_tex = None
        prev_uv_off = None
        prev_uv_scale = None
        if _DRAW_COUNT:
            _DC["runs"] = _DC.get("runs", 0) + 1
            _DC["sprites"] = _DC.get("sprites", 0) + len(sprites)
            _DC["texsw"] = _DC.get("texsw", 0)
        for sp in sprites:
            tex = textures.get(sp.texture_key) if sp.texture_key else white
            if tex is None:
                tex = white
            if tex is not prev_tex:
                tex.use(location=0)
                prev_tex = tex
                if _DRAW_COUNT:
                    _DC["texsw"] = _DC.get("texsw", 0) + 1
            u_color.value = sp.color
            u_center.value = (sp.x, sp.y)
            u_size.value = (sp.w, sp.h)
            u_rot.value = sp.rotation
            # rebind uv only when it changes (defaults for every gameplay
            # sprite, so this fires once per run and never mutates the raster)
            if sp.uv_off != prev_uv_off:
                u_uv_off.value = sp.uv_off
                prev_uv_off = sp.uv_off
            if sp.uv_scale != prev_uv_scale:
                u_uv_scale.value = sp.uv_scale
                prev_uv_scale = sp.uv_scale
            render(strip)

    _PBO_RING = 3

    def _ensure_yuv(self):
        """Lazily build the RGB->yuv420p conversion pass. Built on first use so an
        unused flag costs nothing (two programs, three textures, ~3 MB of FBOs)."""
        if getattr(self, "_yuv_ready", False):
            return
        w, h = self.width, self.height
        if (w & 1) or (h & 1) or h < 12:
            # under 12 rows swscale shortens its chroma filter instead of
            # clamping at the edges, and this conversion stops being its twin
            raise RuntimeError(
                f"R3D_TAIKO_GPU_YUV needs even dimensions and at least 12 rows, "
                f"got {w}x{h}")
        (RY, GY, BY), (RU, GU, BU), (RV, GV, BV) = _yuv_matrix()
        vert = ("#version 330\nin vec2 in_pos;\n"
                "void main(){ gl_Position = vec4(in_pos,0.0,1.0); }")
        # INTEGER math throughout (see rgb_to_yuv420p, the same arithmetic): the
        # shifts must be exact. Texture samples come back as normalised floats,
        # so `int(v*255.0 + 0.5)` recovers the byte -- float32 holds 0..255
        # exactly, and the +0.5 stops the n/255*255 round-trip landing a hair
        # under n and truncating to n-1. Every sum stays inside int32: the
        # largest is the chroma accumulator, under 1.6e8.
        frag_y = f"""#version 330
        uniform sampler2D scene;
        uniform bool u_flip;     // source rows are bottom-first: read them mirrored
        out float outY;
        void main() {{
            ivec2 p = ivec2(gl_FragCoord.xy);
            if (u_flip) p.y = textureSize(scene, 0).y - 1 - p.y;
            vec3 c = texelFetch(scene, p, 0).rgb;
            int r = int(c.r*255.0+0.5), g = int(c.g*255.0+0.5), b = int(c.b*255.0+0.5);
            outY = float(((((({RY}*r + {GY}*g + {BY}*b) + {0x801 << 8}) >> 9) + 32) >> 6)) / 255.0;
        }}"""
        # U and V share the gather (two pixels across, eight rows down), so one
        # pass with two attachments does both.
        # u_flip: the planes are always written top row first (output row 0 is the
        # picture's top row). The scene texture's row 0 is the picture's BOTTOM
        # row, so the shaders read it mirrored -- which also makes the separate
        # full-screen flip pass unnecessary on this path. A CPU frame
        # (yuv_from_rgb) is top-first and is read as it lies.
        frag_uv = f"""#version 330
        uniform sampler2D scene;
        uniform bool u_flip;
        layout(location=0) out float outU;
        layout(location=1) out float outV;
        const int TAP[8] = int[8]({", ".join(str(t) for t in _CHROMA_TAPS)});
        void main() {{
            ivec2 p = ivec2(gl_FragCoord.xy);
            int x = p.x * 2;
            int ymax = textureSize(scene, 0).y - 1;
            int su = 0, sv = 0;
            for (int k = 0; k < 8; ++k) {{
                int y = clamp(p.y * 2 - 3 + k, 0, ymax);   // PICTURE row, top = 0
                if (u_flip) y = ymax - y;
                ivec3 s = ivec3(texelFetch(scene, ivec2(x, y), 0).rgb * 255.0 + 0.5)
                        + ivec3(texelFetch(scene, ivec2(x + 1, y), 0).rgb * 255.0 + 0.5);
                su += TAP[k] * ((({RU}*s.r + {GU}*s.g + {BU}*s.b) + {0x4001 << 9}) >> 10);
                sv += TAP[k] * ((({RV}*s.r + {GV}*s.g + {BV}*s.b) + {0x4001 << 9}) >> 10);
            }}
            outU = float(clamp((su + {1 << 18}) >> 19, 0, 255)) / 255.0;
            outV = float(clamp((sv + {1 << 18}) >> 19, 0, 255)) / 255.0;
        }}"""
        self._yuv_quad = self.ctx.buffer(
            np.array([-1, -1, 3, -1, -1, 3], "f4").tobytes())
        self._yuv_prog_y = self.ctx.program(vertex_shader=vert, fragment_shader=frag_y)
        self._yuv_prog_uv = self.ctx.program(vertex_shader=vert, fragment_shader=frag_uv)
        self._yuv_vao_y = self.ctx.vertex_array(
            self._yuv_prog_y, [(self._yuv_quad, "2f4", "in_pos")])
        self._yuv_vao_uv = self.ctx.vertex_array(
            self._yuv_prog_uv, [(self._yuv_quad, "2f4", "in_pos")])
        self._tex_y = self.ctx.texture((w, h), 1, dtype="f1")
        self._tex_u = self.ctx.texture((w // 2, h // 2), 1, dtype="f1")
        self._tex_v = self.ctx.texture((w // 2, h // 2), 1, dtype="f1")
        self._fbo_y = self.ctx.framebuffer(color_attachments=[self._tex_y])
        self._fbo_uv = self.ctx.framebuffer(
            color_attachments=[self._tex_u, self._tex_v])
        self._yuv_size = w * h * 3 // 2
        self._yuv_ready = True

    def _convert_yuv(self):
        """Run the conversion. Blending MUST be off: these passes write computed
        values, not composites, and a stray blend would silently corrupt the planes."""
        self._ensure_yuv()
        self.ctx.disable(moderngl.BLEND)
        self._scene_tex.use(location=0)
        self._yuv_prog_y["scene"] = 0
        self._yuv_prog_uv["scene"] = 0
        self._yuv_prog_y["u_flip"].value = True       # the scene is bottom-first
        self._yuv_prog_uv["u_flip"].value = True
        self._fbo_y.use()
        self._yuv_vao_y.render(moderngl.TRIANGLES)
        self._fbo_uv.use()
        self._yuv_vao_uv.render(moderngl.TRIANGLES)
        self.ctx.enable(moderngl.BLEND)

    def _readback_ring(self, size: int, n: int) -> list:
        """The n buffers frames are read back into, `size` bytes each.

        R3D_TAIKO_STREAM_READ (on by default on a Mac): declare them
        GL_STREAM_READ. ctx.buffer(reserve=...) asks for GL_STATIC_DRAW ("the
        application fills it once, the GPU draws from it"); a readback buffer
        is the opposite, the GPU writes it and the application reads it once.
        On macOS that declaration decides what a map costs: every
        glMapBufferRange of a STATIC_DRAW buffer took ~0.46 ms (1.1 s over
        the 2,389-frame self-test), with the GPU already done; declared
        STREAM_READ it returns at once. Same bytes either way, only the usage
        hint changes. Found in mania first (its PR #37)."""
        bufs = [self.ctx.buffer(reserve=size) for _ in range(n)]
        if _STREAM_READ:
            try:
                g = _load_gl_c()
                for b in bufs:
                    g.glBindBuffer(_GL_PIXEL_PACK_BUFFER, b.glo)
                    g.glBufferData(_GL_PIXEL_PACK_BUFFER, size, None, _GL_STREAM_READ)
                g.glBindBuffer(_GL_PIXEL_PACK_BUFFER, 0)
            except Exception:  # noqa: BLE001 - a hint, never the ring
                pass
        return bufs

    def read_yuv_async(self) -> "np.ndarray | None":
        """yuv420p twin of read_rgb_async: same PBO ring, same FIFO ordering, but
        1.5 bytes/px instead of 3. Returns a flat uint8 view (Y | U | V) ready to pipe.

        The three planes are read into ONE buffer at their yuv420p offsets via
        read_into's write_offset, so the writer still pushes a single contiguous block
        and nothing downstream has to know it is planar."""
        self._ensure_yuv()
        w, h = self.width, self.height
        ysz, csz = w * h, (w // 2) * (h // 2)
        if self._pbos is None:
            self._pbo_size = self._yuv_size
            self._lat = _PBO_LAT
            n = ((_WRITER_QUEUE_FRAMES + 1 + 1 + self._lat + _PBO_MARGIN)
                 if _MAP_READBACK else self._PBO_RING)
            self._pbos = self._readback_ring(self._yuv_size, n)
            self._mapped = [False] * n
        self._convert_yuv()
        buf = self._unmap_for_write(self._pbo_head % len(self._pbos))
        self._fbo_y.read_into(buf, components=1, alignment=1, write_offset=0)
        self._fbo_uv.read_into(buf, components=1, alignment=1, attachment=0,
                               write_offset=ysz)
        self._fbo_uv.read_into(buf, components=1, alignment=1, attachment=1,
                               write_offset=ysz + csz)
        self._pbo_head += 1
        if self._pbo_head - self._pbo_tail < self._lat:
            return None
        return self._pop_pbo_flat()

    def _pop_pbo_flat(self) -> "np.ndarray":
        """_pop_pbo's flat sibling — yuv420p is planar and not (h, w, c)-shaped."""
        import ctypes as _ct
        idx = self._pbo_tail % len(self._pbos)
        buf = self._pbos[idx]
        self._pbo_tail += 1
        if _MAP_READBACK:
            g = _load_gl_c()
            g.glBindBuffer(_GL_PIXEL_PACK_BUFFER, buf.glo)
            ptr = g.glMapBufferRange(_GL_PIXEL_PACK_BUFFER, 0, self._pbo_size,
                                     _GL_MAP_READ_BIT | _GL_MAP_WRITE_BIT)
            g.glBindBuffer(_GL_PIXEL_PACK_BUFFER, 0)
            if not ptr:
                raise RuntimeError("glMapBufferRange returned NULL")
            self._mapped[idx] = True
            return np.ctypeslib.as_array(
                (_ct.c_uint8 * self._pbo_size).from_address(ptr))
        arr = np.empty(self._pbo_size, dtype="u1")
        buf.read_into(arr)
        return arr

    def yuv_from_rgb(self, rgb) -> "np.ndarray | None":
        """Convert a CPU-composited RGB frame to yuv420p on the GPU.

        For the OUTRO/RESULTS frames, which are built on the CPU and never pass
        through the scene texture. The pure-numpy twin (rgb_to_yuv420p) measured
        **14.01 ms/frame** -- on a short map that is ~320 frames and cost 4.5 s, which
        swamped everything the GPU path had just won. Same shaders, so the result is
        identical to both the gameplay path and to swscale.
        """
        self._ensure_yuv()
        h, w = rgb.shape[:2]
        if (w, h) != (self.width, self.height):
            raise RuntimeError(f"yuv_from_rgb: expected {self.width}x{self.height}, "
                               f"got {w}x{h}")
        if not rgb.flags.c_contiguous:
            rgb = np.ascontiguousarray(rgb)
        if getattr(self, "_yuv_stage", None) is None:
            self._yuv_stage = self.ctx.texture((w, h), 3)
            self._yuv_stage.filter = (moderngl.NEAREST, moderngl.NEAREST)
        # write the array DIRECTLY (buffer protocol) -- .tobytes() was copying
        # 6.2 MB per frame before the upload even started.
        self._yuv_stage.write(memoryview(rgb))
        self.ctx.disable(moderngl.BLEND)
        # bind the STAGING texture where the passes expect the scene
        self._yuv_stage.use(location=0)
        self._yuv_prog_y["scene"] = 0
        self._yuv_prog_uv["scene"] = 0
        self._yuv_prog_y["u_flip"].value = False      # a CPU frame is top-first
        self._yuv_prog_uv["u_flip"].value = False
        self._fbo_y.use()
        self._yuv_vao_y.render(moderngl.TRIANGLES)
        self._fbo_uv.use()
        self._yuv_vao_uv.render(moderngl.TRIANGLES)
        self.ctx.enable(moderngl.BLEND)
        # Queue into the SAME PBO ring the gameplay path uses, instead of reading
        # synchronously. A blocking read here cost 4.27 ms/frame and made GPU_YUV a
        # 6.5% LOSS on a 26 s map (318 of 2076 frames are outro) while being a 42.7%
        # win on a 248 s one. The stall was the sync, not the copies -- removing the
        # allocations only moved it 4.54 -> 4.27. Returns None while the ring fills;
        # the caller drains the tail with read_yuv_drain().
        ysz, csz = w * h, (w // 2) * (h // 2)
        if self._pbos is None:
            self._pbo_size = self._yuv_size
            self._lat = _PBO_LAT
            n = ((_WRITER_QUEUE_FRAMES + 1 + 1 + self._lat + _PBO_MARGIN)
                 if _MAP_READBACK else self._PBO_RING)
            self._pbos = self._readback_ring(self._yuv_size, n)
            self._mapped = [False] * n
        buf = self._unmap_for_write(self._pbo_head % len(self._pbos))
        self._fbo_y.read_into(buf, components=1, alignment=1, write_offset=0)
        self._fbo_uv.read_into(buf, components=1, alignment=1, attachment=0,
                               write_offset=ysz)
        self._fbo_uv.read_into(buf, components=1, alignment=1, attachment=1,
                               write_offset=ysz + csz)
        self._pbo_head += 1
        if self._pbo_head - self._pbo_tail < self._lat:
            return None
        return self._pop_pbo_flat()

    def read_yuv_drain(self) -> list:
        """Flush every frame still in the ring, oldest first."""
        out = []
        while self._pbos is not None and self._pbo_tail < self._pbo_head:
            out.append(self._pop_pbo_flat())
        return out

    def read_rgb_async(self) -> "np.ndarray | None":
        """Queue an async readback of the current fbo into a small PBO
        ring and return the OLDEST completed frame (top-left origin), or
        None while the ring is still filling. Frames come back in strict
        submission order — the render loop pushes them straight to ffmpeg,
        so the byte stream is identical to the synchronous read_rgb path,
        just ~RING-1 frames late. read_drain() flushes the tail. (Ported
        from the std renderer's proven osu_std_renderer/render/gl.py.)"""
        if self._pbos is None:
            size = self.width * self.height * _NC
            self._pbo_size = size
            # READBACK LATENCY: how many frames are queued before we map the
            # oldest. glMapBufferRange BLOCKS until the GPU has finished writing
            # that buffer, so this is the knob that decides whether
            # `readback_block` is paying for a GPU wait or not. Frame ORDER and
            # CONTENT are unaffected -- the ring is strict FIFO and read_drain
            # flushes the tail -- so a deeper pipeline only buffers more frames.
            # R3D_TAIKO_PBO_LAT overrides it; the pool below grows to match.
            self._lat = _PBO_LAT
            # Mapped readback hands out a numpy VIEW onto the PBO, which the
            # compositors mutate and the writer thread pipes. The slot must not
            # be reused until the writer is done with it, so the pool has to be
            # deeper than everything that can hold a frame at once:
            #   writer queue (4) + in-write (1) + in-hand (1) + latency (3)
            # Catch measured that getting this wrong CHANGES THE OUTPUT (its
            # ring=12 vs pool mismatch altered the mp4 md5) — it is a silent
            # corruption, not a crash.
            # Pool depth is DERIVED from the holder budget, not chosen. Every slot
            # that can be occupied at once must exist, or a slot gets reused while
            # something still holds a view into it -- catch proved that silently
            # CHANGES THE OUTPUT (its ring=12 vs pool mismatch altered the mp4 md5).
            # Deriving it means R3D_TAIKO_PBO_LAT cannot drift away from the depth:
            # raising latency raises the pool by the same amount automatically.
            # At the default _lat=3 this evaluates to 4+1+1+3+3 = 12, which is
            # exactly the value it replaces.
            n = ((_WRITER_QUEUE_FRAMES + 1 + 1 + self._lat + _PBO_MARGIN)
                 if _MAP_READBACK else self._PBO_RING)
            self._pbos = self._readback_ring(size, n)
            self._mapped = [False] * n
        # GPU y-flip pass: mirror the scene into _flip_fbo (exact texel copy,
        # blending off), then queue the async read from THAT — the PBO then
        # holds top-left-origin rows directly.
        self.ctx.disable(moderngl.BLEND)
        self._flip_fbo.use()
        self._scene_tex.use(location=0)
        self._flip_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.enable(moderngl.BLEND)
        buf = self._unmap_for_write(self._pbo_head % len(self._pbos))
        self._flip_fbo.read_into(buf, components=_NC, alignment=1)
        self._pbo_head += 1
        if self._pbo_head - self._pbo_tail < self._lat:
            return None
        return self._pop_pbo()

    def _unmap_for_write(self, idx):
        """A MAPPED buffer cannot receive glReadPixels, so unmap on reuse. By the
        time the ring wraps back to this index the writer is long done with it,
        which is exactly what the pool depth above guarantees."""
        buf = self._pbos[idx]
        if _MAP_READBACK and self._mapped[idx]:
            g = _load_gl_c()
            g.glBindBuffer(_GL_PIXEL_PACK_BUFFER, buf.glo)
            g.glUnmapBuffer(_GL_PIXEL_PACK_BUFFER)
            g.glBindBuffer(_GL_PIXEL_PACK_BUFFER, 0)
            self._mapped[idx] = False
        return buf

    def _pop_pbo(self) -> np.ndarray:
        if _MAP_READBACK:
            import ctypes as _ct
            idx = self._pbo_tail % len(self._pbos)
            buf = self._pbos[idx]
            self._pbo_tail += 1
            g = _load_gl_c()
            g.glBindBuffer(_GL_PIXEL_PACK_BUFFER, buf.glo)
            ptr = g.glMapBufferRange(_GL_PIXEL_PACK_BUFFER, 0, self._pbo_size,
                                     _GL_MAP_READ_BIT | _GL_MAP_WRITE_BIT)
            g.glBindBuffer(_GL_PIXEL_PACK_BUFFER, 0)
            if not ptr:
                raise RuntimeError("glMapBufferRange returned NULL")
            self._mapped[idx] = True
            # WRITE bit as well as READ: the compositors mutate the frame in
            # place. The mapped pointer IS the host buffer, so the 6.22 MB copy
            # disappears rather than being made faster. Catch measured
            # glGetBufferSubData 0.663 ms vs glMapBufferRange 0.316 ms for 8.29 MB,
            # with bytes verified identical through both paths.
            arr = np.ctypeslib.as_array(
                (_ct.c_uint8 * self._pbo_size).from_address(ptr))
            return arr.reshape((self.height, self.width, _NC))
        buf = self._pbos[self._pbo_tail % len(self._pbos)]
        self._pbo_tail += 1
        # read straight into a fresh WRITABLE, CONTIGUOUS array (perf: skips
        # the bytes allocation of buf.read(); rows are already top-left origin
        # thanks to the GPU flip pass, so the compositors mutate it in place
        # and the writer thread pipes it zero-copy; byte stream unchanged).
        arr = np.empty((self.height, self.width, 3), dtype="u1")
        buf.read_into(arr)
        return arr

    def read_drain(self) -> list:
        """Return every frame still in flight, oldest first (map end or
        the gameplay->outro boundary)."""
        out = []
        while self._pbos is not None and self._pbo_tail < self._pbo_head:
            out.append(self._pop_pbo())
        return out

    def read_rgb(self) -> np.ndarray:
        """Return HxWx3 uint8, top-left origin (ready for ffmpeg rgb24).

        Note: a 3-component read is faster end-to-end than reading RGBA and
        dropping alpha — the channel-drop forces a strided copy that costs more
        than the faster aligned transfer saves."""
        data = self.fbo.read(components=_NC, alignment=1)
        arr = np.frombuffer(data, dtype="u1").reshape((self.height, self.width, _NC))
        # moderngl reads bottom-left origin; flip to top-left (view; copied once
        # downstream where the frame is made contiguous).
        return np.flipud(arr)

    def release(self) -> None:
        try:
            self.ctx.release()
        except Exception:  # noqa: BLE001
            pass
