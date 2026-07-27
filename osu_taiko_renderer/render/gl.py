"""Minimal moderngl sprite batch for the catch renderer.

Owns a standalone EGL context and an offscreen RGBA framebuffer. Draws
textured/solid quads with straight-alpha blending in painter's order, then
reads back tightly-packed RGB24 for the ffmpeg pipe. Deliberately tiny and
self-contained so it can be discarded when the VRender branch takes over.
"""
from __future__ import annotations

import numpy as np

try:
    import moderngl
except Exception as e:  # noqa: BLE001
    raise RuntimeError("moderngl is required for the catch renderer") from e

from osu_taiko_renderer.beatmap.models import Sprite

_VERT = """
#version 330
in vec2 in_pos;      // unit quad corner [-0.5,0.5]
in vec2 in_uv;
uniform vec2 u_screen;   // (w, h) in px
uniform vec2 u_center;   // sprite center in px (origin top-left)
uniform vec2 u_size;     // sprite w,h in px
uniform float u_rot;     // radians
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
    v_uv = in_uv;
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


class SpriteRenderer:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        # Honor R3D_EGL_DEVICE_INDEX so renders pin to the right GPU (pool
        # isolation: e.g. 1070=index 1 for Pool B). EGL ignores
        # CUDA_VISIBLE_DEVICES, so the device must be selected explicitly.
        import os
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
        self._white = self._make_texture_rgba(np.full((1, 1, 4), 255, dtype="u1"))

        # async PBO ring state (read_rgb_async / read_drain) — ported from
        # the std renderer's proven pipeline (osu_std_renderer/render/gl.py)
        self._pbos: list["moderngl.Buffer"] | None = None
        self._pbo_head = 0
        self._pbo_tail = 0

    # --- texture management ---------------------------------------------------

    def upload_texture(self, key: str, rgba: np.ndarray) -> None:
        """rgba: HxWx4 uint8 array (top-left origin)."""
        if rgba.dtype != np.uint8:
            rgba = rgba.astype("u1")
        if rgba.shape[2] == 3:
            a = np.full(rgba.shape[:2] + (1,), 255, dtype="u1")
            rgba = np.concatenate([rgba, a], axis=2)
        self._textures[key] = self._make_texture_rgba(rgba)

    def has_texture(self, key: str) -> bool:
        return key in self._textures

    def _make_texture_rgba(self, rgba: np.ndarray) -> "moderngl.Texture":
        h, w = rgba.shape[:2]
        tex = self.ctx.texture((w, h), 4, rgba.tobytes())
        tex.build_mipmaps()
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        return tex

    # --- drawing --------------------------------------------------------------

    def begin(self, clear=(0.04, 0.04, 0.06)) -> None:
        self.fbo.use()
        self.ctx.clear(*clear)

    def draw(self, sprites: list[Sprite]) -> None:
        # perf: hoisted locals + cached uniform objects + redundant-bind skip.
        # Per-draw GL state is identical to the old loop, so output is
        # unchanged; only the Python-side overhead per sprite shrinks.
        textures = self._textures
        white = self._white
        u_color, u_center = self._u_color, self._u_center
        u_size, u_rot = self._u_size, self._u_rot
        render = self.vao.render
        strip = moderngl.TRIANGLE_STRIP
        prev_tex = None
        for sp in sprites:
            tex = textures.get(sp.texture_key) if sp.texture_key else white
            if tex is None:
                tex = white
            if tex is not prev_tex:
                tex.use(location=0)
                prev_tex = tex
            u_color.value = sp.color
            u_center.value = (sp.x, sp.y)
            u_size.value = (sp.w, sp.h)
            u_rot.value = sp.rotation
            render(strip)

    _PBO_RING = 3

    def read_rgb_async(self) -> "np.ndarray | None":
        """Queue an async readback of the current fbo into a small PBO
        ring and return the OLDEST completed frame (top-left origin), or
        None while the ring is still filling. Frames come back in strict
        submission order — the render loop pushes them straight to ffmpeg,
        so the byte stream is identical to the synchronous read_rgb path,
        just ~RING-1 frames late. read_drain() flushes the tail. (Ported
        from the std renderer's proven osu_std_renderer/render/gl.py.)"""
        if self._pbos is None:
            size = self.width * self.height * 3
            self._pbos = [self.ctx.buffer(reserve=size)
                          for _ in range(self._PBO_RING)]
        # GPU y-flip pass: mirror the scene into _flip_fbo (exact texel copy,
        # blending off), then queue the async read from THAT — the PBO then
        # holds top-left-origin rows directly.
        self.ctx.disable(moderngl.BLEND)
        self._flip_fbo.use()
        self._scene_tex.use(location=0)
        self._flip_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.enable(moderngl.BLEND)
        buf = self._pbos[self._pbo_head % len(self._pbos)]
        self._flip_fbo.read_into(buf, components=3, alignment=1)
        self._pbo_head += 1
        if self._pbo_head - self._pbo_tail < len(self._pbos):
            return None
        return self._pop_pbo()

    def _pop_pbo(self) -> np.ndarray:
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
        data = self.fbo.read(components=3, alignment=1)
        arr = np.frombuffer(data, dtype="u1").reshape((self.height, self.width, 3))
        # moderngl reads bottom-left origin; flip to top-left (view; copied once
        # downstream where the frame is made contiguous).
        return np.flipud(arr)

    def release(self) -> None:
        try:
            self.ctx.release()
        except Exception:  # noqa: BLE001
            pass
