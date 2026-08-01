"""Storyboard RENDERER — phase 3 of the in-house storyboard system.

Turns the phase-2 command engine's per-time :class:`SpriteState` output into
drawn frames, into the SAME headless-EGL framebuffer the rest of the scene
uses (``render/gl.py`` ``SpriteRenderer``; NO second GL context).

Everything positional/blending is a faithful port of osu!(lazer) master
(2026-07); the sources are cited inline.  Key facts, with the exact source:

* 640x480 -> output transform (``DrawableStoryboard`` +
  ``DrawableStoryboardLayer.LayerElementContainer``):

    - ``DrawableStoryboard``: ``Size = (640, 480)``; ``DrawScale =
      Parent.DrawHeight / 480`` (uniform, fills the output HEIGHT);
      ``Anchor = Origin = Centre``; ``Width = Height * (widescreen ||
      onlyVideo ? 16/9 : 4/3)`` — the widescreen flag only widens the
      *visible* box, it does NOT change the sprite coordinate frame.
    - ``LayerElementContainer``: ``Size = (640, 480)``, ``Anchor = Origin =
      Centre`` — so sprite coords live in a FIXED 640x480 box centred in the
      (wider) widescreen viewport.  Hence, for every layer:

          k        = out_h / 480                       (uniform scale)
          screen_x = out_w/2 + (sb_x - 320) * k
          screen_y = out_h/2 + (sb_y - 240) * k

      The visible/masking box is ``vis_w = (widescreen?853.333:640) * k``
      wide, full height, centred — a 4:3 storyboard on a 16:9 output is
      pillar-boxed; a widescreen one fills the width.  ``DrawableStoryboardLayer.Masking``
      (True for every non-Video layer) clips to that box; we reproduce it
      with a GL scissor.

* Per-sprite geometry (``DrawableStoryboardSprite`` / ...Animation):

    - ``DrawScale = (flipH?-1:1)*Scale.X, (flipV?-1:1)*Scale.Y) * VectorScale``
      and ``Origin = StoryboardExtensions.AdjustOrigin(base.Origin,
      VectorScale, flipH, flipV)``.  The framework maps a local point ``p``
      to ``Position + Rotate(finalScale (.) (p - originLocal), rotation)``.
      We solve that for the sprite CENTRE and feed a positive-size, rotated,
      UV-flipped quad to ``SpriteRenderer`` (algebraically identical — see
      :func:`compute_quad`).  ``AdjustOrigin`` swaps left<->right / top<->bottom
      anchors when the axis is net-mirrored (``flip ^ (vectorScale < 0)``);
      we mirror the CONTENT via a UV flip instead of a negative GL size.
    - ``Update``: ``if (Alpha > 1) Alpha %= 1`` — already applied by the
      engine; alpha <= 0 is invisible (we skip it).

* Additive: ``AddBlendingParameters(..., Additive, ...)`` == framework
  ``BlendingParameters.Additive`` == ``(SrcAlpha, One)`` == ``gl.py``'s
  additive pass.  We draw sprites in strict back-to-front order, splitting
  into maximal same-blend runs so an additive sprite blends in its OWN
  z-slot (NOT hoisted last like the gameplay hit-explosions).

* Layer z-order vs the playfield (``Screens/Play/Player.cs`` +
  ``DimmableStoryboard.cs``): the WHOLE ``DimmableStoryboard`` is a
  ``createUnderlayComponents`` child — i.e. BEHIND the ``DrawableRuleset``
  (playfield/hitobjects/cursor) — EXCEPT the ``Overlay`` layer, which is
  proxied via ``OverlayLayerContainer.CreateProxy()`` into
  ``createOverlayComponents`` ABOVE the ruleset and just BELOW ``HUDOverlay``.
  So: Background/Fail/Pass/Foreground draw under gameplay; Overlay draws over
  gameplay, under the HUD.  ``scene.py`` composites accordingly.

* Dim: ``DimmableStoryboard`` is a ``UserDimContainer`` — it takes the SAME
  background dim envelope (§4.10), so the storyboard is tinted by
  ``1 - DimEnvelope.level(t)`` (the beat-flash is bg-only and NOT applied).

Deferred / stubbed (see the phase-3 report):
  * storyboard SAMPLES — no audio playback yet (engine excludes them anyway).
  * storyboard VIDEO elements — handled by the existing ``video_bg.py`` path,
    not here.
  * ``@2x`` texture display-size halving — dims are the raw pixel dims.
  * pass/fail layer switching by live health — we render the PASSING set
    (Background/Pass/Foreground/Overlay, Fail hidden); the frozen fail path
    does not draw the storyboard.  Both are LOUD-noted, not silent.
  * trigger firing is still stubbed in the engine (hitsound/pass-fail events).
"""
from __future__ import annotations

import logging
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ..beatmap.storyboard import Origin, StoryboardAnimation, LoopType
from .gl import Sprite

log = logging.getLogger(__name__)

__all__ = ["StoryboardRenderer", "compute_quad", "anim_frame_index",
           "ORIGIN_ANCHOR", "storyboard_viewport"]

# Sprite coordinate frame + the widescreen aspect ratios (DrawableStoryboard).
SB_W = 640.0
SB_H = 480.0
_WIDE_ASPECT = 16.0 / 9.0
_NARROW_ASPECT = 4.0 / 3.0

# Origin enum -> (ax, ay) anchor fraction: ax 0=left .5=centre 1=right;
# ay 0=top .5=centre 1=bottom.  LegacyOrigins.cs order; Custom -> TopLeft.
ORIGIN_ANCHOR = {
    Origin.TOP_LEFT: (0.0, 0.0),
    Origin.CENTRE: (0.5, 0.5),
    Origin.CENTRE_LEFT: (0.0, 0.5),
    Origin.TOP_RIGHT: (1.0, 0.0),
    Origin.BOTTOM_CENTRE: (0.5, 1.0),
    Origin.TOP_CENTRE: (0.5, 0.0),
    Origin.CUSTOM: (0.0, 0.0),
    Origin.CENTRE_RIGHT: (1.0, 0.5),
    Origin.BOTTOM_LEFT: (0.0, 1.0),
    Origin.BOTTOM_RIGHT: (1.0, 1.0),
}

_IMAGE_EXTS = (".png", ".jpg", ".jpeg")


# --------------------------------------------------------------------------- #
# Pure geometry (unit-tested; no GL)                                            #
# --------------------------------------------------------------------------- #

def storyboard_viewport(out_w, out_h, widescreen):
    """The visible/masking box of the storyboard in output pixels, as
    ``(x0, y0, w, h)`` with a bottom-left origin (ready for a GL scissor).
    Fills the output HEIGHT; width = (16:9 | 4:3) * height, centred."""
    k = out_h / SB_H
    vis_w = (_WIDE_ASPECT if widescreen else _NARROW_ASPECT) * SB_H * k
    x0 = out_w / 2.0 - vis_w / 2.0
    xi = max(0, int(round(x0)))
    wi = min(int(out_w) - xi, int(round(vis_w)))
    return (xi, 0, max(0, wi), int(out_h))


def compute_quad(origin, x, y, scale_x, scale_y, rotation, flip_h, flip_v,
                 tw, th, k, cx, cy):
    """Map a storyboard sprite to a positive-size rotated quad for
    ``gl.Sprite``.  Exact solve of the framework transform ``screen =
    Position + Rotate(finalScale (.) (p - originLocal_adjusted), rot)`` for
    the sprite centre (see module docstring).

    ``scale_x/scale_y`` are the engine's S*V per-axis magnitudes (may be
    negative when V is negative); ``flip_h/flip_v`` are the P-command
    booleans.  ``tw/th`` texture pixel size; ``k`` the 640x480->screen
    uniform scale; ``cx/cy`` the screen centre.

    Returns ``(scr_cx, scr_cy, w, h, theta_rad, mirror_x, mirror_y)`` — w/h
    positive screen px, mirror_* meaning "flip the texture UV on that axis".
    ``theta`` is clockwise radians (matches gl.py's Y-down rotation)."""
    # finalScale folds the flip booleans onto the (possibly-negative) V scale.
    fsx = -scale_x if flip_h else scale_x
    fsy = -scale_y if flip_v else scale_y
    mirror_x = fsx < 0.0
    mirror_y = fsy < 0.0

    ax, ay = ORIGIN_ANCHOR.get(origin, (0.0, 0.0))
    # StoryboardExtensions.AdjustOrigin: net-mirror swaps the anchor edge.
    if mirror_x:
        ax = 1.0 - ax
    if mirror_y:
        ay = 1.0 - ay

    # centre relative to the (adjusted) origin point, in the sprite's local
    # (unrotated) frame, already scaled+signed by finalScale.
    off_x = fsx * (0.5 - ax) * tw
    off_y = fsy * (0.5 - ay) * th

    theta = math.radians(rotation)
    c = math.cos(theta)
    s = math.sin(theta)
    # rotate the offset about the origin point (clockwise, Y-down).
    rox = off_x * c - off_y * s
    roy = off_x * s + off_y * c

    sb_cx = x + rox
    sb_cy = y + roy
    scr_cx = cx + (sb_cx - 320.0) * k
    scr_cy = cy + (sb_cy - 240.0) * k
    w = abs(fsx) * tw * k
    h = abs(fsy) * th * k
    return (scr_cx, scr_cy, w, h, theta, mirror_x, mirror_y)


def anim_frame_index(playback_pos, frame_count, frame_delay, loop_forever):
    """Frame index at ``playback_pos`` (ms since EarliestTransformTime).

    Port of osu.Framework ``TextureAnimation`` with uniform per-frame
    ``frame_delay``: frame i spans ``[i*delay, (i+1)*delay)``.  LoopForever
    wraps modulo the total duration; LoopOnce clamps to the last frame.
    Guards degenerate (<=1 frame, non-positive delay) to frame 0."""
    n = int(frame_count)
    if n <= 1 or frame_delay <= 0.0:
        return 0
    if playback_pos <= 0.0:
        return 0
    if loop_forever:
        total = n * frame_delay
        pos = playback_pos % total
        return min(int(pos / frame_delay), n - 1)
    return min(int(playback_pos / frame_delay), n - 1)


# --------------------------------------------------------------------------- #
# Texture cache (lazy LRU, alive-set driven)                                    #
# --------------------------------------------------------------------------- #

class _TextureCache:
    """Lazy LRU over ``SpriteRenderer`` textures.  Keyed by the resolved
    (case-insensitive) map-folder-relative path so sprites sharing an image
    upload it ONCE.  Only textures used this frame are uploaded; when the
    resident byte total tops the budget, least-recently-used textures NOT
    needed this frame are evicted.  Never evicts a texture the current frame
    needs (would corrupt the frame) — if the frame alone tops the budget it
    LOUDLY warns instead of dropping sprites."""

    def __init__(self, spr, folder: Path, budget_mb: int = 1024):
        self.spr = spr
        self.folder = Path(folder)
        self.budget = int(budget_mb) * 1024 * 1024
        self._index = _build_file_index(self.folder)     # relpath.lower -> Path
        self._base = _build_basename_index(self._index)  # basename.lower -> Path
        self._resolved: dict[str, str | None] = {}       # sb path -> gl key|None
        self._path_for: dict[str, Path] = {}             # gl key -> disk Path
        self._dims: dict[str, tuple[int, int]] = {}      # gl key -> (w, h)
        self._resident: "OrderedDict[str, int]" = OrderedDict()  # gl key -> bytes
        self._total = 0
        self._missing_logged: set[str] = set()
        self._over_budget_logged = False
        self.uploads = 0
        self.evictions = 0
        self.peak_bytes = 0

    def _resolve(self, sb_path: str) -> str | None:
        """sb_path (parser-cleaned, '/'-separated) -> a gl texture key, or
        None if the image is absent/undecodable.  Cached."""
        if sb_path in self._resolved:
            return self._resolved[sb_path]
        p = self._lookup(sb_path)
        gl_key = ("sbtex:" + sb_path.lower()) if p is not None else None
        self._resolved[sb_path] = gl_key
        if p is not None:
            self._path_for[gl_key] = p
        elif sb_path not in self._missing_logged:
            self._missing_logged.add(sb_path)
            log.warning("storyboard: image not found, skipping: %r", sb_path)
        return gl_key

    def _lookup(self, sb_path: str) -> Path | None:
        norm = sb_path.replace("\\", "/").lower()
        p = self._index.get(norm)
        if p is not None:
            return p
        # extensionless reference -> try the image extensions
        if "." not in norm.rsplit("/", 1)[-1]:
            for ext in _IMAGE_EXTS:
                p = self._index.get(norm + ext)
                if p is not None:
                    return p
        # last resort: match by basename (storyboard prefix vs disk differs)
        return self._base.get(norm.rsplit("/", 1)[-1])

    def get(self, sb_path: str, needed: set) -> tuple[str, int, int] | None:
        """Ensure the texture for ``sb_path`` is resident, mark it needed
        this frame, and return ``(gl_key, w, h)`` (or None if missing)."""
        gl_key = self._resolve(sb_path)
        if gl_key is None:
            return None
        needed.add(gl_key)
        if gl_key in self._resident:
            self._resident.move_to_end(gl_key)
            w, h = self._dims[gl_key]
            return (gl_key, w, h)
        rgba = _decode_image(self._path_for[gl_key])
        if rgba is None:
            self._resolved[sb_path] = None     # demote to known-missing
            if sb_path not in self._missing_logged:
                self._missing_logged.add(sb_path)
                log.warning("storyboard: image failed to decode, skipping: %r",
                            sb_path)
            return None
        h, w = rgba.shape[0], rgba.shape[1]
        # ClampToEdge (lazer WrapMode.ClampToEdge) + mipmaps for clean minify.
        self.spr.upload_texture(gl_key, rgba, clamp=True, mipmaps=True)
        nbytes = int(w) * int(h) * 4
        self._resident[gl_key] = nbytes
        self._dims[gl_key] = (int(w), int(h))
        self._total += nbytes
        self.peak_bytes = max(self.peak_bytes, self._total)
        self.uploads += 1
        return (gl_key, int(w), int(h))

    def evict_unneeded(self, needed: set) -> None:
        """Trim resident textures not needed this frame down to the budget."""
        if self._total <= self.budget:
            return
        for gl_key in list(self._resident):
            if self._total <= self.budget:
                break
            if gl_key in needed:
                continue
            nbytes = self._resident.pop(gl_key)
            self._total -= nbytes
            self.spr.release_texture(gl_key)
            self.evictions += 1
        if self._total > self.budget and not self._over_budget_logged:
            self._over_budget_logged = True
            log.warning("storyboard: this frame's live textures (%.0f MB) "
                        "exceed the %.0f MB cache budget — NOT dropping "
                        "sprites (correctness first); raise the budget if "
                        "VRAM allows.",
                        self._total / 1e6, self.budget / 1e6)


def _build_file_index(folder: Path) -> dict:
    idx = {}
    try:
        for p in folder.rglob("*"):
            if p.is_file():
                rel = p.relative_to(folder).as_posix().lower()
                idx[rel] = p
    except OSError:
        pass
    return idx


def _build_basename_index(index: dict) -> dict:
    base = {}
    for rel, p in index.items():
        base[rel.rsplit("/", 1)[-1]] = p
    return base


def _decode_image(path: Path):
    """Decode to HxWx4 uint8 (top-left origin) or None on failure."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return np.asarray(im.convert("RGBA"), dtype=np.uint8)
    except Exception as e:  # noqa: BLE001 — bad file must not crash the render
        log.warning("storyboard: decode error %s: %s", path, e)
        return None


# --------------------------------------------------------------------------- #
# Renderer                                                                      #
# --------------------------------------------------------------------------- #

class StoryboardRenderer:
    """Owns the storyboard texture cache and draws a frame at time ``t`` into
    the shared ``SpriteRenderer`` framebuffer.

    ``scene.py`` draws the two z-slices at different points of its frame:
    :meth:`draw_underlay` (Background/Fail/Pass/Foreground) just under the
    playfield, and :meth:`draw_overlay` (the Overlay layer) over the
    playfield, under the HUD.  Both share one per-time state build."""

    def __init__(self, spr, engine, folder, out_w, out_h, widescreen,
                 *, passing: bool = True, budget_mb: int = 1024):
        self.spr = spr
        self.engine = engine
        self.out_w = int(out_w)
        self.out_h = int(out_h)
        self.widescreen = bool(widescreen)
        self.passing = passing
        self.cache = _TextureCache(spr, folder, budget_mb=budget_mb)

        self.k = out_h / SB_H
        self.cx = out_w / 2.0
        self.cy = out_h / 2.0
        self.viewport = storyboard_viewport(out_w, out_h, widescreen)

        # per-frame build cache
        self._cache_t: float | None = None
        self._under: list[Sprite] = []
        self._over: list[Sprite] = []

    # -- per-time state build --------------------------------------------------

    def _prepare(self, t: float, brightness: float) -> None:
        if self._cache_t == t:
            return
        self._cache_t = t
        under: list[Sprite] = []
        over: list[Sprite] = []
        needed: set = set()
        for el, st in self.engine.state_at(t):
            layer = el.layer
            # pass/fail visibility (we render the passing set by default).
            if self.passing:
                if layer == "Fail":
                    continue
            else:
                if layer == "Pass":
                    continue
            sp = self._to_sprite(el, st, brightness, needed)
            if sp is None:
                continue
            if layer == "Overlay":
                over.append(sp)
            else:
                under.append(sp)
        self.cache.evict_unneeded(needed)
        self._under = under
        self._over = over

    def _to_sprite(self, el, st, brightness, needed):
        a = st.alpha
        if a <= 0.0:
            return None
        if a > 1.0:
            a = 1.0
        # texture (animation frame selection is phase-3's job)
        if isinstance(el, StoryboardAnimation):
            tl = self.engine.timeline_for(el)
            base_t = tl.earliest_transform_time if tl is not None else 0.0
            idx = anim_frame_index(
                self._cache_t - base_t, el.frame_count, el.frame_delay,
                el.loop_type == LoopType.LOOP_FOREVER)
            path = el.frame_path(idx)
        else:
            path = el.path
        tex = self.cache.get(path, needed)
        if tex is None:
            return None
        gl_key, tw, th = tex
        (scr_cx, scr_cy, w, h, theta,
         mirror_x, mirror_y) = compute_quad(
            el.origin, st.x, st.y, st.scale_x, st.scale_y, st.rotation,
            st.flip_h, st.flip_v, tw, th, self.k, self.cx, self.cy)
        if w <= 0.0 or h <= 0.0:
            return None
        b = brightness
        return Sprite(
            x=scr_cx, y=scr_cy, w=w, h=h, texture_key=gl_key,
            color=(st.r * b, st.g * b, st.b * b, a),
            rotation=theta, additive=st.additive,
            uv_off=(1.0 if mirror_x else 0.0, 1.0 if mirror_y else 0.0),
            uv_scale=(-1.0 if mirror_x else 1.0, -1.0 if mirror_y else 1.0),
        )

    # -- drawing ---------------------------------------------------------------

    def draw_underlay(self, t: float, brightness: float = 1.0) -> None:
        self._prepare(t, brightness)
        self._draw_ordered(self._under)

    def draw_overlay(self, t: float, brightness: float = 1.0) -> None:
        self._prepare(t, brightness)
        self._draw_ordered(self._over)

    def _draw_ordered(self, sprites: list[Sprite]) -> None:
        """Draw strictly back-to-front, honouring per-sprite blend by cutting
        the list into maximal same-blend runs (so additive sprites blend in
        their own z-slot, not hoisted last like gl.py's default draw())."""
        if not sprites:
            return
        # clip to the storyboard viewport (DrawableStoryboardLayer.Masking).
        prev_scissor = self.spr.ctx.scissor
        self.spr.ctx.scissor = self.viewport
        try:
            run: list[Sprite] = [sprites[0]]
            add = sprites[0].additive
            for sp in sprites[1:]:
                if sp.additive == add:
                    run.append(sp)
                else:
                    self.spr.draw(run)
                    run = [sp]
                    add = sp.additive
            self.spr.draw(run)
        finally:
            self.spr.ctx.scissor = prev_scissor

    # -- diagnostics -----------------------------------------------------------

    def stats(self) -> dict:
        return {
            "uploads": self.cache.uploads,
            "evictions": self.cache.evictions,
            "peak_mb": self.cache.peak_bytes / 1e6,
            "resident": len(self.cache._resident),
        }
