"""Additive effects pass over the readback frame: hit explosions and floating
GREAT/OK/MISS judgement text — both additive-blended like osu!lazer Argon.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageChops, ImageFilter

from . import _const as C
from .font import get_font
from .textures import bake_drum_flash, bake_explosion, bake_ring

_JUDGE_TEXT = {"great": "GREAT", "ok": "OK", "miss": "MISS"}
_JUDGE_COL = {"great": C.JUDGE_GREAT, "ok": C.JUDGE_OK, "miss": C.JUDGE_MISS}

# perf: per-(texture, size[, tint]) caches of the resize + float32 conversion
# work that used to run per active popup per frame. The cached arrays hold the
# exact same values the inline computation produced (same resize filter, same
# op order), so the composited output is bit-identical — only recomputation is
# skipped. id() keys are safe because the cache keeps a strong ref to the
# source array (its id can't be reused while cached); bounded by a hard clear.
_SCALE_CACHE: dict = {}     # (id(tex), tw, th, tint) -> (tex, src_rgb, a01, oy, ox)
_SCALE_CACHE_MAX = 256


def _scaled_f32(tex, tw, th, tint=None):
    """(src_rgb float32 [tinted], alpha/255 float32, oy, ox) of `tex` resized
    to tw×th — cached, cropped to the tight alpha>0 bounding box (compositing
    where a==0 is an exact float no-op, so skipping those pixels is
    bit-identical). Values match the previous per-call computation exactly.
    Returns None when the resized texture is fully transparent."""
    key = (id(tex), tw, th, tint)
    hit = _SCALE_CACHE.get(key)
    if hit is not None and hit[0] is tex:
        return hit[1]
    # BILINEAR (not LANCZOS): this runs per active popup per frame and the
    # textures are soft glows/text — the quality difference is invisible, the
    # speedup is ~3x. (One-time bakes stay LANCZOS.)
    im8 = np.asarray(Image.fromarray(tex).resize((tw, th), Image.BILINEAR))
    mask = im8[..., 3] != 0
    rows = mask.any(axis=1)
    if not rows.any():
        entry = None
    else:
        cols = mask.any(axis=0)
        y0 = int(np.argmax(rows))
        y1 = len(rows) - int(np.argmax(rows[::-1]))
        x0 = int(np.argmax(cols))
        x1 = len(cols) - int(np.argmax(cols[::-1]))
        im = im8[y0:y1, x0:x1].astype(np.float32)
        src_rgb, a01 = im[..., :3], im[..., 3:4] / 255.0
        if tint is not None and tint != (1.0, 1.0, 1.0):
            src_rgb = src_rgb * np.asarray(tint, dtype=np.float32)
        entry = (src_rgb, a01, y0, x0)
    if len(_SCALE_CACHE) > _SCALE_CACHE_MAX:
        _SCALE_CACHE.clear()
    _SCALE_CACHE[key] = (tex, entry)
    return entry


def _add_tex(rgb, tex, cx, cy, tw, th, intensity, tint=(1.0, 1.0, 1.0)):
    """Additive-blend RGBA `tex` (resized to tw×th) centred at (cx,cy)."""
    tw, th = int(round(tw)), int(round(th))
    if tw < 1 or th < 1 or intensity <= 0:
        return
    entry = _scaled_f32(tex, tw, th, tint)
    if entry is None:
        return
    src_rgb, a01, oy, ox = entry
    ch, cw = src_rgb.shape[:2]
    H, W = rgb.shape[:2]
    # top-left of the FULL resized tex (as before), then the crop offset
    x0, y0 = int(round(cx - tw / 2)) + ox, int(round(cy - th / 2)) + oy
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(W, x0 + cw), min(H, y0 + ch)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    h, w = dy1 - dy0, dx1 - dx0
    sa = a01[sy0:sy0 + h, sx0:sx0 + w] * intensity
    sc = src_rgb[sy0:sy0 + h, sx0:sx0 + w]
    region = rgb[dy0:dy1, dx0:dx1].astype(np.float32) + sc * sa
    rgb[dy0:dy1, dx0:dx1] = np.clip(region, 0, 255).astype(np.uint8)


_BRIGHT_LUT = None


def bloom(rgb, *, thresh=130, strength=0.55, step=6):
    """Cheap additive bloom (approximates osu!lazer's glow). All work is done in
    C via PIL: downscale → bright-pass (point LUT) → gaussian-blur the small
    image → upscale → saturating add. Avoids per-frame numpy full-frame floats."""
    global _BRIGHT_LUT
    if _BRIGHT_LUT is None:
        _BRIGHT_LUT = [int(max(0, v - thresh) * strength) for v in range(256)]
    img = Image.fromarray(rgb)
    H, W = rgb.shape[:2]
    sw, sh = max(1, W // step), max(1, H // step)
    small = img.resize((sw, sh), Image.BILINEAR).point(_BRIGHT_LUT * 3)
    small = small.filter(ImageFilter.GaussianBlur(radius=max(2, sw * 0.02)))
    up = small.resize((W, H), Image.BILINEAR)
    return np.array(ImageChops.add(img, up))


def _blit_straight(rgb, tex, cx, cy, tw, th, alpha):
    """Straight-alpha composite RGBA `tex` (resized to tw×th) centred at (cx,cy),
    scaled by `alpha` (for fades). For skin judgement images (not additive)."""
    tw, th = int(round(tw)), int(round(th))
    if tw < 1 or th < 1 or alpha <= 0:
        return
    entry = _scaled_f32(tex, tw, th)
    if entry is None:
        return
    src_rgb, a01, oy, ox = entry
    ch, cw = src_rgb.shape[:2]
    H, W = rgb.shape[:2]
    x0, y0 = int(round(cx - tw / 2)) + ox, int(round(cy - th / 2)) + oy
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(W, x0 + cw), min(H, y0 + ch)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    h, w = dy1 - dy0, dx1 - dx0
    sa = a01[sy0:sy0 + h, sx0:sx0 + w] * alpha
    sc = src_rgb[sy0:sy0 + h, sx0:sx0 + w]
    region = rgb[dy0:dy1, dx0:dx1].astype(np.float32)
    region = region * (1 - sa) + sc * sa
    rgb[dy0:dy1, dx0:dx1] = np.clip(region, 0, 255).astype(np.uint8)


def _prebake_add(im8):
    """(rgb f32, alpha/255 f32, oy, ox, full_h, full_w) of an RGBA uint8
    texture, cropped to the tight alpha>0 bbox (additive blend where a==0
    adds exactly +0.0 — skipping is bit-identical). None = fully transparent."""
    mask = im8[..., 3] != 0
    rows = mask.any(axis=1)
    fh, fw = im8.shape[:2]
    if not rows.any():
        return None
    cols = mask.any(axis=0)
    y0 = int(np.argmax(rows))
    y1 = fh - int(np.argmax(rows[::-1]))
    x0 = int(np.argmax(cols))
    x1 = fw - int(np.argmax(cols[::-1]))
    im = im8[y0:y1, x0:x1].astype(np.float32)
    return (im[..., :3], im[..., 3:4] / 255.0, y0, x0, fh, fw)


def _add_prescaled(rgb, pre, cx, cy, intensity):
    """Additive-blend a prescaled texture centred at (cx,cy). `pre` is the
    _prebake_add tuple — the float32 conversion + /255 normalisation used to
    run per call per frame; values are identical."""
    if pre is None:
        return
    sc_full, a01, oy, ox, fh, fw = pre
    ch, cw = sc_full.shape[:2]
    H, W = rgb.shape[:2]
    # top-left of the FULL texture (as before), then the crop offset
    x0, y0 = int(round(cx - fw / 2)) + ox, int(round(cy - fh / 2)) + oy
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(W, x0 + cw), min(H, y0 + ch)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    h, w = dy1 - dy0, dx1 - dx0
    sa = a01[sy0:sy0 + h, sx0:sx0 + w] * intensity
    sc = sc_full[sy0:sy0 + h, sx0:sx0 + w]
    region = rgb[dy0:dy1, dx0:dx1].astype(np.float32) + sc * sa
    rgb[dy0:dy1, dx0:dx1] = np.clip(region, 0, 255).astype(np.uint8)


class ArgonEffects:
    def __init__(self, geo, skin_dir=None):
        self.geo = geo
        # Hit explosion tinted by the NOTE accent (don=red / kat=blue), matching
        # lazer's ArgonHitExplosion. The old CENTRE/RIM_HIT_GRAD are the pink/cyan
        # INPUT-DRUM gradients; with bake_explosion's former solid white core they
        # flashed a WHITE disc on every centre hit (a big don read as a big white
        # circle), and being near-white they wash to solid white where the additive
        # bursts overlap on a dense stream. Near-pure accent colours saturate to
        # clean red / blue instead (the dominant channel clips first).
        self.exp = {
            False: bake_explosion((C.DON_TOP, C.DON_BOT), (255, 32, 32, 255)),  # centre/don
            True: bake_explosion((C.KAT_TOP, C.KAT_BOT), (32, 156, 255, 255)),  # rim/kat
        }
        self._ring = bake_ring(64, 8)   # white judgement RingExplosion ring
        self.font = get_font("SemiBold")
        self._jcache: dict[str, np.ndarray] = {}
        self._rng_cache: dict = {}      # (rt, piece, travel) -> (angle, dist)
        # Skin judgement images (taiko-hit300/100/0) — used for the gameplay
        # popups instead of Torus text when the skin provides them.
        from osu_taiko_renderer.skin.taiko_skin import TaikoSkin
        skin = TaikoSkin(skin_dir)
        self._skin_judge = {}
        for res, name in (("great", "taiko-hit300"), ("ok", "taiko-hit100"),
                          ("miss", "taiko-hit0")):
            img = skin.load(name)
            if img is not None:
                self._skin_judge[res] = img
        # Strong/big-note variants (geki GREAT / big OK): skins ship these as
        # taiko-hit300g / taiko-hit100k. Missing -> fall back to the normal
        # result sprite (see _judge_tex).
        for res, name in (("great_big", "taiko-hit300g"), ("ok_big", "taiko-hit100k")):
            img = skin.load(name)
            if img is not None:
                self._skin_judge[res] = img
        # Honour the skin's judgement graphics whenever it ships ANY of them,
        # not just a plain GREAT — skins deliberately omit taiko-hit300 so a
        # normal GREAT shows nothing (e.g. "+39 nofinish"). Per-result absence
        # is handled as "no popup" in _judge_tex, not an Argon text fallback.
        self._use_skin_judge = bool(self._skin_judge)
        # Pre-scale explosion textures to the two note sizes (resizing every
        # frame per active explosion was a render-time hotspot). Stored as
        # _prebake_add tuples (float32 rgb + alpha/255, alpha-bbox-cropped) so
        # the per-frame additive blend skips the per-call /255 normalisation
        # and all fully-transparent margins (identical values either way).
        self._exp_scaled = {}
        for is_rim in (False, True):
            for big in (False, True):
                d = int(round(geo.big_d if big else geo.note_d))
                im8 = np.asarray(Image.fromarray(self.exp[is_rim])
                                 .resize((d, d), Image.LANCZOS))
                self._exp_scaled[(is_rim, big)] = _prebake_add(im8)
        # Pre-scale the 4 drum-flash quadrants to the drum size.
        dd = int(round(geo.drum_d))
        self._drum_scaled = {}
        for is_rim in (False, True):
            for left in (True, False):
                im8 = np.asarray(Image.fromarray(bake_drum_flash(ring=is_rim, left=left))
                                 .resize((dd, dd), Image.LANCZOS))
                self._drum_scaled[(is_rim, left)] = _prebake_add(im8)

    def _judge_tex(self, result, big=False):
        # Big/strong notes prefer the geki/kiai sprite (taiko-hit300g / -100k)
        # when the skin ships it, else fall back to the normal result sprite.
        if self._use_skin_judge and big and (result + "_big") in self._skin_judge:
            key = result + "_big"
        else:
            key = result
        if key not in self._jcache:
            if self._use_skin_judge:
                # Skin mode: the skin's sprite for this key, or None (no popup)
                # when the skin omits it — e.g. no taiko-hit300 => a normal GREAT
                # shows nothing ("+39 nofinish"). A fully transparent sprite also
                # renders nothing downstream.
                img = self._skin_judge.get(key)
                if img is None:
                    self._jcache[key] = None
                else:
                    th = int(self.geo.note_d * 1.1)
                    tw = max(1, int(th * img.shape[1] / img.shape[0]))
                    self._jcache[key] = np.array(
                        Image.fromarray(img).resize((tw, th), Image.LANCZOS))
            else:
                # ArgonJudgementPiece: plain straight-alpha OsuFont text, no glow
                # halo (the only burst is the separate RingExplosion).
                px = self.geo.note_d * 0.46
                self._jcache[key] = self.font.render(
                    _JUDGE_TEXT[result], px, color=_JUDGE_COL[result],
                    spacing=C.JUDGE_SPACING * self.geo.scale)
        return self._jcache[key]

    @staticmethod
    def _legacy_explosion_anim(age):
        """osu!lazer LegacyHitExplosion.Animate envelope (scale, alpha) at `age`
        ms: FadeInFromZero(120) → FadeOut(180); ScaleTo 0.6 →(96ms)1.1 →(48ms)0.9
        →(24ms)1.0, linear segments. The skin's taiko-hit300/100/0 sprite is the
        legacy HIT EXPLOSION — it bursts AT the hit target, it does not float up
        like Argon's judgement text."""
        if age < 120.0:
            alpha = age / 120.0
        elif age < 300.0:
            alpha = 1.0 - (age - 120.0) / 180.0
        else:
            alpha = 0.0
        if age < 96.0:
            scale = 0.6 + 0.5 * (age / 96.0)                 # 0.6 → 1.1
        elif age < 144.0:
            scale = 1.1 - 0.2 * ((age - 96.0) / 48.0)        # 1.1 → 0.9
        elif age < 168.0:
            scale = 0.9 + 0.1 * ((age - 144.0) / 24.0)       # 0.9 → 1.0
        else:
            scale = 1.0
        return scale, max(0.0, alpha)

    _RING_SPEC = {"great": (4, 4, 1.0), "ok": (4, 0, 0.6)}   # (small,large,travel_x); miss none

    def _ring_burst(self, rgb, res, age, rt, g):
        """lazer taiko ArgonJudgementPiece.RingExplosion: white hollow rings
        burst outward from the hit target, tinted by the result colour,
        additive. travel 58, start_position_ratio 0.6, fade 1000ms OutQuint.
        Seeded on the judged time so it is stable across frames."""
        spec = self._RING_SPEC.get(res)
        if spec is None:
            return
        n_small, n_large, tmult = spec
        ga = max(0.0, (1.0 - age / 1000.0)) ** 5
        if ga <= 0.004:
            return
        import math, random
        col = tuple(c / 255.0 for c in _JUDGE_COL[res][:3])
        sc = g.scale
        travel = 58.0 * sc * tmult
        p = min(age, 600) / 600.0
        rad = 0.6 + 0.4 * (1.0 - (1.0 - p) ** 5)
        pieces = [9.0 * sc] * n_small + [14.0 * sc] * n_large
        for i, size in enumerate(pieces):
            # perf: the seeded draws depend only on (rt, i, travel) — cache
            # them across frames instead of constructing a fresh
            # random.Random per piece per frame (values identical).
            rkey = (int(rt), i, travel)
            hit = self._rng_cache.get(rkey)
            if hit is None:
                rng = random.Random((int(rt) * 1000003) ^ (i * 2654435761))
                hit = (rng.uniform(0.0, 360.0),
                       rng.uniform(travel / 2.0, travel))
                if len(self._rng_cache) > 4096:
                    self._rng_cache.clear()
                self._rng_cache[rkey] = hit
            d, dist = hit
            cur = dist * rad
            _add_tex(rgb, self._ring, g.target_x + math.cos(d) * cur,
                     g.center_y + math.sin(d) * cur, size, size, ga, tint=col)

    def composite(self, rgb, exps, judges, drums=()):
        # rgb arrives as a writable (possibly flipped-view) frame from the PBO
        # pop and is mutated region-by-region in place — the old full-frame
        # ascontiguousarray copy ran on the render thread every frame and is
        # exactly what the writer thread's tobytes() already pays for.
        g = self.geo
        # input-drum press flashes (additive — clean bright pop + glow)
        for is_rim, left, a in drums:
            _add_prescaled(rgb, self._drum_scaled[(is_rim, left)],
                           g.drum_x, g.center_y, a)
        # hit explosions at the target (additive)
        for is_rim, age, big, res in exps:
            if res == "great":
                if age < C.EXPLOSION_GREAT_IN_MS:
                    a = age / C.EXPLOSION_GREAT_IN_MS
                else:
                    f = 1.0 - (age - C.EXPLOSION_GREAT_IN_MS) / C.EXPLOSION_GREAT_OUT_MS
                    a = max(0.0, f) ** 4
            else:
                f = 1.0 - (age - C.EXPLOSION_GREAT_IN_MS) / C.EXPLOSION_OK_OUT_MS
                a = C.EXPLOSION_OK_PEAK * max(0.0, f)
            if a <= 0.001:
                continue
            _add_prescaled(rgb, self._exp_scaled[(is_rim, big)],
                           g.target_x, g.center_y, a)
        # judgement popups. Two distinct lazer behaviours:
        #   * SKIN (legacy) → LegacyHitExplosion: the taiko-hit300/100/0 sprite
        #     BURSTS at the hit target (centred on the drum), scale-punch + fade.
        #     It does NOT float up (the old code reused the Argon text's upward
        #     drift, so the ring hovered near the mascot's feet instead of on the
        #     hit target — issue #117).
        #   * ARGON → ArgonJudgementPiece: GREAT/OK text rises off the target and
        #     fades, with the RingExplosion burst.
        for res, age, rt, big in judges:
            tex = self._judge_tex(res, big)
            if tex is None:                 # skin mode, no sprite for this result
                continue
            h0, w0 = tex.shape[0], tex.shape[1]
            if self._use_skin_judge:
                scale, alpha = self._legacy_explosion_anim(age)
                if alpha <= 0.01:
                    continue
                # straight alpha, centred ON the hit target (no upward float).
                _blit_straight(rgb, tex, g.target_x, g.center_y,
                               w0 * scale, h0 * scale, alpha)
                continue
            self._ring_burst(rgb, res, age, rt, g)   # Argon RingExplosion
            p = age / C.JUDGE_MOVE_MS
            ease = 1.0 - (1.0 - p) ** 5                # OutQuint (move/scale)
            scale = 1.0 + 0.4 * ease
            alpha = max(0.0, (1.0 - p) ** 5)           # FadeOutFromOne, OutQuint
            if alpha <= 0.01:
                continue
            yoff = (0.6 + 0.4 * ease) * g.pf_h
            # straight alpha (lazer draws the judgement as a normal SpriteText —
            # not additive).
            _blit_straight(rgb, tex, g.target_x, g.center_y - yoff,
                           w0 * scale, h0 * scale, alpha)
        return rgb
