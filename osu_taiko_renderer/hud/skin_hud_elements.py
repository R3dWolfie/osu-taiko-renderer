"""Reusable legacy-skin HUD elements so the Argon taiko HUD can render a skin's
digit font + scorebar HP bar per-element, falling back to Argon when the skin
doesn't ship them. Extracted from the legacy skin_hud path.

- SkinDigitFont: loads `<prefix>-0..9` (+ -comma/-dot/-percent/-x) and renders a
  number string; `.render(text, px)` matches ArgonCounter.render's interface so
  the Argon HUD can swap it in at its own draw positions. `.present` is False
  when the skin ships no digit font (then the caller uses ArgonCounter).
- SkinHealthBar: scorebar-bg + scorebar-colour[-N] top HP bar. `.present` False
  when the skin ships no scorebar.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from osu_taiko_renderer.render.envflag import envflag

_SYM = {"-comma": ",", "-dot": ".", "-percent": "%", "-x": "x"}

# Cache each glyph's float32 alpha terms instead of recomputing them on every
# composite. ON BY DEFAULT — it is byte-identical and worth +9.7% fps (508 -> 557.7,
# sprite_build 0.758 -> 0.589, both arms repeated to three decimals). Set
# R3D_TAIKO_NO_GLYPH_TERMS=1 to fall back to the original body, which is kept
# verbatim below both as the kill switch and as the A/B reference arm.
#
# WHY THIS IS THE HUD'S BIGGEST SINGLE COST. cProfile of overlay() alone put
# _alpha_blit at 0.658 s of 2.33 s (28%), 6 calls/frame, ~53 us/call -- more than
# any other function by a factor of three. The reason is the caller below: a number
# is assembled GLYPH BY GLYPH, and the score string changes every frame, so it
# misses _ncache every frame and re-composites all ~7 digits.
#
# The old body ran ~13 numpy dispatches per call on a ~60x40 array. At that size
# dispatch is 62-83% of a numpy call (measured on this box: np.clip alone costs
# 1626 ns before touching a pixel), so the fixed overhead dominated the arithmetic.
#
# Four of those dispatches only ever touched `src`, which is CONSTANT: `parts` come
# straight out of SkinDigitFont._scaled's cache, so the same glyph array is
# composited every frame for the life of the render. Hoisting s.astype, a,
# s[...,:3]*a and (1-a) into a per-glyph cache removes them from the per-call path.
#
# This is exactly the trick hud.py::_pm_terms already uses on the Argon path;
# _alpha_blit simply never got it.
#
# BYTE-IDENTICAL, and not by luck: every hoisted operation is elementwise, and
# elementwise ops commute with cropping, so computing terms on the whole glyph and
# then slicing gives bit-for-bit what computing them on the slice gave. Verified
# across four different x offsets (including partial clips) plus the frozen oracle.
_GLYPH_TERMS = not envflag("R3D_TAIKO_NO_GLYPH_TERMS")
# R3D_TAIKO_NO_GLYPH_SPLIT=1 keeps the cached terms but blends every column,
# so the copy/blend column split can be A/B'd independently of the terms cache.
_GLYPH_SPLIT = not envflag("R3D_TAIKO_NO_GLYPH_SPLIT")
_GTERMS: dict = {}
_GTERMS_MAX = 4096


def _glyph_terms(src):
    """(rgb*a, 1-a, alpha) for `src`, cached on id(src).

    id() is a safe key only because the cache holds a strong reference to `src`
    alongside the terms: an id cannot be recycled while the object is still alive.
    The `hit[0] is src` check catches the one remaining hazard -- an id reused after
    a cached entry was evicted and the original freed."""
    k = id(src)
    hit = _GTERMS.get(k)
    if hit is not None and hit[0] is src:
        return hit[1]
    s = src.astype(np.float32)
    a = s[..., 3:4] / 255.0
    terms = (s[..., :3] * a, 1.0 - a, s[..., 3:4])
    if len(_GTERMS) > _GTERMS_MAX:
        _GTERMS.clear()
    _GTERMS[k] = (src, terms)
    return terms


_GZERO: dict = {}


def _glyph_onzero(src):
    """uint8 result of compositing `src` onto an all-zero RGBA destination.

    Cached per glyph, same id()-with-strong-ref contract as _glyph_terms.

    It must reproduce the blend path's bytes EXACTLY, so it is written as that
    path with the dst terms dropped rather than as "the obvious thing":
      - RGB is `sa` (= src.rgb * a), i.e. PREMULTIPLIED, not src.rgb. Writing
        src.rgb here would look more natural and would be wrong.
      - the final store is `.astype(np.uint8)`, which TRUNCATES. Using np.rint
        would differ by up to 1 LSB on most pixels.
    Both details are why this is derived from _glyph_terms instead of recomputed."""
    k = id(src)
    hit = _GZERO.get(k)
    if hit is not None and hit[0] is src:
        return hit[1]
    sa, _inv, salpha = _glyph_terms(src)
    d = np.zeros(src.shape, np.float32)
    d[..., :3] = sa
    d[..., 3:4] = np.clip(salpha, 0, 255)
    out = d.astype(np.uint8)
    if len(_GZERO) > _GTERMS_MAX:
        _GZERO.clear()
    _GZERO[k] = (src, out)
    return out


def _alpha_blit(dst, src, x, y, clean_from=None):
    """Alpha-composite RGBA `src` onto RGBA `dst` at (x, y) (glyph assembly).

    `clean_from` is an optimisation hint, not a mode: the dst column index from
    which the caller guarantees dst is still untouched (all zero). Compositing
    onto zero is algebraically a plain store -- `sa + 0*inv == sa` and
    `salpha + 0*inv == salpha` -- so those columns need no read, no float
    conversion and no blend, just a copy of a per-glyph array we can cache.

    This is deliberately expressed as a GENERAL split rather than an `if overlap
    == 0` fast path. Glyph runs overlap by `int(px * overlap_frac)` px, which is 0
    for this skin's ScoreOverlap:1 at ~60px but nonzero for others; a skin-specific
    branch would optimise one skin and leave the rest. Splitting each glyph into
    "columns the previous glyph already dirtied" (blend) and "columns still
    virgin" (copy) is correct for every overlap value, and degenerates to a pure
    copy at overlap 0 without testing for it.

    Vertical extents are NOT tracked: glyphs are centred vertically, so a taller
    glyph has clean rows even inside dirty columns. Treating whole columns as
    dirty just blends where it could copy -- conservative, never wrong."""
    sh, sw = src.shape[0], src.shape[1]
    dh, dw = dst.shape[0], dst.shape[1]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(dw, x + sw), min(dh, y + sh)
    if x1 <= x0 or y1 <= y0:
        return
    if _GLYPH_TERMS:
        # split = first dst column that is still virgin, clamped into this glyph.
        # blend [x0, split), copy [split, x1). clean_from=None disables the split.
        split = x1 if clean_from is None else min(x1, max(x0, clean_from))
        if split > x0:
            sa, inv, salpha = _glyph_terms(src)
            ys = slice(y0 - y, y1 - y)
            xs = slice(x0 - x, split - x)
            d = dst[y0:y1, x0:split].astype(np.float32)
            d[..., :3] = sa[ys, xs] + d[..., :3] * inv[ys, xs]
            d[..., 3:4] = np.clip(salpha[ys, xs] + d[..., 3:4] * inv[ys, xs], 0, 255)
            dst[y0:y1, x0:split] = d.astype(np.uint8)
        if x1 > split:
            # One store, from a cached array. No read of dst, no float math.
            dst[y0:y1, split:x1] = _glyph_onzero(src)[y0 - y:y1 - y,
                                                     split - x:x1 - x]
        return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    d = dst[y0:y1, x0:x1].astype(np.float32)
    a = (s[..., 3:4] / 255.0)
    d[..., :3] = s[..., :3] * a + d[..., :3] * (1 - a)
    d[..., 3:4] = np.clip(s[..., 3:4] + d[..., 3:4] * (1 - a), 0, 255)
    dst[y0:y1, x0:x1] = d.astype(np.uint8)


class SkinDigitFont:
    def __init__(self, skin, prefix, extra="", overlap_frac=0.0):
        self.glyphs: dict = {}
        if skin is not None and prefix:
            for d in range(10):
                img = skin.load(f"{prefix}-{d}")
                if img is not None:
                    self.glyphs[str(d)] = img
            for suf in extra.split():
                img = skin.load(f"{prefix}{suf}")
                if img is not None:
                    self.glyphs[_SYM.get(suf, suf)] = img
        self.overlap_frac = float(overlap_frac or 0.0)
        self._gcache: dict = {}
        self._ncache: dict = {}

    @property
    def present(self) -> bool:
        return "0" in self.glyphs

    def _scaled(self, ch, px, ref_h):
        key = (ch, round(px))
        if key in self._gcache:
            return self._gcache[key]
        im = self.glyphs.get(ch)
        if im is not None:
            s = px / ref_h
            w, h = max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))
            im = np.array(Image.fromarray(im).resize((w, h), Image.LANCZOS))  # type: ignore[attr-defined]
        self._gcache[key] = im
        return im

    def render(self, text, px):
        """RGBA (H,W,4) for the number string; cached per (text, px)."""
        text = str(text)
        key = (text, round(px))
        hit = self._ncache.get(key)
        if hit is not None:
            return hit
        ref = self.glyphs.get("0")
        if ref is None:
            return np.zeros((1, 1, 4), np.uint8)
        parts = [g for g in (self._scaled(c, px, ref.shape[0]) for c in text)
                 if g is not None]
        if not parts:
            out = np.zeros((1, 1, 4), np.uint8)
        else:
            ov = int(px * self.overlap_frac)
            total = sum(p.shape[1] for p in parts) - ov * (len(parts) - 1)
            H = max(p.shape[0] for p in parts)
            out = np.zeros((H, max(1, total), 4), np.uint8)
            x = 0
            # `out` is freshly zeroed, so everything from the previous glyph's
            # right edge onwards is still virgin and needs a store, not a blend.
            # dirty_to starts at 0 => the first glyph is a pure copy.
            dirty_to = 0
            for p in parts:
                _alpha_blit(out, p, x, (H - p.shape[0]) // 2,
                            clean_from=dirty_to if _GLYPH_SPLIT else None)
                dirty_to = x + p.shape[1]
                x += p.shape[1] - ov
        if len(self._ncache) > 4096:
            self._ncache.clear()
        self._ncache[key] = out
        return out


class SkinHealthBar:
    # A real scorebar-bg is a wide, thin bar. Skins (esp. catch/taiko ones that
    # don't want an osu!-style HP bar) often ship a 1x1 - or otherwise tiny /
    # non-bar-shaped - transparent scorebar-bg to HIDE it. Such an image carries
    # no usable geometry: its ~square aspect drives bh ~= bw, then the colour
    # frame is scaled by colour.h / bg.h - dividing by a ~1px bg height explodes
    # the fill bar to tens of thousands of px tall (a ~200 MB array re-blitted
    # every frame -> the HUD-blit CPU stall + OOM seen on a catch skin). Reject
    # it (fall back to no skin HP bar; Argon taiko shows none anyway). A plausible
    # bar is >= 3px tall and at least 4x as wide as it is tall.
    _MIN_BAR_H = 3
    _MIN_BAR_ASPECT = 4.0

    def __init__(self, skin):
        self.bg = skin.load("scorebar-bg") if skin is not None else None
        if self.bg is not None and (
                self.bg.shape[0] < self._MIN_BAR_H
                or self.bg.shape[1] < self._MIN_BAR_ASPECT * self.bg.shape[0]):
            self.bg = None
        self.frames = []
        if skin is not None:
            i = 0
            while True:
                f = skin.load(f"scorebar-colour-{i}")
                if f is None:
                    break
                self.frames.append(f)
                i += 1
            if not self.frames:
                c = skin.load("scorebar-colour")
                if c is not None:
                    self.frames = [c]
        self._bg_scaled = None
        self._col_base = None

    # A real legacy scorebar-bg is a WIDE strip (stable's is ~600+ px). Anything
    # narrower than this is not a bar: it is the legacy "hide the scorebar" trick
    # (ship a 1x1 placeholder). 2 of the 9 top community skins by render volume
    # do exactly that.
    _MIN_BG_W = 32

    @property
    def present(self) -> bool:
        """False when the skin has no scorebar, OR ships a degenerate placeholder.

        draw() scales bg to 44% of the frame WIDTH and takes the bar's height
        from bg's aspect ratio, then sizes the colour fill by colour_h / bg_h.
        A 1x1 bg makes both ratios 1.0, so at 1920x1080 it produced

          bh = int(1920 * 0.44 * 1 / 1)   = 844 px tall "bar"
          fh = int(844 * colour_h / 1)    = a colour fill thousands of px tall

        and both were composited on every frame. The placeholder is transparent,
        so the picture shows nothing of it (a level or two of rounding where it
        was blended), but the render ran 6 to 12 times slower.

        Reporting absent matches the skin author's intent (osu!stable draws
        nothing for a 1x1 scorebar) and costs nothing. A skin with a real
        scorebar is unaffected."""
        if self.bg is None:
            return False
        if self.bg.ndim < 2 or self.bg.shape[1] < self._MIN_BG_W:
            return False
        return True

    def draw(self, rgb, w, h, hp, blit) -> int:
        """Draw the HP bar top-left; returns its pixel height (0 if absent) so
        the caller can offset other top-left HUD below it."""
        if not self.present:
            return 0
        if self._bg_scaled is None:
            bw = int(w * 0.44)          # narrower than legacy so it clears the score
            # Absolute ceiling on the bar height: a real HP bar is a thin strip
            # (~40-100px at 1080p); this only ever bites a pathological skin image
            # whose aspect would otherwise scale the bar to thousands of px tall,
            # so it never touches a normal skin but guarantees no skin can blow
            # the per-frame blit array up (belt-and-suspenders to the __init__
            # reject above).
            _cap = max(1, int(h * 0.15))
            bh = max(1, min(int(bw * self.bg.shape[0] / self.bg.shape[1]), _cap))
            self._bg_scaled = np.array(
                Image.fromarray(self.bg).resize((bw, bh), Image.LANCZOS))  # type: ignore[attr-defined]
            if self.frames:
                fr = self.frames[0]
                fh = max(1, min(int(bh * (fr.shape[0] / self.bg.shape[0])), _cap))
                fwf = max(1, int(bw * 0.92))
                self._col_base = np.array(
                    Image.fromarray(fr).resize((fwf, fh), Image.LANCZOS))  # type: ignore[attr-defined]
        bg = self._bg_scaled
        bh = bg.shape[0]
        blit(rgb, bg, 0, 0, "tl")
        if self._col_base is not None:
            hpc = max(0.0, min(1.0, float(hp)))
            fw = max(1, int(self._col_base.shape[1] * hpc))
            blit(rgb, self._col_base[:, :fw], int(bg.shape[1] * 0.04),
                 int(bh * 0.30), "tl")
        return bh
