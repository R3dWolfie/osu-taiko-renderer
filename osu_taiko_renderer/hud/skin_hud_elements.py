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

_SYM = {"-comma": ",", "-dot": ".", "-percent": "%", "-x": "x"}


def _alpha_blit(dst, src, x, y):
    """Alpha-composite RGBA `src` onto RGBA `dst` at (x, y) (glyph assembly)."""
    sh, sw = src.shape[0], src.shape[1]
    dh, dw = dst.shape[0], dst.shape[1]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(dw, x + sw), min(dh, y + sh)
    if x1 <= x0 or y1 <= y0:
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
            im = np.array(Image.fromarray(im).resize((w, h), Image.LANCZOS))
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
            for p in parts:
                _alpha_blit(out, p, x, (H - p.shape[0]) // 2)
                x += p.shape[1] - ov
        if len(self._ncache) > 4096:
            self._ncache.clear()
        self._ncache[key] = out
        return out


class SkinHealthBar:
    def __init__(self, skin):
        self.bg = skin.load("scorebar-bg") if skin is not None else None
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

    @property
    def present(self) -> bool:
        return self.bg is not None

    def draw(self, rgb, w, h, hp, blit) -> int:
        """Draw the HP bar top-left; returns its pixel height (0 if absent) so
        the caller can offset other top-left HUD below it."""
        if self.bg is None:
            return 0
        if self._bg_scaled is None:
            bw = int(w * 0.44)          # narrower than legacy so it clears the score
            bh = max(1, int(bw * self.bg.shape[0] / self.bg.shape[1]))
            self._bg_scaled = np.array(
                Image.fromarray(self.bg).resize((bw, bh), Image.LANCZOS))
            if self.frames:
                fr = self.frames[0]
                fh = max(1, int(bh * (fr.shape[0] / self.bg.shape[0])))
                fwf = max(1, int(bw * 0.92))
                self._col_base = np.array(
                    Image.fromarray(fr).resize((fwf, fh), Image.LANCZOS))
        bg = self._bg_scaled
        bh = bg.shape[0]
        blit(rgb, bg, 0, 0, "tl")
        if self._col_base is not None:
            hpc = max(0.0, min(1.0, float(hp)))
            fw = max(1, int(self._col_base.shape[1] * hpc))
            blit(rgb, self._col_base[:, :fw], int(bg.shape[1] * 0.04),
                 int(bh * 0.30), "tl")
        return bh
