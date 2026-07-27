"""Legacy (true-to-skin) taiko HUD.

When a user .osk skin provides the legacy HUD assets (score-*/combo-* digit
fonts, scorebar HP bar) we draw the HUD from THOSE — score, accuracy, combo, and
HP bar — instead of the Argon HUD. Layout follows osu!stable taiko: HP bar
top-left, score + accuracy top-right, combo bottom-left.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from osu_taiko_renderer.skin.fonts import font as _font
from osu_taiko_renderer.hud.hud import _mod_labels


def _blit(rgb, src, x, y, anchor="tl"):
    h, w = src.shape[:2]
    if "r" in anchor:
        x -= w
    elif "c" in anchor[1:]:
        x -= w // 2
    if "b" in anchor:
        y -= h
    x, y = int(round(x)), int(round(y))
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[..., 3:4].astype(np.uint16)            # 0..255
    inv = 255 - a
    reg = rgb[y0:y1, x0:x1]                       # view into dst
    # integer alpha blend (uint16) — avoids the per-blit float32 round-trip and
    # the final clip; result is bounded so no clamp needed.
    reg[..., :3] = ((reg[..., :3].astype(np.uint16) * inv
                     + s[..., :3].astype(np.uint16) * a + 127) // 255).astype(np.uint8)
    if reg.shape[2] == 4:            # RGBA dst (e.g. the glyph canvas): keep alpha
        da = reg[..., 3].astype(np.uint16)
        reg[..., 3] = (da + ((255 - da) * a[..., 0] + 127) // 255).astype(np.uint8)


class LegacyHud:
    def __init__(self, resolution, meta, bm, first, last, sim, skin):
        self.w, self.h = resolution
        self.meta, self.bm, self.sim = meta, bm, sim
        self.first, self.last = first, max(last, first + 1)
        self.skin = skin
        self.score_glyphs = self._load_font("score", extra="-comma -dot -percent -x")
        self.combo_glyphs = self._load_font("combo", extra="-x")
        self.sb_bg = skin.load("scorebar-bg")
        self.sb_frames = []
        i = 0
        while True:
            f = skin.load(f"scorebar-colour-{i}")
            if f is None:
                break
            self.sb_frames.append(f)
            i += 1
        if not self.sb_frames:
            c = skin.load("scorebar-colour")
            if c is not None:
                self.sb_frames = [c]
        self._gcache: dict = {}      # scaled glyph per (font, char, px)
        self._ncache: dict = {}      # assembled digit-string per (font, text, px)
        self._bg_scaled = None       # scaled scorebar-bg (cached once)
        self._col_base = None        # scaled scorebar-colour at full bar width
        self._results = None
        # Extra HUD info the legacy skin assets don't carry, drawn with a text
        # font like the Argon HUD: the mods row (static, cached), plus pp and the
        # GREAT/OK/MISS tallies (dynamic).
        self.mods = _mod_labels(meta.mods)
        self._mods_img = None
        self._ticache: dict = {}     # cached label-text images (pp/counts labels)

    def _load_font(self, prefix, extra=""):
        g = {}
        for d in range(10):
            img = self.skin.load(f"{prefix}-{d}")
            if img is not None:
                g[str(d)] = img
        sym = {"-comma": ",", "-dot": ".", "-percent": "%", "-x": "x"}
        for suf in extra.split():
            img = self.skin.load(f"{prefix}{suf}")
            if img is not None:
                g[sym[suf]] = img
        return g

    def has_fonts(self):
        return "0" in self.score_glyphs

    def _scaled_glyph(self, glyphs, ch, px, ref_h):
        """Scaled digit/symbol image, cached per (font, char, px) so re-rendering
        a changing number (score) only composites pre-scaled glyphs (no resize)."""
        key = (id(glyphs), ch, round(px))
        if key in self._gcache:
            return self._gcache[key]
        im = glyphs.get(ch)
        if im is not None:
            s = px / ref_h
            w, h = max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))
            im = np.array(Image.fromarray(im).resize((w, h), Image.LANCZOS))
        self._gcache[key] = im
        return im

    def _num(self, text, glyphs, px, overlap_frac=0.0):
        # perf: cache the ASSEMBLED string image (score/acc/combo strings
        # repeat heavily across frames); identical inputs assemble the exact
        # same array, so this only skips recomputation.
        nkey = (id(glyphs), text, round(px), overlap_frac)
        hit = self._ncache.get(nkey)
        if hit is not None:
            return hit
        out = self._num_build(text, glyphs, px, overlap_frac)
        if len(self._ncache) > 4096:
            self._ncache.clear()
        self._ncache[nkey] = out
        return out

    def _num_build(self, text, glyphs, px, overlap_frac=0.0):
        ref = glyphs.get("0")
        if ref is None:
            return np.zeros((1, 1, 4), np.uint8)
        parts = [g for g in (self._scaled_glyph(glyphs, c, px, ref.shape[0]) for c in text)
                 if g is not None]
        if not parts:
            return np.zeros((1, 1, 4), np.uint8)
        ov = int(px * overlap_frac)
        total = sum(p.shape[1] for p in parts) - ov * (len(parts) - 1)
        H = max(p.shape[0] for p in parts)
        out = np.zeros((H, max(1, total), 4), np.uint8)
        x = 0
        for p in parts:
            _blit(out, p, x, (H - p.shape[0]) // 2, "tl")
            x += p.shape[1] - ov
        return out

    def overlay(self, rgb, scene):
        # rgb is mutated in place (writable flipped view from the PBO pop) —
        # the old full-frame ascontiguousarray copy was pure render-thread cost.
        w, h = self.w, self.h
        mx, my = int(w * 0.012), int(h * 0.02)
        # HP bar (scorebar-bg + colour frame by hp), top-left
        if self.sb_bg is not None:
            if self._bg_scaled is None:     # resize the big bg once, not per frame
                bw = int(w * 0.64)          # stable-like HP bar width
                bh = int(bw * self.sb_bg.shape[0] / self.sb_bg.shape[1])
                self._bg_scaled = np.array(Image.fromarray(self.sb_bg)
                                           .resize((bw, bh), Image.LANCZOS))
                if self.sb_frames:
                    fr = self.sb_frames[0]
                    fh = max(1, int(bh * (fr.shape[0] / self.sb_bg.shape[0])))
                    fwf = max(1, int(bw * 0.92))
                    self._col_base = np.array(Image.fromarray(fr)
                                              .resize((fwf, fh), Image.LANCZOS))
            bg = self._bg_scaled
            bh = bg.shape[0]
            _blit(rgb, bg, 0, 0, "tl")
            if self._col_base is not None:
                hp = max(0.0, min(1.0, scene.hp))
                fw = max(1, int(self._col_base.shape[1] * hp))   # slice = free
                _blit(rgb, self._col_base[:, :fw], int(bg.shape[1] * 0.04),
                      int(bh * 0.30), "tl")
        # score top-right
        sc = self._num(str(int(scene.score)), self.score_glyphs, h * 0.052)
        _blit(rgb, sc, w - mx, my, "tr")
        # accuracy top-right, below score
        pct = max(0.0, min(100.0, scene.accuracy * 100.0))
        acc = self._num(f"{pct:.2f}%", self.score_glyphs, h * 0.030)
        _blit(rgb, acc, w - mx, my + sc.shape[0] + int(h * 0.01), "tr")
        # mods row + pp + GREAT/OK/MISS tallies, top-right under accuracy. Legacy
        # skins ship no assets for these, so they're drawn with a text font (same
        # info the Argon HUD shows).
        yb = my + sc.shape[0] + int(h * 0.01) + acc.shape[0] + int(h * 0.014)
        cfg = getattr(self.sim, "cfg", None)

        def _on(name, default=True):
            return cfg is None or getattr(cfg, name, default)

        if _on("show_mods") and self.mods:
            ms = self._mods_strip()
            _blit(rgb, ms, w - mx, yb, "tr")
            yb += ms.shape[0] + int(h * 0.012)
        if getattr(cfg, "show_pp_counter", False) and getattr(scene, "pp", 0) > 0:
            pp = self._text_img(f"{scene.pp:.0f}pp", h * 0.032, (245, 235, 255))
            _blit(rgb, pp, w - mx, yb, "tr")
            yb += pp.shape[0] + int(h * 0.008)
        if _on("show_hit_counter") and hasattr(scene, "counts"):
            great, ok, miss = scene.counts
            for label, val, col in (("GREAT", great, (110, 200, 255)),
                                    ("OK", ok, (130, 255, 160)),
                                    ("MISS", miss, (255, 110, 110))):
                ti = self._text_img(f"{int(val)}x {label}", h * 0.026, col)
                _blit(rgb, ti, w - mx, yb, "tr")
                yb += ti.shape[0] + int(h * 0.004)
        # combo bottom-left
        if scene.combo > 0:
            cb = self._num(f"{int(scene.combo)}x", self.combo_glyphs, h * 0.075)
            _blit(rgb, cb, mx, h - my, "bl")
        return rgb

    def _text_img(self, text, px, color):
        """Small RGBA image of `text` in the HUD font, cached per (text, px)."""
        key = (text, round(px))
        if key in self._ticache:
            return self._ticache[key]
        f = _font(max(8, int(px)))
        tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        bb = tmp.textbbox((0, 0), text, font=f)
        pad = max(2, int(px * 0.22))
        img = Image.new("RGBA", (bb[2] - bb[0] + 2 * pad, bb[3] - bb[1] + 2 * pad),
                        (0, 0, 0, 0))
        ImageDraw.Draw(img).text((pad - bb[0], pad - bb[1]), text, font=f,
                                 fill=color + (255,))
        out = np.array(img)
        self._ticache[key] = out
        return out

    def _mods_strip(self):
        """Mods row as rounded pills (Argon style), rendered once and cached."""
        if self._mods_img is not None:
            return self._mods_img
        h = self.h
        f = _font(max(8, int(h * 0.026)))
        ph = int(h * 0.04)
        gap = int(h * 0.006)
        tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        items = []
        for m in self.mods:
            bb = tmp.textbbox((0, 0), m, font=f)
            pad = int((bb[3] - bb[1]) * 0.6)
            items.append((m, bb, pad, (bb[2] - bb[0]) + 2 * pad))
        total = sum(it[3] for it in items) + gap * (len(items) - 1)
        img = Image.new("RGBA", (max(1, total), ph), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        x = 0
        for m, bb, pad, wpill in items:
            d.rounded_rectangle([x, 0, x + wpill, ph], radius=int(h * 0.01),
                                fill=(40, 42, 56, 235))
            d.text((x + pad - bb[0], (ph - (bb[3] - bb[1])) // 2 - bb[1]), m,
                   font=f, fill=(235, 235, 245, 255))
            x += wpill + gap
        self._mods_img = np.array(img)
        return self._mods_img

    def draw_results(self, rgb, op):
        """Simple legacy results: dark scrim + score / accuracy / max combo (+
        GREAT/OK/MISS counts) in the skin's own fonts, cross-faded in."""
        if self._results is None:
            w, h = self.w, self.h
            ov = np.zeros((h, w, 4), np.float32)
            ov[..., 3] = 255  # opaque black bg: clean results, no gameplay-HUD bleed
            ov8 = ov.astype(np.uint8)
            m = self.meta
            cx = w // 2
            # FEATURED player avatar — circular chip centred above the score.
            try:
                from osu_taiko_renderer.hud.lb_cards import bake_avatar_circle
                _chip = bake_avatar_circle(
                    int(h * 0.14), getattr(m, "player_name", ""),
                    getattr(self, "featured_avatar_bytes", None))
                _blit(ov8, np.array(_chip), cx, int(h * 0.05), "tc")
            except Exception:  # noqa: BLE001 — avatar never breaks results
                pass
            sc = self._num(str(int(m.score)), self.score_glyphs, h * 0.10)
            _blit(ov8, sc, cx, int(h * 0.30), "tc")
            pct = max(0.0, min(100.0, float(getattr(m, "accuracy", 0.0))))
            acc = self._num(f"{pct:.2f}%", self.score_glyphs, h * 0.05)
            _blit(ov8, acc, cx, int(h * 0.45), "tc")
            mc = self._num(f"{int(getattr(m, 'max_combo', 0))}x",
                           self.combo_glyphs, h * 0.06)
            _blit(ov8, mc, cx, int(h * 0.55), "tc")
            # GREAT/OK/MISS counts with the skin's judgement images, if any
            cw = int(w * 0.16)
            for i, (img_name, val) in enumerate((
                    ("taiko-hit300", int(getattr(m, "count_300", 0))),
                    ("taiko-hit100", int(getattr(m, "count_100", 0))),
                    ("taiko-hit0", int(getattr(m, "count_miss", 0))))):
                jx = cx + (i - 1) * cw
                jimg = self.skin.load(img_name)
                if jimg is not None:
                    th = int(h * 0.045)
                    tw = int(th * jimg.shape[1] / jimg.shape[0])
                    j = np.array(Image.fromarray(jimg).resize((max(1, tw), th), Image.LANCZOS))
                    _blit(ov8, j, jx, int(h * 0.68), "tc")
                num = self._num(str(val), self.score_glyphs, h * 0.035)
                _blit(ov8, num, jx, int(h * 0.75), "tc")
            self._results = ov8
        ov = self._results
        a = (ov[..., 3:4].astype(np.float32) / 255.0) * max(0.0, min(1.0, op))
        out = rgb.astype(np.float32) * (1 - a) + ov[..., :3].astype(np.float32) * a
        out = np.clip(out, 0, 255).astype(np.uint8)
        # flank leaderboard: composited per-frame at the results opacity (fades
        # in with the cross-fade). Only when a board was attached; fail-soft.
        if getattr(self, "board", None) is not None:
            try:
                from osu_taiko_renderer.hud.lb_cards import draw_board
                pim = Image.fromarray(out, "RGB").convert("RGBA")
                draw_board(pim, self.board, max(0.0, min(1.0, op)))
                out = np.asarray(pim.convert("RGB"))
            except Exception:  # noqa: BLE001 — leaderboard never breaks a render
                pass
        return out
