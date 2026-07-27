"""Argon taiko HUD: score (top-left), accuracy + PP (top-right), combo
(bottom-left), key counter B1–B4 (bottom-right), song progress (bottom).
Numbers use the Argon counter font; labels use Torus — matching lazer's
ArgonScoreCounter / ArgonAccuracyCounter / ArgonComboCounter / etc.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .counter import ArgonCounter
from .font import get_font

_LABEL = (188, 200, 214)          # muted Torus label colour
_WHITE = (255, 255, 255)
_ACCENT = (0x66, 0xcc, 0xff)      # Argon blue

# osu! mod bitmask → acronym (display order matters: difficulty then time then fl)
_MODS = [(1 << 1, "EZ"), (1 << 4, "HR"), (1 << 0, "NF"), (1 << 3, "HD"),
         (1 << 8, "HT"), (1 << 9, "NC"), (1 << 6, "DT"), (1 << 10, "FL"),
         (1 << 5, "SD"), (1 << 14, "PF"), (1 << 12, "SO"), (1 << 7, "RX")]


def mod_acronyms(mods: int) -> list[str]:
    out = [a for bit, a in _MODS if mods & bit]
    if (mods & (1 << 9)) and "DT" in out:    # NC implies DT bit; show NC only
        out.remove("DT")
    return out


# Argon grade colours (OsuColour.ForRank).
_GRADE_COL = {"SS": (255, 221, 85), "X": (255, 221, 85), "SSH": (200, 220, 255),
              "S": (255, 204, 34), "SH": (200, 220, 255),
              "A": (0xb3, 0xd9, 0x44), "B": (0x66, 0xcc, 0xff),
              "C": (0xcb, 0x3c, 0xec), "D": (0xed, 0x11, 0x21)}


def _fillband(rgb, x0, x1, y, h, color, alpha):
    """Alpha-fill a horizontal band [x0,x1) × [y,y+h) of rgb with color."""
    W = rgb.shape[1]
    x0, x1 = max(0, int(x0)), min(W, int(x1))
    if x1 <= x0:
        return
    seg = rgb[y:y + h, x0:x1].astype(np.float32)
    seg = seg * (1 - alpha) + np.array(color, np.float32) * alpha
    rgb[y:y + h, x0:x1] = np.clip(seg, 0, 255).astype(np.uint8)


# perf: per-texture cache of the float32 conversion + alpha terms used by
# _blit. The HUD composites the same (font/counter-cached) RGBA arrays every
# frame; precomputing `1-a` and `rgb*a` once per texture halves the per-blit
# math while producing bit-identical results (same float32 ops, elementwise —
# cropping commutes with them). id() keys are safe because the cache keeps a
# strong ref to the source (id can't be reused while cached); hard-clear bound.
_PM_CACHE: dict = {}
_PM_CACHE_MAX = 1024


def _pm_terms(src):
    """(1-a, rgb*a, oy, ox) of `src`, cropped to the tight alpha>0 bounding
    box (blending where a==0 is an exact float no-op: x*1.0 + s*0.0 == x, so
    skipping those pixels is bit-identical — font/wedge textures carry large
    transparent pads). None = fully transparent (nothing to composite)."""
    key = id(src)
    hit = _PM_CACHE.get(key)
    if hit is not None and hit[0] is src:
        return hit[1]
    mask = src[..., 3] != 0
    rows = mask.any(axis=1)
    if not rows.any():
        terms = None
    else:
        cols = mask.any(axis=0)
        y0 = int(np.argmax(rows))
        y1 = len(rows) - int(np.argmax(rows[::-1]))
        x0 = int(np.argmax(cols))
        x1 = len(cols) - int(np.argmax(cols[::-1]))
        s = src[y0:y1, x0:x1].astype(np.float32)
        a = s[..., 3:4] / 255.0
        terms = (1 - a, s[..., :3] * a, y0, x0)
    if len(_PM_CACHE) > _PM_CACHE_MAX:
        _PM_CACHE.clear()
    _PM_CACHE[key] = (src, terms)
    return terms


def _blit(rgb, src, x, y, anchor="tl"):
    """Alpha-composite RGBA uint8 `src` onto RGB uint8 `rgb`. anchor picks the
    reference corner: tl, tr, bl, br, tc, bc, cc."""
    h, w = src.shape[:2]
    if "r" in anchor:
        x -= w
    elif "c" in anchor[1:]:
        x -= w // 2
    if "b" in anchor:
        y -= h
    x, y = int(round(x)), int(round(y))
    terms = _pm_terms(src)
    if terms is None:
        return
    inv, sa, oy, ox = terms
    # shift to the cropped rect (anchoring above used the FULL src shape,
    # exactly as before)
    x += ox
    y += oy
    h, w = inv.shape[:2]
    H, W = rgb.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    crop = (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x))
    region = rgb[y0:y1, x0:x1].astype(np.float32)
    region = region * inv[crop] + sa[crop]
    rgb[y0:y1, x0:x1] = np.clip(region, 0, 255).astype(np.uint8)


class ArgonHud:
    def __init__(self, resolution, meta, bm, first, last, sim, cfg=None):
        self.w, self.h = resolution
        self.meta = meta
        self.bm = bm
        self.first = first
        self.last = max(last, first + 1)
        self.sim = sim
        self.cfg = cfg
        self.counter = ArgonCounter()
        self.bold = get_font("Bold")
        self.semi = get_font("SemiBold")
        from pathlib import Path
        wp = Path(__file__).resolve().parent / "glyphs" / "argon_wedge.png"
        self.wedge = np.array(Image.open(wp).convert("RGBA")) if wp.is_file() else None
        self._wedge_scaled = None
        self._results = None        # cached results-screen overlay (RGBA)
        # Per-element skin HUD: the skin's digit font (score/combo) + scorebar HP
        # bar when present, else Argon fallback. Argon keeps its own chrome
        # (progress / key counter / hit-error / title) that the legacy HUD lacks.
        from osu_taiko_renderer.hud.skin_hud_elements import (SkinDigitFont,
                                                              SkinHealthBar)
        _sk = getattr(sim, "skin", None)
        self._sfont = SkinDigitFont(_sk, getattr(_sk, "score_prefix", "score"),
                                    "-comma -dot -percent -x",
                                    (getattr(_sk, "score_overlap", 0) or 0) / 100.0)
        self._cfont = SkinDigitFont(_sk, getattr(_sk, "combo_prefix", "combo"),
                                    "-x", (getattr(_sk, "combo_overlap", 0) or 0) / 100.0)
        self._hpbar = SkinHealthBar(_sk)

    def _label(self, text, px, color=_LABEL):
        return self.bold.render(text, px, color=color)

    def _num(self, font, text, px):
        """Skin digit font if the skin ships one, else the Argon LED counter."""
        if font is not None and font.present:
            return font.render(text, px)
        return self.counter.render(text, px)

    def _key_active(self, t, window=110):
        """Whether each key (B1..B4 = rim-L, centre-L, centre-R, rim-R) had a
        press within the last `window` ms — lights its activity bar."""
        import bisect
        out = []
        for z in ("rl", "cl", "cr", "rr"):
            edges = self.sim._zedges[z]
            i = bisect.bisect_right(edges, t) - 1
            out.append(i >= 0 and (t - edges[i]) <= window)
        return out

    def overlay(self, rgb: np.ndarray, scene) -> np.ndarray:
        # rgb is mutated in place (writable flipped view from the PBO pop) —
        # the old full-frame ascontiguousarray copy was pure render-thread cost.
        w, h, t = self.w, self.h, scene.time_ms
        mx, my = int(w * 0.018), int(h * 0.03)
        lab_px = max(11, int(h * 0.016))
        # skin scorebar HP bar (top-left) — Argon has none; push the top-left
        # score block below it. No skin scorebar -> no bar (unchanged Argon).
        hp_h = self._hpbar.draw(rgb, w, h, scene.hp, _blit) if self._hpbar.present else 0

        # --- score (top-RIGHT, lazer default taiko layout) ---
        sc = self._num(self._sfont, str(int(scene.score)), h * 0.056)
        _blit(rgb, sc, w - mx, my, "tr")

        # --- accuracy (top-right, directly below the score) ---
        pct = max(0.0, min(100.0, scene.accuracy * 100.0))
        ah = h * 0.05
        ay = my + sc.shape[0] + int(h * 0.010)
        if self._sfont.present:
            _blit(rgb, self._sfont.render(f"{pct:.2f}%", ah * 0.62),
                  w - mx, ay, "tr")
        else:
            whole = str(int(pct))
            frac = f"{pct:06.2f}".split(".")[1]
            pcttex = self.counter.render("%", ah * 0.5)
            _blit(rgb, pcttex, w - mx, ay, "tr")
            fx = w - mx - pcttex.shape[1]
            fractex = self.counter.render(frac, ah * 0.5)
            _blit(rgb, fractex, fx, ay, "tr")
            dottex = self.counter.render(".", ah * 0.5)
            _blit(rgb, dottex, fx - fractex.shape[1], ay, "tr")
            wholetex = self.counter.render(whole, ah)
            _blit(rgb, wholetex, fx - fractex.shape[1] - dottex.shape[1], ay, "tr")

        # --- pp (top-right, below accuracy) ---
        ppy = ay + int(ah) + int(lab_px * 0.6)
        _blit(rgb, self._label("PP", lab_px), w - mx, ppy, "tr")
        pptex = self.counter.render(str(int(round(scene.pp))), h * 0.03)
        _blit(rgb, pptex, w - mx, ppy + int(lab_px * 1.2), "tr")

        # --- mods (top-right, below PP): coloured acronym pills ---
        mods = mod_acronyms(int(getattr(self.meta, "mods", 0) or 0))
        if mods:
            my2 = ppy + int(lab_px * 1.2) + int(h * 0.03) + int(h * 0.012)
            px = w - mx
            pill_px = max(12, int(h * 0.02))
            for ac in reversed(mods):
                tex = self.bold.render(ac, pill_px, color=_WHITE)
                pad = int(pill_px * 0.4)
                pw, ph = tex.shape[1] + pad * 2, tex.shape[0] + pad
                x0 = px - pw
                rgb[my2:my2 + ph, x0:px] = (44, 50, 60)
                _blit(rgb, tex, x0 + pad, my2 + pad // 2, "tl")
                px = x0 - int(w * 0.006)

        # --- hit counter (mid-left): GREAT / OK / MISS, per --hit-counter ---
        if getattr(self.cfg, "show_hit_counter", True) and getattr(scene, "counts", None) is not None:
            _g, _o, _m = scene.counts
            _cnp = h * 0.055
            _cyy = int(h * 0.46)
            for _v, _lb, _cl in ((_g, "GREAT", (95, 200, 255)),
                                 (_o, "OK", (150, 235, 90)),
                                 (_m, "MISS", (255, 95, 95))):
                _nt = self.bold.render(str(int(_v)), _cnp, color=_cl)
                _blit(rgb, _nt, mx, _cyy, "tl")
                _blit(rgb, self._label(_lb, lab_px, color=_cl),
                      mx, _cyy + _nt.shape[0] - int(lab_px * 0.1), "tl")
                _cyy += _nt.shape[0] + int(lab_px * 1.7)

        # --- combo (bottom-left) ---
        ctex = self._num(self._cfont, f"{int(scene.combo)}x", h * 0.072)
        _blit(rgb, ctex, mx, h - my, "bl")
        _blit(rgb, self._label("COMBO", lab_px, color=_ACCENT),
              mx, h - my - ctex.shape[0] - int(lab_px * 0.4), "bl")

        # --- key counter B1–B4 (bottom-right): activity bar / label / count,
        # each column centred (ArgonKeyCounter layout) ---
        counts = self.sim.key_counts(t)
        active = self._key_active(t)
        kw = int(w * 0.042)
        right = w - mx
        bar_w, bar_h = int(kw * 0.62), max(2, int(h * 0.004))
        for i in range(4):           # left→right B1..B4
            c = counts[i]
            cx = right - (3 - i) * kw - kw // 2
            num = self.bold.render(str(c), int(h * 0.026), color=_WHITE)
            lab = self.bold.render(f"B{i + 1}", lab_px, color=_LABEL)
            base_y = h - my
            _blit(rgb, num, cx, base_y, "bc")
            _blit(rgb, lab, cx, base_y - num.shape[0] - 2, "bc")
            by = base_y - num.shape[0] - lab.shape[0] - 8
            col = _WHITE if active[i] else (70, 78, 90)
            rgb[by:by + bar_h, cx - bar_w // 2:cx + bar_w // 2] = col

        # --- song progress (bottom centre) ---
        frac = max(0.0, min(1.0, (t - self.first) / (self.last - self.first)))
        bar_y = h - int(h * 0.012)
        bx0, bx1 = int(w * 0.16), int(w * 0.84)
        rgb[bar_y:bar_y + 3, bx0:bx1] = (70, 78, 90)
        fillx = bx0 + int((bx1 - bx0) * frac)
        rgb[bar_y:bar_y + 3, bx0:fillx] = _ACCENT
        _blit(rgb, self.semi.render(_fmt_time(t - self.first), int(h * 0.016),
                                    color=_LABEL), bx0, bar_y - 4, "bl")
        _blit(rgb, self.semi.render(_fmt_time(self.last - self.first),
                                    int(h * 0.016), color=_LABEL),
              bx1, bar_y - 4, "br")

        # --- hit-error / UR bar (bottom centre, above the progress bar) ---
        self._draw_hit_error(rgb, t)

        # --- player / title (top centre, spectator-style) ---
        title = f"{self.bm.artist} - {self.bm.title} [{self.bm.version}]".strip(" -")
        info = self.semi.render(f"{self.meta.player_name}  ·  {title}",
                                int(h * 0.018), color=_LABEL)
        _blit(rgb, info, w // 2, my, "tc")
        return rgb

    def _draw_hit_error(self, rgb, t):
        """Horizontal hit-error meter (lazer BarHitErrorMeter): OK/GREAT window
        bands, centre line, recent-hit ticks (fading), and the UR value."""
        w, h = self.w, self.h
        gw = getattr(self.sim, "great_w", 35.0)
        ow = getattr(self.sim, "ok_w", 80.0)
        if ow <= 0:
            return
        cx = w // 2
        half = int(w * 0.11)
        by = h - int(h * 0.05)
        bh = max(4, int(h * 0.012))
        scale = half / ow
        great_blue = (0x66, 0xcc, 0xff)
        ok_green = (0x88, 0xb3, 0x00)
        # window bands (dim base, then OK band, then GREAT band on top)
        gwp = int(gw * scale)
        rgb[by:by + bh, cx - half:cx + half] = (40, 46, 56)
        _fillband(rgb, cx - int(ow * scale), cx + int(ow * scale), by, bh, ok_green, 0.5)
        _fillband(rgb, cx - gwp, cx + gwp, by, bh, great_blue, 0.6)
        # centre line
        rgb[by - 3:by + bh + 3, cx - 1:cx + 1] = _WHITE
        # recent hit ticks (fade with age)
        for err, res, age in self.sim.recent_errors(t):
            x = int(cx + max(-ow, min(ow, err)) * scale)
            a = max(0.0, 1.0 - age / 3500.0)
            col = great_blue if res == "great" else ok_green
            ty0, ty1 = by - int(bh * 0.8), by + bh + int(bh * 0.8)
            seg = rgb[ty0:ty1, max(0, x - 1):x + 2].astype(np.float32)
            seg = seg * (1 - a) + np.array(col, np.float32) * a
            rgb[ty0:ty1, max(0, x - 1):x + 2] = np.clip(seg, 0, 255).astype(np.uint8)
        # UR value
        ur = self.sim.ur_at(t)
        if ur > 0:
            tex = self.bold.render(f"{ur:.2f} UR", int(h * 0.016), color=_LABEL)
            _blit(rgb, tex, cx, by - bh - 4, "bc")

    # --- results screen (Argon ranking panel) -------------------------------

    def _compose(self, ov, src, x, y, anchor="tl"):
        """Alpha-composite RGBA uint8 `src` onto RGBA float overlay `ov`."""
        h, w = src.shape[:2]
        if "r" in anchor:
            x -= w
        elif "c" in anchor[1:]:
            x -= w // 2
        if "b" in anchor:
            y -= h
        x, y = int(round(x)), int(round(y))
        H, W = ov.shape[:2]
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        s = src[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
        a = s[..., 3:4] / 255.0
        reg = ov[y0:y1, x0:x1]
        reg[..., :3] = reg[..., :3] * (1 - a) + s[..., :3] * a
        reg[..., 3:4] = np.clip(reg[..., 3:4] + a * (255 - reg[..., 3:4]), 0, 255)

    def _build_results(self):
        w, h = self.w, self.h
        m, bm = self.meta, self.bm
        ov = np.zeros((h, w, 4), np.float32)
        ov[..., 3] = 255                               # opaque black bg: clean results, no gameplay-HUD bleed (like std/catch)
        cx = w // 2
        grade = str(getattr(m, "grade", "D") or "D")
        mods_i = int(getattr(m, "mods", 0) or 0)
        silver = bool(mods_i & ((1 << 3) | (1 << 10)))      # HD or FL → silver SS/S
        if silver and grade.upper() in ("SS", "S"):
            gcol = _GRADE_COL["SSH"]
        else:
            gcol = _GRADE_COL.get(grade.upper(), _WHITE)
        # FEATURED player avatar — circular chip, top-centre, above the title.
        try:
            from osu_taiko_renderer.hud.lb_cards import bake_avatar_circle
            _av_px = int(h * 0.11)
            _chip = bake_avatar_circle(_av_px, m.player_name,
                                       getattr(self, "featured_avatar_bytes", None))
            self._compose(ov, np.array(_chip), cx, int(h * 0.045), "tc")
        except Exception:  # noqa: BLE001 — avatar never breaks the results card
            pass
        # title / difficulty / player (top) — nudged down to clear the avatar chip
        title = f"{bm.artist} - {bm.title}".strip(" -")
        self._compose(ov, self.semi.render(title, int(h * 0.026), color=_WHITE),
                      cx, int(h * 0.135), "tc")
        self._compose(ov, self.bold.render(
            f"[{bm.version}]   played by {m.player_name}", int(h * 0.018),
            color=_LABEL), cx, int(h * 0.175), "tc")
        # big grade letter
        self._compose(ov, self.bold.render(grade, int(h * 0.24), color=gcol),
                      cx, int(h * 0.15), "tc")
        # score (segmented), centred
        sctex = self.counter.render(str(int(m.score)), h * 0.085)
        self._compose(ov, sctex, cx, int(h * 0.43), "tc")
        # accuracy | max combo | pp   (label above value)
        pct = max(0.0, min(100.0, float(getattr(m, "accuracy", 0.0))))
        pp = int(round(getattr(self.sim, "_final_pp", 0.0)))
        cells = [("ACCURACY", f"{pct:.2f}".replace("100.00", "100") + "%"),
                 ("MAX COMBO", f"{int(getattr(m, 'max_combo', 0))}x"),
                 ("PP", str(pp))]
        cw = int(w * 0.16)
        y_lab = int(h * 0.56)
        for i, (lab, val) in enumerate(cells):
            ccx = cx + (i - 1) * cw
            self._compose(ov, self.bold.render(lab, int(h * 0.016), color=_LABEL),
                          ccx, y_lab, "tc")
            self._compose(ov, self.counter.render(val, h * 0.044),
                          ccx, y_lab + int(h * 0.028), "tc")
        # GREAT / OK / MISS counts (coloured)
        g3 = (int(getattr(m, "count_300", 0)), int(getattr(m, "count_100", 0)),
              int(getattr(m, "count_miss", 0)))
        judge = [("GREAT", g3[0], _GRADE_COL["B"]), ("OK", g3[1], (0x88, 0xb3, 0x00)),
                 ("MISS", g3[2], (0xed, 0x11, 0x21))]
        y_j = int(h * 0.69)
        for i, (lab, val, col) in enumerate(judge):
            ccx = cx + (i - 1) * cw
            self._compose(ov, self.bold.render(lab, int(h * 0.017), color=col),
                          ccx, y_j, "tc")
            self._compose(ov, self.counter.render(str(val), h * 0.04),
                          ccx, y_j + int(h * 0.026), "tc")
        # mods
        mods = mod_acronyms(int(getattr(m, "mods", 0) or 0))
        if mods:
            row = "  ".join(mods)
            self._compose(ov, self.bold.render(row, int(h * 0.022), color=_ACCENT),
                          cx, int(h * 0.82), "tc")
        return ov.astype(np.uint8)

    def draw_results(self, rgb, op):
        """Cross-fade the Argon results overlay onto the frozen final frame, then
        composite the per-map leaderboard flank cards (fading in with `op`)."""
        if self._results is None:
            self._results = self._build_results()
        ov = self._results
        a = (ov[..., 3:4].astype(np.float32) / 255.0) * max(0.0, min(1.0, op))
        out = rgb.astype(np.float32) * (1 - a) + ov[..., :3].astype(np.float32) * a
        out = np.clip(out, 0, 255).astype(np.uint8)
        # flank leaderboard: composited per-frame at the results opacity so the
        # cards unfurl with the fade. Only when a board was attached; fully
        # fail-soft — a board must never break a render.
        if getattr(self, "board", None) is not None:
            try:
                from osu_taiko_renderer.hud.lb_cards import draw_board
                pim = Image.fromarray(out, "RGB").convert("RGBA")
                draw_board(pim, self.board, max(0.0, min(1.0, op)))
                out = np.asarray(pim.convert("RGB"))
            except Exception:  # noqa: BLE001 — leaderboard never breaks a render
                pass
        return out


def _fmt_time(ms):
    s = max(0, int(ms // 1000))
    return f"{s // 60}:{s % 60:02d}"
