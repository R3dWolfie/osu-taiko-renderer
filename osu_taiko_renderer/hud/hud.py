"""Taiko HUD + results card, drawn procedurally with PIL (no skin assets yet).

Layout mirrors the catch/mania renderers — score + accuracy + grade top-right,
a mods row, pp + judgement tallies, a big centred combo, an HP bar + player/
title top-left, a progress bar, and a results card — but with taiko semantics
(GREAT / OK / MISS, taiko grade). scene.counts is (great, ok, miss).
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

# osu mod bit -> short label (Nightcore supersedes the DT label).
_MODS = [
    (2, "EZ"), (8, "HD"), (16, "HR"), (512, "NC"), (64, "DT"),
    (256, "HT"), (1024, "FL"), (1, "NF"), (16384, "PF"), (32, "SD"),
    (128, "RX"), (4096, "SO"), (2048, "AT"),
]

_GRADE_COLOURS = {
    "SS": (240, 220, 120), "S": (240, 220, 120),
    "A": (110, 220, 130), "B": (110, 180, 220),
    "C": (200, 130, 220), "D": (220, 110, 110),
}


def _grade(acc: float) -> str:
    if acc >= 1.0:
        return "SS"
    if acc >= 0.98:
        return "S"
    if acc >= 0.94:
        return "A"
    if acc >= 0.90:
        return "B"
    if acc >= 0.85:
        return "C"
    return "D"


def _mod_labels(mods: int) -> list[str]:
    out, seen = [], set()
    for bit, name in _MODS:
        if mods & bit and name not in seen:
            if name == "DT" and (mods & 512):
                continue
            if name == "SD" and (mods & 16384):
                continue  # perfect icon already covers it (PF = SD|Perfect)
            out.append(name); seen.add(name)
    return out


class TaikoHud:
    def __init__(self, resolution, meta, beatmap, first_ms, last_ms, cfg=None):
        self.cfg = cfg
        self.w, self.h = resolution
        self.meta = meta
        self.bm = beatmap
        self.first_ms = first_ms
        self.last_ms = max(last_ms, first_ms + 1)
        self.mods = _mod_labels(meta.mods)

    def _on(self, n):
        return self.cfg is None or getattr(self.cfg, n, True)

    def overlay(self, rgb: np.ndarray, scene) -> np.ndarray:
        img = Image.fromarray(rgb, "RGB")
        d = ImageDraw.Draw(img)
        w, h = self.w, self.h
        pad = int(w * 0.012)
        right = w - pad

        # score (top-right)
        y = pad
        if self._on("show_score"):
            f = _font(int(h * 0.058))
            txt = f"{scene.score:,}"
            bb = d.textbbox((0, 0), txt, font=f)
            d.text((right - (bb[2] - bb[0]), y), txt, font=f, fill=(255, 255, 255))
            y += (bb[3] - bb[1]) + int(h * 0.014)

        # accuracy + grade (under score)
        if self._on("show_score"):
            af = _font(int(h * 0.036))
            atxt = f"{scene.accuracy * 100:.2f}%"
            ab = d.textbbox((0, 0), atxt, font=af)
            aw, ah = ab[2] - ab[0], ab[3] - ab[1]
            d.text((right - aw, y), atxt, font=af, fill=(235, 235, 245))
            if self._on("show_grade"):
                g = _grade(scene.accuracy)
                gf = _font(int(h * 0.05))
                gb = d.textbbox((0, 0), g, font=gf)
                d.text((right - aw - (gb[2] - gb[0]) - int(w * 0.01),
                        y - int(h * 0.008)), g, font=gf, fill=_GRADE_COLOURS.get(g, (230, 230, 230)))
            y += ah + int(h * 0.016)

        # mods row (text pills)
        if self._on("show_mods") and self.mods:
            mf = _font(int(h * 0.026))
            mx = right
            for m in reversed(self.mods):
                mx = self._pill(d, m, mx, y, mf, align_right=True)
                mx -= int(w * 0.006)
            y += int(h * 0.05)

        # pp + judgement tallies (top-right)
        if self.cfg is not None and getattr(self.cfg, "show_pp_counter", False) and scene.pp > 0:
            pf = _font(int(h * 0.032))
            txt = f"{scene.pp:.0f}pp"
            bb = d.textbbox((0, 0), txt, font=pf)
            d.text((right - (bb[2] - bb[0]), y), txt, font=pf, fill=(245, 235, 255))
            y += int(h * 0.045)
        if self.cfg is None or getattr(self.cfg, "show_hit_counter", True):
            great, ok, miss = scene.counts
            hf = _font(int(h * 0.026))
            for label, val, col in (("GREAT", great, (110, 200, 255)),
                                    ("OK", ok, (130, 255, 160)),
                                    ("MISS", miss, (255, 110, 110))):
                txt = f"{val}x {label}"
                bb = d.textbbox((0, 0), txt, font=hf)
                d.text((right - (bb[2] - bb[0]), y), txt, font=hf, fill=col)
                y += int(h * 0.032)

        # combo (big, centred)
        if scene.combo > 0 and self._on("show_combo"):
            cf = _font(int(h * 0.11))
            txt = f"{scene.combo}"
            bb = d.textbbox((0, 0), txt, font=cf)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            d.text((w * 0.5 - tw / 2 - bb[0], h * 0.46 - th / 2 - bb[1]), txt,
                   font=cf, fill=(247, 247, 248))

        # HP bar (top-left)
        if self._on("show_hp_bar"):
            bx, by, bw, bh = pad, int(h * 0.018), int(w * 0.33), max(8, int(h * 0.014))
            d.rounded_rectangle([bx, by, bx + bw, by + bh], radius=bh // 2, fill=(35, 35, 48))
            fillw = max(1, int(bw * max(0.0, min(1.0, scene.hp))))
            d.rounded_rectangle([bx, by, bx + fillw, by + bh], radius=bh // 2, fill=(120, 220, 150))

        # player + title (top-left, under HP)
        ty = int(h * 0.05)
        d.text((pad, ty), self.meta.player_name, font=_font(int(h * 0.026)), fill=(245, 245, 250))
        title = f"{self.bm.artist} - {self.bm.title} [{self.bm.version}]".strip(" -")
        tf = _font(int(h * 0.021))
        title = _ellipsize(d, title, tf, w - pad * 2)
        d.text((pad, ty + int(h * 0.032)), title, font=tf, fill=(190, 190, 205))

        # progress bar (bottom)
        frac = max(0.0, min(1.0, (scene.time_ms - self.first_ms) / (self.last_ms - self.first_ms)))
        d.rectangle([0, h - 6, w, h], fill=(30, 30, 40))
        d.rectangle([0, h - 6, int(w * frac), h], fill=(150, 200, 255))

        # watermark
        wm = getattr(self.cfg, "watermark", "") if self.cfg else ""
        if wm:
            wf = _font(int(h * 0.022))
            wb = d.textbbox((0, 0), wm, font=wf)
            d.text((right - (wb[2] - wb[0]), h - pad - (wb[3] - wb[1]) - int(h * 0.01)),
                   wm, font=wf, fill=(238, 238, 245))
        return np.asarray(img)

    def _pill(self, d, text, x_right, y, font, align_right=True):
        bb = d.textbbox((0, 0), text, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        px = int(th * 0.5)
        W = tw + px * 2
        x0 = x_right - W if align_right else x_right
        d.rounded_rectangle([x0, y, x0 + W, y + int(self.h * 0.04)],
                            radius=int(self.h * 0.01), fill=(40, 42, 56))
        d.text((x0 + px - bb[0], y + (int(self.h * 0.04) - th) // 2 - bb[1]),
               text, font=font, fill=(235, 235, 245))
        return x0


def draw_results(rgb, meta, bm, opacity: float):
    """Taiko results card: grade / score / accuracy / max combo / GREAT-OK-MISS,
    same vertical stack + grade colours as the catch/mania cards."""
    a = max(0.0, min(1.0, opacity))
    img = Image.fromarray(rgb, "RGB").convert("RGBA")
    img = Image.alpha_composite(img, Image.new("RGBA", img.size, (0, 0, 0, int(0.7 * a * 255))))
    W, H = img.size
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx = W // 2
    y = int(H * 0.10)
    A = int(a * 255)

    def line(size, text, color, gap):
        nonlocal y
        font = _font(size)
        bb = d.textbbox((0, 0), text, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        d.text((cx - tw // 2 - bb[0], y - bb[1]), text, font=font, fill=(*color, A))
        y += th + gap

    great, ok = meta.count_300, meta.count_100
    miss = meta.count_miss
    total = great + ok + miss
    acc = (great + ok * 0.5) / total if total else 1.0
    grade = _grade(acc)
    line(220, grade, _GRADE_COLOURS.get(grade, (200, 200, 220)), int(H * 0.02))
    line(96, f"{meta.score:,}", (255, 255, 255), 10)
    line(56, f"{acc * 100:.2f}%", (235, 235, 245), 10)
    line(40, f"Max combo {meta.max_combo}x", (200, 200, 220), 24)

    cells = [("GREAT", great, (110, 200, 255)),
             ("OK", ok, (130, 255, 160)),
             ("MISS", miss, (240, 80, 80))]
    f36 = _font(36)
    rendered = [(f"{lab}: {cnt}", col) for lab, cnt, col in cells]
    widths = [d.textbbox((0, 0), t, font=f36)[2] for t, _ in rendered]
    gap = 28
    x = cx - (sum(widths) + gap * (len(rendered) - 1)) // 2
    for (t, col), wd in zip(rendered, widths):
        d.text((x, y), t, font=f36, fill=(*col, A))
        x += wd + gap

    title = f"{bm.artist} - {bm.title} [{bm.version}]".strip(" -")
    rf = _font(26)
    full = _ellipsize(d, f"{meta.player_name}  ·  {title}", rf, int(W * 0.92))
    fb = d.textbbox((0, 0), full, font=rf)
    d.text((cx - (fb[2] - fb[0]) // 2, y + 70), full, font=rf, fill=(180, 180, 200, A))
    return np.asarray(Image.alpha_composite(img, layer).convert("RGB"))


def _ellipsize(draw, text: str, font, max_w: int) -> str:
    if draw.textbbox((0, 0), text, font=font)[2] <= max_w:
        return text
    while text and draw.textbbox((0, 0), text + "…", font=font)[2] > max_w:
        text = text[:-1]
    return (text + "…") if text else ""


from osu_taiko_renderer.skin.fonts import font as _font  # skin-aware, host-robust font resolver
