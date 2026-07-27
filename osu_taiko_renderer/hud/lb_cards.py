"""Compact ranked FLANK CARDS for the osu!taiko results screen — the per-map
render leaderboard, ported from the osu!catch renderer's lb_cards.py (itself a
port of the osu!STANDARD renderer's flank cards) so taiko reaches parity.

The DATA + avatar layer is leaderboard.py (query_leaderboard / build_board /
rows_from_osu_json / resolve_avatar_bytes — a verbatim port of the std module).
THIS module bakes each LeaderboardEntry into a rounded card image and lays the
board out around catch's centred results text stack, then composites it onto the
results frame. Everything is PIL (catch composites its results on the CPU), so
the cards drop straight on — no GL sprite atlas like std uses, and hence no GL
slide-in: the board fades in with the results opacity, with a light per-card
stagger keyed off that opacity so it still unfurls rather than popping in.

Adaptations vs the std flank cards:
  * fonts   = catch's own results font (fonts.font), NOT std's bundled Nunito,
              so the cards match catch's featured text stack.
  * palette = catch's Fruit/Drop/Droplet/Miss judgment colours + catch's grade
              pill colours (hud._GRADE_COLOURS), so the flanks read as the same
              UI as catch's results card. Miss = count_miss + count_katu (the
              missed-droplets), exactly as catch's featured card counts it.
"""
from __future__ import annotations

import colorsys
import hashlib
import os
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

from osu_taiko_renderer.skin.fonts import font as _load_font  # catch's skin-aware, host-robust resolver

# render DB — same file + env override the std renderer uses (a symlink resolves
# this path to state/db.sqlite on the render box). Local-only, read-only.
DB_PATH = os.environ.get("R3D_RENDER_DB",
                         "/home/red/.local/state/mania-ordr/db.sqlite")

LB_CARD_W = 214.0                 # compact card size (virtual 1080-space px)
LB_CARD_H = 596.0

# Fraction of the frame width reserved DOWN THE CENTRE for catch's featured
# results stack; the flank cards start just outside it. hud.draw_results reads
# this same constant to shrink the featured stack to a compact form (2×2
# judgment grid + player-only caption) when a board is present, so the stack
# never collides with the flanks. Only in effect when a board is shown → the
# default (boardless) results card is untouched.
CENTER_CLEAR_FRAC = 0.40

# taiko judgment palette (0..1) — GREAT / OK / MISS for the flank card rows.
RESULT_COLORS = {
    "GREAT": (110 / 255, 200 / 255, 255 / 255),
    "OK":    (130 / 255, 255 / 255, 160 / 255),
    "MISS":  (240 / 255,  80 / 255,  80 / 255),
}

# grade PILL colours — catch's results grade palette (hud._GRADE_COLOURS) as
# 0..1 so the flank pill matches catch's featured grade letter.
FOR_RANK = {
    "SS": (240 / 255, 220 / 255, 120 / 255),
    "S":  (240 / 255, 220 / 255, 120 / 255),
    "A":  (110 / 255, 220 / 255, 130 / 255),
    "B":  (110 / 255, 180 / 255, 220 / 255),
    "C":  (200 / 255, 130 / 255, 220 / 255),
    "D":  (220 / 255, 110 / 255, 110 / 255),
}
FOR_RANK["X"] = FOR_RANK["XH"] = FOR_RANK["SSH"] = FOR_RANK["SS"]
FOR_RANK["SH"] = FOR_RANK["S"]

# the arc-gradient cyan std uses for the NEW BEST pill (its ForRank S adjacent)
_ARC_CYAN = (0x7C / 255, 0xF6 / 255, 0xFF / 255)


# --- small helpers (ported from lazer_results.py) ----------------------------------

def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _hsv(h: float, s: float, v: float) -> tuple[float, float, float]:
    return colorsys.hsv_to_rgb(h, s, v)


def _clip(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n - 1] + "…"


def _clip_words(s: str, n: int) -> str:
    """Ellipsize on a WORD boundary: the whole string when it fits, else cut at
    the last word break that fits (mid-word only for one giant unbroken word),
    dropping any dangling separator before the '…' (2026-07-22 polish — the
    banner used to chop titles mid-word: 'Shooti…')."""
    s = s or ""
    if len(s) <= n:
        return s
    cut = s[:n - 1].rstrip()
    sp = cut.rfind(" ")
    if sp >= max(8, (n - 1) // 3):
        cut = cut[:sp]
    return cut.rstrip(" -–—·:|(") + "…"


def _name_hash(name: str) -> int:
    """Stable, process-independent username hash → non-negative int (hashlib,
    not builtin hash() which is PYTHONHASHSEED-salted) so avatars are
    deterministic across runs/machines."""
    key = (name or "?").strip().lower().encode("utf-8", "ignore")
    return int(hashlib.md5(key).hexdigest(), 16)


def avatar_initials(name: str) -> str:
    """1–2 uppercase initials from the first two word-tokens (else the first
    char; '?' when empty)."""
    name = (name or "").strip()
    if not name:
        return "?"
    tokens = [t for t in re.split(r"[\s_.\-]+", name) if t]
    if len(tokens) >= 2:
        return (tokens[0][0] + tokens[1][0]).upper()
    return tokens[0][0].upper()


def _norm_grade(g) -> str:
    s = str(g or "").strip().upper()
    return {"X": "SS", "XH": "SS", "SSH": "SS", "SH": "S", "F": "D"}.get(s, s)


# --- bakers ------------------------------------------------------------------------

def bake_round_panel(w: int, h: int, radius: int, top, bot, alpha: float,
                     border=None) -> Image.Image:
    """Rounded panel with a vertical top→bot gradient at `alpha` + optional
    border. Returns a PIL RGBA image."""
    w = max(int(w), 2)
    h = max(int(h), 2)
    grad = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    px = grad.load()
    a = int(round(_clamp01(alpha) * 255))
    for y in range(h):
        f = y / max(h - 1, 1)
        r = int(round(_lerp(top[0], bot[0], f) * 255))
        g = int(round(_lerp(top[1], bot[1], f) * 255))
        b = int(round(_lerp(top[2], bot[2], f) * 255))
        for x in range(w):
            px[x, y] = (r, g, b, a)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius=radius,
                                           fill=255)
    grad.putalpha(mask)
    if border is not None:
        br = tuple(int(round(c * 255)) for c in border)
        ImageDraw.Draw(grad).rounded_rectangle(
            [0, 0, w - 1, h - 1], radius=radius, outline=(*br, 160), width=2)
    return grad


def bake_avatar_square(px: int, name: str, avatar_bytes: bytes | None = None,
                       radius_frac: float = 0.2) -> Image.Image:
    """A rounded-square avatar chip. From Discord/osu PNG bytes (cover-fit +
    rounded-square mask) when available, else the deterministic username-hued
    square with centred initials — the same fallback std uses, so a missing/
    unfetchable avatar still reads as that player. Never raises."""
    px = max(int(px), 16)
    rad = max(int(px * radius_frac), 2)
    img = None
    if avatar_bytes:
        try:
            src = Image.open(BytesIO(avatar_bytes)).convert("RGBA")
            sw, sh = src.size
            scale = px / max(min(sw, sh), 1)
            nw, nh = max(int(sw * scale + 0.5), px), max(int(sh * scale + 0.5), px)
            src = src.resize((nw, nh), Image.LANCZOS)
            lft, top = (nw - px) // 2, (nh - px) // 2
            img = src.crop((lft, top, lft + px, top + px))
        except Exception:  # noqa: BLE001 — corrupt/animated → procedural
            img = None
    if img is None:
        h = _name_hash(name)
        hue = (h % 360) / 360.0
        sat = 0.42 + ((h >> 9) % 18) / 100.0
        c0 = _hsv(hue, sat, 0.62)
        c1 = _hsv((hue + 0.06) % 1.0, min(sat + 0.08, 1.0), 0.34)
        img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
        dd = ImageDraw.Draw(img)
        for y in range(px):
            f = y / max(px - 1, 1)
            col = tuple(int(round(_lerp(c0[i], c1[i], f) * 255)) for i in range(3))
            dd.line([(0, y), (px, y)], fill=(*col, 255))
        ini = avatar_initials(name)
        font = _load_font(int(px * (0.42 if len(ini) >= 2 else 0.52)))
        try:
            x0, y0, x1, y1 = font.getbbox(ini)
        except AttributeError:
            x1, y1 = font.getsize(ini); x0 = y0 = 0     # type: ignore
        dd.text(((px - (x1 - x0)) / 2 - x0, (px - (y1 - y0)) / 2 - y0), ini,
                font=font, fill=(255, 255, 255, 235))
    mask = Image.new("L", (px, px), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, px - 1, px - 1], radius=rad,
                                           fill=255)
    img.putalpha(mask)
    return img


def bake_avatar_circle(px: int, name: str,
                       avatar_bytes: bytes | None = None) -> Image.Image:
    """A CIRCULAR avatar chip for the taiko results FEATURED card: osu/Discord
    PNG bytes cover-fit + disc-clipped + soft HSV ring when available, else the
    deterministic username-hued disc with centred initials (the same fallback
    the flank cards use). Mirrors std/catch's lazer_results.bake_avatar. Never
    raises — a corrupt/animated/absent avatar degrades to the procedural disc."""
    px = max(int(px), 8)
    h = _name_hash(name)
    hue = (h % 360) / 360.0
    sat = 0.42 + ((h >> 9) % 18) / 100.0
    if avatar_bytes:
        try:
            src = Image.open(BytesIO(avatar_bytes)).convert("RGBA")
            sw, sh = src.size
            scale = px / max(min(sw, sh), 1)
            nw = max(int(sw * scale + 0.5), px)
            nh = max(int(sh * scale + 0.5), px)
            src = src.resize((nw, nh), Image.LANCZOS)
            lft, top = (nw - px) // 2, (nh - px) // 2
            img = src.crop((lft, top, lft + px, top + px))
            mask = Image.new("L", (px, px), 0)
            ImageDraw.Draw(mask).ellipse([0, 0, px - 1, px - 1], fill=255)
            img.putalpha(mask)
            d = ImageDraw.Draw(img)
            ring = _hsv(hue, max(sat - 0.16, 0.0), 0.92)
            rc = tuple(int(round(c * 255)) for c in ring)
            lw = max(int(px * 0.045), 2)
            off = lw * 0.5
            d.ellipse([off, off, px - 1 - off, px - 1 - off],
                      outline=(*rc, 150), width=lw)
            return img
        except Exception:  # noqa: BLE001 — corrupt/animated → procedural disc
            pass
    # deterministic username-hued disc with centred initials + soft ring
    c0 = _hsv(hue, sat, 0.62)
    c1 = _hsv((hue + 0.06) % 1.0, min(sat + 0.08, 1.0), 0.34)
    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for y in range(px):
        f = y / max(px - 1, 1)
        col = tuple(int(round(_lerp(c0[i], c1[i], f) * 255)) for i in range(3))
        d.line([(0, y), (px, y)], fill=(*col, 255))
    mask = Image.new("L", (px, px), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, px - 1, px - 1], fill=255)
    img.putalpha(mask)
    d = ImageDraw.Draw(img)
    ring = _hsv(hue, max(sat - 0.16, 0.0), 0.92)
    rc = tuple(int(round(c * 255)) for c in ring)
    lw = max(int(px * 0.045), 2)
    off = lw * 0.5
    d.ellipse([off, off, px - 1 - off, px - 1 - off],
              outline=(*rc, 150), width=lw)
    ini = avatar_initials(name)
    font = _load_font(int(px * (0.46 if len(ini) >= 2 else 0.56)))
    try:
        x0, y0, x1, y1 = font.getbbox(ini)
    except AttributeError:
        x1, y1 = font.getsize(ini); x0 = y0 = 0     # type: ignore
    d.text(((px - (x1 - x0)) / 2 - x0, (px - (y1 - y0)) / 2 - y0), ini,
           font=font, fill=(255, 255, 255, 240))
    return img


def _pil_text(draw, font, text, x, y, fill, align="l") -> int:
    """Draw one text run at (x, top y) with l/r/m horizontal alignment,
    measured via getbbox. Returns text height."""
    try:
        x0, y0, x1, y1 = font.getbbox(text)
    except AttributeError:
        x1, y1 = font.getsize(text); x0 = y0 = 0        # type: ignore
    w = x1 - x0
    tx = x - x0 if align == "l" else (x - w - x0 if align == "r"
                                      else x - w / 2 - x0)
    col = tuple(int(round(c * 255)) for c in fill)
    draw.text((tx, y - y0), text, font=font, fill=(*col, 255))
    return y1 - y0


def bake_lb_card(entry, avatar_bytes, W_px: int, H_px: int, k: float,
                 font_loader, score_loader) -> Image.Image:
    """Bake one compact leaderboard card — rank header, rounded-square avatar,
    name, the Great/OK/Miss rows, Max Combo, Accuracy, mods badges, the big
    score, and the grade pill. `entry` is a leaderboard.LeaderboardEntry.
    Returns a PIL RGBA image."""
    W = max(int(W_px), 60)
    H = max(int(H_px), 120)
    rad = max(int(18 * k), 4)
    img = bake_round_panel(W, H, rad, (0.13, 0.14, 0.19), (0.06, 0.06, 0.09),
                           0.95, border=(0.30, 0.33, 0.40))
    d = ImageDraw.Draw(img)
    pad = int(15 * k)

    def font(v):
        return font_loader(max(int(v * k), 8))

    y = pad
    # rank header (gold for #1)
    rcol = (1.0, 0.84, 0.4) if entry.rank == 1 else (0.80, 0.84, 0.94)
    _pil_text(d, font(24), f"#{entry.rank}", W / 2, y, rcol, "m")
    y += int(32 * k)
    # rounded-square avatar
    av = min(W - 2 * pad, int(118 * k))
    ax = (W - av) // 2
    img.alpha_composite(bake_avatar_square(av, entry.player_name, avatar_bytes),
                        (ax, int(y)))
    y += av + int(8 * k)
    # player name
    _pil_text(d, font(21), _clip(entry.player_name, 12), W / 2, y,
              (0.96, 0.97, 1.0), "m")
    y += int(32 * k)
    lx, rx = pad, W - pad
    # vertically BALANCE the stats block: centre the judgment/combo/acc/mods
    # block in the space between the name and the bottom-anchored score,
    # instead of leaving one big empty gap above the score (2026-07-22 polish).
    ms = (entry.mods_str or "").strip()
    has_mods = bool(ms and ms.upper() != "NM")
    block_h = 3 * int(23 * k) + int(8 * k) + int(23 * k) + int(30 * k)
    if has_mods:
        block_h += int(22 * k)
    slack = (H - int(84 * k)) - y - block_h
    y += max(0, int(slack * 0.5))
    # taiko judgment rows — GREAT / OK / MISS (great=count_300, ok=count_100,
    # miss=count_miss). No count_katu fold: taiko has no droplets.
    for lbl, val, col in (("Great", entry.counts[0], RESULT_COLORS["GREAT"]),
                          ("OK", entry.counts[1], RESULT_COLORS["OK"]),
                          ("Miss", entry.counts[3], RESULT_COLORS["MISS"])):
        _pil_text(d, font(16), lbl, lx, y, (0.62, 0.66, 0.76), "l")
        _pil_text(d, font(16), str(val), rx, y, col, "r")
        y += int(23 * k)
    y += int(8 * k)
    _pil_text(d, font(15), "Max Combo", lx, y, (0.62, 0.66, 0.76), "l")
    _pil_text(d, font(15), f"{entry.max_combo}x", rx, y, (0.96, 0.86, 0.42), "r")
    y += int(23 * k)
    _pil_text(d, font(15), "Accuracy", lx, y, (0.62, 0.66, 0.76), "l")
    _pil_text(d, font(15), f"{entry.accuracy:.2f}%", rx, y, (0.96, 0.86, 0.42), "r")
    y += int(30 * k)
    # mods badges (small pills) — reuse the .osr mods string (hoisted above
    # for the block-height measure)
    if has_mods:
        mods = [m for m in ms.split(",") if m][:5]
        mf = font(13)
        pill_h = int(22 * k)
        gap = int(6 * k)
        widths = []
        for m in mods:
            try:
                bb = mf.getbbox(m); w = bb[2] - bb[0]
            except AttributeError:
                w = mf.getsize(m)[0]                     # type: ignore
            widths.append(w + int(14 * k))
        total = sum(widths) + gap * (len(mods) - 1)
        mx = W / 2 - total / 2
        for m, w in zip(mods, widths):
            d.rounded_rectangle([mx, y, mx + w, y + pill_h], radius=pill_h // 2,
                                fill=(88, 70, 140, 235))
            _pil_text(d, mf, m, mx + w / 2, y + int(3 * k), (0.96, 0.95, 1.0), "m")
            mx += w + gap
    # big score + grade pill anchored to the bottom
    sy = H - int(84 * k)
    _pil_text(d, score_loader(max(int(30 * k), 10)), f"{entry.score:,}", W / 2,
              sy, (1.0, 1.0, 1.0), "m")
    grade = _norm_grade(entry.grade)
    gcol = FOR_RANK.get(grade, (0.8, 0.8, 0.85))
    glabel = grade if grade else "?"
    gpw = int(58 * k)
    gph = int(30 * k)
    gpx = W / 2 - gpw / 2
    gpy = H - int(42 * k)
    gc = tuple(int(round(c * 255)) for c in gcol)
    d.rounded_rectangle([gpx, gpy, gpx + gpw, gpy + gph], radius=gph // 2,
                        fill=(*gc, 255))
    lum = 0.299 * gcol[0] + 0.587 * gcol[1] + 0.114 * gcol[2]
    gfg = (0.06, 0.06, 0.09) if lum > 0.62 else (1.0, 1.0, 1.0)
    _pil_text(d, font(20), glabel, W / 2, gpy + int(4 * k), gfg, "m")
    return img


def bake_moment_pill(text: str, scale: float) -> Image.Image:
    """The rank-moment ribbon — NEW #1 = gold, NEW BEST = arc cyan."""
    up = "NEW #1" if "#1" in text else "NEW BEST"
    bg = (1.0, 0.80, 0.28) if "#1" in text else _ARC_CYAN
    font = _load_font(max(int(20 * scale), 9))
    try:
        x0, y0, x1, y1 = font.getbbox(up)
    except AttributeError:
        x1, y1 = font.getsize(up); x0 = y0 = 0          # type: ignore
    tw, th = x1 - x0, y1 - y0
    padx, pady = int(16 * scale), int(8 * scale)
    W = tw + 2 * padx
    H = th + 2 * pady
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    bgc = tuple(int(round(c * 255)) for c in bg)
    d.rounded_rectangle([0, 0, W - 1, H - 1], radius=H // 2, fill=(*bgc, 255))
    d.text((padx - x0, pady - y0), up, font=font, fill=(20, 18, 30, 255))
    return img


# --- board assembly + layout (once per render) -------------------------------------

@dataclass
class _BakedCard:
    img: Image.Image
    cx: float                 # rest-position centre (screen px)
    cy: float
    stagger: int              # entrance order (0 = nearest the centre stack)
    out_dir: int              # +1 slides right on entry, -1 slides left


@dataclass
class BakedBoard:
    cards: list                          # list[_BakedCard]
    banner: tuple | None                 # (PIL image, cx, cy) | None
    compact: bool = False                # ask draw_results to shrink the stack
                                         # (a side has 2+ cards → they crowd the
                                         # centre; 1/side leaves room for the
                                         # full featured caption)


def _avatar_bytes_for(entry, resolve_fn):
    """Avatar bytes for a flank card: a pre-fetched osu PNG (osu-global path),
    else a Discord avatar via the render-DB path, else None → procedural chip.
    Never raises."""
    try:
        ap = getattr(entry, "avatar_png", None)
        if ap:
            with open(ap, "rb") as fh:
                return fh.read() or None
        did = getattr(entry, "discord_user_id", None)
        if resolve_fn is not None and did:
            return resolve_fn(did)
    except Exception:  # noqa: BLE001 — avatars never break a bake
        return None
    return None


def bake_board(board, W: int, H: int, title: str, *,
               resolve_avatar_fn=None, max_per_side: int = 3,
               center_clear_px: float | None = None,
               cy_frac: float = 0.52) -> "BakedBoard | None":
    """Bake the flank cards + rank-moment banner and lay them out around the
    results screen's centre. Returns None when there are no flanks (solo
    render / empty board) — the caller then draws the plain results card,
    UNCHANGED.

    The board is CENTRED on the current play: cards ranked just ABOVE go to the
    LEFT (closest score nearest the centre), just BELOW to the RIGHT, marching
    outward. Cards auto-scale to fit the screen width around a centre clearance
    that keeps the featured content clear.

    `center_clear_px` overrides the cleared centre width (build_taiko_board
    passes the lazer featured panel's width so the cards hug the panel edges
    exactly like the std results screen); None → the legacy text-stack
    clearance (CENTER_CLEAR_FRAC). `cy_frac` = card-centre height fraction
    (0.5 = the panel's vertical centre)."""
    left = list(getattr(board, "left", []) or [])
    right = list(getattr(board, "right", []) or [])
    left = left[-max_per_side:] if max_per_side > 0 else []
    right = right[:max_per_side] if max_per_side > 0 else []
    if not left and not right:
        return None

    k = H / 1080.0
    outer_margin = int(W * 0.015)
    if center_clear_px is not None:
        center_clear = float(center_clear_px)
    else:
        center_clear = W * CENTER_CLEAR_FRAC   # legacy centred-stack clearance
    n = max(len(left), len(right), 1)
    half_avail = max((W - center_clear) / 2.0 - outer_margin, 1.0)
    slot = half_avail / n
    card_w = slot * 0.86
    # never upscale past native size; floor so a sparse board stays readable
    scale = max(0.42, min(card_w / LB_CARD_W, k))
    cw = LB_CARD_W * scale
    ch = LB_CARD_H * scale
    gap = max(slot - cw, 14 * scale)
    cy = H * cy_frac
    cl = (W - center_clear) / 2.0      # right edge of the innermost LEFT card
    cr = (W + center_clear) / 2.0      # left edge of the innermost RIGHT card

    cards: list = []
    # left: innermost first (rank R-1, nearest the centre), marching left
    for i, entry in enumerate(reversed(left)):
        img = bake_lb_card(entry, _avatar_bytes_for(entry, resolve_avatar_fn),
                           cw, ch, scale, _load_font, _load_font)
        cx = cl - gap - cw / 2.0 - i * (cw + gap)
        cards.append(_BakedCard(img, cx, cy, i, -1))
    for i, entry in enumerate(right):
        img = bake_lb_card(entry, _avatar_bytes_for(entry, resolve_avatar_fn),
                           cw, ch, scale, _load_font, _load_font)
        cx = cr + gap + cw / 2.0 + i * (cw + gap)
        cards.append(_BakedCard(img, cx, cy, i, +1))

    banner = _bake_banner(board, title, scale, W, H)
    compact = max(len(left), len(right)) >= 2
    return BakedBoard(cards=cards, banner=banner, compact=compact)


def _bake_banner(board, title, scale, W, H):
    """'#RANK on <map>' + the NEW #1 / NEW BEST pill, centred along the top edge
    (above catch's grade letter, so it never collides with the stack)."""
    rank_txt = f"#{board.rank} on {_clip_words(title, 48)}"
    f = _load_font(max(int(26 * scale), 13))
    try:
        bx0, by0, bx1, by1 = f.getbbox(rank_txt)
    except AttributeError:
        bx1, by1 = f.getsize(rank_txt); bx0 = by0 = 0   # type: ignore
    tw, th = bx1 - bx0, by1 - by0
    pill = bake_moment_pill(board.moment, scale) if getattr(board, "moment", None) else None
    gap = int(14 * scale) if pill else 0
    pw = pill.width if pill else 0
    ph = pill.height if pill else 0
    pad = int(12 * scale)
    Wb = tw + gap + pw + 2 * pad
    Hb = max(th, ph) + 2 * pad
    img = Image.new("RGBA", (max(Wb, 2), max(Hb, 2)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x = pad
    d.text((x - bx0, (Hb - th) // 2 - by0), rank_txt, font=f, fill=(228, 232, 244, 255))
    x += tw + gap
    if pill is not None:
        img.alpha_composite(pill, (int(x), (Hb - ph) // 2))
    return (img, W / 2.0, H * 0.05)


# --- per-frame compositing ---------------------------------------------------------

_SLIDE_OFFSET = 64.0   # px each card slides inward during the fade-in


def _paste(base: Image.Image, card: Image.Image, cx: float, cy: float,
           alpha: float) -> None:
    if alpha <= 0.003:
        return
    if alpha < 1.0:
        a = card.getchannel("A").point(lambda v: int(v * alpha))
        card = card.copy()
        card.putalpha(a)
    base.alpha_composite(card, (int(cx - card.width / 2.0),
                                int(cy - card.height / 2.0)))


def draw_board(base_rgba: Image.Image, baked: "BakedBoard | None",
               op: float) -> Image.Image:
    """Composite the flank cards + banner onto `base_rgba` (a PIL RGBA image) at
    the results-fade opacity `op`. A light per-card stagger keyed off `op` makes
    the board unfurl from the centre outward; once `op` saturates to 1 every
    card sits at rest and the draw is the static board. No-op when `baked` is
    None (solo render / leaderboard off → catch's plain results card)."""
    if baked is None:
        return base_rgba
    for c in baked.cards:
        head = c.stagger * 0.12                     # outer cards start later
        p = _clamp01((op - head) / max(1e-3, 1.0 - head))
        ease = 1.0 - (1.0 - p) ** 4                 # OutQuart
        if ease <= 0.003:
            continue
        x = c.cx + c.out_dir * (1.0 - ease) * _SLIDE_OFFSET
        _paste(base_rgba, c.img, x, c.cy, ease)
    if baked.banner is not None:
        bimg, bx, by = baked.banner
        _paste(base_rgba, bimg, bx, by, _clamp01(op))
    return base_rgba


# --- orchestration (called once from render_core) ----------------------------------

def build_taiko_board(cfg, meta, bm, replay_md5: str):
    """Build + bake the taiko results-screen leaderboard for one render, or None.

    Default source is the R3D render DB (best-per-player OTHER renders of this
    map). When cfg.leaderboard_source == "osu" AND cfg.leaderboard_json is set,
    the osu! GLOBAL scores the bot pre-fetched are used instead; any problem
    (missing/empty/invalid file) SILENTLY falls back to the render DB so the
    default path is never disturbed. Fully fail-soft — never raises."""
    from osu_taiko_renderer.hud import leaderboard as lb

    W, H = cfg.resolution
    src = str(getattr(cfg, "leaderboard_source", "r3d") or "r3d").lower()
    lb_json = getattr(cfg, "leaderboard_json", None)
    rows = None
    used_osu = False
    if src == "osu" and lb_json:
        try:
            import json
            raw = json.loads(Path(lb_json).read_text(encoding="utf-8"))
            if raw:
                rows = lb.rows_from_osu_json(raw)
                used_osu = bool(rows)
        except Exception:  # noqa: BLE001 — unreadable JSON → render-DB fallback
            rows = None
            used_osu = False
    if not rows:
        rows = lb.query_leaderboard(DB_PATH, meta.beatmap_md5, replay_md5)

    cur_score = int(getattr(meta, "score", 0) or 0)
    # prev best = this player's own best still in `rows` (the current replay is
    # already excluded by md5), so the NEW BEST moment can fire.
    cur_name = (getattr(meta, "player_name", "") or "").strip().lower()
    prev_best = None
    for r in rows:
        if (r.get("player_name") or "").strip().lower() == cur_name:
            prev_best = int(r.get("score") or 0)
            break

    board = lb.build_board(rows, meta.player_name, cur_score,
                           prev_best_score=prev_best, max_per_side=3)
    if not board.left and not board.right:
        return None

    title = f"{bm.artist} - {bm.title} [{bm.version}]".strip(" -")
    # lay the flanks out around the LAZER featured panel (std results-screen
    # geometry): the cleared centre = the panel width + a small breathing gap
    # each side, cards vertically centred on the panel. Falls back to the
    # legacy clearance if the lazer module is unavailable.
    try:
        from osu_taiko_renderer.hud.lazer_results import PANEL_W as _PANEL_W
    except Exception:  # noqa: BLE001 — stripped checkout → legacy layout
        _PANEL_W = None
    kw = {}
    if _PANEL_W is not None:
        kw = {"center_clear_px": _PANEL_W * (H / 1080.0), "cy_frac": 0.5}
    baked = bake_board(board, W, H, title,
                       resolve_avatar_fn=lb.resolve_avatar_bytes,
                       max_per_side=3, **kw)

    import sys
    moment = f" [{board.moment}]" if board.moment else ""
    srclbl = "osu!global" if used_osu else "render DB"
    print(f"board:  #{board.rank}/{board.n_players} on this map — "
          f"{len(board.left)} left + {len(board.right)} right{moment} "
          f"({srclbl})", file=sys.stderr)
    return baked
