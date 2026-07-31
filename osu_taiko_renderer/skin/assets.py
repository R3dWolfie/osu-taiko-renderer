"""Procedural taiko textures (before skin wiring).

Notes are drawn as two layers (like osu!taiko's taikohitcircle +
taikohitcircleoverlay): a tinted `note_body` (red don / blue kat) plus an
untinted white `note_rim` on top. Plus a `drum`/hit-target, a `ring` glow, and
a `bar` for drumroll bodies.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw

_TEX = 192

# Argon-ish taiko note colours (tint applied to note_body).
DON_COLOR = (0.90, 0.26, 0.28, 1.0)   # red, centre
KAT_COLOR = (0.27, 0.62, 0.92, 1.0)   # blue, rim
DRUMROLL_COLOR = (0.97, 0.76, 0.24, 1.0)  # yellow


def _note_body() -> np.ndarray:
    """White filled disc with a soft inner highlight (tinted per note)."""
    img = Image.new("RGBA", (_TEX, _TEX), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = 14
    d.ellipse([pad, pad, _TEX - pad, _TEX - pad], fill=(255, 255, 255, 255))
    # top highlight for a bit of dimension
    d.ellipse([_TEX * 0.28, _TEX * 0.20, _TEX * 0.62, _TEX * 0.46],
              fill=(255, 255, 255, 70))
    return np.array(img)


def _note_rim() -> np.ndarray:
    """White outline ring sized to sit just outside the body."""
    img = Image.new("RGBA", (_TEX, _TEX), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = 8
    d.ellipse([pad, pad, _TEX - pad, _TEX - pad], outline=(255, 255, 255, 255),
              width=max(4, _TEX // 22))
    return np.array(img)


def _default_hitcircle_base(n: int = 128) -> np.ndarray:
    """osu!'s DEFAULT legacy taikohitcircle: a plain filled white disc. It is
    multiply-tinted to the don/kat/drumroll colour, with taikohitcircleoverlay
    composited on top (see compose_skin_note). Used as the base when a user skin
    ships an overlay but no base circle — osu's user->default element resolution
    (rather than dropping to a full Argon note). Supersampled for a clean edge;
    128px matches the common overlay size so the overlay maps ~1:1."""
    ss = 4
    N = n * ss
    m = 2 * ss                          # tiny margin so the disc isn't clipped
    img = Image.new("L", (N, N), 0)
    ImageDraw.Draw(img).ellipse([m, m, N - 1 - m, N - 1 - m], fill=255)
    alpha = np.asarray(img.resize((n, n), Image.LANCZOS), dtype=np.uint8)
    out = np.zeros((n, n, 4), dtype=np.uint8)
    out[..., :3] = 255
    out[..., 3] = alpha
    return out


def _drum() -> np.ndarray:
    """The left hit-target: a faintly-filled disc with one crisp ring.

    osu!taiko's hit target is a single clean circle marking where notes
    land — not a busy multi-ring 'target'. We draw a soft translucent fill
    (so notes pop against it) plus one bright outline ring.
    """
    img = Image.new("RGBA", (_TEX, _TEX), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = 6
    d.ellipse([pad, pad, _TEX - pad, _TEX - pad], fill=(255, 255, 255, 28),
              outline=(255, 255, 255, 235), width=max(3, _TEX // 26))
    return np.array(img)


def _drum_big() -> np.ndarray:
    """Faint big-note reference ring drawn behind the hit target."""
    img = Image.new("RGBA", (_TEX, _TEX), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = 8
    d.ellipse([pad, pad, _TEX - pad, _TEX - pad],
              outline=(255, 255, 255, 90), width=max(2, _TEX // 40))
    return np.array(img)


def _ring() -> np.ndarray:
    """Soft glow ring (tinted for the hit-target highlight / explosions)."""
    img = Image.new("RGBA", (_TEX, _TEX), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = 6
    d.ellipse([pad, pad, _TEX - pad, _TEX - pad], outline=(255, 255, 255, 255),
              width=max(6, _TEX // 16))
    return np.array(img)


def _flash() -> np.ndarray:
    """Soft filled disc with a feathered edge — tinted for hit explosions."""
    n = _TEX
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    cx = cy = (n - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (n / 2.0)
    a = np.clip(1.0 - r, 0.0, 1.0) ** 1.6      # feather toward the edge
    img = np.zeros((n, n, 4), dtype=np.uint8)
    img[..., 0] = 255
    img[..., 1] = 255
    img[..., 2] = 255
    img[..., 3] = (a * 255).astype(np.uint8)
    return img


def _bar() -> np.ndarray:
    """Horizontal rounded bar for drumroll bodies (stretched in w)."""
    h = _TEX
    img = Image.new("RGBA", (_TEX, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = h // 2
    d.rounded_rectangle([2, h * 0.20, _TEX - 2, h * 0.80], radius=r,
                        fill=(255, 255, 255, 255))
    return np.array(img)


# --- input drum (4-zone press visualiser) ------------------------------------
# osu!taiko's left-side drum: an inner disc split left/right = the two CENTRE
# (don/red) inputs, an outer ring split left/right = the two RIM (kat/blue)
# inputs. Idle it's dim; the quadrant matching a pressed key flashes.

def _idrum_geom():
    n = _TEX
    c = (n - 1) / 2.0
    r_out = n / 2.0 - 6
    r_in = r_out * 0.55
    return n, c, r_out, r_in


def _zone_mask(*, ring: bool, left: bool) -> np.ndarray:
    """White RGBA texture masked to one drum quadrant (inner half-disc for a
    centre key, outer half-annulus for a rim key)."""
    n, c, r_out, r_in = _idrum_geom()
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    m = (r <= r_out) & (r > r_in) if ring else (r <= r_in)
    m &= (xx <= c) if left else (xx > c)
    img = np.zeros((n, n, 4), dtype=np.uint8)
    img[..., :3] = 255
    img[..., 3] = np.where(m, 255, 0).astype(np.uint8)
    return img


def _idrum_base() -> np.ndarray:
    """Dim idle drum: faint red centre disc + faint blue rim ring + white
    outlines (outer circle, inner circle, vertical divider)."""
    n, c, r_out, r_in = _idrum_geom()
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    buf = np.zeros((n, n, 4), dtype=np.float32)
    ring = (r <= r_out) & (r > r_in)
    inner = r <= r_in
    buf[ring] = [30, 48, 74, 205]      # faint blue rim
    buf[inner] = [86, 30, 36, 215]     # faint red centre
    img = Image.fromarray(buf.astype(np.uint8), "RGBA")
    d = ImageDraw.Draw(img)
    lw = max(3, n // 30)
    d.ellipse([c - r_out, c - r_out, c + r_out, c + r_out],
              outline=(238, 238, 248, 255), width=lw)
    d.ellipse([c - r_in, c - r_in, c + r_in, c + r_in],
              outline=(238, 238, 248, 235), width=max(2, n // 44))
    d.line([c, c - r_out + lw, c, c + r_out - lw],
           fill=(238, 238, 248, 220), width=max(2, n // 54))
    return np.array(img)


# --- R3D intro logo splash ---------------------------------------------------
# The glossy beveled 'R' tile shown during the intro (show_logo), fading out
# as the first note begins its scroll-in. Ported from the std renderer
# (osu_std_renderer.render.textures.bake_logo_tile) via the catch port
# (osu_catch_renderer.assets.bake_logo_tile): load R3D's REAL logo asset
# (own IP, license-clean -- the SAME logo.png the std/catch splashes use, so
# the splash is identical across modes) and fall back to a simple procedural
# red 'R' tile only if the asset is missing.
LOGO_TILE_RED = (216, 44, 54)


def bake_logo_tile(size: int = 256) -> np.ndarray:
    """RGBA tile for the intro splash. Prefers assets/logo.png (the real R3D
    logo); procedural fallback (rounded red tile + white R) only if missing."""
    try:
        lp = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "logo.png")
        im = Image.open(lp).convert("RGBA").resize((size, size), Image.LANCZOS)
        return np.asarray(im, dtype=np.uint8).copy()
    except Exception:
        pass
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    drw = ImageDraw.Draw(img)
    drw.rounded_rectangle([0, 0, size - 1, size - 1],
                          radius=int(size * 0.18), fill=LOGO_TILE_RED + (255,))
    try:
        from osu_taiko_renderer.skin.fonts import font as _font
        f = _font(int(size * 0.66))
        box = f.getbbox("R")
        rw, rh = box[2] - box[0], box[3] - box[1]
        drw.text(((size - rw) / 2.0 - box[0], (size - rh) / 2.0 - box[1]),
                 "R", font=f, fill=(255, 255, 255, 255))
    except Exception:
        pass
    return np.asarray(img, dtype=np.uint8).copy()


def logo_glow_rgba(size: int = 128) -> np.ndarray:
    """Soft white radial glow behind the logo tile (alpha falloff), tinted at
    draw time. Same bake as osu_catch_renderer.assets.catch_glow_rgba / the
    std renderer's "glow" so the splash halo matches across modes. (Taiko's
    sprite pass is straight-alpha only, so scene.py draws it non-additively —
    visually equivalent over the dark intro playfield.)"""
    import math
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    px = img.load()
    c = (size - 1) / 2.0
    for y in range(size):
        for x in range(size):
            d = math.hypot(x - c, y - c) / c
            a = max(0.0, 1.0 - d)
            a = a * a * a
            px[x, y] = (255, 255, 255, int(255 * a))
    return np.array(img)


def _fallback_drum_half(color, *, inner: bool, n: int = 192) -> np.ndarray:
    """Default input-drum press half (LEFT side): the left half of the drum
    circle, flat right edge on the centre line. inner=True -> filled CENTRE disc
    (don, red); inner=False -> RIM annulus (kat, blue). Baked when a legacy skin
    ships taiko-bar-left but omits taiko-drum-inner/-outer, so key presses still
    light the drum (osu! falls back to its default drum halves)."""
    W = n // 2
    yy, xx = np.mgrid[0:n, 0:W].astype(np.float32)
    cx, cy = float(W), (n - 1) / 2.0            # circle centre at the RIGHT edge
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (n / 2.0)
    m = (r <= 0.62) if inner else ((r > 0.62) & (r <= 1.0))
    out = np.zeros((n, W, 4), dtype=np.uint8)
    out[..., 0], out[..., 1], out[..., 2] = color
    out[..., 3] = np.where(m, 255, 0).astype(np.uint8)
    return out


def build_textures(skin_dir=None) -> dict[str, np.ndarray]:
    """Faithful Argon taiko textures (ported from lazer; see argon/). When a
    user .osk `skin_dir` provides legacy taiko note images, they override the
    Argon notes (user→Argon→wiki chain)."""
    from osu_taiko_renderer.argon import _const as C
    from osu_taiko_renderer.argon import textures as AT
    tex = {
        "argon_don": AT.bake_note(C.DON_TOP, C.DON_BOT, symbol="chevron"),
        "argon_kat": AT.bake_note(C.KAT_TOP, C.KAT_BOT, symbol="chevron"),
        "argon_don_glow": AT.bake_note_glow(C.DON_TOP, C.DON_BOT),
        "argon_kat_glow": AT.bake_note_glow(C.KAT_TOP, C.KAT_BOT),
        "argon_drumroll": AT.bake_note(C.DRUMROLL_TOP, C.DRUMROLL_BOT, symbol=None),
        "argon_drumroll_body": AT.bake_drumroll_body(),
        "argon_drumroll_glow": AT.bake_note_glow(C.DRUMROLL_TOP, C.DRUMROLL_BOT),
        "argon_tick": AT.bake_tick(),
        "argon_swell": AT.bake_note(C.SWELL_TOP, C.SWELL_BOT, symbol="asterisk"),
        "argon_swell_ring": AT.bake_swell_ring(),
        "argon_swell_glow": AT.bake_note_glow(C.SWELL_TOP, C.SWELL_BOT),
        "argon_note_flash": AT.bake_note_flash(),
        "argon_hit_target": AT.bake_hit_target(),
        "argon_drum_idle": AT.bake_drum_idle(),
        "argon_drum_centre_l": AT.bake_drum_flash(ring=False, left=True),
        "argon_drum_centre_r": AT.bake_drum_flash(ring=False, left=False),
        "argon_drum_rim_l": AT.bake_drum_flash(ring=True, left=True),
        "argon_drum_rim_r": AT.bake_drum_flash(ring=True, left=False),
        "argon_explosion_centre": AT.bake_explosion(C.CENTRE_HIT_GRAD, C.CENTRE_HIT_GLOW),
        "argon_explosion_rim": AT.bake_explosion(C.RIM_HIT_GRAD, C.RIM_HIT_GLOW),
        "argon_barline_anchor": AT.bake_barline_anchor(),
        "argon_barline_anchor_f": AT.bake_barline_anchor()[::-1].copy(),
    }
    # --- user .osk note override (legacy taiko skin) ---
    from osu_taiko_renderer.skin.taiko_skin import TaikoSkin
    skin = TaikoSkin(skin_dir)
    hc = skin.load("taikohitcircle")
    ov = skin.load("taikohitcircleoverlay")
    bc = skin.load("taikobigcircle")
    bov = skin.load("taikobigcircleoverlay")
    DON, KAT, GOLD = (235, 105, 85), (116, 177, 207), (252, 140, 70)  # lazer note colours
    # osu resolves a MISSING taikohitcircle to the DEFAULT skin's base circle,
    # NOT to a foreign-mode note. A skin that ships taikohitcircleoverlay but no
    # base (overlay-only, like osu_13811400) is still a legacy note: composite
    # the skin's overlay onto a default legacy base disc instead of falling back
    # to a full Argon note. A present-but-blank overlay is already resolved to
    # its -0 animation frame by TaikoSkin.find (osu -0-blanks-static rule).
    # When the skin ships an overlay but no matching CIRCLE, the base is a
    # synthesised default disc whose pixel size is arbitrary — tell compose to
    # fit the overlay to the note (fit_overlay) so the rim is concentric and
    # note-sized rather than mis-scaled/off-centre (see compose_skin_note).
    hc_fit = hc is None and ov is not None
    bc_fit = bc is None and bov is not None
    if hc_fit:
        hc = _default_hitcircle_base()
    if bc_fit:
        bc = _default_hitcircle_base()
    if hc is not None:
        tex["argon_don"] = AT.compose_skin_note(hc, ov, DON, fit_overlay=hc_fit)
        tex["argon_kat"] = AT.compose_skin_note(hc, ov, KAT, fit_overlay=hc_fit)
        tex["argon_drumroll"] = AT.compose_skin_note(hc, ov, GOLD, fit_overlay=hc_fit)
    # big-note textures: skin's taikobigcircle if present, else reuse the normal
    # note (the scene scales it — same as the Argon path).
    if bc is not None:
        tex["argon_don_big"] = AT.compose_skin_note(bc, bov, DON, fit_overlay=bc_fit)
        tex["argon_kat_big"] = AT.compose_skin_note(bc, bov, KAT, fit_overlay=bc_fit)
    else:
        tex["argon_don_big"] = tex["argon_don"]
        tex["argon_kat_big"] = tex["argon_kat"]
    # --- mascot (pippidon): idle/kiai/clear/fail animation frames ->
    # pippidon_<state>_<i> texture keys (drawn behind the notes in scene.py) ---
    for _st in ("idle", "kiai", "clear", "fail"):
        _i = 0
        while True:
            _fr = skin.load(f"pippidon{_st}{_i}")
            if _fr is None:
                break
            tex[f"pippidon_{_st}_{_i}"] = _fr
            _i += 1
        if _i == 0:                               # static (un-numbered) -> single frame 0
            _fr = skin.load(f"pippidon{_st}")
            if _fr is not None and _fr.shape[0] > 2 and _fr.shape[1] > 2:
                tex[f"pippidon_{_st}_0"] = _fr    # (skip 1x1 "hidden" placeholders)
    # --- the rest of the legacy taiko playfield (drum, hit target, judgements,
    # bar line, drumroll) — loaded raw; the scene/compositor use them when the
    # skin provides them, else fall back to Argon. ---
    for key, name in (
        ("skin_drum_idle", "taiko-bar-left"),
        ("skin_drum_inner", "taiko-drum-inner"),
        ("skin_drum_outer", "taiko-drum-outer"),
        ("skin_hit_target", "taiko-bar-right"),
        ("skin_hit_glow", "taiko-bar-right-glow"),
        ("skin_barline", "taiko-barline"),
        ("skin_roll_mid", "taiko-roll-middle"),
        ("skin_roll_end", "taiko-roll-end"),
        ("skin_hit_great", "taiko-hit300"),
        ("skin_hit_ok", "taiko-hit100"),
        ("skin_hit_miss", "taiko-hit0"),
        # legacy denden (swell): osu!'s LegacySwell uses the std spinner sprites.
        ("skin_spinner_circle", "spinner-circle"),
        ("skin_spinner_approach", "spinner-approachcircle"),
        ("skin_spinner_warning", "spinner-warning"),
    ):
        img = skin.load(name)
        if img is not None:
            tex[key] = img
    # A legacy skin can ship taiko-bar-left (the drum background) yet omit the
    # taiko-drum-inner/-outer press halves. osu! then flashes its DEFAULT drum
    # halves, so key presses still light the drum — bake red centre (don) + blue
    # rim (kat) fallbacks (colours = LegacyHit centre/rim) so it isn't static.
    if "skin_drum_idle" in tex:                 # skin has taiko-bar-left
        if "skin_drum_inner" not in tex:
            tex["skin_drum_inner"] = _fallback_drum_half((235, 69, 44), inner=True)
        if "skin_drum_outer" not in tex:
            tex["skin_drum_outer"] = _fallback_drum_half((67, 142, 172), inner=False)
    # mirrored drum halves for the right-side press (legacy drum-inner/outer are
    # left-half graphics).
    for base in ("skin_drum_inner", "skin_drum_outer"):
        if base in tex:
            tex[base + "_r"] = np.ascontiguousarray(tex[base][:, ::-1])
    # mirrored drumroll end cap: taiko-roll-end is the RIGHT (round-right) cap;
    # flipped it becomes the LEFT (round-left) head cap so the roll reads as a
    # rounded bar built from the skin's own art (osu! legacy drumroll look).
    if "skin_roll_end" in tex:
        tex["skin_roll_end_l"] = np.ascontiguousarray(tex["skin_roll_end"][:, ::-1])
    return tex
