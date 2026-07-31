"""Bake Argon taiko element textures (RGBA uint8) faithfully from the lazer
drawable definitions in `_const`. The GL sprite path tints by a single solid
colour, so per-element vertical gradients / glows are baked into the pixels here.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import _const as C

_N = 256   # base texture resolution for a normal note


def _vgrad(n, top, bot):
    """Vertical RGBA gradient (top→bottom), shape (n,n,4) float 0..1."""
    ts = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]      # (n,1)
    g = np.empty((n, 4), dtype=np.float32)
    for i in range(4):
        g[:, i] = (top[i] + (bot[i] - top[i]) * ts[:, 0]) / 255.0
    return np.repeat(g[:, None, :], n, axis=1)                    # (n,n,4)


def _radius(n):
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    return np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / (n / 2.0), c


def _over(dst, src_rgb, src_a):
    """Alpha-composite src (rgb float (n,n,3), a float (n,n)) onto dst RGBA."""
    a = src_a[..., None]
    dst[..., :3] = dst[..., :3] * (1 - a) + src_rgb * a
    dst[..., 3] = dst[..., 3] + src_a * (1 - dst[..., 3])


def _chevron_mask(n, asterisk=False, y_off=0.0):
    """White FontAwesome-AngleLeft chevron (or Asterisk for swell) coverage.
    `y_off` shifts the chevron vertically as a fraction of `n` (negative = up)
    for optical centring; the asterisk stays on the geometric centre."""
    img = Image.new("L", (n, n), 0)
    d = ImageDraw.Draw(img)
    c = n / 2.0
    cy = c + y_off * n
    if asterisk:
        # 6-arm asterisk (FontAwesome Solid asterisk) — swell, on the icon box.
        import math
        h = C.ICON_SIZE * n
        w = h * C.ICON_X_SCALE
        lw = int(max(2, h * 0.22))
        for k in range(6):
            ang = math.pi / 6 + k * math.pi / 3
            dx, dy = math.cos(ang) * w * 0.55, math.sin(ang) * h * 0.55
            d.line([(c - dx, c - dy), (c + dx, c + dy)], fill=255, width=lw)
    else:
        # note / drumroll-tick chevron — sized to the real game (see _const).
        h = C.CHEVRON_SIZE * n
        w = h * C.CHEVRON_X_SCALE
        lw = int(max(3, h * 0.20))
        pts = [(c + w / 2, cy - h / 2), (c - w / 2, cy), (c + w / 2, cy + h / 2)]
        d.line(pts, fill=255, width=lw, joint="curve")
        for p in pts:                       # round the caps
            d.ellipse([p[0] - lw / 2, p[1] - lw / 2, p[0] + lw / 2, p[1] + lw / 2],
                      fill=255)
    return np.array(img).astype(np.float32) / 255.0


def _as_rgba(im):
    """Force an image array to (H, W, 4) uint8 so skins that ship RGB or
    greyscale note art don't break compositing."""
    if im.ndim == 2:
        im = np.stack([im, im, im, np.full_like(im, 255)], axis=-1)
    elif im.shape[-1] == 3:
        im = np.concatenate(
            [im, np.full(im.shape[:2] + (1,), 255, im.dtype)], axis=-1)
    return im


def _content_bbox(arr, thr=20):
    """(x0, y0, x1, y1) of the opaque content (alpha > thr) in an RGBA array,
    or None when fully transparent. Used to align/scale a note's base + overlay
    by their VISIBLE geometry rather than their (possibly padded) canvas."""
    a = arr[..., 3]
    ys, xs = np.where(a > thr)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _content_center(arr):
    bb = _content_bbox(arr)
    if bb is None:
        return (arr.shape[1] - 1) / 2.0, (arr.shape[0] - 1) / 2.0
    return (bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0


def compose_skin_note(hc, overlay, tint, n=_N, *, fit_overlay=False):
    """A legacy-skin taiko note: `taikohitcircle` multiply-tinted by the note
    colour (don red / kat blue / drumroll gold) with the untinted white
    `taikohitcircleoverlay` composited on top. hc/overlay are RGBA arrays.

    The overlay (rim) is a SEPARATE skin sprite that osu draws Origin=Centre on
    the circle — i.e. concentric with it. We compose the two into one texture, so
    they must land on the SAME centre. Align by VISIBLE-CONTENT centre (not the
    canvas), so an overlay with asymmetric transparent padding still sits
    concentric with the circle instead of drifting off to one side.

    `fit_overlay`: set when `hc` is a SYNTHESISED default base (the skin shipped
    an overlay but no matching circle, e.g. taikobigcircleoverlay without
    taikobigcircle). The default base's pixel size is arbitrary, so the raw
    overlay/circle ratio is meaningless (and blows the rim up ~2x for an @2x
    overlay). Instead scale the overlay so its content spans the base's content
    diameter — a concentric rim at the note size, @2x-independent."""
    hc = _as_rgba(hc)
    base = np.array(Image.fromarray(hc).resize((n, n), Image.LANCZOS)).astype(np.float32)
    base[..., :3] *= np.array(tint, np.float32) / 255.0
    if overlay is not None:
        overlay = _as_rgba(overlay)
        if fit_overlay:
            # default base: size the overlay to the base's content diameter and
            # align by CONTENT centre, so a padded / @2x / off-centre overlay
            # still sits concentric with the note at the right size.
            obb, bbb = _content_bbox(overlay), _content_bbox(base)
            o_d = max(obb[2] - obb[0], obb[3] - obb[1]) + 1 if obb else overlay.shape[0]
            b_d = max(bbb[2] - bbb[0], bbb[3] - bbb[1]) + 1 if bbb else n
            osz = max(1, int(round(overlay.shape[0] * (b_d / o_d))))
            o = np.array(Image.fromarray(overlay).resize((osz, osz), Image.LANCZOS)).astype(np.float32)
            bcx, bcy = _content_center(base)
            ocx, ocy = _content_center(o)
            offx, offy = int(round(bcx - ocx)), int(round(bcy - ocy))
        else:
            # skin ships BOTH circle + overlay: keep osu's native-size ratio and
            # canvas-centring (Origin=Centre) so a deliberately larger/smaller or
            # off-centre rim is reproduced exactly as osu draws it.
            osz = max(1, int(round(n * (overlay.shape[0] / hc.shape[0]))))
            o = np.array(Image.fromarray(overlay).resize((osz, osz), Image.LANCZOS)).astype(np.float32)
            offx = offy = (n - osz) // 2
        by0, bx0 = max(0, offy), max(0, offx)
        by1, bx1 = min(n, offy + osz), min(n, offx + osz)
        if by1 > by0 and bx1 > bx0:
            oc = o[by0 - offy:by1 - offy, bx0 - offx:bx1 - offx]
            reg = base[by0:by1, bx0:bx1]
            a = oc[..., 3:4] / 255.0
            reg[..., :3] = reg[..., :3] * (1 - a) + oc[..., :3] * a
            reg[..., 3:4] = reg[..., 3:4] + a * (255.0 - reg[..., 3:4])
    return np.clip(base, 0, 255).astype(np.uint8)


def bake_note(top, bot, *, symbol="chevron", n=_N):
    """ArgonCirclePiece: black core (a=190) + thick ring (accent×0.5α) + thin
    ring (full accent) + centre symbol. `top`/`bot` = vertical accent gradient."""
    r, _ = _radius(n)
    out = np.zeros((n, n, 4), dtype=np.float32)
    grad = _vgrad(n, top, bot)              # (n,n,4), alpha from colour (255)
    # 1) black core, full circle, alpha 190
    core = (r <= 1.0).astype(np.float32) * (C.CORE_RGBA[3] / 255.0)
    _over(out, np.zeros((n, n, 3), np.float32), core)
    # 2) thick ring (ring1): outer 0.5714 of radius, accent × 0.5 alpha
    in1 = 1.0 - C.RING1_THICKNESS * 2.0     # thickness is relative to diameter
    m1 = ((r >= in1) & (r <= 1.0)).astype(np.float32) * C.RING1_ALPHA
    _over(out, grad[..., :3], m1)
    # 3) thin ring (ring2): outer 0.1428 of radius, full accent
    in2 = 1.0 - C.RING2_THICKNESS * 2.0
    m2 = ((r >= in2) & (r <= 1.0)).astype(np.float32)
    _over(out, grad[..., :3], m2)
    # 4) symbol
    if symbol:
        yoff = C.ICON_Y_OFFSET if symbol == "chevron" else 0.0
        sym = _chevron_mask(n, asterisk=(symbol == "asterisk"), y_off=yoff)
        _over(out, np.ones((n, n, 3), np.float32), sym)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def bake_swell_ring(n=512):
    """DefaultSwell target ring: two concentric borders (thick 1.4 + thin 1.0,
    YellowDark@0.25 additive in lazer). Baked WHITE and 2x supersampled at high
    resolution so it stays CRISP when the swell scales it up to 5× the note
    (a low-res un-antialiased ring stair-steps badly when magnified); the caller
    tints it YellowDark + draws it at reduced alpha to match lazer's faint gold
    ring rather than the old hard white one."""
    ss = 2
    S = n * ss
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    m = int(S * 0.02)
    # thick outer border (target_ring_thick_border) near the edge
    d.ellipse([m, m, S - m, S - m], outline=(255, 255, 255, 255),
              width=max(2, int(S * 0.013)))
    # thin inner border (target_ring_thin_border), just inside the thick one
    gap = int(S * 0.045)
    d.ellipse([m + gap, m + gap, S - m - gap, S - m - gap],
              outline=(255, 255, 255, 200), width=max(1, int(S * 0.007)))
    img = img.resize((n, n), Image.LANCZOS)      # antialias down to base res
    return np.array(img)


def bake_barline_anchor(n=64):
    """ArgonBarLine major top/bottom anchor: a vertical white gradient
    (transparent at the far end → opaque white toward the bar). Row 0 = top
    (transparent), row n-1 = bottom (white)."""
    a = np.linspace(0.0, 1.0, n, dtype=np.float32)        # 0 (top) → 1 (bottom)
    out = np.zeros((n, 2, 4), np.float32)
    out[..., :3] = 1.0
    out[:, :, 3] = a[:, None]
    return (out * 255).astype(np.uint8)


def bake_tick(n=_N):
    """White chevron for drumroll ticks (ArgonTickPiece, AngleLeft)."""
    m = _chevron_mask(n)
    out = np.zeros((n, n, 4), np.float32)
    out[..., :3] = 1.0
    out[..., 3] = m
    return (out * 255).astype(np.uint8)


def bake_note_glow(top, bot, n=_N):
    """Soft accent-coloured halo drawn behind a note (cheap stand-in for
    lazer's bloom). A filled disc of the mid accent colour, gaussian-blurred."""
    mid = tuple((top[i] + bot[i]) // 2 for i in range(3))
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = int(n * 0.16)
    d.ellipse([pad, pad, n - pad, n - pad], fill=(mid[0], mid[1], mid[2], 255))
    img = img.filter(ImageFilter.GaussianBlur(radius=n * 0.10))
    arr = np.array(img).astype(np.float32)
    arr[..., 3] *= 0.55                       # overall halo strength
    return arr.astype(np.uint8)


def bake_note_flash(n=_N):
    """Additive white hit-flash disc (ArgonCirclePiece flash layer)."""
    r, _ = _radius(n)
    a = (r <= 1.0).astype(np.float32)
    out = np.zeros((n, n, 4), np.float32)
    out[..., :3] = 1.0
    out[..., 3] = a
    return (out * 255).astype(np.uint8)


def _glow(n, color, radius_frac):
    """Soft additive glow disc: filled circle blurred by radius_frac*n."""
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = int(n * 0.30)
    d.ellipse([pad, pad, n - pad, n - pad], fill=(color[0], color[1], color[2], 255))
    img = img.filter(ImageFilter.GaussianBlur(radius=n * radius_frac))
    return np.array(img).astype(np.uint8)


def bake_hit_target(n=_N):
    """ArgonHitTarget: two faint additive white circles (note size + ×0.85)."""
    r, _ = _radius(n)
    out = np.zeros((n, n, 4), np.float32)
    outer = (r <= 1.0).astype(np.float32) * C.HIT_TARGET_CIRCLE_ALPHA
    inner = (r <= C.HIT_TARGET_INNER_SCALE).astype(np.float32) * C.HIT_TARGET_CIRCLE_ALPHA
    out[..., :3] = 1.0
    out[..., 3] = np.clip(outer + inner, 0, 1)
    return (out * 255).astype(np.uint8)


def _hgrad(n, left, right):
    """Horizontal RGBA gradient (left→right), shape (n,n,4) float 0..1."""
    ts = np.linspace(0.0, 1.0, n, dtype=np.float32)[None, :]
    g = np.empty((4, n), dtype=np.float32)
    for i in range(4):
        g[i] = (left[i] + (right[i] - left[i]) * ts[0]) / 255.0
    return np.transpose(np.repeat(g[:, None, :], n, axis=1), (1, 2, 0))


def bake_drum_idle(n=_N):
    """ArgonInputDrum idle: gray rim circle (51/255) + gray centre disc
    (64/255, size 0.7) + central vertical split bars."""
    r, c = _radius(n)
    out = np.zeros((n, n, 4), np.float32)
    rim = (r <= 1.0).astype(np.float32)
    _over(out, np.full((n, n, 3), C.DRUM_RIM_GRAY[0] / 255.0), rim)
    centre = (r <= (1.0 - C.DRUM_RIM_SIZE)).astype(np.float32)
    _over(out, np.full((n, n, 3), C.DRUM_CENTRE_GRAY[0] / 255.0), centre)
    # vertical split: thin tall bar + shorter bar (over the centre)
    sw = max(1, int(C.DRUM_MIDDLE_SPLIT / C.BASE_HEIGHT * n))
    x0 = int(c - sw / 2)
    yy = np.arange(n)
    mask_full = (np.abs(np.arange(n) - c) <= sw / 2)
    out[:, x0:x0 + sw, :3] = C.DRUM_SPLIT_GRAY_A[0] / 255.0
    out[:, x0:x0 + sw, 3] = np.maximum(out[:, x0:x0 + sw, 3], rim[:, x0:x0 + sw])
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def bake_ring(size: int = 64, thickness: int = 8) -> np.ndarray:
    """White hollow ring (CircularContainer w/ BorderThickness) for the
    judgement RingExplosion. Tinted + additive-blended at draw time."""
    from PIL import Image, ImageDraw
    ss = 2
    S = size * ss
    Tk = max(ss, thickness * ss)
    m = ss * 3
    img = Image.new("RGBA", (S + 2 * m, S + 2 * m), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse([m, m, m + S, m + S],
                                outline=(255, 255, 255, 255), width=Tk)
    img = img.resize((size + 6, size + 6), Image.LANCZOS)
    return np.array(img)


def bake_drum_flash(*, ring: bool, left: bool, n=_N):
    """Additive press-flash for one drum half (ArgonInputDrumHalf): a FLAT accent
    fill (rim annulus or centre half-disc, no inner gradient) with the glow living
    strictly OUTSIDE the filled area — the halo is a blurred silhouette minus the
    fill itself, so the highlight reads flat and only bleeds outward.

    Fills are chosen SATURATED (see _const) because the flash composites additively
    over the dark idle drum; the compositor's LANCZOS downscale to drum size
    anti-aliases the hard shape/split edges. NOTE: an earlier version used a soft
    radial falloff to avoid rapid-mash "bars"; Red's Argon spec is a flat fill, so
    the hard two-semicircle split (matching lazer's drum + the idle split) is
    intentional here."""
    r, c = _radius(n)
    if ring:
        fill = C.RIM_HIT_FILL
        glow_c = C.RIM_HIT_GLOW
        shape = ((r > (1.0 - C.DRUM_RIM_SIZE)) & (r <= 1.0)).astype(np.float32)
    else:
        fill = C.CENTRE_HIT_FILL
        glow_c = C.CENTRE_HIT_GLOW
        shape = (r <= (1.0 - C.DRUM_RIM_SIZE)).astype(np.float32)
    half = ((np.arange(n)[None, :] <= c) if left else
            (np.arange(n)[None, :] > c)).astype(np.float32)
    shape = shape * half

    out = np.zeros((n, n, 4), np.float32)

    # 1) OUTER glow only: blur the fill silhouette, subtract the fill so nothing
    # is added inside the highlight (no inner gradient) — halo bleeds outward.
    blur = C.DRUM_GLOW_RADIUS / C.BASE_HEIGHT * n * 0.6
    sm = Image.fromarray((shape * 255.0).astype(np.uint8), "L").filter(
        ImageFilter.GaussianBlur(radius=blur))
    glow_a = np.clip(np.asarray(sm, np.float32) / 255.0 - shape, 0.0, 1.0)
    out[..., 0], out[..., 1], out[..., 2] = (glow_c[0] / 255.0,
                                             glow_c[1] / 255.0, glow_c[2] / 255.0)
    out[..., 3] = glow_a * C.DRUM_GLOW_STRENGTH

    # 2) FLAT fill on top: one uniform accent colour across the highlighted area.
    fill_rgb = np.array(fill[:3], np.float32) / 255.0
    out[..., :3] = out[..., :3] * (1 - shape[..., None]) + fill_rgb * shape[..., None]
    out[..., 3] = np.maximum(out[..., 3], shape)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def bake_drumroll_body(n=_N):
    """ArgonElongatedCirclePiece body cross-section: the note's vertical profile
    (gold thin ring / thick ring×0.5 / black core / …) made uniform across the
    width, so stretching it horizontally yields a straight capsule body. The
    rounded gold ends are drawn separately as cap circles."""
    yy = np.abs(np.arange(n) - (n - 1) / 2.0) / (n / 2.0)     # 0 centre .. 1 edge
    top, bot = C.DRUMROLL_TOP, C.DRUMROLL_BOT
    col = np.empty((n, 4), np.float32)
    ts = np.linspace(0.0, 1.0, n)
    for i in range(4):
        col[:, i] = (top[i] + (bot[i] - top[i]) * ts) / 255.0   # vertical gradient
    out = np.zeros((n, 4), np.float32)
    core = (yy <= 1.0) * (C.CORE_RGBA[3] / 255.0)
    out[:, 3] = np.maximum(out[:, 3], core)
    in1 = 1.0 - C.RING1_THICKNESS * 2.0
    m1 = ((yy >= in1) & (yy <= 1.0)) * C.RING1_ALPHA
    in2 = 1.0 - C.RING2_THICKNESS * 2.0
    m2 = ((yy >= in2) & (yy <= 1.0)).astype(np.float32)
    rows = np.zeros((n, 4), np.float32)
    a = core.copy()
    rgb = np.zeros((n, 3), np.float32)                          # core black
    rgb = rgb * (1 - m1[:, None]) + col[:, :3] * m1[:, None]; a = a + m1 * (1 - a)
    rgb = rgb * (1 - m2[:, None]) + col[:, :3] * m2[:, None]; a = np.maximum(a, m2)
    rows[:, :3] = rgb
    rows[:, 3] = a
    body = np.repeat(rows[:, None, :], 8, axis=1)               # 8px wide, stretched by caller
    return (np.clip(body, 0, 1) * 255).astype(np.uint8)


def bake_explosion(grad, glow_color, n=_N, inner_scale=0.35, inner_alpha=0.0):
    """ArgonHitExplosion: an accent-coloured burst at the hit target — a filled
    accent disc + accent glow (don red / kat blue), so a big don reads as a larger
    SATURATED RED. The old inner white@0.85 covered the accent almost entirely, so
    every big centre hit flashed a solid WHITE disc over the note; worse, the
    additive explosions STACK on a dense stream and any white core washes the whole
    burst to white. A pure accent disc instead saturates to clean red/blue when it
    stacks (the red channel clips first), matching lazer's accent-tinted explosion.
    inner_alpha>0 re-adds a soft white hot-centre if ever wanted."""
    r, _ = _radius(n)
    out = np.zeros((n, n, 4), np.float32)
    g = _glow(n, glow_color, C.EXPLOSION_GLOW_RADIUS / 200.0)
    out = g.astype(np.float32) / 255.0
    outer = (r <= 1.0).astype(np.float32)
    grd = _vgrad(n, grad[0], grad[1])
    _over(out, grd[..., :3], outer * 0.95)                # accent disc dominates
    inner = (r <= inner_scale).astype(np.float32) * inner_alpha
    _over(out, np.ones((n, n, 3), np.float32), inner)     # soft white hot-centre
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)
