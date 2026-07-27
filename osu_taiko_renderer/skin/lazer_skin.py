"""Procedural osu!lazer-style base UI elements (no skin sprites).

Red's parity target is osu!lazer's built-in look as the BASE; custom skins
layer on later. Colours/geometry are measured directly from the lazer
reference render (Night05 @ EZ): the catcher plate is a violet->magenta
vertical-gradient trapezoid (top wider than bottom) with a white outline.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

# Measured from the reference: top fill (171,85,254) -> bottom fill (246,8,254),
# trapezoid bottom width / top width = 235/284 ~= 0.827, height/topwidth = 48/284.
_TOP_RGB = (171, 85, 254)
_BOT_RGB = (246, 8, 254)
CATCHER_BOTTOM_RATIO = 235 / 284
CATCHER_ASPECT = 48 / 284          # height / top-width


def catcher_rgba(top_w: int = 568) -> np.ndarray:
    """RGBA texture of the lazer catcher plate, plate filling the top edge."""
    plate_h = int(round(top_w * CATCHER_ASPECT))
    pad = max(4, top_w // 90)                      # room for the white outline
    W, H = top_w + pad * 2, plate_h + pad * 2
    bot_w = top_w * CATCHER_BOTTOM_RATIO
    cx = W / 2.0
    y0, y1 = pad, pad + plate_h
    top = [(cx - top_w / 2, y0), (cx + top_w / 2, y0)]
    bot = [(cx + bot_w / 2, y1), (cx - bot_w / 2, y1)]
    poly = [top[0], top[1], bot[0], bot[1]]

    # vertical gradient, masked to the trapezoid
    grad = np.zeros((H, W, 4), np.uint8)
    for y in range(H):
        f = min(1.0, max(0.0, (y - y0) / max(1, plate_h)))
        grad[y, :, 0] = int(_TOP_RGB[0] + (_BOT_RGB[0] - _TOP_RGB[0]) * f)
        grad[y, :, 1] = int(_TOP_RGB[1] + (_BOT_RGB[1] - _TOP_RGB[1]) * f)
        grad[y, :, 2] = int(_TOP_RGB[2] + (_BOT_RGB[2] - _TOP_RGB[2]) * f)
    grad[:, :, 3] = 255                              # opaque fill (mask clips it)
    fill = Image.fromarray(grad)
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).polygon(poly, fill=255)
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    out.paste(fill, (0, 0), mask)
    # white rounded outline
    ow = max(2, top_w // 110)
    ImageDraw.Draw(out).polygon(poly, outline=(255, 255, 255, 255), width=ow)
    return np.asarray(out)
