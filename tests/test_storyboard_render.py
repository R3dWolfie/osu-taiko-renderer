"""Storyboard RENDERER (phase 3): the pure-math transform layer.

Covers the 640x480 -> output transform + widescreen/letterbox viewport, the
9 origin anchors combined with the flip origin-adjust, the exact centre solve
for non-centre / rotated / mirrored sprites, and the animation frame index
(LoopForever wrap vs LoopOnce clamp).  Every expectation traces to
osu!(lazer) master (cited in render/storyboard_render.py).  A GL smoke test
proves pixels come out when an EGL context is available (skips otherwise).
"""
import math

from osu_taiko_renderer.beatmap.storyboard import Origin
from osu_taiko_renderer.render.storyboard_render import (
    ORIGIN_ANCHOR, anim_frame_index, compute_quad, storyboard_viewport)


def _close(a, b, tol=1e-4):
    return abs(a - b) <= tol


# --------------------------------------------------------------------------- #
# 640x480 -> output transform + viewport                                        #
# --------------------------------------------------------------------------- #

def test_centre_sprite_maps_to_screen_centre():
    # 1920x1080: k = 1080/480 = 2.25, centre (960,540).
    k, cx, cy = 1080 / 480.0, 960.0, 540.0
    scr_cx, scr_cy, w, h, theta, mx, my = compute_quad(
        Origin.CENTRE, 320, 240, 1.0, 1.0, 0.0, False, False, 100, 100,
        k, cx, cy)
    assert _close(scr_cx, 960.0) and _close(scr_cy, 540.0)
    assert _close(w, 225.0) and _close(h, 225.0)       # 100 * 2.25
    assert not mx and not my and _close(theta, 0.0)


def test_topleft_corner_position():
    # TopLeft origin at osu (0,0): the CORNER sits at screen (240,0);
    # the sprite CENTRE is corner + (w/2,h/2).
    k, cx, cy = 2.25, 960.0, 540.0
    scr_cx, scr_cy, w, h, *_ = compute_quad(
        Origin.TOP_LEFT, 0, 0, 1.0, 1.0, 0.0, False, False, 100, 100, k, cx, cy)
    corner_x = scr_cx - w / 2.0
    corner_y = scr_cy - h / 2.0
    assert _close(corner_x, 960.0 + (0 - 320) * k)     # 240
    assert _close(corner_y, 540.0 + (0 - 240) * k)     # 0
    assert _close(scr_cx, 352.5) and _close(scr_cy, 112.5)


def test_fliph_keeps_footprint_mirrors_content():
    # A flipped TopLeft sprite occupies the SAME footprint as the unflipped
    # one (StoryboardExtensions.AdjustOrigin swaps the anchor edge); only the
    # UV is mirrored.
    k, cx, cy = 2.25, 960.0, 540.0
    base = compute_quad(Origin.TOP_LEFT, 0, 0, 1.0, 1.0, 0.0, False, False,
                        100, 100, k, cx, cy)
    flip = compute_quad(Origin.TOP_LEFT, 0, 0, 1.0, 1.0, 0.0, True, False,
                        100, 100, k, cx, cy)
    assert _close(base[0], flip[0]) and _close(base[1], flip[1])   # same centre
    assert _close(base[2], flip[2]) and _close(base[3], flip[3])   # same size
    assert flip[5] is True and base[5] is False                    # mirror_x


def test_negative_vscale_mirrors_like_fliph():
    # negative per-axis scale (from a V command) mirrors + adjusts origin the
    # same way the flipH boolean does (finalScale.X < 0 path).
    k, cx, cy = 2.25, 960.0, 540.0
    neg = compute_quad(Origin.TOP_LEFT, 0, 0, -1.0, 1.0, 0.0, False, False,
                       100, 100, k, cx, cy)
    flip = compute_quad(Origin.TOP_LEFT, 0, 0, 1.0, 1.0, 0.0, True, False,
                        100, 100, k, cx, cy)
    assert _close(neg[0], flip[0]) and _close(neg[2], flip[2])
    assert neg[5] is True and _close(neg[2], 225.0)    # positive size


def test_double_flip_cancels():
    # flipH AND negative X scale cancel (flip ^ (scale<0) == False): no mirror.
    q = compute_quad(Origin.TOP_LEFT, 0, 0, -1.0, 1.0, 0.0, True, False,
                     100, 100, 2.25, 960.0, 540.0)
    assert q[5] is False


def test_rotation_pivots_about_origin_not_centre():
    # TopLeft origin, 90 deg CW, at playfield-centre position: the centre must
    # swing to origin + Rotate((w/2,h/2)). Rotate((50,50), 90cw) = (-50, 50).
    k, cx, cy = 1.0, 0.0, 0.0     # identity screen map (osu == screen)
    scr_cx, scr_cy, w, h, theta, *_ = compute_quad(
        Origin.TOP_LEFT, 320, 240, 1.0, 1.0, 90.0, False, False, 100, 100,
        k, cx, cy)
    # osu centre via identity: cx + (sb-320)*1 == sb-320 ... position is (320,240)
    # so screen origin point == (0,0); centre offset rotated:
    #   off=(50,50); 90cw -> (off.x*cos - off.y*sin, off.x*sin + off.y*cos)
    c, s = math.cos(math.radians(90)), math.sin(math.radians(90))
    ox, oy = 50.0, 50.0
    exp_x = (320 - 320) + (ox * c - oy * s)
    exp_y = (240 - 240) + (ox * s + oy * c)
    assert _close(scr_cx, exp_x) and _close(scr_cy, exp_y)
    assert _close(theta, math.radians(90))


def test_scale_multiplies_size():
    _, _, w, h, *_ = compute_quad(
        Origin.CENTRE, 320, 240, 2.0, 0.5, 0.0, False, False, 100, 80,
        2.25, 960.0, 540.0)
    assert _close(w, 100 * 2.0 * 2.25) and _close(h, 80 * 0.5 * 2.25)


def test_all_nine_origins_anchor_fractions():
    # sanity of the LegacyOrigins.cs table (ax: 0 L .5 C 1 R; ay: 0 T .5 C 1 B)
    assert ORIGIN_ANCHOR[Origin.TOP_LEFT] == (0.0, 0.0)
    assert ORIGIN_ANCHOR[Origin.CENTRE] == (0.5, 0.5)
    assert ORIGIN_ANCHOR[Origin.CENTRE_LEFT] == (0.0, 0.5)
    assert ORIGIN_ANCHOR[Origin.TOP_RIGHT] == (1.0, 0.0)
    assert ORIGIN_ANCHOR[Origin.BOTTOM_CENTRE] == (0.5, 1.0)
    assert ORIGIN_ANCHOR[Origin.TOP_CENTRE] == (0.5, 0.0)
    assert ORIGIN_ANCHOR[Origin.CUSTOM] == (0.0, 0.0)    # -> TopLeft (lazer)
    assert ORIGIN_ANCHOR[Origin.CENTRE_RIGHT] == (1.0, 0.5)
    assert ORIGIN_ANCHOR[Origin.BOTTOM_LEFT] == (0.0, 1.0)
    assert ORIGIN_ANCHOR[Origin.BOTTOM_RIGHT] == (1.0, 1.0)


def test_bottom_right_origin_places_corner():
    # BottomRight origin at (640,480) -> that corner at screen bottom-right.
    k, cx, cy = 2.25, 960.0, 540.0
    scr_cx, scr_cy, w, h, *_ = compute_quad(
        Origin.BOTTOM_RIGHT, 640, 480, 1.0, 1.0, 0.0, False, False, 50, 50,
        k, cx, cy)
    corner_x = scr_cx + w / 2.0     # right edge
    corner_y = scr_cy + h / 2.0     # bottom edge
    assert _close(corner_x, 960.0 + (640 - 320) * k)   # 1680
    assert _close(corner_y, 540.0 + (480 - 240) * k)   # 1080


def test_widescreen_viewport_fills_16by9():
    # widescreen storyboard on a 16:9 output fills the width exactly.
    x0, y0, w, h = storyboard_viewport(1920, 1080, widescreen=True)
    assert (x0, y0) == (0, 0) and w == 1920 and h == 1080


def test_narrow_viewport_pillarboxes_on_16by9():
    # 4:3 storyboard on 16:9 output -> centred 1440-wide box (pillarbox).
    x0, y0, w, h = storyboard_viewport(1920, 1080, widescreen=False)
    assert _close(w, 1440) and _close(x0, 240) and h == 1080


def test_widescreen_viewport_720p():
    x0, y0, w, h = storyboard_viewport(1280, 720, widescreen=True)
    # 853.333 * (720/480) = 1280 -> fills width
    assert w == 1280 and x0 == 0 and h == 720


# --------------------------------------------------------------------------- #
# animation frame index                                                         #
# --------------------------------------------------------------------------- #

def test_anim_loop_forever_wraps():
    # N=4, delay=100ms, total=400ms.
    assert anim_frame_index(0, 4, 100, True) == 0
    assert anim_frame_index(150, 4, 100, True) == 1
    assert anim_frame_index(399, 4, 100, True) == 3
    assert anim_frame_index(450, 4, 100, True) == 0      # 450 % 400 = 50 -> 0
    assert anim_frame_index(825, 4, 100, True) == 0      # 825 % 400 = 25 -> 0


def test_anim_loop_once_clamps():
    assert anim_frame_index(250, 4, 100, False) == 2
    assert anim_frame_index(1000, 4, 100, False) == 3    # clamp to last
    assert anim_frame_index(399, 4, 100, False) == 3


def test_anim_degenerate_and_negative():
    assert anim_frame_index(500, 1, 100, True) == 0      # single frame
    assert anim_frame_index(500, 4, 0.0, True) == 0      # zero delay guard
    assert anim_frame_index(-50, 4, 100, True) == 0      # before start


# --------------------------------------------------------------------------- #
# GL smoke (skips without an EGL device)                                        #
# --------------------------------------------------------------------------- #

def test_storyboard_gl_smoke():
    try:
        from osu_taiko_renderer.render.gl import SpriteRenderer
        spr = SpriteRenderer(320, 240)
    except Exception:  # noqa: BLE001 — no EGL device on this box
        print("SKIP (no GL context)")
        return
    import tempfile
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from osu_taiko_renderer.beatmap.storyboard import parse_storyboard_text
    from osu_taiko_renderer.render.storyboard_engine import StoryboardEngine
    from osu_taiko_renderer.render.storyboard_render import StoryboardRenderer

    try:
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            # a solid red 8x8 sprite, centred, held opaque 0..1000ms
            Image.fromarray(
                np.tile(np.array([220, 30, 30, 255], np.uint8), (8, 8, 1))
            ).save(folder / "dot.png")
            doc = ("osu file format v14\n[Events]\n"
                   'Sprite,Background,Centre,"dot.png",320,240\n'
                   " F,0,0,1000,1\n S,0,0,1000,10\n")   # scale 10 -> 80x80 px
            eng = StoryboardEngine(parse_storyboard_text(doc))
            sbr = StoryboardRenderer(spr, eng, folder, 320, 240,
                                     widescreen=False)
            spr.begin(clear=(0.0, 0.0, 0.0))
            sbr.draw_underlay(500.0, 1.0)
            sbr.draw_overlay(500.0, 1.0)
            rgb = spr.read_rgb()
            assert rgb.shape == (240, 320, 3)
            # centre pixel must be the red-ish sprite
            px = rgb[120, 160].astype(int)
            assert px[0] > 120 and px[0] > px[1] and px[0] > px[2], px
            assert sbr.stats()["uploads"] == 1
    finally:
        spr.release()
