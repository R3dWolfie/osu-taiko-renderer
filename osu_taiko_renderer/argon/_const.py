"""Argon taiko constants — ported verbatim from ppy/osu source.

Every value here cites its origin in osu!lazer so the look is faithful, not
invented. Colours are 0-255 RGBA tuples; sizes are the lazer relative fractions.
See ARGON_PORT.md for the source-file map.
"""
from __future__ import annotations

# --- geometry (TaikoPlayfield / TaikoHitObject) ------------------------------
BASE_HEIGHT = 200.0                 # TaikoPlayfield.BASE_HEIGHT
REF_SCREEN_H = 768.0                # lazer reference height for relative scaling
DEFAULT_SIZE = 0.475                # TaikoHitObject.DEFAULT_SIZE (note Ø / playfield_H)
STABLE_SIZE = 128.0 / 200.0        # osu!stable legacy note Ø / playfield_H (=0.64): the
                                   # FIXED on-screen size stable rescales every legacy
                                   # taikohitcircle to (128px on the 200px reference),
                                   # independent of the source sprite's pixel dimensions.
STRONG_SCALE = 1.0 / 0.65          # TaikoStrongableHitObject.STRONG_SCALE (big note)
DEFAULT_STRONG_SIZE = DEFAULT_SIZE * STRONG_SCALE
INPUT_DRUM_WIDTH = 180.0           # TaikoPlayfield.INPUT_DRUM_WIDTH (local units)
HIT_TARGET_OFFSET = -24.0          # TaikoPlayfield hit_target_offset
DRUM_INNER_SCALE = 0.9             # ArgonInputDrum InternalChild Scale

# Playfield placement on screen. At 16:9 the relative height equals the base
# (200/768); centre sits ≈0.41 down (TaikoPlayfieldAdjustmentContainer Y=135/480
# top + half height). We compute these from the screen in geometry.py.
PLAYFIELD_TOP_FRAC = 135.0 / 480.0       # 0.28125

# --- note pieces (ArgonCirclePiece / RingPiece) ------------------------------
CORE_RGBA = (0, 0, 0, 190)               # ArgonCirclePiece inner black Circle
RING1_THICKNESS = 20.0 / 70.0            # RingPiece(20/70): thick ring, accent×0.5α
RING2_THICKNESS = 5.0 / 70.0             # RingPiece(5/70): thin ring, full accent
RING1_ALPHA = 0.5                        # AccentColour.MultiplyAlpha(0.5)
ICON_SIZE = 20.0 / 70.0                  # ArgonCirclePiece.ICON_SIZE (icon BOX; swell asterisk)
ICON_X_SCALE = 0.8                       # SpriteIcon Scale (0.8, 1) — asterisk
# Note chevron, TUNED to the real game (reference/argon_baseline measured: the
# visible ‹ is h/d≈0.27, w/d≈0.135, centred). lazer's ICON_SIZE is the icon BOX;
# the FontAwesome AngleLeft glyph only fills part of it, so a box-filling procedural
# chevron came out too big/wide. total_h = size*(1+stroke) , total_w = (size*xscale + size*stroke).
CHEVRON_SIZE = 0.225                     # vertex→tip vertical span (frac of note)
CHEVRON_X_SCALE = 0.40                   # width/height — narrow ‹ (was 0.8, ~2× too wide)
ICON_Y_OFFSET = 0.0                      # chevron vertical nudge (frac; -=up). Game is
                                         # vertically CENTRED, so 0 matches it.
HIT_FLASH_PEAK = 0.9                     # flash.FadeTo(0.9) on hit
HIT_FLASH_MS = 500                       # .FadeOut(500, OutQuint)

# Accent gradients (top → bottom), from the *CirclePiece loaders.
DON_TOP, DON_BOT = (241, 0, 0, 255), (167, 0, 0, 255)
KAT_TOP, KAT_BOT = (0, 161, 241, 255), (0, 111, 167, 255)
DRUMROLL_TOP, DRUMROLL_BOT = (241, 161, 0, 255), (167, 111, 0, 255)
SWELL_TOP, SWELL_BOT = (240, 201, 0, 255), (167, 139, 0, 255)

# --- input drum (ArgonInputDrum) ---------------------------------------------
DRUM_MIDDLE_SPLIT = 6.0                   # px (local) divider width
DRUM_RIM_SIZE = 0.3                       # rim_size; centre = 1 - rim_size = 0.7
DRUM_RIM_GRAY = (51, 51, 51, 255)         # OsuColour.Gray(51/255) rim base
DRUM_CENTRE_GRAY = (64, 64, 64, 255)      # OsuColour.Gray(64/255) centre base
DRUM_SPLIT_GRAY_A = (38, 38, 38, 255)     # Gray(38/255)
DRUM_SPLIT_GRAY_B = (48, 48, 48, 255)     # Gray(48/255)
RIM_HIT_GRAD = ((227, 248, 255, 255), (198, 245, 255, 255))   # horizontal (unused now)
RIM_HIT_GLOW = (126, 215, 253, 255)
CENTRE_HIT_GRAD = ((255, 227, 236, 255), (255, 198, 211, 255))
CENTRE_HIT_GLOW = (255, 147, 199, 255)
# The press highlight itself is a FLAT accent fill (no inner gradient); the glow
# lives strictly OUTSIDE it (ArgonInputDrumHalf = flat Circle + EdgeEffect halo).
# Fills are additive over the dark idle drum (gray ~51-64), so they're chosen
# SATURATED — fill ≈ target − idle-gray — to land on a flat cyan/pink instead of
# clipping to white: rim→~(140,215,255), centre→~(255,150,200).
RIM_HIT_FILL = (89, 164, 204, 255)         # flat cyan kat highlight
CENTRE_HIT_FILL = (191, 86, 136, 255)      # flat pink don highlight
DRUM_GLOW_STRENGTH = 1.0                   # additive strength of the outer halo
DRUM_GLOW_RADIUS = 50                      # EdgeEffect glow radius (local px @200H)
DRUM_PRESS_ALPHA = 0.5                     # +0.5 on press
DRUM_PRESS_DOWN_MS = 40
DRUM_PRESS_UP_MS = 200

# --- hit target (ArgonHitTarget) ---------------------------------------------
HIT_TARGET_BORDER = 4.0                    # bar thickness (local px)
HIT_TARGET_CIRCLE_ALPHA = 0.1             # additive white circles
HIT_TARGET_INNER_SCALE = 0.85

# --- hit explosion (ArgonHitExplosion) ---------------------------------------
EXPLOSION_INNER_SCALE = 0.85
EXPLOSION_GLOW_RADIUS = 45
EXPLOSION_GLOW_ALPHA = 0.5
EXPLOSION_GREAT_IN_MS, EXPLOSION_GREAT_OUT_MS = 30, 450
EXPLOSION_OK_PEAK, EXPLOSION_OK_OUT_MS = 0.2, 200

# --- bar line (ArgonBarLine) -------------------------------------------------
BARLINE_MAJOR_ALPHA, BARLINE_MINOR_ALPHA = 1.0, 0.5
BARLINE_MAJOR_EXT = 10.0                   # additive top/bottom anchor px

# --- judgement (ArgonJudgementPiece) + OsuColour.ForHitResult ----------------
JUDGE_GREAT = (0x66, 0xcc, 0xff, 255)      # Blue
JUDGE_OK = (0x88, 0xb3, 0x00, 255)         # Green
JUDGE_GOOD = (0xb3, 0xd9, 0x44, 255)       # GreenLight
JUDGE_MEH = (0xff, 0xcc, 0x22, 255)        # Yellow
JUDGE_MISS = (0xed, 0x11, 0x21, 255)       # Red
JUDGE_FONT_SIZE = 20                        # OsuFont size
JUDGE_SPACING = 10                          # Spacing (10,0)
JUDGE_MOVE_MS = 800
RING_BURST_THICK = 4
RING_BURST_SMALL, RING_BURST_LARGE = 9, 14
RING_BURST_TRAVEL = 58
RING_BURST_FADE_MS = 1000


def lerp_rgba(a, b, t):
    """Linear interpolate two RGBA tuples (t in 0..1)."""
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(4))
