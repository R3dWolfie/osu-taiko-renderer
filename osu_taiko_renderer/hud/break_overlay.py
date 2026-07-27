"""osu!lazer's BreakOverlay, ported onto the taiko renderer's CPU HUD
compositing (numpy RGB frames, argon.hud-style blits).

This is the taiko-engine sibling of the catch reference implementation
(osu-catch/osu_catch_renderer/break_overlay.py, commit d8ccb60) — same
lazer semantics; the drawing rides this engine's own stack: RGBA-numpy
sprite blits onto the readback frame (argon/hud.py's compositing model),
the REAL Torus BMFont glyphs (argon/font.py) for the text lines and the
Argon counter cells (argon/counter.py, lit only — no wireframe backing,
that's a score-counter decoration) for the countdown digits.

Source of truth — ppy/osu master (read 2026-07-23, keep in sync):
  osu.Game/Screens/Play/BreakOverlay.cs        fade/slide timings, progress
                                               bar semantics, layout Y=±15
  osu.Game/Screens/Play/BreakTracker.cs        which breaks count (HasEffect),
                                               Period = (Start, End - FADE)
  osu.Game/Screens/Play/Break/BreakInfo.cs     "CURRENT PROGRESS" + info lines
  osu.Game/Screens/Play/Break/BreakInfoLine.cs label Yellow/value YellowLight,
                                               2px split margins, acc format
  osu.Game/Screens/Play/Break/RemainingTimeCounter.cs  ceil(ms/1000) seconds
  osu.Game/Screens/Play/Break/BreakArrows.cs   chevron pair geometry/offsets
  osu.Game/Screens/Play/Break/GlowIcon.cs      sharp icon + BlueLighter glow
  osu.Game/Screens/Play/Break/BlurredIcon.cs   blur-only + additive + a=0.7
  osu.Game/Beatmaps/Timing/BreakPeriod.cs      MIN_BREAK_DURATION = 650

The exact lazer timeline (BreakOverlay.updateDisplay, absolute from b.Start;
the tracker's Period trims BREAK_FADE_DURATION=325ms off the END, so with
D = period duration = break duration - 325):
  t'=0..325   fadeContainer.FadeIn(325) [linear]; arrows slide in (OutQuint,
              325ms); counter X -50->0 / info X +50->0 (OutQuint, 325ms);
              progress-bar CONTAINER width 0 -> 0.3 rel (OutQuint, 325ms)
  t'=0..D+325 counter counts (D+325 = full break duration) -> 0, linear;
              display = ceil(count/1000)
  every frame bar width DampContinuously(current, target, halfTime=40ms);
              target = max(0, (Period.End - now - 325) / D)  [reaches 0
              already 325ms BEFORE the fade-out starts]
  t'=D        fadeContainer.FadeOut(325); arrows slide back out (OutQuint,
              325ms); bar container width snaps to 0 — gone at t'=D+325,
              exactly the break's end.

ARROWS — deliberately NOT blinking: lazer master's BreakArrows only slide
in/out and hold (Show/Hide MoveToX, Easing.OutQuint). The pair per side is
the sharp GlowIcon (60px chevron, sigma-10 BlueLighter glow) in front of
the big BlurredIcon (130px, sigma-20, blur-only, additive, alpha 0.7).
Cursor parallax has no analogue in a fixed render and is dropped — the
same call as the catch reference.

VALUES — live, not snapshotted: BreakOverlay.LoadComplete BindTo()s the
ScoreProcessor's Accuracy/Rank bindables; the wiring samples the sim's
running scene.accuracy each frame and the grade comes from the engine's
own taiko cutoffs (lazer_results.taiko_grade — the single grade source),
silvered under HD/FL like lazer's AdjustRank. Accuracy is formatted with
lazer's FormatAccuracy floor (never rounded up), as in the catch port.

lazer taiko shows this exact screen-centre overlay (BreakOverlay is a
Player-level component, ruleset-independent), so the geometry is the
full-frame centre — not the drum lane.

Z-ORDER: lazer's BreakOverlay is a LATER overlay-component child than
HUDOverlay (Player.createOverlayComponents) — render.py composites this
AFTER hud.overlay() on BOTH HUD variants (Argon + legacy skin, one wiring
point), so it sits above every HUD element. Absent from multi-player
composites (versus lanes are stitched from single renders downstream; no
lazer analogue applies there).

Coordinates are lazer's 768-tall UI space scaled by lk = screen_h/768; the
arrows' X offsets are RELATIVE TO WIDTH (GlowIcon RelativePositionAxes =
Axes.X), matching upstream. All times are MAP-time ms (this sim runs on
the map-time axis; lazer runs these transforms on the rate-adjusted
FrameStableClock, which is the map timeline — DT/HT inherently correct).
Frames must arrive in monotonic map-time order (they do — render_core's
_emit_gameplay pops the PBO ring in submission order)."""
from __future__ import annotations

import math
from bisect import bisect_right

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from osu_taiko_renderer.argon.counter import ArgonCounter
from osu_taiko_renderer.argon.font import get_font

# --- lazer constants (files cited in the module docstring) -------------------
MIN_BREAK_DURATION = 650.0        # BreakPeriod.MIN_BREAK_DURATION (HasEffect)
BREAK_FADE_MS = MIN_BREAK_DURATION / 2.0   # BreakOverlay.BREAK_FADE_DURATION
REMAINING_MAX_W = 0.3             # remaining_time_container_max_size
VERTICAL_MARGIN = 15.0            # BreakOverlay.vertical_margin (lazer px)
BAR_H = 8.0                       # remainingTimeBox Height
DAMP_HALF_MS = 40.0               # Interpolation.DampContinuously halfTime
SLIDE_X = 50.0                    # counter/info MoveToX slide distance

GLOW_ICON_SIZE = 60.0             # BreakArrows glow_icon_*
GLOW_ICON_SIGMA = 10.0
GLOW_ICON_FINAL = 0.22            # X offsets, RELATIVE TO WIDTH
GLOW_ICON_OFFSCREEN = 0.6
BLUR_ICON_SIZE = 130.0            # BreakArrows blurred_icon_*
BLUR_ICON_SIGMA = 20.0
BLUR_ICON_FINAL = 0.38
BLUR_ICON_OFFSCREEN = 0.7
BLUR_ICON_ALPHA = 0.7

BLUE_LIGHTER = (0xDD, 0xFF, 0xFF)   # OsuColour.BlueLighter (glow colour)
YELLOW = (0xFF, 0xCC, 0x22)         # OsuColour.Yellow (info labels)
YELLOW_LIGHT = (0xFF, 0xDD, 0x55)   # OsuColour.YellowLight (info values)
SHADOW_GRAY = (51, 51, 51)          # OsuColour.Gray(0.2f)
SHADOW_ALPHA = 0.8                  # .Opacity(0.8f)
SHADOW_RADIUS = 260.0               # EdgeEffect shadow radius (lazer px)
SHADOW_CORE_W, SHADOW_CORE_H = 80.0, 4.0   # the CircularContainer core

COUNTER_SIZE = 33.0               # RemainingTimeCounter OsuFont.Numeric 33
TITLE_SIZE = 15.0                 # "CURRENT PROGRESS" bold 15
LINE_SIZE = 17.0                  # BreakInfoLine label/value size
LINE_MARGIN = 2.0                 # BreakInfoLine margin each side of centre
FLOW_SPACING = 5.0                # BreakInfo FillFlow Spacing(5)

# Torus px per lazer-font-size unit: o!f font size == em box px, and the
# BMFont bakes carry their own native size — render(text, px) takes the
# TARGET pixel height of the em box, so lazer size × lk is direct.
LAZER_UI_HEIGHT = 768.0

# lazer's HD/FL AdjustRank turns X/S silver; the GradeDisplay renders
# ScoreRank.GetLocalisableDescription(): X="SS", XH="Silver SS",
# SH="Silver S" (osu-resources Localisation/Web).
_HD, _FL = 1 << 3, 1 << 10


def _out_quint(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return 1.0 - (1.0 - u) ** 5


def grade_display(accuracy: float, mods: int) -> str:
    """The break overlay's Grade line text: the engine's own taiko grade
    (lazer_results.taiko_grade — single source for cutoffs) mapped to
    lazer's rank display strings, with the HD/FL silver adjustment lazer
    applies. Lazy import: lazer_results is a heavy results-screen module."""
    from osu_taiko_renderer.hud.lazer_results import taiko_grade
    g = taiko_grade(max(0.0, min(1.0, accuracy)))
    if mods & (_HD | _FL):
        if g == "SS":
            return "Silver SS"
        if g == "S":
            return "Silver S"
    return g


class LazerBreakOverlay:
    """Stateful per-render overlay: bakes the static art once, then draws
    per frame during effective breaks (duration >= MIN_BREAK_DURATION).
    The damped bar width is replayed statefully like lazer's always-running
    Update()."""

    def __init__(self, w: int, h: int, breaks, mods: int = 0):
        self.w, self.h = int(w), int(h)
        self.mods = int(mods or 0)
        self.lk = self.h / LAZER_UI_HEIGHT
        # BreakTracker.Breaks: only HasEffect breaks, Period end trimmed by
        # BREAK_FADE_DURATION. We keep (start, D) with D = period duration;
        # the overlay is on screen over [start, start + D + 325].
        self.periods = sorted(
            (float(s), float(e - s) - BREAK_FADE_MS)
            for s, e in (breaks or ())
            if (e - s) >= MIN_BREAK_DURATION)
        self._starts = [p[0] for p in self.periods]
        # DampContinuously state (remainingTimeBox.Width, RELATIVE 0..1)
        self._bar_w = 0.0
        self._last_t: float | None = None
        if not self.periods:
            return                     # no effective breaks -> never draws
        self.counter = ArgonCounter()
        self._bold = get_font("Bold")
        self._reg = get_font("Regular")
        self._shadow = self._bake_shadow()
        # arrows: right-pointing bakes, mirrored for the left-pointing pair
        glow = self._bake_glow_icon()
        self._glow_r = glow
        self._glow_l = glow[:, ::-1].copy()
        blur = self._bake_blurred_icon()   # premultiplied float RGB field
        self._blur_r = blur
        self._blur_l = blur[:, ::-1].copy()
        self._pill_cache: dict = {}

    # --- bakes ---------------------------------------------------------------

    def _bake_shadow(self) -> np.ndarray:
        """The fadeContainer's first child: an invisible 80x4 pill whose
        EdgeEffect SHADOW (radius 260, gray(0.2) @ 0.8) is the big soft dark
        blob behind the centre block. Approximated as a quadratic falloff
        over the radius from the pill edge (o!f edge-effect profile) —
        identical math to the catch reference."""
        lk = self.lk
        R = SHADOW_RADIUS * lk
        cw, ch = SHADOW_CORE_W * lk, SHADOW_CORE_H * lk
        W = int(math.ceil(cw + 2 * R))
        H = int(math.ceil(ch + 2 * R))
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = ch / 2.0                      # pill corner radius
        qx = np.abs(xx - W / 2.0) - (cw / 2.0 - r)
        qy = np.abs(yy - H / 2.0) - (ch / 2.0 - r)
        d = (np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
             + np.minimum(np.maximum(qx, qy), 0.0) - r)
        fall = np.clip(1.0 - d / R, 0.0, 1.0) ** 2
        rgba = np.zeros((H, W, 4), np.uint8)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = SHADOW_GRAY
        rgba[..., 3] = np.round(fall * SHADOW_ALPHA * 255.0).astype(np.uint8)
        return rgba

    def _chevron_mask(self, size_px: int) -> Image.Image:
        """FontAwesome Solid.ChevronRight silhouette: a bold '>' polyline
        (glyph aspect ~0.63 in a square SpriteIcon cell), round caps/joint."""
        s = size_px
        m = Image.new("L", (s, s), 0)
        d = ImageDraw.Draw(m)
        w = max(2, int(round(s * 0.17)))
        pts = [(0.36 * s, 0.14 * s), (0.67 * s, 0.50 * s),
               (0.36 * s, 0.86 * s)]
        d.line(pts, fill=255, width=w, joint="curve")
        for px, py in (pts[0], pts[2]):
            d.ellipse([px - w / 2, py - w / 2, px + w / 2, py + w / 2],
                      fill=255)
        return m

    def _bake_glow_icon(self) -> np.ndarray:
        """GlowIcon: sharp white chevron over its BlueLighter gaussian glow
        (GlowingDrawable: blurred silhouette tinted GlowColour, original on
        top). RGBA uint8, right-pointing."""
        lk = self.lk
        s = max(4, int(round(GLOW_ICON_SIZE * lk)))
        sigma = GLOW_ICON_SIGMA * lk
        pad = int(math.ceil(3 * sigma)) + 1
        cv = Image.new("L", (s + 2 * pad, s + 2 * pad), 0)
        cv.paste(self._chevron_mask(s), (pad, pad))
        glow_a = cv.filter(ImageFilter.GaussianBlur(sigma))
        glow = Image.new("RGBA", cv.size, BLUE_LIGHTER + (0,))
        glow.putalpha(glow_a)
        sharp = Image.new("RGBA", cv.size, (255, 255, 255, 0))
        sharp.putalpha(cv)
        return np.asarray(Image.alpha_composite(glow, sharp)).copy()

    def _bake_blurred_icon(self) -> np.ndarray:
        """BlurredIcon: blur-only (DrawOriginal=false), additive, alpha 0.7.
        Kept as a premultiplied float RGB field ready for additive blending
        (BlendingParameters.Additive)."""
        lk = self.lk
        s = max(4, int(round(BLUR_ICON_SIZE * lk)))
        sigma = BLUR_ICON_SIGMA * lk
        pad = int(math.ceil(3 * sigma)) + 1
        cv = Image.new("L", (s + 2 * pad, s + 2 * pad), 0)
        cv.paste(self._chevron_mask(s), (pad, pad))
        a = (np.asarray(cv.filter(ImageFilter.GaussianBlur(sigma)),
                        np.float32) / 255.0) * BLUR_ICON_ALPHA
        col = np.array(BLUE_LIGHTER, np.float32)
        return a[..., None] * col[None, None, :]

    def _bar_pill(self, w_px: int, h_px: int) -> np.ndarray:
        """remainingTimeBox: a white fully-rounded Circle, h = min(8, w) —
        an antialiased rounded-rect alpha field (supersampled), cached per
        integer size (the damped width revisits sizes)."""
        key = (w_px, h_px)
        hit = self._pill_cache.get(key)
        if hit is None:
            ss = 4
            im = Image.new("L", (w_px * ss, h_px * ss), 0)
            ImageDraw.Draw(im).rounded_rectangle(
                [0, 0, w_px * ss - 1, h_px * ss - 1],
                radius=h_px * ss // 2, fill=255)
            im = im.resize((w_px, h_px), Image.LANCZOS)
            rgba = np.full((h_px, w_px, 4), 255, np.uint8)
            rgba[..., 3] = np.asarray(im)
            if len(self._pill_cache) > 1024:
                self._pill_cache.clear()
            self._pill_cache[key] = rgba
            hit = rgba
        return hit

    # --- compositing helpers (argon/hud.py's model, alpha-scaled) ------------

    @staticmethod
    def _paste(rgb: np.ndarray, src: np.ndarray, cx: float, cy: float,
               alpha: float) -> None:
        """Alpha-over composite of an RGBA uint8 sprite CENTRED at (cx, cy),
        scaled by the overlay alpha. Clips partial/off-frame boxes."""
        if alpha <= 0.004:
            return
        h, w = src.shape[:2]
        x = int(round(cx - w / 2.0))
        y = int(round(cy - h / 2.0))
        H, W = rgb.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        s = src[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
        a = (s[..., 3:4] / 255.0) * alpha
        region = rgb[y0:y1, x0:x1].astype(np.float32)
        region = region * (1.0 - a) + s[..., :3] * a
        rgb[y0:y1, x0:x1] = np.clip(region, 0, 255).astype(np.uint8)

    @staticmethod
    def _add(rgb: np.ndarray, field: np.ndarray, cx: float, cy: float,
             alpha: float) -> None:
        """Additive blend of a premultiplied float-RGB field centred at
        (cx, cy) — BlendingParameters.Additive for the BlurredIcons."""
        if alpha <= 0.004:
            return
        fh, fw = field.shape[:2]
        x0 = int(round(cx - fw / 2.0))
        y0 = int(round(cy - fh / 2.0))
        H, W = rgb.shape[:2]
        ix0, iy0 = max(x0, 0), max(y0, 0)
        ix1, iy1 = min(x0 + fw, W), min(y0 + fh, H)
        if ix1 <= ix0 or iy1 <= iy0:
            return
        sub = field[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] * alpha
        base = rgb[iy0:iy1, ix0:ix1].astype(np.float32)
        rgb[iy0:iy1, ix0:ix1] = np.clip(base + sub, 0.0, 255.0
                                        ).astype(np.uint8)

    # --- per-frame -----------------------------------------------------------

    def draw(self, rgb: np.ndarray, t_ms: float, accuracy: float) -> None:
        """Compose the overlay for map time t_ms onto the numpy RGB frame
        (mutated in place, like the HUD blits). Called every frame (the bar
        damp runs continuously, like lazer's Update); cheap no-op outside
        break windows — the frame bytes are untouched then."""
        if not self.periods:
            return
        t = float(t_ms)
        dt = 16.7 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t

        # active period: overlay lives over [start, start + D + FADE]
        idx = bisect_right(self._starts, t) - 1
        cur = None
        if idx >= 0:
            s0, D = self.periods[idx]
            if t <= s0 + D + BREAK_FADE_MS:
                cur = (s0, D)

        # remainingTimeBox.Width — DampContinuously toward
        # max(0, (Period.End - now - FADE) / D), EVERY frame, in/out of breaks
        if cur is None:
            target = 0.0
        else:
            s0, D = cur
            target = max(0.0, (s0 + D - t - BREAK_FADE_MS) / D) if D > 0 else 0.0
        self._bar_w = target + (self._bar_w - target) * (0.5 ** (dt / DAMP_HALF_MS))

        if cur is None:
            return
        s0, D = cur
        tp = t - s0                       # time since break start
        # fadeContainer alpha: linear FadeIn/FadeOut over BREAK_FADE_MS
        if tp >= D:
            alpha = max(0.0, 1.0 - (tp - D) / BREAK_FADE_MS)
        else:
            alpha = min(1.0, tp / BREAK_FADE_MS)
        if alpha <= 0.004:
            return

        lk = self.lk
        cx, cy = self.w / 2.0, self.h / 2.0
        p_in = _out_quint(tp / BREAK_FADE_MS)

        # 1) shadow blob (first fadeContainer child)
        self._paste(rgb, self._shadow, cx, cy, alpha)

        # 2) progress bar: container width 0 -> 0.3 (OutQuint, 325ms), snap
        #    to 0 at t'=D; pill width rides the damped fraction
        wc = REMAINING_MAX_W * p_in if tp < D else 0.0
        bw = int(round(wc * self.w * max(0.0, min(1.0, self._bar_w))))
        if bw >= 2:
            bh = max(1, int(round(min(BAR_H * lk, bw))))
            self._paste(rgb, self._bar_pill(bw, bh), cx, cy, alpha)

        # 3) remaining-time counter: ceil(count/1000); count runs linearly
        #    from the FULL break duration to 0 at the break's end. Digits =
        #    the engine's Argon counter cells, lit only (wire_alpha=0 — the
        #    wireframe ghost is score-counter decoration, not break art).
        count = max(0.0, (D + BREAK_FADE_MS) - tp)
        text = str(int(math.ceil(count / 1000.0)))
        digits = self.counter.render(text, COUNTER_SIZE * lk, wire_alpha=0.0)
        dx = -SLIDE_X * lk * (1.0 - p_in)          # MoveToX(-50 -> 0)
        self._paste(rgb, digits, cx + dx,
                    cy - VERTICAL_MARGIN * lk - digits.shape[0] / 2.0, alpha)

        # 4) BreakInfo (slides +50 -> 0): title, then Accuracy / Grade lines
        #    split 2px either side of centre; values LIVE like lazer's
        #    bindables (constant mid-break in practice). Torus glyphs —
        #    labels Regular, values/title Bold, exactly lazer's weights.
        dxi = SLIDE_X * lk * (1.0 - p_in)
        cxi = cx + dxi
        y0 = cy + VERTICAL_MARGIN * lk
        title = self._bold.render("CURRENT PROGRESS",
                                  TITLE_SIZE * lk, color=(255, 255, 255))
        self._paste(rgb, title, cxi, y0 + TITLE_SIZE * lk / 2.0, alpha)
        acc = max(0.0, min(1.0, float(accuracy)))
        acc_txt = f"{math.floor(acc * 10000.0) / 100.0:.2f}%"  # FormatAccuracy
        rows = [("Accuracy", acc_txt),
                ("Grade", grade_display(acc, self.mods))]
        ly = y0 + (TITLE_SIZE + FLOW_SPACING) * lk
        for label, value in rows:
            mid = ly + LINE_SIZE * lk / 2.0
            lab = self._reg.render(label, LINE_SIZE * lk, color=YELLOW)
            val = self._bold.render(value, LINE_SIZE * lk,
                                    color=YELLOW_LIGHT)
            self._paste(rgb, lab,
                        cxi - LINE_MARGIN * lk - lab.shape[1] / 2.0, mid,
                        alpha)
            self._paste(rgb, val,
                        cxi + LINE_MARGIN * lk + val.shape[1] / 2.0, mid,
                        alpha)
            ly += LINE_SIZE * lk

        # 5) arrows, topmost: slide in over the fade (OutQuint), hold, slide
        #    back out from t'=D. X offsets are fractions of the WIDTH.
        if tp >= D:
            po = _out_quint((tp - D) / BREAK_FADE_MS)
            g_off = GLOW_ICON_FINAL + (GLOW_ICON_OFFSCREEN - GLOW_ICON_FINAL) * po
            b_off = BLUR_ICON_FINAL + (BLUR_ICON_OFFSCREEN - BLUR_ICON_FINAL) * po
        else:
            g_off = GLOW_ICON_OFFSCREEN + (GLOW_ICON_FINAL - GLOW_ICON_OFFSCREEN) * p_in
            b_off = BLUR_ICON_OFFSCREEN + (BLUR_ICON_FINAL - BLUR_ICON_OFFSCREEN) * p_in
        # origins CentreRight/CentreLeft: the offset is the icon's inner
        # LAYOUT edge (AutoSize box = the 60/130px icon; the glow/blur
        # overhang is draw-only, like o!f's inflated draw quad) -> shift
        # each sprite centre outward by half the LAYOUT size, not the
        # padded canvas
        g_half = GLOW_ICON_SIZE * lk / 2.0
        b_half = BLUR_ICON_SIZE * lk / 2.0
        self._add(rgb, self._blur_r, cx - b_off * self.w - b_half, cy, alpha)
        self._add(rgb, self._blur_l, cx + b_off * self.w + b_half, cy, alpha)
        self._paste(rgb, self._glow_r, cx - g_off * self.w - g_half, cy, alpha)
        self._paste(rgb, self._glow_l, cx + g_off * self.w + g_half, cy, alpha)
