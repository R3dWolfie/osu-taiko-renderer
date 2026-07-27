"""osu!taiko OsuModFlashlight (TaikoModFlashlight) overlay for the R3D taiko
renderer's CPU compositing path.

lazer parity (osu.Game.Rulesets.Taiko/Mods/TaikoModFlashlight.cs +
osu.Game/Rulesets/Mods/ModFlashlight.cs):
- The lit region is CIRCULAR and FIXED on the playfield hit target — taiko has
  no cursor to follow (unlike OsuModFlashlight which tracks the cursor).
- Size steps with combo: >200 -> 0.8x, >100 -> 0.9x, else 1.0x of the default.
- On a combo change the size transforms over FLASHLIGHT_FADE_DURATION = 800 ms
  with Easing.OutQuint.
- Outside the lit radius the screen is solid black; the HUD (score/combo/health)
  is drawn ABOVE the flashlight so it stays fully visible.

Everything the overlay needs is a pure function of precomputed sim state
(result times `_rt`, cumulative combo `_cum`, playfield `geo`), so a frame's
darkening is a cheap per-frame lookup + one bounded blit.
"""
from __future__ import annotations

import bisect

import numpy as np

MOD_FLASHLIGHT = 1 << 10

FLASHLIGHT_FADE_DURATION = 800.0    # ms (ModFlashlight.FLASHLIGHT_FADE_DURATION)
# Lit RADIUS at base combo as a factor of the playfield (lane) height. taiko's
# flashlight reveals roughly the lane plus a note or two of lead-in. Tunable.
FL_RADIUS_FACTOR = 1.2
_CORE = 0.45                        # fraction of radius that stays fully lit
_RAMP_CACHE_MAX = 256


def flashlight_scale_for(combo: int) -> float:
    """TaikoModFlashlight combo size multiplier (ModFlashlight breakpoints)."""
    if combo > 200:
        return 0.8
    if combo > 100:
        return 0.9
    return 1.0


def _out_quint(p: float) -> float:
    """Easing.OutQuint."""
    q = 1.0 - p
    return 1.0 - q * q * q * q * q


class TaikoFlashlight:
    """Per-render flashlight overlay. Construct once; call composite() per
    gameplay frame with the frame RGB and its map time."""

    def __init__(self, geo, rt, cum, mods):
        self.on = bool(int(mods or 0) & MOD_FLASHLIGHT)
        self.cx = float(getattr(geo, "target_x", 0.0))
        self.cy = float(getattr(geo, "center_y", 0.0))
        self.base_r = FL_RADIUS_FACTOR * float(getattr(geo, "pf_h", 0.0))
        # (change_time_ms, scale) breakpoints — only where the scale changes.
        tl = [(-1e18, 1.0)]
        cur = 1.0
        if self.on:
            for i, t in enumerate(rt or []):
                try:
                    s = flashlight_scale_for(int(cum[i][0]))
                except (IndexError, TypeError):
                    continue
                if s != cur:
                    tl.append((float(t), s))
                    cur = s
        self._tl = tl
        self._tl_t = [b[0] for b in tl]
        self._ramp_cache: dict[int, np.ndarray] = {}

    def _scale_at(self, t: float) -> float:
        i = bisect.bisect_right(self._tl_t, t) - 1
        if i < 0:
            return 1.0
        change_t, target = self._tl[i]
        if t >= change_t + FLASHLIGHT_FADE_DURATION:
            return target
        prev = self._tl[i - 1][1] if i > 0 else 1.0
        p = (t - change_t) / FLASHLIGHT_FADE_DURATION
        return prev + (target - prev) * _out_quint(max(0.0, min(1.0, p)))

    def _ramp(self, ri: int) -> np.ndarray:
        """(2ri x 2ri) float32 black-overlay ALPHA: 0 (fully lit) inside the
        core, smoothstep up to 1 (opaque black) at radius `ri`. Cached by
        integer radius (constant except during ~800 ms combo fades)."""
        cached = self._ramp_cache.get(ri)
        if cached is not None:
            return cached
        n = 2 * ri
        c = (n - 1) / 2.0
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        dn = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / float(ri)   # 1.0 at radius
        x = np.clip((dn - _CORE) / (1.0 - _CORE), 0.0, 1.0)
        alpha = (x * x * (3.0 - 2.0 * x)).astype(np.float32)       # smoothstep
        if len(self._ramp_cache) < _RAMP_CACHE_MAX:
            self._ramp_cache[ri] = alpha
        return alpha

    def composite(self, frame: np.ndarray, t: float) -> np.ndarray:
        """Return `frame` (HxWx3 uint8) darkened to the flashlight spotlight:
        solid black except the lit disc at the hit target. Never mutates the
        input (safe on read-only arrays). No-op (returns frame) when FL is off."""
        if not self.on:
            return frame
        ri = int(round(self.base_r * self._scale_at(t)))
        if ri < 1:
            return np.zeros_like(frame)
        ramp = self._ramp(ri)
        h, w = frame.shape[:2]
        cx, cy = int(round(self.cx)), int(round(self.cy))
        x0, y0 = cx - ri, cy - ri
        # frame region covered by the disc bbox, clipped to the frame
        fx0, fy0 = max(0, x0), max(0, y0)
        fx1, fy1 = min(w, cx + ri), min(h, cy + ri)
        out = np.zeros_like(frame)
        if fx1 <= fx0 or fy1 <= fy0:
            return out                       # disc entirely off-screen -> black
        a = ramp[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0][..., None]
        src = frame[fy0:fy1, fx0:fx1].astype(np.float32) * (1.0 - a)
        out[fy0:fy1, fx0:fx1] = src.astype(np.uint8)
        return out
