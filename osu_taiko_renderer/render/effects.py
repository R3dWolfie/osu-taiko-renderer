"""Effect lifecycle math for the taiko renderer's intro splash.

Ported from the std renderer (osu_std_renderer/render/effects.py) — the same
port the catch renderer carries (osu_catch_renderer/effects.py) — so the R3D
'R' logo splash is IDENTICAL across modes (the V2 porting-guide coherence
rule): same fade envelope, same timing, same asset. scene.py draws from these
pure functions. Only the LOGO section is ported (taiko has no bg-triangles /
seizure card / etc.).
"""
from __future__ import annotations

# --- intro logo -----------------------------------------------------------------
LOGO_FADE_IN_MS = 300.0
LOGO_FADE_OUT_MS = 500.0       # ends exactly as gameplay's first approach
LOGO_MIN_WINDOW_MS = 700.0     # not enough intro to read the logo -> skip
LOGO_MAX_ALPHA = 0.92
LOGO_UI_SIZE = 220.0           # tile edge in the 1080-space


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def logo_alpha(t: float, t_start: float, gameplay_in: float) -> float | None:
    """The intro splash alpha at map time t, or None when the logo phase is
    inactive. Window = [t_start, gameplay_in] (render start -> the first
    object's approach start): fade in over LOGO_FADE_IN_MS, hold, fade out over
    LOGO_FADE_OUT_MS ENDING at gameplay_in. Windows too short to read
    (< LOGO_MIN_WINDOW_MS) show nothing."""
    if gameplay_in - t_start < LOGO_MIN_WINDOW_MS:
        return None
    if t < t_start or t >= gameplay_in:
        return None
    a_in = _clamp01((t - t_start) / LOGO_FADE_IN_MS)
    a_out = _clamp01((gameplay_in - t) / LOGO_FADE_OUT_MS)
    a = LOGO_MAX_ALPHA * min(a_in, a_out)
    return a if a > 0.0 else None


def logo_scale(t: float, t_start: float) -> float:
    """Gentle settle: 1.06 -> 1.0 over the first 600 ms (quad-out)."""
    p = _clamp01((t - t_start) / 600.0)
    ease = 1.0 - (1.0 - p) * (1.0 - p)
    return 1.06 - 0.06 * ease
