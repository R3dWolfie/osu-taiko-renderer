"""Robust, skin-aware font resolution.

osu! skins draw HUD *numbers* from PNG glyphs (score-*/combo-*), but free text
(player name, song title, results card) needs a real scalable font. We resolve
one once per render:

  1. a font bundled INSIDE the skin dir (``*.ttf``/``*.otf``) — "per-skin font",
  2. else a known system path (DejaVu/Noto/Arial across distros + macOS),
  3. else whatever ``fc-match`` reports for ``sans-serif:bold``.

If none resolves we warn LOUDLY (once) and use PIL's bitmap default — which
ignores the requested size and renders ~10px text. Relying on a bare font NAME
(``ImageFont.truetype("DejaVuSans-Bold.ttf", 96)``) silently hits that path on
any host lacking that exact name (e.g. Arch ships Noto, not DejaVu), so we never
do that.
"""
from __future__ import annotations

import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

_SYSTEM_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Debian/Ubuntu (FoofPC)
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",              # Arch
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",           # Fedora/Bazzite
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/noto/NotoSans-Bold.ttf",               # Arch (Noto)
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",                         # macOS
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]

_skin_font_path: str | None = None
_warned = False


def set_skin_font(skin_dir) -> None:
    """Point the resolver at a skin (called once at the start of a render).
    Picks a font bundled in the skin dir if present; otherwise system fonts."""
    global _skin_font_path
    _skin_font_path = _find_skin_font(skin_dir)
    font.cache_clear()


def _find_skin_font(skin_dir) -> str | None:
    if not skin_dir:
        return None
    d = Path(skin_dir)
    if not d.is_dir():
        return None
    fonts: list[Path] = []
    for pat in ("*.ttf", "*.otf", "*.TTF", "*.OTF"):
        fonts.extend(d.glob(pat))
    if not fonts:
        return None
    fonts.sort()
    bold = [f for f in fonts if "bold" in f.name.lower()]
    return str((bold or fonts)[0])


def _resolve_path() -> str | None:
    if _skin_font_path and Path(_skin_font_path).is_file():
        return _skin_font_path
    for p in _SYSTEM_CANDIDATES:
        if Path(p).is_file():
            return p
    try:
        out = subprocess.run(
            ["fc-match", "-f", "%{file}", "sans-serif:bold"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if out and Path(out).is_file():
            return out
    except Exception:  # noqa: BLE001 — fc-match absent / errored
        pass
    return None


@lru_cache(maxsize=128)
def font(size: int):
    path = _resolve_path()
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:  # noqa: BLE001
            pass
    global _warned
    if not _warned:
        _warned = True
        print("[catch-renderer] WARN: no scalable font found (skin/system); "
              "HUD/results text will render tiny via PIL's bitmap default. "
              "Install DejaVu/Noto or bundle a .ttf in the skin.", file=sys.stderr)
    return ImageFont.load_default()
