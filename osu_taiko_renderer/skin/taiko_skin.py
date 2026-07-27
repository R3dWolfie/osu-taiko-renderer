"""User .osk skin resolution for taiko.

Resolves osu!taiko legacy skin elements (per the osu! skinning wiki) from an
extracted .osk directory: case-insensitive, prefers @2x. Each element falls back
to the Argon procedural default (handled by the caller) when the skin doesn't
provide it — the user→default→wiki chain, taiko edition.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


class TaikoSkin:
    def __init__(self, skin_dir):
        self.dir = Path(skin_dir) if skin_dir else None
        self._files: dict[str, Path] = {}
        if self.dir and self.dir.is_dir():
            for p in self.dir.iterdir():
                if p.is_file():
                    self._files[p.name.lower()] = p
        # skin.ini [Fonts] digit-font prefixes (default score/combo). Lazer skins
        # often override to e.g. "main" so the digits are main-0.png...
        self.score_prefix = "score"
        self.combo_prefix = "combo"
        self.score_overlap = 0
        self.combo_overlap = 0
        self._parse_ini()

    def _parse_ini(self) -> None:
        ini = self._files.get("skin.ini")
        if ini is None:
            return
        try:
            txt = ini.read_text(encoding="utf-8-sig", errors="replace")
        except Exception:  # noqa: BLE001
            return
        section = ""
        for line in txt.splitlines():
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].strip().lower()
                continue
            if section != "fonts" or ":" not in line:
                continue
            k, _, v = line.partition(":")
            k, v = k.strip().lower(), v.strip()
            if k == "scoreprefix" and v:
                self.score_prefix = v
            elif k == "comboprefix" and v:
                self.combo_prefix = v
            elif k in ("scoreoverlap", "combooverlap"):
                try:
                    setattr(self, k.replace("overlap", "_overlap"), int(v))
                except ValueError:
                    pass

    def find(self, name: str) -> Path | None:
        """Path for an element. Static file wins (name@2x.png then name.png,
        case-insensitive); else frame 0 of an animation (name-0@2x.png then
        name-0.png) — many skins ship judgement graphics only as animation
        sequences (taiko-hit100-0.png, taiko-hit0-0.png, ...)."""
        for cand in (f"{name}@2x.png", f"{name}.png",
                     f"{name}-0@2x.png", f"{name}-0.png"):
            p = self._files.get(cand.lower())
            if p is not None:
                return p
        return None

    def has(self, name: str) -> bool:
        return self.find(name) is not None

    def load(self, name: str) -> np.ndarray | None:
        """RGBA uint8 array for an element, or None if absent/unreadable."""
        p = self.find(name)
        if p is None:
            return None
        try:
            return np.array(Image.open(p).convert("RGBA"))
        except Exception:  # noqa: BLE001
            return None
