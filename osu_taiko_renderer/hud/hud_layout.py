"""Per-skin HUD layout from the .osk's lazer ``MainHUDComponents.json``.

Maps lazer skinnable HUD components to our taiko HUD elements and resolves each
to a screen position, so a skin that repositions its score / accuracy /
judgement counter / combo / key counter in the lazer editor is honoured. Falls
back (element missing, no JSON, unreadable) to the built-in lazer-default
layout — nothing here can hard-fail a render.
"""
import json
from pathlib import Path

# osu!framework Anchor flag bits -> fraction along each axis.
_XBITS = ((8, 0.0), (16, 0.5), (32, 1.0))   # x0 left / x1 centre / x2 right
_YBITS = ((1, 0.0), (2, 0.5), (4, 1.0))     # y0 top  / y1 centre / y2 bottom


def _frac(anchor):
    ax = next((v for b, v in _XBITS if anchor & b), 0.0)
    ay = next((v for b, v in _YBITS if anchor & b), 0.0)
    return ax, ay


# lazer component type (last dotted segment) -> our element id.
_TYPE_MAP = {
    "LegacyScoreCounter": "score", "DefaultScoreCounter": "score",
    "ArgonScoreCounter": "score", "GameplayScoreCounter": "score",
    "LegacyAccuracyCounter": "accuracy", "DefaultAccuracyCounter": "accuracy",
    "ArgonAccuracyCounter": "accuracy",
    "JudgementCounterDisplay": "counter",
    "LegacyComboCounter": "combo", "LegacyDefaultComboCounter": "combo",
    "DefaultComboCounter": "combo", "ArgonComboCounter": "combo",
    "LegacyKeyCounterDisplay": "keys", "KeyCounterDisplay": "keys",
    "ArgonKeyCounterDisplay": "keys",
}

REF_H = 1080.0   # lazer skins are near-universally laid out at 1080p


class SkinHudLayout:
    def __init__(self, skin_dir):
        self.elems = {}
        if not skin_dir:
            return
        p = Path(skin_dir) / "MainHUDComponents.json"
        if not p.is_file():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return
        comps = []

        def walk(o):
            if isinstance(o, dict):
                if "Type" in o and "Anchor" in o:
                    comps.append(o)
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for x in o:
                    walk(x)

        walk(data.get("DrawableInfo", data))
        for c in comps:
            t = str(c.get("Type", "")).split(",")[0].split(".")[-1]
            eid = _TYPE_MAP.get(t)
            if not eid or eid in self.elems:   # first occurrence wins
                continue
            pos = c.get("Position") or {}
            try:
                self.elems[eid] = {
                    "anchor": _frac(int(c.get("Anchor", 0) or 0)),
                    "origin": _frac(int(c.get("Origin", 0) or 0)),
                    "pos": (float(pos.get("x", 0.0)), float(pos.get("y", 0.0))),
                }
            except Exception:
                continue

    def has(self, eid):
        return eid in self.elems

    def place(self, eid, ew, eh, W, H, mx=0, my=0):
        """Top-left screen px for an element sized ``ew`` x ``eh`` on a ``W`` x
        ``H`` frame, honouring the skin's anchor/origin/offset. Inset from the
        edges by (mx, my) so corner-flush (pos 0,0) elements don't kiss the
        border. ``None`` when the skin doesn't position this element."""
        e = self.elems.get(eid)
        if e is None:
            return None
        s = H / REF_H
        ax, ay = e["anchor"]
        ox, oy = e["origin"]
        px, py = e["pos"]
        axp = ax * W + (0.5 - ax) * 2.0 * mx + px * s
        ayp = ay * H + (0.5 - ay) * 2.0 * my + py * s
        return axp - ox * ew, ayp - oy * eh
