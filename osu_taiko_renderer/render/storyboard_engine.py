"""Storyboard COMMAND ENGINE — phase 2 of the in-house storyboard system.

Consumes the phase-1 data model (``osu_taiko_renderer.beatmap.storyboard``) and,
for a time ``t`` (ms), produces every sprite's full transform state.  This is a
pure time -> state evaluator: NO drawing, NO textures, NO ffmpeg (phase 3).

Everything here is a faithful port of osu!(lazer) master (2026-07):

* Easings — ``osu.Framework/Graphics/Transforms/DefaultEasingFunction.cs``
  (all 36, incl. OutPow10 and the elastic/back/bounce families), reached via
  the storyboard easing index -> ``osu.Framework.Graphics.Easing`` (identical
  ordering 0..35, ``Easing.cs``).  The pre-decrement/compound mutations of the
  C# source are reproduced exactly.
* Interpolation — ``osu.Framework/Utils/Interpolation.cs`` ``ValueAt``.  The
  scalar/vector path does NOT clamp the eased fraction (elastic/back overshoot
  is intentional); the colour path DOES clamp to [0,1] and interpolates in
  LINEAR (gamma-correct) RGB (``Color4Extensions.ToLinear``/``ToSRGB``,
  ``GAMMA = 2.4``).
* Per-property timelines / transform overwrite semantics — the framework
  ``TargetGroupingTransformTracker`` applies a member's transforms in
  ``(StartTime, insertion)`` order, and when a later-starting command begins it
  fully takes over the property (older overlapping commands are cut off at the
  new one's StartTime).  Net observable rule, per property:
    - before the first command  -> that command's start value (ApplyInitialValue)
    - governed by the last command whose StartTime <= t
        * t within [start, end) -> interpolate (command's easing)
        * t >= end              -> clamp to that command's end value
                                   (the framework's AppliedToEnd freeze; we do
                                    NOT reproduce the frame-rate-dependent
                                    forward-jump overshoot — a deliberate,
                                    deterministic deviation matching danser and
                                    steady-state lazer playback)
* Sprite lifetime — ``StoryboardSprite.StartTime`` (with lazer's alpha-0
  optimisation) to ``EndTimeForDisplay``; ``DrawableStoryboardSprite`` alive
  window is half-open ``[StartTime, EndTimeForDisplay)`` (framework
  ``Drawable.IsAlive``).
* M splits into independent X and Y timelines; S (uniform) and V (per-axis)
  are separate ``Scale`` / ``VectorScale`` members combined multiplicatively;
  R is converted radians -> DEGREES (``float.RadiansToDegrees``); C is /255.
* Loops — unrolled to absolute-time commands.  Period (framework
  ``iterDuration``) = group ``Duration`` = ``maxChildEnd - minChildStart``;
  iteration k of child c spans ``loopStart + c.start + k*period`` ->
  ``loopStart + c.end + k*period`` for ``total_iterations`` iterations.  (When
  ``minChildStart == 0`` — the overwhelmingly common case — the period equals
  the max child end time.)
* Triggers — the timeline machinery is implemented (a fired trigger's children
  fold in as absolute commands at ``fire_time + child.start``), but firing
  detection is STUBBED: no trigger fires unless one is supplied explicitly via
  ``trigger_fires``.  Firing needs hitsound / pass-fail events that this phase
  does not have.  See ``build_sprite_timeline``.

Unit conventions (explicit, per the phase-3 hand-off):
  position (x, y)   : storyboard/osu! pixels, 640x480 space, sprite-declared
                      (x, y) is the default when there is no move timeline.
  scale (sx, sy)    : positive magnitudes = S * V per axis (flips are separate
                      booleans, not sign — phase 3 mirrors + adjusts origin).
  rotation          : DEGREES, clockwise (matches lazer ``Drawable.Rotation``).
  colour (r, g, b)  : 0..1 sRGB.
  alpha             : 0..1 after the stable "flicker" wrap (``if a>1: a%=1``);
                      a value < 0 means invisible (phase 3 clamps low to 0).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..beatmap.storyboard import (
    Command, CommandType, Easing, ParameterType, Loop, Trigger,
    Storyboard, StoryboardSprite, StoryboardAnimation, StoryboardVideo)

__all__ = [
    "apply_easing", "SpriteState", "SpriteTimeline",
    "build_sprite_timeline", "StoryboardEngine",
    "srgb_to_linear", "linear_to_srgb",
]


# --------------------------------------------------------------------------- #
# Easing — DefaultEasingFunction.ApplyEasing (osu.Framework, master)           #
# --------------------------------------------------------------------------- #

_ELASTIC_CONST = 2 * math.pi / .3
_ELASTIC_CONST2 = .3 / 4
_BACK_CONST = 1.70158
_BACK_CONST2 = _BACK_CONST * 1.525
_BOUNCE_CONST = 1 / 2.75

_EXPO_OFFSET = 2.0 ** -10
_ELASTIC_OFFSET_FULL = 2.0 ** -11
_ELASTIC_OFFSET_HALF = 2.0 ** -10 * math.sin((.5 - _ELASTIC_CONST2) * _ELASTIC_CONST)
_ELASTIC_OFFSET_QUARTER = 2.0 ** -10 * math.sin((.25 - _ELASTIC_CONST2) * _ELASTIC_CONST)
_IN_OUT_ELASTIC_OFFSET = 2.0 ** -10 * math.sin((1 - _ELASTIC_CONST2 * 1.5) * _ELASTIC_CONST / 1.5)


def apply_easing(easing, time: float) -> float:
    """Port of ``DefaultEasingFunction.ApplyEasing`` for a normalised
    ``time`` in [0, 1].  ``easing`` may be an :class:`Easing` member or a raw
    int; any value outside 0..35 falls through to linear (the C# ``default``),
    matching lazer's blind cast of unknown storyboard easing indices."""
    e = int(easing)

    if e == 0:  # None
        return time
    if e in (2, 3):  # In, InQuad
        return time * time
    if e in (1, 4):  # Out, OutQuad
        return time * (2 - time)
    if e == 5:  # InOutQuad
        if time < .5:
            return time * time * 2
        time -= 1
        return time * time * -2 + 1
    if e == 6:  # InCubic
        return time * time * time
    if e == 7:  # OutCubic
        time -= 1
        return time * time * time + 1
    if e == 8:  # InOutCubic
        if time < .5:
            return time * time * time * 4
        time -= 1
        return time * time * time * 4 + 1
    if e == 9:  # InQuart
        return time * time * time * time
    if e == 10:  # OutQuart
        time -= 1
        return 1 - time * time * time * time
    if e == 11:  # InOutQuart
        if time < .5:
            return time * time * time * time * 8
        time -= 1
        return time * time * time * time * -8 + 1
    if e == 12:  # InQuint
        return time * time * time * time * time
    if e == 13:  # OutQuint
        time -= 1
        return time * time * time * time * time + 1
    if e == 14:  # InOutQuint
        if time < .5:
            return time * time * time * time * time * 16
        time -= 1
        return time * time * time * time * time * 16 + 1
    if e == 15:  # InSine
        return 1 - math.cos(time * math.pi * .5)
    if e == 16:  # OutSine
        return math.sin(time * math.pi * .5)
    if e == 17:  # InOutSine
        return .5 - .5 * math.cos(math.pi * time)
    if e == 18:  # InExpo
        return 2.0 ** (10 * (time - 1)) + _EXPO_OFFSET * (time - 1)
    if e == 19:  # OutExpo
        return -(2.0 ** (-10 * time)) + 1 + _EXPO_OFFSET * time
    if e == 20:  # InOutExpo
        if time < .5:
            return .5 * (2.0 ** (20 * time - 10) + _EXPO_OFFSET * (2 * time - 1))
        return 1 - .5 * (2.0 ** (-20 * time + 10) + _EXPO_OFFSET * (-2 * time + 1))
    if e == 21:  # InCirc
        return 1 - math.sqrt(1 - time * time)
    if e == 22:  # OutCirc
        time -= 1
        return math.sqrt(1 - time * time)
    if e == 23:  # InOutCirc
        time *= 2
        if time < 1:
            return .5 - .5 * math.sqrt(1 - time * time)
        time -= 2
        return .5 * math.sqrt(1 - time * time) + .5
    if e == 24:  # InElastic
        return (-(2.0 ** (-10 + 10 * time))
                * math.sin((1 - _ELASTIC_CONST2 - time) * _ELASTIC_CONST)
                + _ELASTIC_OFFSET_FULL * (1 - time))
    if e == 25:  # OutElastic
        return (2.0 ** (-10 * time)
                * math.sin((time - _ELASTIC_CONST2) * _ELASTIC_CONST)
                + 1 - _ELASTIC_OFFSET_FULL * time)
    if e == 26:  # OutElasticHalf
        return (2.0 ** (-10 * time)
                * math.sin((.5 * time - _ELASTIC_CONST2) * _ELASTIC_CONST)
                + 1 - _ELASTIC_OFFSET_HALF * time)
    if e == 27:  # OutElasticQuarter
        return (2.0 ** (-10 * time)
                * math.sin((.25 * time - _ELASTIC_CONST2) * _ELASTIC_CONST)
                + 1 - _ELASTIC_OFFSET_QUARTER * time)
    if e == 28:  # InOutElastic
        time *= 2
        if time < 1:
            return -.5 * (2.0 ** (-10 + 10 * time)
                          * math.sin((1 - _ELASTIC_CONST2 * 1.5 - time) * _ELASTIC_CONST / 1.5)
                          - _IN_OUT_ELASTIC_OFFSET * (1 - time))
        time -= 1
        return .5 * (2.0 ** (-10 * time)
                     * math.sin((time - _ELASTIC_CONST2 * 1.5) * _ELASTIC_CONST / 1.5)
                     - _IN_OUT_ELASTIC_OFFSET * time) + 1
    if e == 29:  # InBack
        return time * time * ((_BACK_CONST + 1) * time - _BACK_CONST)
    if e == 30:  # OutBack
        time -= 1
        return time * time * ((_BACK_CONST + 1) * time + _BACK_CONST) + 1
    if e == 31:  # InOutBack
        time *= 2
        if time < 1:
            return .5 * time * time * ((_BACK_CONST2 + 1) * time - _BACK_CONST2)
        time -= 2
        return .5 * (time * time * ((_BACK_CONST2 + 1) * time + _BACK_CONST2) + 2)
    if e == 32:  # InBounce
        time = 1 - time
        if time < _BOUNCE_CONST:
            return 1 - 7.5625 * time * time
        if time < 2 * _BOUNCE_CONST:
            time -= 1.5 * _BOUNCE_CONST
            return 1 - (7.5625 * time * time + .75)
        if time < 2.5 * _BOUNCE_CONST:
            time -= 2.25 * _BOUNCE_CONST
            return 1 - (7.5625 * time * time + .9375)
        time -= 2.625 * _BOUNCE_CONST
        return 1 - (7.5625 * time * time + .984375)
    if e == 33:  # OutBounce
        if time < _BOUNCE_CONST:
            return 7.5625 * time * time
        if time < 2 * _BOUNCE_CONST:
            time -= 1.5 * _BOUNCE_CONST
            return 7.5625 * time * time + .75
        if time < 2.5 * _BOUNCE_CONST:
            time -= 2.25 * _BOUNCE_CONST
            return 7.5625 * time * time + .9375
        time -= 2.625 * _BOUNCE_CONST
        return 7.5625 * time * time + .984375
    if e == 34:  # InOutBounce
        if time < .5:
            return .5 - .5 * apply_easing(Easing.OUT_BOUNCE, 1 - time * 2)
        return apply_easing(Easing.OUT_BOUNCE, (time - .5) * 2) * .5 + .5
    if e == 35:  # OutPow10
        time -= 1
        return time * (time ** 10) + 1

    return time  # unknown index -> linear (C# switch default)


# --------------------------------------------------------------------------- #
# sRGB <-> linear (Color4Extensions, GAMMA = 2.4)                              #
# --------------------------------------------------------------------------- #

_GAMMA = 2.4


def srgb_to_linear(c: float) -> float:
    if c == 1:
        return 1.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** _GAMMA


def linear_to_srgb(c: float) -> float:
    if c == 1:
        return 1.0
    return 12.92 * c if c < 0.0031308 else 1.055 * (c ** (1.0 / _GAMMA)) - 0.055


# --------------------------------------------------------------------------- #
# Per-property timelines                                                        #
# --------------------------------------------------------------------------- #
#
# A segment is (start, end, start_value, end_value, easing, idx); value shape
# depends on the property (float / (r,g,b) / (sx,sy)).  ``idx`` is a per-sprite
# monotonic insertion counter (root commands in file order first, then loop
# unrolls, then fired triggers) breaking StartTime ties — the later insertion
# wins, matching the framework's TransformID ordering.

def _governing(segs, t):
    """The last segment (by sorted (start, idx)) whose start <= t, or None."""
    gov = None
    for s in segs:
        if s[0] <= t:
            gov = s
        else:
            break
    return gov


def _eval_scalar(segs, default, t):
    if not segs:
        return default
    gov = _governing(segs, t)
    if gov is None:
        return segs[0][2]  # ApplyInitialValue: first command's start value
    start, end, sv, ev, easing, _ = gov
    if t >= end:
        return ev  # AppliedToEnd freeze
    if sv == ev:
        return sv
    duration = end - start
    current = t - start
    if current == 0 or duration == 0:
        return sv
    tt = apply_easing(easing, current / duration)  # NOT clamped (overshoot ok)
    return sv + tt * (ev - sv)


def _eval_vector(segs, default, t):
    if not segs:
        return default
    gov = _governing(segs, t)
    if gov is None:
        return segs[0][2]
    start, end, sv, ev, easing, _ = gov
    if t >= end:
        return ev
    duration = end - start
    current = t - start
    if current == 0 or duration == 0:
        return sv
    tt = apply_easing(easing, current / duration)
    return (sv[0] + tt * (ev[0] - sv[0]), sv[1] + tt * (ev[1] - sv[1]))


def _eval_colour(segs, default, t):
    if not segs:
        return default
    gov = _governing(segs, t)
    if gov is None:
        return segs[0][2]
    start, end, sv, ev, easing, _ = gov
    if t >= end:
        return ev
    if sv == ev:
        return sv
    duration = end - start
    current = t - start
    if duration == 0 or current == 0:
        return sv
    tt = apply_easing(easing, current / duration)
    tt = 0.0 if tt < 0.0 else (1.0 if tt > 1.0 else tt)  # colour path clamps
    out = []
    for i in range(3):
        sl = srgb_to_linear(sv[i])
        el = srgb_to_linear(ev[i])
        out.append(linear_to_srgb(sl + tt * (el - sl)))
    return (out[0], out[1], out[2])


def _eval_param(segs, t):
    """FlipH / FlipV / Additive: boolean 'on' timeline.

    Each parameter command's start value is 'on' (True); its end value is 'on'
    only when the command is instant (start == end), else 'off' (lazer:
    ``AddFlipH(..., true, start == end)`` / ``AddBlendingParameters(...,
    Additive, start == end ? Additive : Inherit)``).  ``ApplyInitialValue``
    turns the member permanently 'on' iff the FIRST (min start) command is
    instant."""
    if not segs:
        return False
    # initial value: first command by (start, idx); on iff instant
    first = min(segs, key=lambda s: (s[0], s[5]))
    initial = first[0] == first[1]
    gov = _governing(segs, t)
    if gov is None:
        return initial
    start, end, _sv, _ev, _e, _i = gov
    if t < end:
        return True          # start value ('on') during the command
    return start == end      # end value: 'on' only if instant


# --------------------------------------------------------------------------- #
# Sprite state + timeline                                                       #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SpriteState:
    """A sprite's fully-resolved transform at one instant (see module unit
    conventions)."""

    x: float
    y: float
    scale_x: float
    scale_y: float
    rotation: float          # degrees
    r: float                 # 0..1 sRGB
    g: float
    b: float
    alpha: float             # 0..1 (post-flicker); < 0 => invisible
    flip_h: bool
    flip_v: bool
    additive: bool


def _norm_command(cmd: Command):
    """(start, end) with lazer's ``endTime < startTime -> endTime = startTime``
    clamp (StoryboardCommand ctor)."""
    start = cmd.start_time
    end = cmd.end_time if cmd.end_time >= start else start
    return start, end


class SpriteTimeline:
    """Pre-resolved per-property timelines for one sprite, with an
    O(active-commands) :meth:`state_at`.  Construct via
    :func:`build_sprite_timeline`."""

    __slots__ = (
        "element", "default_x", "default_y",
        "_x", "_y", "_scale", "_vscale", "_rot", "_colour", "_alpha",
        "_fliph", "_flipv", "_additive",
        "start_time", "end_time", "earliest_transform_time",
    )

    def __init__(self, element):
        self.element = element
        self.default_x = float(element.x)
        self.default_y = float(element.y)
        self._x = []
        self._y = []
        self._scale = []
        self._vscale = []
        self._rot = []
        self._colour = []
        self._alpha = []
        self._fliph = []
        self._flipv = []
        self._additive = []
        self.start_time = 0.0
        self.end_time = 0.0
        self.earliest_transform_time = 0.0

    # -- alive window (framework Drawable.IsAlive: half-open) --

    def is_alive(self, t: float) -> bool:
        return self.start_time <= t < self.end_time

    # -- full state --

    def state_at(self, t: float) -> SpriteState:
        s = _eval_scalar(self._scale, 1.0, t)
        vx, vy = _eval_vector(self._vscale, (1.0, 1.0), t)
        r, g, b = _eval_colour(self._colour, (1.0, 1.0, 1.0), t)
        alpha = _eval_scalar(self._alpha, 1.0, t)
        if alpha > 1:
            alpha %= 1  # DrawableStoryboardSprite.Update stable-flicker wrap
        return SpriteState(
            x=_eval_scalar(self._x, self.default_x, t),
            y=_eval_scalar(self._y, self.default_y, t),
            scale_x=s * vx,
            scale_y=s * vy,
            rotation=_eval_scalar(self._rot, 0.0, t),
            r=r, g=g, b=b,
            alpha=alpha,
            flip_h=_eval_param(self._fliph, t),
            flip_v=_eval_param(self._flipv, t),
            additive=_eval_param(self._additive, t),
        )


def _rad_to_deg(v: float) -> float:
    return v * (180.0 / math.pi)


def build_sprite_timeline(sprite: StoryboardSprite, trigger_fires=None) -> SpriteTimeline:
    """Resolve a sprite's raw command groups (root + loops + optionally fired
    triggers) into a :class:`SpriteTimeline`.

    ``trigger_fires`` — optional ``{trigger_index: [fire_time_ms, ...]}`` where
    ``trigger_index`` indexes ``sprite.triggers``.  Firing is STUBBED (no
    auto-detection); pass explicit fire times to fold a trigger's children in
    (absolute = fire_time + child.start).  Omitted => no trigger contributes
    (documented phase-2 stub)."""
    tl = SpriteTimeline(sprite)
    counter = [0]

    def add_command(cmd: Command, base: float):
        """Fold one command (times offset by ``base``) into its timeline(s)."""
        idx = counter[0]
        counter[0] += 1
        start, end = _norm_command(cmd)
        start += base
        end += base
        ct = cmd.type
        if ct is CommandType.FADE:
            tl._alpha.append((start, end, cmd.start_value[0], cmd.end_value[0], cmd.easing, idx))
        elif ct is CommandType.MOVE:
            tl._x.append((start, end, cmd.start_value[0], cmd.end_value[0], cmd.easing, idx))
            tl._y.append((start, end, cmd.start_value[1], cmd.end_value[1], cmd.easing, idx))
        elif ct is CommandType.MOVE_X:
            tl._x.append((start, end, cmd.start_value[0], cmd.end_value[0], cmd.easing, idx))
        elif ct is CommandType.MOVE_Y:
            tl._y.append((start, end, cmd.start_value[0], cmd.end_value[0], cmd.easing, idx))
        elif ct is CommandType.SCALE:
            tl._scale.append((start, end, cmd.start_value[0], cmd.end_value[0], cmd.easing, idx))
        elif ct is CommandType.VECTOR_SCALE:
            tl._vscale.append((start, end,
                               (cmd.start_value[0], cmd.start_value[1]),
                               (cmd.end_value[0], cmd.end_value[1]), cmd.easing, idx))
        elif ct is CommandType.ROTATE:
            tl._rot.append((start, end, _rad_to_deg(cmd.start_value[0]),
                            _rad_to_deg(cmd.end_value[0]), cmd.easing, idx))
        elif ct is CommandType.COLOUR:
            sv = (cmd.start_value[0] / 255.0, cmd.start_value[1] / 255.0, cmd.start_value[2] / 255.0)
            ev = (cmd.end_value[0] / 255.0, cmd.end_value[1] / 255.0, cmd.end_value[2] / 255.0)
            tl._colour.append((start, end, sv, ev, cmd.easing, idx))
        elif ct is CommandType.PARAMETER:
            target = {ParameterType.HORIZONTAL_FLIP: tl._fliph,
                      ParameterType.VERTICAL_FLIP: tl._flipv,
                      ParameterType.ADDITIVE_BLEND: tl._additive}.get(cmd.parameter)
            if target is not None:
                target.append((start, end, True, start == end, cmd.easing, idx))

    # -- root commands (file order) --
    for cmd in sprite.commands.commands:
        add_command(cmd, 0.0)

    # -- loops: unroll to absolute-time commands --
    loop_display_ends = []
    loop_group_starts = []
    # (loop_start_abs, iter0_alpha_segs) for the alpha lifetime optimisation
    loop_alpha_iter0 = []
    for loop in sprite.loops:
        children = loop.commands
        if not children:
            continue
        norm = [_norm_command(c) for c in children]
        min_child_start = min(s for s, _ in norm)
        max_child_end = max(en for _, en in norm)
        period = max_child_end - min_child_start          # framework iterDuration
        group_start = loop.start_time + min_child_start   # StoryboardCommandGroup.StartTime
        loop_group_starts.append(group_start)
        loop_display_ends.append(group_start + period * loop.total_iterations)
        # collect iteration-0 alpha children for the StartTime optimisation
        it0 = []
        for cmd, (cs, ce) in zip(children, norm):
            if cmd.type is CommandType.FADE:
                it0.append((loop.start_time + cs, loop.start_time + ce,
                            cmd.start_value[0], cmd.end_value[0]))
        loop_alpha_iter0.append(it0)
        for k in range(loop.total_iterations):
            offset = loop.start_time + k * period
            for cmd in children:
                add_command(cmd, offset)

    # -- fired triggers (STUB: only when explicitly supplied) --
    if trigger_fires:
        for ti, trig in enumerate(sprite.triggers):
            for fire_time in trigger_fires.get(ti, ()):  # absolute fire time
                for cmd in trig.commands:
                    add_command(cmd, fire_time)

    # -- sort every timeline by (start, idx) --
    for segs in (tl._x, tl._y, tl._scale, tl._vscale, tl._rot, tl._colour,
                 tl._alpha, tl._fliph, tl._flipv, tl._additive):
        segs.sort(key=lambda s: (s[0], s[5]))

    # -- lifetime (StoryboardSprite.EarliestTransformTime / EndTimeForDisplay) --
    root_starts = [_norm_command(c)[0] for c in sprite.commands.commands]
    root_ends = [_norm_command(c)[1] for c in sprite.commands.commands]
    earliest_candidates = root_starts + loop_group_starts
    end_candidates = root_ends + loop_display_ends
    tl.earliest_transform_time = min(earliest_candidates) if earliest_candidates else 0.0
    tl.end_time = max(end_candidates) if end_candidates else tl.earliest_transform_time
    tl.start_time = _compute_start_time(sprite, loop_alpha_iter0, tl.earliest_transform_time)
    return tl


def _compute_start_time(sprite, loop_alpha_iter0, earliest_transform_time):
    """StoryboardSprite.StartTime: lazer's alpha-0 lifetime optimisation.

    Collect alpha commands (root, then per-loop iteration-0) up to and
    including the first that is visible-at-start-or-end; if the absolute
    earliest alpha starts fully invisible (start value 0) and a visible alpha
    exists, the sprite only 'starts' at that first visible alpha."""
    def visible(a):  # (start, end, sv, ev): StartValue > 0 || EndValue > 0
        return a[2] > 0 or a[3] > 0

    collected = []
    root_alpha = [(_norm_command(c)[0], _norm_command(c)[1], c.start_value[0], c.end_value[0])
                  for c in sprite.commands.commands if c.type is CommandType.FADE]
    root_alpha.sort(key=lambda a: (a[0], a[1]))  # SortedList (StartTime, EndTime)
    for a in root_alpha:
        collected.append(a)
        if visible(a):
            break
    for it0 in loop_alpha_iter0:
        it0_sorted = sorted(it0, key=lambda a: (a[0], a[1]))
        for a in it0_sorted:
            collected.append(a)
            if visible(a):
                break

    if collected:
        first_alpha = min(collected, key=lambda a: a[0])
        reals = [a for a in collected if visible(a)]
        first_real = min(reals, key=lambda a: a[0]) if reals else None
        if first_alpha[2] == 0 and first_real is not None:
            return first_real[0]
    return earliest_transform_time


# --------------------------------------------------------------------------- #
# Whole-storyboard engine                                                       #
# --------------------------------------------------------------------------- #

class StoryboardEngine:
    """Builds timelines for every drawable sprite in a :class:`Storyboard` and
    evaluates them at a time ``t``.

    ``sprites`` is ordered back-to-front (draw order): layers by descending
    depth (Video 4 .. Overlay -2^31), elements within a layer in insertion
    order.  Samples and command-less videos are NOT included here (phase 3
    reads them off ``storyboard`` directly)."""

    def __init__(self, storyboard: Storyboard, trigger_fires=None):
        self.storyboard = storyboard
        self.sprites: list[tuple[StoryboardSprite, SpriteTimeline]] = []
        for layer in sorted(storyboard.layers.values(), key=lambda l: -l.depth):
            for el in layer.elements:
                if isinstance(el, StoryboardSprite) and el.has_commands:
                    self.sprites.append((el, build_sprite_timeline(el, trigger_fires)))

    def state_at(self, t: float):
        """``[(element, SpriteState), ...]`` for every sprite alive at ``t``,
        in back-to-front draw order."""
        return [(el, tl.state_at(t)) for el, tl in self.sprites if tl.is_alive(t)]

    def timeline_for(self, element) -> SpriteTimeline | None:
        for el, tl in self.sprites:
            if el is element:
                return tl
        return None
