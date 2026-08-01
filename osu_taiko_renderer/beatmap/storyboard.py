"""Storyboard PARSER — phase 1 of the in-house storyboard engine.

Faithful port of osu!(lazer)'s LegacyStoryboardDecoder
(osu.Game/Beatmaps/Formats/LegacyStoryboardDecoder.cs, master 2026-07) plus
the line plumbing it inherits from LegacyDecoder.cs (sections, comment
stripping, per-line fail-soft) and the multi-stream merge of
Decoder.Decode() / WorkingBeatmapCache.GetStoryboard().

Scope: turn the ``[Events]`` block of the .osu and an external .osb into a
complete DATA MODEL — sprites / animations / videos / samples and their raw
command timelines (plain commands, loops, triggers).  NO interpolation, NO
loop unrolling, NO trigger firing, NO drawing: the command engine (next
phase) consumes these structures.

Merge order (Decoder.cs Decode(): ``otherStreams.Prepend(primaryStream)``;
WorkingBeatmapCache.GetStoryboard() passes the .osu as the primary stream):
the .osu ``[Events]`` is parsed FIRST, then the .osb is parsed into the
same storyboard, appending to the same layers.  Decoder state ($variables,
format version, current sprite) lives on the decoder instance and persists
across both streams — so an .osb's ``[Variables]`` cannot affect the .osu
(it was already parsed), exactly like lazer.  Every element records which
stream it came from (``ElementSource.BEATMAP`` vs ``.OSB`` — lazer's
StoryboardElementSource.Beatmap/Shared).

Values are stored RAW (as written in the file); interpretation belongs to
the command engine.  Deliberate parser-vs-lazer value differences (each is
a decode-time bake in lazer that we defer so no information is lost):
  * ``R`` rotate values stay RADIANS (lazer converts to degrees,
    LegacyStoryboardDecoder.cs:266).
  * ``C`` colour channels stay 0..255 (lazer normalises /255, :305-307).
  * ``M`` move stays ONE command holding (x, y) pairs (lazer splits it
    into an X and a Y timeline, :276-277).
  * ``P`` parameter commands keep kind + easing/times only (lazer bakes
    the "instant when startTime == endTime" end-value trick, :317-328).

Everything else is ported bug-for-bug: depth counting over spaces AND
underscores, ``depth < 2`` resetting the target group to the sprite's root
commands, L/T groups hoisting to the sprite regardless of depth, blank
endTime = startTime, omitted end values = start values, trigger group
number NEGATION (stable parity, :211-212), loop count ``max(0,
count-1)+1`` total iterations, the v<6 frameDelay transform (:176-178),
numeric layer/origin/event aliases with lazer's exact fallbacks, variable
expansion with the no-progress guard, and one-line-one-failure isolation
(a malformed line logs + skips; the parse never crashes).
"""
from __future__ import annotations

import logging
import posixpath
import re
import sys
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path

log = logging.getLogger(__name__)

LATEST_VERSION = 14  # LegacyDecoder.cs:21

# Parsing.cs:8-10
MAX_COORDINATE_VALUE = 131072.0
MAX_PARSE_VALUE = 2147483647.0

_DOUBLE_MAX = sys.float_info.max  # C# double.MaxValue

# SupportedExtensions.cs — VIDEO_EXTENSIONS
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".flv", ".mpg", ".wmv", ".m4v")


# --- enums -----------------------------------------------------------------

class ElementSource(Enum):
    """Which stream an element came from (StoryboardElementSource)."""

    BEATMAP = "beatmap"  # the primary .osu stream
    OSB = "osb"          # the external .osb (lazer: "Shared")


class LayerType(IntEnum):
    """LegacyStoryLayer.cs — the numeric aliases used by the file format."""

    BACKGROUND = 0
    FAIL = 1
    PASS = 2
    FOREGROUND = 3
    OVERLAY = 4
    VIDEO = 5


class Origin(IntEnum):
    """LegacyOrigins.cs — declaration order == stable's numeric values."""

    TOP_LEFT = 0
    CENTRE = 1
    CENTRE_LEFT = 2
    TOP_RIGHT = 3
    BOTTOM_CENTRE = 4
    TOP_CENTRE = 5
    CUSTOM = 6        # unsupported: parses to TOP_LEFT like lazer (:346-381)
    CENTRE_RIGHT = 7
    BOTTOM_LEFT = 8
    BOTTOM_RIGHT = 9


class Easing(IntEnum):
    """osu.Framework Graphics/Easing.cs — storyboard easing indices 0..35."""

    NONE = 0
    OUT = 1
    IN = 2
    IN_QUAD = 3
    OUT_QUAD = 4
    IN_OUT_QUAD = 5
    IN_CUBIC = 6
    OUT_CUBIC = 7
    IN_OUT_CUBIC = 8
    IN_QUART = 9
    OUT_QUART = 10
    IN_OUT_QUART = 11
    IN_QUINT = 12
    OUT_QUINT = 13
    IN_OUT_QUINT = 14
    IN_SINE = 15
    OUT_SINE = 16
    IN_OUT_SINE = 17
    IN_EXPO = 18
    OUT_EXPO = 19
    IN_OUT_EXPO = 20
    IN_CIRC = 21
    OUT_CIRC = 22
    IN_OUT_CIRC = 23
    IN_ELASTIC = 24
    OUT_ELASTIC = 25
    OUT_ELASTIC_HALF = 26
    OUT_ELASTIC_QUARTER = 27
    IN_OUT_ELASTIC = 28
    IN_BACK = 29
    OUT_BACK = 30
    IN_OUT_BACK = 31
    IN_BOUNCE = 32
    OUT_BOUNCE = 33
    IN_OUT_BOUNCE = 34
    OUT_POW10 = 35


class CommandType(Enum):
    """The value-command letters (:236-332). L/T are groups, not commands."""

    FADE = "F"          # values: (alpha,)
    MOVE = "M"          # values: (x, y)
    MOVE_X = "MX"       # values: (x,)
    MOVE_Y = "MY"       # values: (y,)
    SCALE = "S"         # values: (scale,)
    VECTOR_SCALE = "V"  # values: (sx, sy)
    ROTATE = "R"        # values: (radians,)
    COLOUR = "C"        # values: (r, g, b) in 0..255
    PARAMETER = "P"     # values: (); see .parameter


class ParameterType(Enum):
    """P-command kinds (:315-329)."""

    HORIZONTAL_FLIP = "H"
    VERTICAL_FLIP = "V"
    ADDITIVE_BLEND = "A"


class LoopType(IntEnum):
    """AnimationLoopType (StoryboardAnimation.cs)."""

    LOOP_FOREVER = 0
    LOOP_ONCE = 1


# --- command model ----------------------------------------------------------

@dataclass
class Command:
    """One raw storyboard command.  Values are stored exactly as written
    (see module docstring for per-type arity + units).  The command engine
    interpolates/applies; the parser only captures."""

    type: CommandType
    easing: int  # Easing member when 0..35, raw int otherwise (lazer casts blindly, :230)
    start_time: float
    end_time: float
    start_value: tuple[float, ...] = ()
    end_value: tuple[float, ...] = ()
    parameter: ParameterType | None = None  # P commands only


@dataclass
class CommandGroup:
    """A flat list of commands (a sprite's root StoryboardCommandGroup)."""

    commands: list[Command] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.commands)


@dataclass
class Loop(CommandGroup):
    """``L,<startTime>,<loopCount>`` — children replay ``total_iterations``
    times at ``start_time + i * loop_length`` (unrolled by the engine).
    ``repeat_count`` uses lazer semantics: ``max(0, fileValue - 1)``
    (StoryboardLoopingGroup: TotalIterations = repeatCount + 1)."""

    start_time: float = 0.0
    repeat_count: int = 0

    @property
    def total_iterations(self) -> int:
        return self.repeat_count + 1


@dataclass
class Trigger(CommandGroup):
    """``T,<triggerType>[,<startTime>[,<endTime>[,<groupNumber>]]]`` —
    children fire on the named trigger (HitSound…, Passing, Failing);
    the engine handles firing.  Omitted times span all of time
    (double.MinValue/MaxValue, :209-210); group number is NEGATED like
    stable (:211-212)."""

    trigger_name: str = ""
    start_time: float = -_DOUBLE_MAX
    end_time: float = _DOUBLE_MAX
    group_number: int = 0


# --- element model ----------------------------------------------------------

@dataclass(kw_only=True)
class StoryboardSprite:
    """``Sprite,<layer>,<origin>,"<path>",<x>,<y>`` (+ command timeline)."""

    source: ElementSource
    layer: str            # resolved layer NAME ("Background"…"Overlay"/"Video")
    origin: Origin
    path: str             # cleaned map-folder-relative path, case preserved
    x: float
    y: float
    commands: CommandGroup = field(default_factory=CommandGroup)
    loops: list[Loop] = field(default_factory=list)
    triggers: list[Trigger] = field(default_factory=list)

    def add_loop(self, start_time: float, repeat_count: int) -> Loop:
        group = Loop(start_time=start_time, repeat_count=repeat_count)
        self.loops.append(group)
        return group

    def add_trigger(self, name: str, start_time: float, end_time: float,
                    group_number: int) -> Trigger:
        group = Trigger(trigger_name=name, start_time=start_time,
                        end_time=end_time, group_number=group_number)
        self.triggers.append(group)
        return group

    @property
    def has_commands(self) -> bool:
        """StoryboardSprite.HasCommands — root or loop commands (triggers
        alone don't make a sprite drawable in lazer)."""
        return bool(self.commands.commands) or any(l.commands for l in self.loops)

    @property
    def command_count(self) -> int:
        """Total raw commands including loop + trigger children."""
        return (len(self.commands.commands)
                + sum(len(l.commands) for l in self.loops)
                + sum(len(t.commands) for t in self.triggers))


@dataclass(kw_only=True)
class StoryboardAnimation(StoryboardSprite):
    """``Animation,…,<frameCount>,<frameDelay>[,<loopType>]`` — frame files
    are ``path0.ext`` … ``path(frameCount-1).ext`` (see frame_path())."""

    frame_count: int = 0
    frame_delay: float = 0.0
    loop_type: LoopType = LoopType.LOOP_FOREVER

    def frame_path(self, index: int) -> str:
        """Frame file for `index`: the number goes before the extension."""
        stem, dot, ext = self.path.rpartition(".")
        if not dot:
            return f"{self.path}{index}"
        return f"{stem}{index}.{ext}"


@dataclass(kw_only=True)
class StoryboardVideo(StoryboardSprite):
    """``Video,<startTime>,"<path>"`` — a sprite subclass in lazer
    (StoryboardVideo.cs: origin Centre, position zero); commands under a
    Video event attach to it."""

    start_time: float = 0.0


@dataclass(kw_only=True)
class StoryboardSample:
    """``Sample,<time>,<layer>,"<path>",<volume>`` (:186-194)."""

    source: ElementSource
    layer: str
    time: float
    path: str
    volume: float = 100.0


@dataclass
class StoryboardLayerData:
    """One render layer (lazer StoryboardLayer): higher depth draws first
    (further back).  Fail is hidden while passing; Pass while failing."""

    name: str
    depth: int
    visible_when_passing: bool = True
    visible_when_failing: bool = True
    masking: bool = True  # Video layer: False (StoryboardVideoLayer)
    elements: list = field(default_factory=list)


def _default_layers() -> dict[str, StoryboardLayerData]:
    """Storyboard.cs ctor — the fixed layers, in creation order."""
    return {
        "Video": StoryboardLayerData("Video", 4, masking=False),
        "Background": StoryboardLayerData("Background", 3),
        "Fail": StoryboardLayerData("Fail", 2, visible_when_passing=False),
        "Pass": StoryboardLayerData("Pass", 1, visible_when_failing=False),
        "Foreground": StoryboardLayerData("Foreground", 0),
        "Overlay": StoryboardLayerData("Overlay", -(2 ** 31)),
    }


@dataclass
class Storyboard:
    """The parsed storyboard: fixed layers (+ on-demand extras), the
    variable table, and the [General] flags the renderer needs."""

    layers: dict[str, StoryboardLayerData] = field(default_factory=_default_layers)
    variables: dict[str, str] = field(default_factory=dict)
    use_skin_sprites: bool = False
    widescreen: bool = False
    background_offset: tuple[float, float] | None = None
    format_version: int = LATEST_VERSION
    _min_layer_depth: int = field(default=0, repr=False)

    def get_layer(self, name: str) -> StoryboardLayerData:
        """Storyboard.GetLayer — creates unknown layers below Foreground."""
        layer = self.layers.get(name)
        if layer is None:
            self._min_layer_depth -= 1
            layer = StoryboardLayerData(name, self._min_layer_depth)
            self.layers[name] = layer
        return layer

    @property
    def elements(self):
        """All elements, iterated in layer creation order."""
        for layer in self.layers.values():
            yield from layer.elements

    def counts(self) -> dict[str, int]:
        """Model census (report/tests): element + command totals."""
        n = {"sprites": 0, "animations": 0, "videos": 0, "samples": 0,
             "commands": 0, "loops": 0, "triggers": 0}
        for el in self.elements:
            if isinstance(el, StoryboardAnimation):
                n["animations"] += 1
            elif isinstance(el, StoryboardVideo):
                n["videos"] += 1
            elif isinstance(el, StoryboardSprite):
                n["sprites"] += 1
            elif isinstance(el, StoryboardSample):
                n["samples"] += 1
            if isinstance(el, StoryboardSprite):
                n["commands"] += el.command_count
                n["loops"] += len(el.loops)
                n["triggers"] += len(el.triggers)
        return n


# --- low-level parse helpers (Parsing.cs / LegacyDecoder.cs) -----------------

def _parse_float(text: str, limit: float = MAX_PARSE_VALUE) -> float:
    if "_" in text:  # Python-only numeric literal; C# float.Parse rejects
        raise ValueError(f"invalid number: {text!r}")
    value = float(text)
    if value < -limit:
        raise ValueError(f"value too low: {text!r}")
    if value > limit:
        raise ValueError(f"value too high: {text!r}")
    if value != value:  # NaN
        raise ValueError(f"not a number: {text!r}")
    return value


_parse_double = _parse_float  # same rules; C# just uses a wider type


def _parse_int(text: str) -> int:
    if "_" in text:  # Python-only numeric literal; C# int.Parse rejects
        raise ValueError(f"invalid number: {text!r}")
    value = int(text.strip())
    if not -2147483647 <= value <= 2147483647:
        raise ValueError(f"value out of range: {text!r}")
    return value


def _clean_filename(path: str) -> str:
    """LegacyDecoder.CleanFilename — collapse doubled backslashes (stable
    user-error compat), trim quotes, standardise separators to '/'."""
    return path.replace("\\\\", "\\").strip('"').replace("\\", "/")


def _split_key_val(line: str, sep: str = ":") -> tuple[str, str]:
    """LegacyDecoder.SplitKeyVal (trimming variant)."""
    first, _, rest = line.partition(sep)
    return first.strip(), rest.strip()


# Sections LegacyDecoder recognises (unknown headers fall back to General —
# C# Enum.TryParse leaves `section` at default(Section) on failure).
_SECTIONS = frozenset({
    "General", "Editor", "Metadata", "Difficulty", "Events", "TimingPoints",
    "Colours", "HitObjects", "Variables", "Fonts", "CatchTheBeat", "Mania",
})

# LegacyEventType.cs (names + numeric aliases, case-sensitive like Enum.TryParse)
_EVENT_TYPES = {
    "Background": 0, "Video": 1, "Break": 2, "Colour": 3,
    "Sprite": 4, "Sample": 5, "Animation": 6,
}
_EV_BACKGROUND, _EV_VIDEO, _EV_SPRITE, _EV_SAMPLE, _EV_ANIMATION = 0, 1, 4, 5, 6

_LAYER_NAMES = {m.name.capitalize(): m for m in LayerType}  # Background..Video
_ORIGIN_NAMES = {
    "TopLeft": Origin.TOP_LEFT, "Centre": Origin.CENTRE,
    "CentreLeft": Origin.CENTRE_LEFT, "TopRight": Origin.TOP_RIGHT,
    "BottomCentre": Origin.BOTTOM_CENTRE, "TopCentre": Origin.TOP_CENTRE,
    "Custom": Origin.CUSTOM, "CentreRight": Origin.CENTRE_RIGHT,
    "BottomLeft": Origin.BOTTOM_LEFT, "BottomRight": Origin.BOTTOM_RIGHT,
}

_FORMAT_VERSION_RE = re.compile(r"^\s*osu file format v(\d+)")


def _parse_layer(value: str) -> str:
    """parseLayer (:344): Enum.Parse then ToString.  Names map to
    themselves; numeric 0..5 map to layer names; an out-of-range numeric
    becomes a CUSTOM layer named by the number (bug-compatible: undefined
    C# enum values stringify to their number).  Unknown names raise (the
    line is then skipped).  Enum.Parse trims whitespace."""
    value = value.strip()
    if value in _LAYER_NAMES:
        return value
    number = _parse_int(value)  # ValueError on non-numeric unknowns
    try:
        return LayerType(number).name.capitalize()
    except ValueError:
        return str(number)


def _parse_origin(value: str) -> Origin:
    """parseOrigin (:346-382): unknown numerics AND Custom → TopLeft;
    unknown names raise (line skipped).  Enum.Parse trims whitespace."""
    value = value.strip()
    origin = _ORIGIN_NAMES.get(value)
    if origin is None:
        number = _parse_int(value)
        try:
            origin = Origin(number)
        except ValueError:
            return Origin.TOP_LEFT
    return Origin.TOP_LEFT if origin is Origin.CUSTOM else origin


def _parse_loop_type(value: str) -> LoopType:
    """parseAnimationLoopType (:384-388): defined names/numerics pass,
    undefined numerics → LoopForever, unknown names raise (line skipped)."""
    value = value.strip()
    if value == "LoopForever":
        return LoopType.LOOP_FOREVER
    if value == "LoopOnce":
        return LoopType.LOOP_ONCE
    number = _parse_int(value)
    try:
        return LoopType(number)
    except ValueError:
        return LoopType.LOOP_FOREVER


# --- the decoder -------------------------------------------------------------

class _StoryboardDecoder:
    """Instance state mirrors LegacyStoryboardDecoder's fields: the current
    sprite, the current command group, and the $variables table — all of
    which persist across the .osu and .osb streams."""

    def __init__(self, storyboard: Storyboard, variables: dict[str, str] | None = None):
        self.storyboard = storyboard
        self.sprite: StoryboardSprite | None = None
        self.group: CommandGroup | None = None
        if variables:
            storyboard.variables.update(variables)

    # -- stream level (LegacyDecoder.ParseStreamInto) --

    def parse_stream(self, lines, source: ElementSource) -> None:
        section = "General"
        for raw_line in lines:
            line = raw_line.rstrip("\n").rstrip("\r")
            if not line.strip() or line.lstrip().startswith("//"):
                continue
            if section != "Metadata":
                idx = line.find("//")
                if idx > 0:
                    line = line[:idx]
            line = line.rstrip()
            if line.startswith("[") and line.endswith("]"):
                name = line[1:-1]
                if name in _SECTIONS:
                    section = name
                else:
                    # bug-compatible: TryParse failure leaves default(Section)
                    log.warning("storyboard: unknown section %r", line)
                    section = "General"
                continue
            try:
                self._parse_line(section, line, source)
            except Exception as exc:  # noqa: BLE001 — per-line fail-soft
                log.warning("storyboard: failed to parse line %r: %s", line, exc)

    def _parse_line(self, section: str, line: str, source: ElementSource) -> None:
        if section == "General":
            self._handle_general(line)
        elif section == "Events":
            self._handle_events(line, source)
        elif section == "Variables":
            self._handle_variables(line)
        # all other sections: not storyboard data

    # -- [General] (:78-92) --

    def _handle_general(self, line: str) -> None:
        key, value = _split_key_val(line)
        if key == "UseSkinSprites":
            self.storyboard.use_skin_sprites = value == "1"
        elif key == "WidescreenStoryboard":
            self.storyboard.widescreen = _parse_int(value) == 1

    # -- [Variables] (:390-394) --

    def _handle_variables(self, line: str) -> None:
        key, _, value = line.partition("=")  # SplitKeyVal('=', shouldTrim: false)
        self.storyboard.variables[key] = value

    def _decode_variables(self, line: str) -> str:
        """decodeVariables (:400-412) — repeated whole-table substitution
        with a no-progress guard (handles $a=$b chains, survives cycles)."""
        variables = self.storyboard.variables
        while "$" in line:
            orig = line
            for key, value in variables.items():
                line = line.replace(key, value)
            if line == orig:
                break
        return line

    # -- [Events] (:94-342) --

    def _handle_events(self, line: str, source: ElementSource) -> None:
        line = self._decode_variables(line)

        depth = 0
        for ch in line:
            if ch in (" ", "_"):
                depth += 1
            else:
                break
        line = line[depth:]
        split = line.split(",")

        if depth == 0:
            self._handle_element(split, source)
        else:
            self._handle_command(split, depth)

    def _handle_element(self, split: list[str], source: ElementSource) -> None:
        sb = self.storyboard
        self.sprite = None  # before type validation, like :114

        token = split[0].strip()  # Enum.TryParse trims whitespace
        etype = _EVENT_TYPES.get(token)
        if etype is None:
            try:  # Enum.TryParse accepts any numeric string
                etype = _parse_int(token)
            except ValueError:
                raise ValueError(f"unknown event type: {token!r}") from None

        if etype == _EV_BACKGROUND:
            # filename belongs to the beatmap parser; only the offset here
            if len(split) > 4:
                sb.background_offset = (_parse_float(split[3]),
                                        _parse_float(split[4]))

        elif etype == _EV_VIDEO:
            offset = _parse_int(split[1])
            path = _clean_filename(split[2])
            if posixpath.splitext(path)[1].lower() not in VIDEO_EXTENSIONS:
                return  # image mis-declared as Video (:142-148)
            video = StoryboardVideo(source=source, layer="Video",
                                    origin=Origin.CENTRE, path=path,
                                    x=0.0, y=0.0, start_time=float(offset))
            sb.get_layer("Video").elements.append(video)
            self.sprite = video

        elif etype == _EV_SPRITE:
            layer = _parse_layer(split[1])
            origin = _parse_origin(split[2])
            path = _clean_filename(split[3])
            x = _parse_float(split[4], MAX_COORDINATE_VALUE)
            y = _parse_float(split[5], MAX_COORDINATE_VALUE)
            sprite = StoryboardSprite(source=source, layer=layer,
                                      origin=origin, path=path, x=x, y=y)
            sb.get_layer(layer).elements.append(sprite)
            self.sprite = sprite

        elif etype == _EV_ANIMATION:
            layer = _parse_layer(split[1])
            origin = _parse_origin(split[2])
            path = _clean_filename(split[3])
            x = _parse_float(split[4], MAX_COORDINATE_VALUE)
            y = _parse_float(split[5], MAX_COORDINATE_VALUE)
            frame_count = _parse_int(split[6])
            frame_delay = _parse_double(split[7])
            if sb.format_version < 6:
                # "random as hell but taken straight from osu-stable" (:176-178)
                frame_delay = round(0.015 * frame_delay) * 1.186 * (1000 / 60)
            loop_type = (_parse_loop_type(split[8]) if len(split) > 8
                         else LoopType.LOOP_FOREVER)
            animation = StoryboardAnimation(
                source=source, layer=layer, origin=origin, path=path, x=x,
                y=y, frame_count=frame_count, frame_delay=frame_delay,
                loop_type=loop_type)
            sb.get_layer(layer).elements.append(animation)
            self.sprite = animation

        elif etype == _EV_SAMPLE:
            time = _parse_double(split[1])
            layer = _parse_layer(split[2])
            path = _clean_filename(split[3])
            volume = _parse_float(split[4]) if len(split) > 4 else 100.0
            sb.get_layer(layer).elements.append(StoryboardSample(
                source=source, layer=layer, time=time, path=path,
                volume=volume))
        # Break (2) / Colour (3) / unknown numerics: no-op, like the C# switch

    def _handle_command(self, split: list[str], depth: int) -> None:
        if depth < 2:
            self.group = self.sprite.commands if self.sprite else None

        ctype = split[0]

        if ctype == "T":
            name = split[1]
            start = _parse_double(split[2]) if len(split) > 2 else -_DOUBLE_MAX
            end = _parse_double(split[3]) if len(split) > 3 else _DOUBLE_MAX
            # stable negates the group number (:211-212)
            group_number = -_parse_int(split[4]) if len(split) > 4 else 0
            self.group = (self.sprite.add_trigger(name, start, end, group_number)
                          if self.sprite else None)
            return

        if ctype == "L":
            start = _parse_double(split[1])
            repeat = _parse_int(split[2])
            self.group = (self.sprite.add_loop(start, max(0, repeat - 1))
                          if self.sprite else None)
            return

        if split[3] == "":  # blank endTime = startTime (:227-228)
            split[3] = split[2]
        easing_raw = _parse_int(split[1])
        easing = Easing(easing_raw) if 0 <= easing_raw <= 35 else easing_raw
        start = _parse_double(split[2])
        end = _parse_double(split[3])

        parameter = None
        if ctype in ("F", "S", "R", "MX", "MY"):
            sv = (_parse_float(split[4]),)
            ev = (_parse_float(split[5]),) if len(split) > 5 else sv
        elif ctype in ("M", "V"):
            sx = _parse_float(split[4])
            sy = _parse_float(split[5])
            ex = _parse_float(split[6]) if len(split) > 6 else sx
            ey = _parse_float(split[7]) if len(split) > 7 else sy
            sv, ev = (sx, sy), (ex, ey)
        elif ctype == "C":
            sr = _parse_float(split[4])
            sg = _parse_float(split[5])
            sb_ = _parse_float(split[6])
            er = _parse_float(split[7]) if len(split) > 7 else sr
            eg = _parse_float(split[8]) if len(split) > 8 else sg
            eb = _parse_float(split[9]) if len(split) > 9 else sb_
            sv, ev = (sr, sg, sb_), (er, eg, eb)
        elif ctype == "P":
            kind = split[4]
            if kind not in ("A", "H", "V"):
                return  # unhandled parameter kinds are dropped (:315-329)
            parameter = ParameterType(kind)
            sv = ev = ()
        else:
            raise ValueError(f"unknown command type: {ctype!r}")

        if self.group is not None:  # None → orphan command, dropped (:240 `?.`)
            self.group.commands.append(Command(
                type=CommandType(ctype), easing=easing, start_time=start,
                end_time=end, start_value=sv, end_value=ev,
                parameter=parameter))


# --- public API ---------------------------------------------------------------

def _detect_format_version(lines: list[str]) -> int | None:
    for line in lines:
        stripped = line.lstrip("\ufeff").strip()
        if not stripped:
            continue
        match = _FORMAT_VERSION_RE.match(stripped)
        return int(match.group(1)) if match else None
    return None


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8-sig", errors="replace").splitlines()


def find_osb(folder: Path) -> Path | None:
    """First .osb (case-insensitive suffix, sorted for determinism) in a
    map folder — WorkingBeatmapCache picks the set's .osb similarly."""
    candidates = sorted(p for p in folder.iterdir()
                        if p.is_file() and p.suffix.lower() == ".osb")
    return candidates[0] if candidates else None


def parse_storyboard(osu_path: str | Path | None,
                     osb_path: str | Path | None = None,
                     variables: dict[str, str] | None = None) -> Storyboard:
    """Parse the .osu [Events] + the map's .osb into one Storyboard.

    `osu_path` — the beatmap (primary stream, parsed FIRST); may be None
    for a standalone .osb.  `osb_path` — the external storyboard (parsed
    second); when None it is auto-discovered next to the .osu.
    `variables` — optional seed $variables (rarely needed outside tests).
    Fail-soft: malformed lines log + skip; this never raises for bad data.
    """
    osu_lines = osb_lines = None
    if osu_path is not None:
        osu_path = Path(osu_path)
        osu_lines = _read_lines(osu_path)
        if osb_path is None:
            osb_path = find_osb(osu_path.parent)
    if osb_path is not None:
        osb_lines = _read_lines(Path(osb_path))
    return parse_storyboard_text(osu_lines, osb_lines, variables)


def parse_storyboard_text(osu_lines: list[str] | str | None,
                          osb_lines: list[str] | str | None = None,
                          variables: dict[str, str] | None = None) -> Storyboard:
    """parse_storyboard over in-memory text (tests / already-read files)."""
    if isinstance(osu_lines, str):
        osu_lines = osu_lines.splitlines()
    if isinstance(osb_lines, str):
        osb_lines = osb_lines.splitlines()

    storyboard = Storyboard()
    version = None
    if osu_lines is not None:
        version = _detect_format_version(osu_lines)
    if version is None and osb_lines is not None:
        version = _detect_format_version(osb_lines)
    storyboard.format_version = version if version is not None else LATEST_VERSION

    decoder = _StoryboardDecoder(storyboard, variables)
    # merge order (Decoder.Decode): primary .osu first, then the shared .osb
    if osu_lines is not None:
        decoder.parse_stream(osu_lines, ElementSource.BEATMAP)
    if osb_lines is not None:
        decoder.parse_stream(osb_lines, ElementSource.OSB)
    return storyboard
