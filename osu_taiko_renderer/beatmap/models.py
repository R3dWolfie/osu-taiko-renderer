"""Data model for the osu!taiko renderer.

Taiko has no playfield x: hit objects are a 1-D stream in time that scrolls
right -> left to a fixed hit target near the left edge. Time is absolute ms
from the start of the audio.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class TaikoType(Enum):
    DON = "don"            # red centre hit (no whistle/clap)
    KAT = "kat"            # blue rim hit (whistle or clap)
    DRUMROLL = "drumroll"  # slider -> a rolling bar held across a duration
    DRUMROLL_TICK = "tick" # individual beat inside a drumroll
    SWELL = "swell"        # spinner -> alternating-hit shaker (denden)


@dataclass(frozen=True)
class TaikoObject:
    """One taiko hit object on the scroll stream.

    `time_ms` is when it must be hit (reaches the hit target). `big` marks a
    finish/large note (needs both keys). For DRUMROLL/SWELL `end_ms` is the
    release/end time; for SWELL `required_hits` is the hit count to clear it.
    `scroll_vel` is px/ms the note travels (from its timing point's effective
    SV), so notes under different SV move at different speeds like in lazer.
    """
    time_ms: int
    kind: TaikoType
    big: bool = False
    end_ms: int | None = None
    required_hits: int = 0
    scroll_vel: float = 0.0
    new_combo: bool = False


@dataclass(frozen=True)
class TaikoFrame:
    """One replay frame: which keys are pressed. Taiko has 4 inputs — two don
    centre keys (left/right) and two kat rim keys (left/right). `don`/`kat` are
    the collapsed booleans the judge uses; `cl/cr/rl/rr` keep the individual
    centre-left/centre-right/rim-left/rim-right state for the input-drum
    visualisation (which quadrant flashes)."""
    time_ms: int
    don: bool
    kat: bool
    big: bool = False   # both don (or both kat) held → a big-note hit
    cl: bool = False    # centre-left  (bit 1)
    cr: bool = False    # centre-right (bit 4)
    rl: bool = False    # rim-left     (bit 2)
    rr: bool = False    # rim-right    (bit 8)


@dataclass
class TaikoBeatmap:
    objects: list[TaikoObject]
    cs: float = 5.0
    ar: float = 9.0
    od: float = 5.0
    hp: float = 5.0
    audio_filename: str | None = None
    background: str | None = None
    breaks: list = field(default_factory=list)   # [(start_ms, end_ms)]
    title: str = ""
    artist: str = ""
    version: str = ""
    rate: float = 1.0   # DT/NC=1.5, HT=0.75; object times stay in map-time
    bar_lines: list = field(default_factory=list)  # [(time_ms, scroll_vel, major)]
    kiai_ranges: list = field(default_factory=list)  # [(start_ms, end_ms)]
    timing: object = None                            # _Timing (beat grid for kiai)

    @property
    def length_ms(self) -> int:
        return max((o.end_ms or o.time_ms for o in self.objects), default=0)


@dataclass(frozen=True)
class ReplayMeta:
    mode: int
    beatmap_md5: str
    player_name: str
    mods: int
    score: int
    max_combo: int
    count_300: int   # taiko GREAT
    count_100: int   # taiko OK
    count_50: int    # unused in taiko (kept for .osr layout parity)
    count_katu: int
    count_miss: int
    accuracy: float
    grade: str
    game_version: int = 0
    # ((time_ms, life 0..1), …) from the .osr life-bar graph, if the replay
    # carries one (stable replays usually do; lazer/API ones often don't).
    life_bar: tuple = ()
    # last meaningful input-frame time — used to detect a fail/quit (the replay
    # stops recording at death) and end the video there instead of the map end.
    replay_end_ms: int = 0


@dataclass
class RenderConfig:
    resolution: tuple[int, int] = (1920, 1080)
    fps: int = 60
    encoder: str = "auto"
    encoder_device: str | None = None
    skin_dir: Path | None = None
    default_skin_dir: Path | None = None
    # timing / outro
    lead_in_ms: int = 1500
    tail_ms: int = 1500
    skip_intro: bool = True
    show_countdown: bool = False
    show_results: bool = True
    results_ms: int = 4500
    letterbox_breaks: bool = True
    # taiko scroll: ms a 1.0x-SV note is visible crossing from spawn to target.
    # Lower = faster scroll. Scaled per-note by its effective SV.
    scroll_time_ms: int = 1600
    # cosmetic
    show_hit_explosion: bool = True
    show_kiai: bool = True
    watermark: str = ""
    # audio
    music_volume: int = 100
    general_volume: int = 100
    audio_offset_ms: int = 0
    # background
    bg_dim_intro: int = 0
    bg_dim_game: int = 70
    bg_dim_breaks: int = 0
    bg_blur: int = 0
    # HUD toggles
    show_combo: bool = True
    show_score: bool = True
    show_hp_bar: bool = True
    show_grade: bool = True
    show_mods: bool = True
    show_pp_counter: bool = True
    show_hit_counter: bool = True
    # intro R3D "R" splash (parity with std/catch show_logo; off by default so
    # existing renders are unchanged)
    show_logo: bool = False
    # results-screen map leaderboard (parity with std/catch): the featured play
    # flanked by compact ranked cards of the OTHER renders of this map. Default
    # source = the local render DB; "osu" reads the bot-written osu! global
    # scores JSON (falls back to the DB when absent). Default-on but a no-op when
    # the map has no other renders, so existing renders are unchanged.
    show_leaderboard: bool = True
    leaderboard_source: str = "r3d"      # r3d | osu
    leaderboard_json: Path | None = None
    # FEATURED results-card avatar: the current player's REAL osu! avatar PNG
    # (service passes it). None -> the engine draws the procedural username chip.
    featured_avatar_png: Path | None = None


@dataclass
class Sprite:
    """A single textured/coloured quad to draw this frame (back-to-front)."""
    x: float                 # screen px, center
    y: float                 # screen px, center
    w: float
    h: float
    texture_key: str | None = None
    color: tuple[float, float, float, float] = (1, 1, 1, 1)
    rotation: float = 0.0


@dataclass
class SceneState:
    """Everything to draw for one frame, plus HUD numbers."""
    sprites: list[Sprite] = field(default_factory=list)
    combo: int = 0
    score: int = 0
    accuracy: float = 1.0
    hp: float = 1.0
    time_ms: int = 0
    pp: float = 0.0
    counts: tuple = (0, 0, 0)   # (great, ok, miss)


# osu!taiko geometry / timing -------------------------------------------------

# The input drum (4-zone press visualiser) sits at the far left; the hit
# target (judgement circle the notes land on) sits in front of it.
INPUT_DRUM_X_FRAC = 0.072
# The hit target sits this fraction across the playfield from the left.
HIT_TARGET_X_FRAC = 0.20
# Playfield band vertical center (fraction of height) + band height fraction.
PLAYFIELD_Y_FRAC = 0.32
PLAYFIELD_H_FRAC = 0.22


def od_to_hit_windows_ms(od: float) -> tuple[float, float]:
    """(GREAT, OK) hit-window half-widths in ms for taiko, from OD.

    osu!taiko (stable) windows: GREAT = 50 - 3*OD, OK = 120 - 8*OD (ms),
    measured from the note time. MISS is anything outside OK.
    """
    great = 50.0 - 3.0 * od
    ok = 120.0 - 8.0 * od
    return max(great, 20.0), max(ok, 50.0)
