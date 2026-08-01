"""Storyboard PARSER (phase 1): the LegacyStoryboardDecoder port.

Covers: every command type (F/M/MX/MY/S/V/R/C/P), loop + trigger nesting,
$variable expansion, .osu+.osb merge order (primary .osu first), all 9
origins + 5 layers (names AND numeric aliases), the blank-endTime /
omitted-endValue shorthands, animations (incl. the v<6 frameDelay
transform), video/sample/background events, and malformed-line fail-soft.
NO command execution here — the parser only captures raw structures.
"""
import math
import tempfile
from pathlib import Path

from osu_taiko_renderer.beatmap.storyboard import (
    CommandType, Easing, ElementSource, LoopType, Origin,
    ParameterType, Storyboard, StoryboardAnimation, StoryboardSample,
    StoryboardSprite, StoryboardVideo, parse_storyboard,
    parse_storyboard_text, _DOUBLE_MAX)


def _sb(events: str) -> Storyboard:
    """Parse an [Events] body (osu-side unless noted)."""
    return parse_storyboard_text(
        f"osu file format v14\n\n[Events]\n{events}\n")


def _one_sprite(events: str, layer: str = "Background") -> StoryboardSprite:
    sb = _sb(events)
    (el,) = sb.layers[layer].elements
    return el


SPRITE = 'Sprite,Background,Centre,"sb/x.png",320,240\n'


# --- elements ----------------------------------------------------------------

def test_sprite_basic():
    spr = _one_sprite('Sprite,Background,Centre,"SB\\dir\\pic.png",320,240.5')
    assert isinstance(spr, StoryboardSprite)
    assert spr.layer == "Background"
    assert spr.origin is Origin.CENTRE
    assert spr.path == "SB/dir/pic.png"  # quotes trimmed, '\' standardised
    assert (spr.x, spr.y) == (320.0, 240.5)
    assert spr.source is ElementSource.BEATMAP
    assert not spr.has_commands


def test_unquoted_and_doubled_backslash_path():
    # stable user-error compat: '\\\\' collapses (CleanFilename)
    spr = _one_sprite('Sprite,Background,Centre,SB\\\\pic.png,0,0')
    assert spr.path == "SB/pic.png"


def test_all_nine_origins():
    names = {
        "TopLeft": Origin.TOP_LEFT, "TopCentre": Origin.TOP_CENTRE,
        "TopRight": Origin.TOP_RIGHT, "CentreLeft": Origin.CENTRE_LEFT,
        "Centre": Origin.CENTRE, "CentreRight": Origin.CENTRE_RIGHT,
        "BottomLeft": Origin.BOTTOM_LEFT,
        "BottomCentre": Origin.BOTTOM_CENTRE,
        "BottomRight": Origin.BOTTOM_RIGHT,
    }
    for name, want in names.items():
        assert _one_sprite(f'Sprite,Background,{name},"a.png",0,0').origin is want
    # numeric aliases use LegacyOrigins declaration order
    for num, want in [("0", Origin.TOP_LEFT), ("1", Origin.CENTRE),
                      ("2", Origin.CENTRE_LEFT), ("3", Origin.TOP_RIGHT),
                      ("4", Origin.BOTTOM_CENTRE), ("5", Origin.TOP_CENTRE),
                      ("7", Origin.CENTRE_RIGHT), ("8", Origin.BOTTOM_LEFT),
                      ("9", Origin.BOTTOM_RIGHT)]:
        assert _one_sprite(f'Sprite,Background,{num},"a.png",0,0').origin is want
    # Custom (6) and undefined numerics resolve to TopLeft, like lazer
    assert _one_sprite('Sprite,Background,Custom,"a.png",0,0').origin is Origin.TOP_LEFT
    assert _one_sprite('Sprite,Background,6,"a.png",0,0').origin is Origin.TOP_LEFT
    assert _one_sprite('Sprite,Background,42,"a.png",0,0').origin is Origin.TOP_LEFT
    # unknown origin NAME → whole line skipped (Enum.Parse throws)
    assert not list(_sb('Sprite,Background,MiddleIsh,"a.png",0,0').elements)


def test_all_five_layers():
    for name in ("Background", "Fail", "Pass", "Foreground", "Overlay"):
        sb = _sb(f'Sprite,{name},Centre,"a.png",0,0')
        assert len(sb.layers[name].elements) == 1
    # numeric aliases (LegacyStoryLayer)
    for num, name in [("0", "Background"), ("1", "Fail"), ("2", "Pass"),
                      ("3", "Foreground"), ("4", "Overlay")]:
        sb = _sb(f'Sprite,{num},Centre,"a.png",0,0')
        assert len(sb.layers[name].elements) == 1
    # out-of-range numeric → custom layer named by the number (bug-compat)
    sb = _sb('Sprite,7,Centre,"a.png",0,0')
    assert len(sb.layers["7"].elements) == 1
    assert sb.layers["7"].depth == -1  # created below Foreground
    # unknown layer NAME → line skipped
    assert not list(_sb('Sprite,Backgroundish,Centre,"a.png",0,0').elements)


def test_layer_visibility_flags():
    sb = Storyboard()
    assert not sb.layers["Fail"].visible_when_passing
    assert sb.layers["Fail"].visible_when_failing
    assert not sb.layers["Pass"].visible_when_failing
    assert sb.layers["Pass"].visible_when_passing
    assert not sb.layers["Video"].masking


# --- plain commands ----------------------------------------------------------

def test_fade_command():
    spr = _one_sprite(SPRITE + " F,0,1000,2000,0,1")
    (cmd,) = spr.commands.commands
    assert cmd.type is CommandType.FADE
    assert cmd.easing is Easing.NONE
    assert (cmd.start_time, cmd.end_time) == (1000.0, 2000.0)
    assert cmd.start_value == (0.0,)
    assert cmd.end_value == (1.0,)


def test_move_command_full_and_partial():
    spr = _one_sprite(SPRITE + " M,1,0,500,320,240,0,480\n"
                               " M,2,500,600,10,20,30\n"   # endY omitted
                               " M,3,600,700,1,2")          # both ends omitted
    m1, m2, m3 = spr.commands.commands
    assert m1.type is CommandType.MOVE
    assert m1.easing is Easing.OUT
    assert m1.start_value == (320.0, 240.0) and m1.end_value == (0.0, 480.0)
    assert m2.start_value == (10.0, 20.0) and m2.end_value == (30.0, 20.0)
    assert m3.start_value == (1.0, 2.0) and m3.end_value == (1.0, 2.0)


def test_move_x_y_commands():
    spr = _one_sprite(SPRITE + " MX,0,0,100,10,20\n MY,0,0,100,-5")
    mx, my = spr.commands.commands
    assert mx.type is CommandType.MOVE_X
    assert mx.start_value == (10.0,) and mx.end_value == (20.0,)
    assert my.type is CommandType.MOVE_Y
    assert my.start_value == (-5.0,) and my.end_value == (-5.0,)


def test_scale_and_vector_scale():
    spr = _one_sprite(SPRITE + " S,0,0,100,1,2\n V,0,0,100,1,1.5,2")
    s, v = spr.commands.commands
    assert s.type is CommandType.SCALE
    assert s.start_value == (1.0,) and s.end_value == (2.0,)
    assert v.type is CommandType.VECTOR_SCALE
    assert v.start_value == (1.0, 1.5) and v.end_value == (2.0, 1.5)


def test_rotate_keeps_radians():
    spr = _one_sprite(SPRITE + " R,0,0,100,0,3.14159")
    (r,) = spr.commands.commands
    assert r.type is CommandType.ROTATE
    assert math.isclose(r.end_value[0], 3.14159)  # RAW radians, no deg bake


def test_colour_full_and_partial():
    spr = _one_sprite(SPRITE + " C,0,0,100,255,128,0,0,64,255\n"
                               " C,0,100,200,10,20,30,40")  # only endRed given
    c1, c2 = spr.commands.commands
    assert c1.type is CommandType.COLOUR
    assert c1.start_value == (255.0, 128.0, 0.0)  # RAW 0..255
    assert c1.end_value == (0.0, 64.0, 255.0)
    assert c2.start_value == (10.0, 20.0, 30.0)
    assert c2.end_value == (40.0, 20.0, 30.0)


def test_parameter_commands():
    spr = _one_sprite(SPRITE + " P,0,0,100,H\n P,0,0,100,V\n P,0,0,0,A\n"
                               " P,0,0,100,Z")  # unknown kind → dropped
    kinds = [c.parameter for c in spr.commands.commands]
    assert kinds == [ParameterType.HORIZONTAL_FLIP,
                     ParameterType.VERTICAL_FLIP,
                     ParameterType.ADDITIVE_BLEND]
    assert all(c.type is CommandType.PARAMETER and c.start_value == ()
               for c in spr.commands.commands)


def test_blank_end_time_means_start_time():
    spr = _one_sprite(SPRITE + " F,0,1500,,1")
    (cmd,) = spr.commands.commands
    assert cmd.start_time == 1500.0 and cmd.end_time == 1500.0


def test_easing_parse():
    spr = _one_sprite(SPRITE + " F,35,0,100,1\n F,99,0,100,1")
    e35, e99 = (c.easing for c in spr.commands.commands)
    assert e35 is Easing.OUT_POW10
    assert e99 == 99 and not isinstance(e99, Easing)  # raw, like lazer's cast


# --- loops + triggers ----------------------------------------------------------

def test_loop_parse():
    spr = _one_sprite(SPRITE +
                      " L,1000,5\n"
                      "  F,0,0,300,0,1\n"
                      "  M,0,0,300,0,0,10,10\n"
                      " F,0,9000,9500,1,0")  # depth-1 → back to root group
    (loop,) = spr.loops
    assert loop.start_time == 1000.0
    assert loop.repeat_count == 4          # lazer: max(0, count-1)
    assert loop.total_iterations == 5
    assert [c.type for c in loop.commands] == [CommandType.FADE,
                                               CommandType.MOVE]
    assert [c.type for c in spr.commands.commands] == [CommandType.FADE]


def test_loop_count_clamps_to_one_iteration():
    spr = _one_sprite(SPRITE + " L,0,0\n  F,0,0,100,1")
    assert spr.loops[0].repeat_count == 0
    assert spr.loops[0].total_iterations == 1


def test_trigger_parse():
    spr = _one_sprite(SPRITE +
                      " T,HitSoundClap,4000,5000,2\n"
                      "  F,0,0,100,0,1\n"
                      " T,Passing\n"
                      "  C,0,0,100,255,0,0")
    t1, t2 = spr.triggers
    assert t1.trigger_name == "HitSoundClap"
    assert (t1.start_time, t1.end_time) == (4000.0, 5000.0)
    assert t1.group_number == -2  # stable NEGATES the group number
    assert [c.type for c in t1.commands] == [CommandType.FADE]
    assert t2.trigger_name == "Passing"
    assert t2.start_time == -_DOUBLE_MAX and t2.end_time == _DOUBLE_MAX
    assert t2.group_number == 0
    assert [c.type for c in t2.commands] == [CommandType.COLOUR]


def test_underscore_indentation():
    spr = _one_sprite(SPRITE + "_L,0,2\n__F,0,0,100,1")
    assert len(spr.loops) == 1
    assert len(spr.loops[0].commands) == 1


def test_nested_group_reset_bug_compat():
    # a nested L at depth 2 still hoists to the SPRITE (lazer: AddLoopingGroup
    # is called on storyboardSprite regardless of depth)
    spr = _one_sprite(SPRITE + " L,0,2\n  L,100,3\n   F,0,0,100,1")
    assert len(spr.loops) == 2
    assert len(spr.loops[0].commands) == 0
    assert len(spr.loops[1].commands) == 1


# --- variables ------------------------------------------------------------------

def test_variable_expansion():
    text = ("osu file format v14\n"
            "[Variables]\n"
            "$bg=\"sb/back.png\"\n"
            "$pos=320,240\n"
            "[Events]\n"
            "Sprite,Background,Centre,$bg,$pos\n")
    sb = parse_storyboard_text(text)
    (spr,) = sb.layers["Background"].elements
    assert spr.path == "sb/back.png"
    assert (spr.x, spr.y) == (320.0, 240.0)
    assert sb.variables["$bg"] == '"sb/back.png"'


def test_variable_chained_and_unresolvable():
    text = ("[Variables]\n"
            "$a=$b\n"
            "$b=128\n"
            "[Events]\n"
            "Sprite,Background,Centre,x.png,$a,$c\n")  # $c undefined
    sb = parse_storyboard_text(None, text)
    # $a → $b → 128 resolves; $c never resolves → float('$c') fails → skipped
    assert not list(sb.elements)
    text_ok = text.replace(",$c", ",0")
    sb = parse_storyboard_text(None, text_ok)
    (spr,) = sb.layers["Background"].elements
    assert spr.x == 128.0


def test_variable_seed_param():
    sb = parse_storyboard_text(
        "[Events]\nSprite,Background,Centre,$img,0,0\n",
        variables={"$img": "seeded.png"})
    (spr,) = sb.layers["Background"].elements
    assert spr.path == "seeded.png"


# --- merge order -----------------------------------------------------------------

OSU_DOC = ('osu file format v14\n'
           '[General]\n'
           'WidescreenStoryboard: 1\n'
           '[Events]\n'
           'Sprite,Foreground,Centre,"from_osu.png",0,0\n'
           ' F,0,0,100,1\n')
OSB_DOC = ('[Events]\n'
           'Sprite,Foreground,Centre,"from_osb.png",1,1\n'
           ' F,0,0,100,1\n')


def test_merge_osu_then_osb():
    sb = parse_storyboard_text(OSU_DOC, OSB_DOC)
    els = sb.layers["Foreground"].elements
    # primary .osu stream parses FIRST, .osb appends after
    # (Decoder.Decode: otherStreams.Prepend(primaryStream))
    assert [e.path for e in els] == ["from_osu.png", "from_osb.png"]
    assert els[0].source is ElementSource.BEATMAP
    assert els[1].source is ElementSource.OSB
    assert sb.widescreen


def test_osu_only_and_osb_only():
    assert [e.path for e in parse_storyboard_text(OSU_DOC).elements] \
        == ["from_osu.png"]
    only_osb = parse_storyboard_text(None, OSB_DOC)
    (el,) = only_osb.elements
    assert el.path == "from_osb.png" and el.source is ElementSource.OSB


def test_parse_storyboard_files_and_autodiscovery():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "map.osu").write_text(OSU_DOC, encoding="utf-8")
        (folder / "set.osb").write_text(OSB_DOC, encoding="utf-8")
        sb = parse_storyboard(folder / "map.osu")  # .osb auto-discovered
        assert [e.path for e in sb.layers["Foreground"].elements] \
            == ["from_osu.png", "from_osb.png"]
        # explicit path + osb-only entry points
        sb2 = parse_storyboard(None, folder / "set.osb")
        assert [e.path for e in sb2.elements] == ["from_osb.png"]


# --- animation / video / sample / background ---------------------------------------

def test_animation_parse():
    sb = _sb('Animation,Foreground,Centre,"sb/fx.png",100,200,10,50,LoopOnce\n'
             'Animation,Foreground,Centre,"b.png",0,0,4,20\n'      # default loop
             'Animation,Foreground,Centre,"c.png",0,0,4,20,1\n'    # numeric
             'Animation,Foreground,Centre,"d.png",0,0,4,20,9\n')   # undefined num
    a1, a2, a3, a4 = sb.layers["Foreground"].elements
    assert isinstance(a1, StoryboardAnimation)
    assert (a1.frame_count, a1.frame_delay) == (10, 50.0)
    assert a1.loop_type is LoopType.LOOP_ONCE
    assert a1.frame_path(0) == "sb/fx0.png"
    assert a1.frame_path(9) == "sb/fx9.png"
    assert a2.loop_type is LoopType.LOOP_FOREVER
    assert a3.loop_type is LoopType.LOOP_ONCE
    assert a4.loop_type is LoopType.LOOP_FOREVER


def test_animation_old_format_frame_delay():
    text = ('osu file format v5\n[Events]\n'
            'Animation,Foreground,Centre,"a.png",0,0,4,100\n')
    (anim,) = parse_storyboard_text(text).elements
    # LegacyStoryboardDecoder.cs:176-178 ("random as hell", from osu-stable)
    assert math.isclose(anim.frame_delay, round(0.015 * 100) * 1.186 * (1000 / 60))


def test_video_event():
    sb = _sb('Video,5000,"movie.mp4"\n F,0,0,100,0,1\n'
             'Video,0,"picture.jpg"\n')  # non-video ext: ignored (lazer parity)
    els = sb.layers["Video"].elements
    (vid,) = els
    assert isinstance(vid, StoryboardVideo)
    assert vid.start_time == 5000.0 and vid.path == "movie.mp4"
    assert len(vid.commands.commands) == 1  # commands attach to the video


def test_sample_event():
    sb = _sb('Sample,12000,Background,"sb/boom.wav",80\n'
             'Sample,500,3,"quiet.ogg"\n')
    (s1,) = sb.layers["Background"].elements
    assert isinstance(s1, StoryboardSample)
    assert (s1.time, s1.path, s1.volume) == (12000.0, "sb/boom.wav", 80.0)
    (s2,) = sb.layers["Foreground"].elements
    assert s2.volume == 100.0  # default


def test_background_offset():
    sb = _sb('0,0,"bg.jpg",42,-7\n')
    assert sb.background_offset == (42.0, -7.0)
    sb2 = _sb('0,0,"bg.jpg"\n')  # no offset fields
    assert sb2.background_offset is None
    assert not list(sb2.elements)  # backgrounds are not storyboard elements


def test_general_flags():
    text = ("osu file format v14\n[General]\n"
            "UseSkinSprites: 1\nWidescreenStoryboard: 0\n[Events]\n")
    sb = parse_storyboard_text(text)
    assert sb.use_skin_sprites and not sb.widescreen
    assert parse_storyboard_text(
        "[General]\nWidescreenStoryboard: 1\n").widescreen


# --- fail-soft -----------------------------------------------------------------------

def test_malformed_lines_fail_soft():
    sb = _sb(
        "Sproite,Background,Centre,a.png,0,0\n"   # unknown event type
        + SPRITE +
        " Q,0,0,100,1\n"                          # unknown command type
        " F,0,abc,100,1\n"                        # bad number
        " F,0,0,100,NaN\n"                        # NaN rejected (Parsing.cs)
        " F,0,0,100\n"                            # missing value field
        " M,0,0,100,1\n"                          # missing y
        " L,500\n"                                # loop missing count
        " T\n"                                    # trigger missing name
        " F,0,0,100,0,1\n")                       # valid — still lands
    (spr,) = sb.layers["Background"].elements
    assert [c.type for c in spr.commands.commands] == [CommandType.FADE]
    assert spr.commands.commands[0].end_value == (1.0,)
    assert not spr.loops and not spr.triggers


def test_orphan_commands_dropped():
    # commands with no current sprite are silently dropped (lazer `?.`)
    sb = _sb(" F,0,0,100,1\n L,0,5\n  F,0,0,100,1\n")
    assert not list(sb.elements)


def test_sprite_coordinate_limit():
    # Sprite x/y use MAX_COORDINATE_VALUE (131072) — beyond it, line skipped
    assert not list(_sb('Sprite,Background,Centre,"a.png",999999,0').elements)
    assert list(_sb('Sprite,Background,Centre,"a.png",131072,0').elements)


def test_group_targeting_survives_failed_sprite_line():
    # bug-compat: a failed Sprite line leaves currentCommandsGroup pointing at
    # the previous sprite's loop; depth-2 commands keep landing there
    spr_events = (SPRITE +
                  " L,0,2\n"
                  "Sprite,Nope,Centre,a.png,0,0\n"   # fails: unknown layer
                  "  F,0,0,100,1\n")                 # depth 2 → previous loop
    (spr,) = _sb(spr_events).layers["Background"].elements
    assert len(spr.loops[0].commands) == 1


def test_comments_blanks_and_unknown_sections():
    text = ("osu file format v14\n"
            "[Events]\n"
            "// full-line comment\n"
            "\n"
            'Sprite,Background,Centre,"a.png",0,0 // trailing comment\n'
            " F,0,0,100,1\n"
            "[SomethingNew]\n"
            "Ignored: yes\n"
            "[Events]\n"
            'Sprite,Background,Centre,"b.png",0,0\n')
    sb = parse_storyboard_text(text)
    els = sb.layers["Background"].elements
    assert [e.path for e in els] == ["a.png", "b.png"]
    assert len(els[0].commands.commands) == 1


def test_counts_census():
    sb = parse_storyboard_text(OSU_DOC, OSB_DOC)
    n = sb.counts()
    assert n["sprites"] == 2 and n["commands"] == 2
    assert n["animations"] == n["videos"] == n["samples"] == 0


# --- the everything-fixture (hand-written .osb with every construct) ---------------

FIXTURE_OSB = """[Variables]
$c=255,255,255
$fg=Foreground

[Events]
//Storyboard Layer 0 (Background)
Sprite,Background,TopLeft,"sb\\all.png",0,0
 F,0,0,1000,0,1
 M,9,0,1000,0,0,640,480
 MX,18,1000,2000,640,0
 MY,27,2000,3000,480,0
 S,35,3000,4000,1,2
 V,0,4000,5000,1,1,2,0.5
 R,0,5000,6000,0,6.283
 C,0,6000,7000,$c,0,0,0
 P,0,7000,7000,A
 P,0,7000,8000,H
 P,0,7000,8000,V
 L,8000,3
  F,0,0,250,0,1
  F,0,250,500,1,0
 T,HitSoundWhistle,0,10000,1
  S,0,0,100,1,1.2
Animation,Pass,BottomRight,"sb\\anim.png",320,240,8,41.7,LoopForever
 F,0,0,500,1
Sprite,Fail,CentreRight,"fail.png",1,2
 F,0,0,500,1
Sprite,Overlay,BottomLeft,"top.png",3,4
 F,0,0,500,1
Sprite,$fg,TopCentre,"mid.png",5,6
 F,0,0,500,1
Sample,1234,Foreground,"hit.wav",65
Video,-500,"intro.avi"
"""


def test_fixture_every_construct():
    sb = parse_storyboard_text(None, FIXTURE_OSB)
    n = sb.counts()
    assert n["sprites"] == 4 and n["animations"] == 1
    assert n["videos"] == 1 and n["samples"] == 1
    assert n["loops"] == 1 and n["triggers"] == 1
    # 11 root + 2 loop + 1 trigger on the big sprite, +1 on each of the others
    assert n["commands"] == 14 + 4
    (big,) = sb.layers["Background"].elements
    types = [c.type for c in big.commands.commands]
    assert types == [CommandType.FADE, CommandType.MOVE, CommandType.MOVE_X,
                     CommandType.MOVE_Y, CommandType.SCALE,
                     CommandType.VECTOR_SCALE, CommandType.ROTATE,
                     CommandType.COLOUR, CommandType.PARAMETER,
                     CommandType.PARAMETER, CommandType.PARAMETER]
    assert big.commands.commands[7].start_value == (255.0, 255.0, 255.0)
    assert big.has_commands
    assert big.command_count == 14
    # every element came from the OSB stream
    assert all(e.source is ElementSource.OSB for e in sb.elements)
    # variable-expanded layer routed correctly
    assert [e.path for e in sb.layers["Foreground"].elements
            if isinstance(e, StoryboardSprite)] == ["mid.png"]
