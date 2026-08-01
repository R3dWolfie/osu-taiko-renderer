"""Storyboard COMMAND ENGINE (phase 2): the time -> state evaluator.

Covers the ~36 easings (endpoints + analytic + pinned transcendental values),
per-property timeline semantics (initial clamp / interpolate / end clamp /
overlap override), M->X/Y split, S*V scale composition, R radians->degrees,
C /255 + LINEAR-space colour interpolation, the alpha flicker wrap, P
flip/additive (span vs instant-permanent), loop unrolling (period + iterations
+ lifetime), the alpha-0 lifetime optimisation, the endTime<startTime clamp,
the trigger-firing stub, and whole-storyboard draw order + alive filtering.

Every numeric expectation traces to osu!(lazer)/osu-framework master (cited in
osu_taiko_renderer/render/storyboard_engine.py).
"""
import math

from osu_taiko_renderer.beatmap.storyboard import (
    CommandType, Easing, ParameterType, parse_storyboard_text)
from osu_taiko_renderer.render.storyboard_engine import (
    apply_easing, build_sprite_timeline, srgb_to_linear, linear_to_srgb,
    SpriteState, StoryboardEngine)


# --------------------------------------------------------------------------- #
# helpers                                                                       #
# --------------------------------------------------------------------------- #

def _sprite(events: str, x=0, y=0, layer="Foreground"):
    """Parse one sprite (declared at x,y) carrying the given command lines."""
    head = f'Sprite,{layer},Centre,"a.png",{x},{y}\n'
    doc = "osu file format v14\n[Events]\n" + head + events
    sb = parse_storyboard_text(doc)
    (el,) = [e for e in sb.elements if hasattr(e, "commands")]
    return el


def _tl(events: str, x=0, y=0, trigger_fires=None):
    return build_sprite_timeline(_sprite(events, x, y), trigger_fires)


def _close(a, b, tol=1e-6):
    return abs(a - b) <= tol


# --------------------------------------------------------------------------- #
# easings                                                                       #
# --------------------------------------------------------------------------- #

def test_easing_endpoints_all_36():
    # every easing is normalised to hit exactly 0 at t=0 and 1 at t=1
    for i in range(36):
        assert _close(apply_easing(i, 0.0), 0.0, 1e-9), (i, apply_easing(i, 0.0))
        assert _close(apply_easing(i, 1.0), 1.0, 1e-9), (i, apply_easing(i, 1.0))


def test_easing_none_is_linear_and_enum_accepted():
    assert apply_easing(Easing.NONE, 0.37) == 0.37
    assert apply_easing(0, 0.37) == 0.37
    # out-of-range easing index -> linear (C# blind cast / switch default)
    assert apply_easing(99, 0.42) == 0.42


def test_easing_analytic_values():
    assert _close(apply_easing(Easing.IN_QUAD, 0.5), 0.25)
    assert _close(apply_easing(Easing.OUT_QUAD, 0.5), 0.75)
    assert _close(apply_easing(Easing.IN_CUBIC, 0.5), 0.125)
    assert _close(apply_easing(Easing.OUT_CUBIC, 0.5), 1 + (-0.5) ** 3)
    assert _close(apply_easing(Easing.IN_QUART, 0.5), 0.0625)
    assert _close(apply_easing(Easing.IN_QUINT, 0.5), 0.03125)
    assert _close(apply_easing(Easing.IN_SINE, 0.5), 1 - math.cos(math.pi * 0.25))
    assert _close(apply_easing(Easing.OUT_SINE, 0.5), math.sin(math.pi * 0.25))
    assert _close(apply_easing(Easing.IN_OUT_SINE, 0.5), 0.5)
    assert _close(apply_easing(Easing.IN_CIRC, 0.5), 1 - math.sqrt(0.75))
    # OutPow10: --time * time^10 + 1  ->  (-0.5)*(-0.5)^10 + 1
    assert _close(apply_easing(Easing.OUT_POW10, 0.5), (-0.5) * ((-0.5) ** 10) + 1)
    # InOutBounce is symmetric about (0.5, 0.5)
    assert _close(apply_easing(Easing.IN_OUT_BOUNCE, 0.5), 0.5)


def test_easing_elastic_back_overshoot():
    # elastic/back deliberately leave [0,1] mid-curve (framework does not clamp)
    assert max(apply_easing(Easing.OUT_ELASTIC, i / 100) for i in range(101)) > 1.0
    assert min(apply_easing(Easing.IN_BACK, i / 100) for i in range(101)) < 0.0


def test_srgb_linear_roundtrip():
    for c in (0.0, 0.02, 0.25, 0.5, 0.75, 1.0):
        assert _close(linear_to_srgb(srgb_to_linear(c)), c, 1e-9)
    assert srgb_to_linear(1.0) == 1.0 and linear_to_srgb(1.0) == 1.0


# --------------------------------------------------------------------------- #
# scalar timeline semantics                                                     #
# --------------------------------------------------------------------------- #

def test_alpha_before_between_after():
    tl = _tl(" F,0,1000,2000,0.3,0.7\n")
    # before first command -> clamp to its START value (ApplyInitialValue)
    assert _close(tl.state_at(500).alpha, 0.3)
    # midpoint linear
    assert _close(tl.state_at(1500).alpha, 0.5)
    # after last -> clamp to END value
    assert _close(tl.state_at(2500).alpha, 0.7)
    # endpoints
    assert _close(tl.state_at(1000).alpha, 0.3)
    assert _close(tl.state_at(2000).alpha, 0.7)


def test_overlapping_alpha_later_command_overrides():
    # cmd B starts later and fully takes over the Alpha member from its start,
    # even while cmd A's [0,1000] window is still "active" (framework cut-off).
    tl = _tl(" F,0,0,1000,0,1\n F,0,500,700,0.2,0.8\n")
    assert _close(tl.state_at(300).alpha, 0.3)   # governed by A
    assert _close(tl.state_at(600).alpha, 0.5)   # governed by B (0.2 + .5*.6)
    assert _close(tl.state_at(800).alpha, 0.8)   # B ended -> clamp B end value
    assert _close(tl.state_at(5000).alpha, 0.8)


def test_alpha_flicker_wrap_above_one():
    tl = _tl(" F,0,0,1000,0,2\n")
    assert _close(tl.state_at(500).alpha, 1.0)   # exactly 1 -> untouched
    assert _close(tl.state_at(750).alpha, 0.5)   # 1.5 % 1
    assert _close(tl.state_at(1000).alpha, 0.0)  # 2.0 % 1


# --------------------------------------------------------------------------- #
# position: M split + MX/MY + declared-position default                         #
# --------------------------------------------------------------------------- #

def test_move_splits_into_x_and_y():
    tl = _tl(" M,0,0,1000,100,200,300,400\n")
    st = tl.state_at(500)
    assert _close(st.x, 200) and _close(st.y, 300)
    assert _close(tl.state_at(1000).x, 300) and _close(tl.state_at(1000).y, 400)


def test_mx_my_only_affect_their_axis():
    # MX drives X; Y has no timeline -> stays at the declared sprite position
    tl = _tl(" MX,0,0,1000,10,20\n", x=111, y=222)
    st = tl.state_at(500)
    assert _close(st.x, 15) and _close(st.y, 222)


def test_default_position_when_no_move():
    tl = _tl(" F,0,0,1000,1,1\n", x=320, y=240)
    st = tl.state_at(500)
    assert _close(st.x, 320) and _close(st.y, 240)


# --------------------------------------------------------------------------- #
# scale (S uniform) * vector scale (V per-axis)                                 #
# --------------------------------------------------------------------------- #

def test_scale_uniform_and_vector_compose():
    tl = _tl(" S,0,0,1000,2,2\n V,0,0,1000,1,3,1,3\n")
    st = tl.state_at(500)
    assert _close(st.scale_x, 2 * 1) and _close(st.scale_y, 2 * 3)


def test_scale_default_is_one():
    st = _tl(" F,0,0,1000,1,1\n").state_at(500)
    assert _close(st.scale_x, 1.0) and _close(st.scale_y, 1.0)


def test_vector_scale_interpolates_per_axis():
    tl = _tl(" V,0,0,1000,1,1,3,5\n")
    st = tl.state_at(500)
    assert _close(st.scale_x, 2.0) and _close(st.scale_y, 3.0)


# --------------------------------------------------------------------------- #
# rotation radians -> degrees                                                   #
# --------------------------------------------------------------------------- #

def test_rotation_radians_to_degrees():
    tl = _tl(" R,0,0,1000,0,3.141592653589793\n")
    assert _close(tl.state_at(0).rotation, 0.0)
    assert _close(tl.state_at(500).rotation, 90.0, 1e-4)   # half of 180deg
    assert _close(tl.state_at(1000).rotation, 180.0, 1e-4)


# --------------------------------------------------------------------------- #
# colour: /255 + LINEAR-space interpolation                                     #
# --------------------------------------------------------------------------- #

def test_colour_normalised_and_endpoints():
    tl = _tl(" C,0,0,1000,255,128,0,0,64,255\n")
    st0 = tl.state_at(0)
    assert _close(st0.r, 1.0) and _close(st0.g, 128 / 255) and _close(st0.b, 0.0)
    st1 = tl.state_at(1000)
    assert _close(st1.r, 0.0) and _close(st1.g, 64 / 255) and _close(st1.b, 1.0)


def test_colour_interpolates_in_linear_space():
    # (255,0,0) -> (0,0,255), linear easing, half-way.
    tl = _tl(" C,0,0,1000,255,0,0,0,0,255\n")
    st = tl.state_at(500)
    # expected = round-trip through linear space, NOT a naive sRGB 0.5 lerp
    exp = linear_to_srgb(srgb_to_linear(1.0) + 0.5 * (srgb_to_linear(0.0) - srgb_to_linear(1.0)))
    assert _close(st.r, exp) and _close(st.b, exp)
    assert st.r > 0.6   # linear-space midpoint is ~0.735, well above naive 0.5
    assert _close(st.g, 0.0)


def test_colour_default_white():
    st = _tl(" F,0,0,1000,1,1\n").state_at(500)
    assert _close(st.r, 1.0) and _close(st.g, 1.0) and _close(st.b, 1.0)


# --------------------------------------------------------------------------- #
# parameters: flip H/V + additive (span vs instant-permanent)                   #
# --------------------------------------------------------------------------- #

def test_flip_span_is_on_only_within_window():
    tl = _tl(" F,0,0,4000,1,1\n P,0,1000,2000,H\n")
    assert tl.state_at(500).flip_h is False    # before
    assert tl.state_at(1500).flip_h is True    # within [1000,2000)
    assert tl.state_at(2500).flip_h is False   # after -> end value off


def test_flip_instant_is_permanent():
    # an instant P (blank end -> end==start) applies regardless of time
    tl = _tl(" F,0,0,4000,1,1\n P,0,3000,,V\n")
    assert tl.state_at(1000).flip_v is True    # permanent, even before 3000
    assert tl.state_at(3500).flip_v is True


def test_additive_span_and_default():
    tl = _tl(" F,0,0,4000,1,1\n P,0,1000,2000,A\n")
    assert tl.state_at(500).additive is False
    assert tl.state_at(1500).additive is True
    assert tl.state_at(2500).additive is False   # -> Inherit (off)


# --------------------------------------------------------------------------- #
# endTime < startTime clamp                                                     #
# --------------------------------------------------------------------------- #

def test_end_before_start_is_clamped_to_instant():
    # R with end(400) < start(800) -> lazer clamps end to start (instant @800)
    tl = _tl(" F,0,0,2000,1,1\n R,0,800,400,0,3.141592653589793\n")
    assert _close(tl.state_at(700).rotation, 0.0)     # before -> start value
    assert _close(tl.state_at(900).rotation, 180.0, 1e-4)  # at/after -> end value


# --------------------------------------------------------------------------- #
# loops                                                                         #
# --------------------------------------------------------------------------- #

def test_loop_unrolls_with_period_and_iterations():
    tl = _tl(" L,1000,3\n"
             "  F,0,0,500,0,1\n"
             "  F,0,500,1000,1,0\n")
    # period = maxChildEnd(1000) - minChildStart(0) = 1000; 3 iterations
    assert _close(tl.end_time, 4000.0)                 # EndTimeForDisplay
    assert _close(tl.state_at(1250).alpha, 0.5)        # iter0 fade-in half
    assert _close(tl.state_at(2250).alpha, 0.5)        # iter1 fade-in half
    assert _close(tl.state_at(3250).alpha, 0.5)        # iter2 fade-in half
    assert _close(tl.state_at(1750).alpha, 0.5)        # iter0 fade-out half
    assert not tl.is_alive(4000.0)                     # half-open end


def test_loop_period_uses_group_duration_not_bare_max_end():
    # children start at relative 200 -> period = 400-200 = 200 (NOT 400).
    tl = _tl(" L,1000,2\n  F,0,200,400,0,1\n")
    assert _close(tl.end_time, 1600.0)                 # 1200 + 200*2
    assert _close(tl.state_at(1300).alpha, 0.5)        # iter0 [1200,1400]
    assert _close(tl.state_at(1500).alpha, 0.5)        # iter1 [1400,1600]
    assert not tl.is_alive(1600.0)


def test_loop_folds_into_shared_property_timeline():
    # a root command and a loop command on the same property coexist / order
    tl = _tl(" F,0,0,1000,1,1\n"
             " L,2000,2\n  S,0,0,500,1,2\n")
    assert _close(tl.state_at(500).scale_x, 1.0)       # default (no scale yet)
    assert _close(tl.state_at(2250).scale_x, 1.5)      # loop iter0 [2000,2500]


# --------------------------------------------------------------------------- #
# lifetime: alpha-0 optimisation                                                #
# --------------------------------------------------------------------------- #

def test_lifetime_alpha_zero_optimisation():
    # sprite is invisible 0..5000, then fades in -> StartTime jumps to 5000
    tl = _tl(" F,0,0,5000,0,0\n F,0,5000,6000,0,1\n")
    assert _close(tl.earliest_transform_time, 0.0)
    assert _close(tl.start_time, 5000.0)
    assert _close(tl.end_time, 6000.0)
    assert not tl.is_alive(1000.0)
    assert tl.is_alive(5500.0)


def test_lifetime_without_alpha_uses_earliest_transform():
    tl = _tl(" S,0,1000,2000,1,2\n")
    assert _close(tl.start_time, 1000.0)
    assert _close(tl.end_time, 2000.0)


# --------------------------------------------------------------------------- #
# triggers: firing is stubbed                                                   #
# --------------------------------------------------------------------------- #

def test_trigger_does_not_fire_by_default():
    spr = _sprite(" F,0,0,1000,1,1\n"
                  " T,HitSoundClap,0,10000,0\n"
                  "  S,0,0,500,1,5\n")
    tl = build_sprite_timeline(spr)                    # no trigger_fires
    # trigger child (scale 1->5) must NOT contribute; scale stays default 1
    assert _close(tl.state_at(2000).scale_x, 1.0)


def test_trigger_machinery_folds_when_fired_explicitly():
    spr = _sprite(" F,0,0,1000,1,1\n"
                  " T,HitSoundClap,0,10000,0\n"
                  "  S,0,0,500,1,5\n")
    tl = build_sprite_timeline(spr, trigger_fires={0: [2000.0]})
    # fired at 2000 -> child S,0..500 becomes [2000,2500], scale 1->5
    assert _close(tl.state_at(2250).scale_x, 3.0)      # half of 1->5
    assert _close(tl.state_at(2500).scale_x, 5.0)


# --------------------------------------------------------------------------- #
# whole-storyboard engine                                                       #
# --------------------------------------------------------------------------- #

def test_engine_draw_order_and_alive_filter():
    doc = ("osu file format v14\n[Events]\n"
           'Sprite,Overlay,Centre,"front.png",0,0\n'
           " F,0,0,1000,1,1\n"
           'Sprite,Background,Centre,"back.png",0,0\n'
           " F,0,0,1000,1,1\n"
           'Sprite,Foreground,Centre,"mid.png",0,0\n'
           " F,0,5000,6000,1,1\n")   # not alive at t=500
    eng = StoryboardEngine(parse_storyboard_text(doc))
    # back-to-front: Background (depth 3) before Overlay (depth -2^31)
    paths = [el.path for el, _ in eng.sprites]
    assert paths.index("back.png") < paths.index("front.png")
    # alive filter at t=500: mid.png (starts 5000) excluded
    alive = [el.path for el, _ in eng.state_at(500)]
    assert "back.png" in alive and "front.png" in alive and "mid.png" not in alive


def test_engine_state_at_returns_sprite_state():
    doc = ("osu file format v14\n[Events]\n"
           'Sprite,Foreground,Centre,"a.png",50,60\n'
           " F,0,0,1000,1,1\n M,0,0,1000,0,0,100,100\n")
    eng = StoryboardEngine(parse_storyboard_text(doc))
    (el, st), = eng.state_at(500)
    assert isinstance(st, SpriteState)
    assert _close(st.x, 50) and _close(st.y, 50) and _close(st.alpha, 1.0)


def test_engine_skips_commandless_and_trigger_only_sprites():
    doc = ("osu file format v14\n[Events]\n"
           'Sprite,Foreground,Centre,"nocmd.png",0,0\n'
           'Sprite,Foreground,Centre,"trig.png",0,0\n'
           " T,Passing\n  F,0,0,100,0,1\n")
    eng = StoryboardEngine(parse_storyboard_text(doc))
    # neither is drawable (no root/loop commands -> HasCommands False)
    assert eng.sprites == []
