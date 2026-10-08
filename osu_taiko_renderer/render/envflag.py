"""Parse an R3D_* switch the way a reader expects.

`bool(os.environ.get(X))` is True for the STRING "0", so `R3D_FOO=0` turns the
feature ON. That has cost an invalid A/B before (both arms instrumented). Every
switch added with the speed work goes through this instead."""
from __future__ import annotations

import os
import sys

_OFF = ("", "0", "false", "no", "off")


def envflag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in _OFF



# ---- which render-path speedups are on with no switch set --------------------
# The same rule as std's render/perf.py: each speedup stays a switch (R3D_X=0
# off, R3D_X=1 on, on any platform), a platform where the whole set has been
# built, timed and checked frame by frame gets it by default, and
# R3D_TAIKO_STOCK=1 turns the whole set off at once: the stock render path, for
# regression gates ("does stock still equal main?") and for bisecting.
STOCK = envflag("R3D_TAIKO_STOCK")
# macOS is where every one of them was built, timed and compared frame by frame
# against the stock path. Anywhere else nothing is on until that platform has
# been checked and this rule is widened.
FAST_DEFAULT = sys.platform == "darwin" and not STOCK


def _fast(name: str) -> bool:
    return envflag(name, FAST_DEFAULT) and not STOCK


# The switches that depend on each other are resolved ONCE, here, and every
# module imports the result, so two modules cannot disagree. Each module used to
# read its own copy, and one of them (the flashlight) read R3D_TAIKO_GPU_FL
# without the "needs GPU_FX" half: GPU_FL=1 alone turned the CPU spotlight off
# and never drew the GPU one.
GPU_FX = _fast("R3D_TAIKO_GPU_FX")                     # effects in the GL pass
# Not switches of their own any more, because the half-set was a wrong picture:
# effects without the skin judgements dropped the legacy-lane hit flash, and the
# flashlight recomputed in GLSL instead of sampled was a level off in places.
GPU_SJ = GPU_FX                                        # + skin judgements
GPU_FL = _fast("R3D_TAIKO_GPU_FL") and GPU_FX          # flashlight in the pass
FL_EXACT = GPU_FL                                      # ... sampled, not recomputed
GPU_NUM = _fast("R3D_TAIKO_GPU_NUM") and GPU_FX        # HUD numbers in the pass
GPU_HUD = _fast("R3D_TAIKO_GPU_HUD") and GPU_FX        # the whole HUD in the pass
GPU_BREAK = _fast("R3D_TAIKO_GPU_BREAK") and GPU_HUD   # break overlay's shadow
MERGE_RUNS = _fast("R3D_TAIKO_MERGE_RUNS") and GPU_FX  # fewer blend runs per frame
# round to nearest when a blend is stored to 8 bits, as GL (and catch, and std)
# do, instead of truncating; with it the CPU and GPU composites agree
ROUND = _fast("R3D_TAIKO_ROUND")
MAP_READBACK = _fast("R3D_MAP_READBACK") and sys.platform == "darwin"
SOCKET_PIPE = _fast("R3D_MAC_SOCKET_PIPE") and sys.platform == "darwin"
RESULTS_AHEAD = _fast("R3D_TAIKO_RESULTS_AHEAD")
# Opt-in only, like INSTANCED below: measured 5-17% SLOWER than the set above on
# replays with breaks, because a frame inside a break still needs the CPU and
# takes the slow road (4-7% faster on replays without). Asked for, gl.py still
# checks the local ffmpeg.
GPU_YUV = envflag("R3D_TAIKO_GPU_YUV") and not STOCK
# not in the default set: one draw call per blend run. It was slower than the
# per-sprite path on taiko's short runs, and an additive sprite's fractional
# alpha lands one level apart as a vertex attribute (4 frames of the fixture).
INSTANCED = envflag("R3D_TAIKO_INSTANCED") and not STOCK

# Not a render-path speedup, and on by default nowhere: it changes the SOUND
# (render.py, "loudness by one fixed gain"). The engine's own switch wins over
# the node-wide R3D_FIXED_GAIN that every engine reads.
FIXED_GAIN = (envflag("R3D_TAIKO_FIXED_GAIN", envflag("R3D_FIXED_GAIN"))
              and not STOCK)

# Is anything other than the stock render path in use? __main__ runs a render
# that failed with any of these on again on the stock path.
ANY_FAST = any((GPU_FX, GPU_FL, GPU_NUM, GPU_HUD, GPU_BREAK, MERGE_RUNS, ROUND,
                MAP_READBACK, SOCKET_PIPE, RESULTS_AHEAD, GPU_YUV, INSTANCED))
