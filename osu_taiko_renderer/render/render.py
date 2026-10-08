"""Phase 1 orchestrator: parse -> simulate -> per-frame GL draw + HUD -> ffmpeg.

Owns a small ffmpeg subprocess (raw rgb24 on stdin) so it stays decoupled
from osu_renderer's encode FIFO machinery. HUD text is composited on the CPU
with PIL after GL readback — cheap and avoids a GL text pass for Phase 1.
"""
from __future__ import annotations

import hashlib
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
import pathlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from osu_taiko_renderer.skin.assets import build_textures
from osu_taiko_renderer.beatmap.beatmap import parse_beatmap
from osu_taiko_renderer.render.gl import SpriteRenderer
from osu_taiko_renderer.render import preview_hw as _phw
from osu_taiko_renderer.render.gl import _GPU_YUV as _GPU_YUV_R
from osu_taiko_renderer.render.gl import rgb_to_yuv420p as _rgb_to_yuv420p
from osu_taiko_renderer.beatmap.models import RenderConfig
from osu_taiko_renderer.beatmap.replay import parse_replay
from osu_taiko_renderer.render.scene import TaikoSim
from osu_taiko_renderer.render.envflag import envflag
import osu_taiko_renderer.render.envflag as _sw

log = logging.getLogger(__name__)


class TaikoRenderError(RuntimeError):
    pass


class _FrameWriter:
    """ffmpeg stdin writer thread — ported from the std renderer's proven
    FfmpegPipe (osu_std_renderer/record/encode.py), minus the process
    ownership (this renderer already owns its ffmpeg Popen).

    Frames are handed to the thread over a small bounded queue: the
    serialisation (`tobytes` — a negative-stride flip copy) and the blocking
    pipe write happen OFF the render thread, overlapping the next frame's
    draw. Order is FIFO so the byte stream ffmpeg sees is unchanged. The
    queue bounds memory (~4 frames) and provides natural backpressure when
    ffmpeg is the bottleneck; writer errors surface on the next push()
    instead of deadlocking the producer.

    R3D_FRAME_MD5=1 hashes every raw frame writer-side (blake2b) and prints
    one digest at close — bit-identical output proof across perf changes
    (same env/mechanism as the std renderer)."""

    _QUEUE_FRAMES = 4

    def __init__(self, proc):
        self._stdin = proc.stdin
        self._q: "queue.Queue" = queue.Queue(maxsize=self._QUEUE_FRAMES)
        self._werr: BaseException | None = None
        self._hash = None
        self._hash_frames = 0
        if os.environ.get("R3D_FRAME_MD5"):
            self._hash = hashlib.blake2b(digest_size=16)
        self._thread = threading.Thread(target=self._writer,
                                        name="ffmpeg-writer", daemon=True)
        self._thread.start()

    def _writer(self) -> None:
        while True:
            frame = self._q.get()
            if frame is None:
                return
            if self._werr is not None:
                continue          # drain (never write after an error)
            try:
                # perf: a C-contiguous frame (outro frames, repeated frozen
                # frames) is written zero-copy via its buffer; only flipped
                # gameplay views need the tobytes() flip copy. Bytes on the
                # pipe are identical either way.
                if frame.flags.c_contiguous:
                    data = memoryview(frame).cast("B")
                else:
                    data = frame.tobytes()
                if self._hash is not None:
                    self._hash.update(data)
                    self._hash_frames += 1
                self._stdin.write(data)
            except BaseException as e:  # noqa: BLE001 — surfaced on push()
                self._werr = e

    def push(self, frame_rgb) -> None:
        """Queue one frame. Re-raises the writer thread's error, so a dead
        ffmpeg surfaces here just like the old synchronous write did
        (BrokenPipeError included)."""
        if self._werr is not None:
            raise self._werr
        self._q.put(frame_rgb)

    def close(self) -> None:
        self._q.put(None)
        self._thread.join()
        if self._hash is not None:
            print(f"frame-stream-hash: {self._hash.hexdigest()} "
                  f"({self._hash_frames} frames)", file=sys.stderr, flush=True)


def render_taiko(
    osr_path: Path,
    beatmap_dir: Path,
    output_path: Path,
    cfg: RenderConfig | None = None,
    *,
    progress_callback=None,
) -> Path:
    cfg = cfg or RenderConfig()
    _mark("entry")
    frames, meta = parse_replay(osr_path)
    osu_path = _find_osu(beatmap_dir, meta.beatmap_md5)
    bm = parse_beatmap(osu_path, mods=meta.mods)
    _mark("parse_replay+beatmap")
    if not bm.objects:
        raise TaikoRenderError(f"no hit objects parsed from {osu_path.name}")
    audio = bm.audio_filename and (beatmap_dir / bm.audio_filename)
    audio = audio if (audio and audio.is_file()) else None
    bg = bm.background and (beatmap_dir / bm.background)
    bg = bg if (bg and bg.is_file()) else None
    return render_core(bm, frames, meta, output_path, cfg, audio=audio, bg=bg,
                       progress_callback=progress_callback, osu_path=osu_path)


def _draw_lazer_results(cache, rgb, meta, bm, opacity, *, age_ms=None,
                        board=None, osu_path=None, sim=None):
    """Composite the ported osu!(lazer) results screen (CatchLazerResults from
    lazer_results.py) over `rgb`. Built ONCE on the first outro frame, then
    re-drawn each frame at `opacity`/`age_ms` (mirrors catch's hud.draw_results
    dispatcher). Fully fail-soft: any build/draw failure is caught, logged
    LOUDLY, and the plain frame is returned so a render never crashes on the
    results screen."""
    try:
        scr = cache.get("scr")
        if scr is False:                 # an earlier bake failed → plain frame
            return rgb
        if scr is None:
            # perf: render_core kicks a background thread that builds this
            # instance (and pre-bakes its animation assets) DURING gameplay.
            # Wait only for the build-published event — never the full
            # prebake — then fall back to the original inline build if the
            # thread failed (identical output either way).
            evt = cache.get("evt")
            if evt is not None:
                evt.wait()
                scr = cache.get("pre")
            if scr is None:
                from osu_taiko_renderer.hud.lazer_results import CatchLazerResults
                scr = CatchLazerResults((rgb.shape[1], rgb.shape[0]), meta, bm,
                                        board=board, osu_path=osu_path, sim=sim)
            cache["scr"] = scr
        return scr.render_frame(rgb, opacity, age_ms)
    except Exception as e:  # noqa: BLE001 — results must never kill a render
        import traceback
        print("[taiko-renderer] !!! LAZER RESULTS SCREEN FAILED — leaving the "
              f"plain final frame: {e}", file=sys.stderr)
        traceback.print_exc()
        cache["scr"] = False
        return rgb


def _build_storyboard(cfg, renderer, osu_path, w, h):
    """Construct the storyboard renderer when --storyboard is on, else None.

    Gated on cfg.load_storyboard (DEFAULT OFF) — while off this returns None
    and the frame loop takes its exact single-draw path, so live renders are
    byte-identical. Auto-discovers the map's .osb next to the .osu. Fully
    fail-soft: any parse/build problem logs LOUDLY and renders without the
    storyboard rather than crashing."""
    if not getattr(cfg, "load_storyboard", False) or osu_path is None:
        return None
    try:
        from osu_taiko_renderer.beatmap.storyboard import parse_storyboard
        from osu_taiko_renderer.render.storyboard_engine import StoryboardEngine
        from osu_taiko_renderer.render.storyboard_render import StoryboardRenderer
        sb_data = parse_storyboard(osu_path)
        engine = StoryboardEngine(sb_data)
        if not engine.sprites:
            print("[taiko-renderer] storyboard: no drawable sprites — "
                  "rendering without storyboard", file=sys.stderr, flush=True)
            return None
        sbr = StoryboardRenderer(renderer, engine, Path(osu_path).parent,
                                 w, h, widescreen=sb_data.widescreen)
        c = sb_data.counts()
        print(f"[taiko-renderer] storyboard: {len(engine.sprites)} drawable "
              f"sprites ({c['sprites']} sprite, {c['animations']} animation, "
              f"{c['videos']} video, {c['samples']} sample event(s) NOT played "
              f"— storyboard audio deferred), widescreen={sb_data.widescreen}",
              file=sys.stderr, flush=True)
        return sbr
    except Exception as e:  # noqa: BLE001 — a storyboard must never break a render
        import traceback
        print(f"[taiko-renderer] WARNING: storyboard load failed ({e!r}) — "
              "rendering without storyboard", file=sys.stderr)
        traceback.print_exc()
        return None


_PERF = os.environ.get("R3D_TAIKO_PERF")
_STAGE = os.environ.get("R3D_TAIKO_STAGE")
# R3D_TAIKO_OUTRO_CPROF=1: cProfile ONLY the results-screen composite.
# Wrapped around the call rather than the process because a top-level profile drowns
# it in 1758 gameplay frames, and the outro is only ~318 frames of a different shape.
_OUTRO_CPROF = envflag("R3D_TAIKO_OUTRO_CPROF")
if _OUTRO_CPROF:
    import atexit as _ocp_atx
    import cProfile as _ocp_mod
    import io as _ocp_io
    import pstats as _ocp_pstats

    _OCP = _ocp_mod.Profile()

    @_ocp_atx.register
    def _ocp_dump():
        import sys as _o
        buf = _ocp_io.StringIO()
        _ocp_pstats.Stats(_OCP, stream=buf).sort_stats("tottime").print_stats(30)
        print("[outro-cprof] results composite only:", file=_o.stderr)
        print(buf.getvalue(), file=_o.stderr, flush=True)


_dump_env = os.environ.get("R3D_TAIKO_DUMP")
_DUMP = None
if _dump_env:
    _parts = _dump_env.split(",")
    _DUMP = (_parts[0], {int(x): True for x in _parts[1:]})
_ST: dict = {}
# HARNESS: R3D_TAIKO_SBPROF=1 splits stage buckets that enclose more than their
# name suggests. THREE of this session's buckets lied: `hud`=0.000 (the work had
# moved into sprite_build), `gl_draw` (its window enclosed overlay_gl), and
# `sb_fx` (93% of it was sim.active_effects, not sprite building at all).
# A stage bucket measures a WINDOW, not a function -- split before optimising.
_SBPROF = envflag("R3D_TAIKO_SBPROF")
_SB = {"hud_gl": 0.0, "fl_params": 0.0, "n": 0}
if _SBPROF:
    import atexit as _sb_atx

    @_sb_atx.register
    def _sb_dump():
        import sys as _sb
        n = max(1, _SB["n"])
        print(f"[sb-prof] sprite_build split over {n} frames: "
              f"hud.overlay_gl={_SB['hud_gl']/n*1e3:.4f} ms  "
              f"flashlight_params={_SB['fl_params']/n*1e3:.4f} ms",
              file=_sb.stderr, flush=True)


def _acc(k, dt):
    _ST[k] = _ST.get(k, 0.0) + dt
_MARKS: list = []


def _mark(label: str) -> None:
    """HARNESS: R3D_TAIKO_PERF=1 stamps a wall mark. Printed as a phase table at
    the end, so the PROLOGUE and the encoder tail are visible -- taiko's printed
    `done: N frames in Xs` starts AFTER setup and ends AFTER proc.wait(), which
    is what let a contaminated run read as a 515s serial tail."""
    if _PERF:
        _MARKS.append((label, time.monotonic()))


def render_core(
    bm,
    frames,
    meta,
    output_path: Path,
    cfg: RenderConfig,
    *,
    audio: Path | None = None,
    bg: Path | None = None,
    progress_callback=None,
    osu_path: Path | None = None,
) -> Path:
    """Render from already-parsed beatmap/frames/meta. Shared by the osr path
    and tests."""
    from osu_taiko_renderer.skin.fonts import set_skin_font
    set_skin_font(cfg.skin_dir)

    # Phase 1: procedural taiko textures + simple HUD (no skin asset wiring yet).
    skin = None
    sim = TaikoSim(bm, frames, cfg, skin=skin, has_bg=bg is not None, meta=meta)
    _mark("sim_built")
    if cfg.show_pp_counter and osu_path is not None:
        sim.compute_pp_curve(osu_path, meta.mods)
    # --pp: pin the FINAL pp (results card + live-counter endpoint) to the EXACT
    # value passed by the dispatch layer (osu's OFFICIAL pp). The live curve
    # keeps its rosu/score-progress SHAPE — build_scene computes the live counter
    # as `_final_pp * (score / final_score)`, so overriding _final_pp scales the
    # whole curve by a constant and only moves the ENDPOINT (mirrors the score
    # endpoint-anchor, #91). Set unconditionally (independent of show_pp_counter)
    # so the results card shows it even when the live counter is off — EVERY pp
    # consumer reads sim._final_pp: the live counter (build_scene), the ported
    # results screen (CatchLazerResults via _compute_stars_pp), and the argon
    # results cell.
    if getattr(cfg, "pp_override", None) is not None:
        sim._final_pp = float(cfg.pp_override)
    # --sr: pin the results card's star-rating pill to the EXACT star rating
    # passed by the dispatch layer (osu's OFFICIAL SR). Static display value —
    # there is no live SR counter, so this only feeds the results screen. The
    # ported results screen (CatchLazerResults via _compute_stars_pp) prefers
    # sim._final_stars over the rosu SR. No-op when --sr is absent.
    if getattr(cfg, "sr_override", None) is not None:
        sim._final_stars = float(cfg.sr_override)

    # ── SCORE FIDELITY: one lazer-standardised scale everywhere (#155/#115) ──
    # The .osr header total means different things per source (stable ScoreV1,
    # osu-web legacy export of a lazer play = classic display total, lazer
    # standardised). score_fidelity converts the header under every
    # interpretation with lazer's own taiko math and picks the one consistent
    # with our client-agnostic sim (sim._std_sim_final). The sim's ScoreV3 curve
    # is then RE-PINNED so the in-video counter ENDS EXACTLY on that number, and
    # meta.score is swapped so the results screen + leaderboard card show the
    # same value. Result: a STABLE taiko play now shows ScoreV2 (~1M scale), not
    # the raw ScoreV1 total (millions). The authoritative total is exported via a
    # `<output>.mp4.score.json` sidecar (bot → renders.score_v3). Fail-soft: any
    # problem leaves the header score displayed as before.
    score_fid: dict | None = None
    if osu_path is not None and getattr(meta, "score", 0):
        try:
            from osu_taiko_renderer.beatmap.score_fidelity import (
                compute_candidates, resolve_authoritative)
            from osu_taiko_renderer.render.scene import mods_score_multiplier
            _mod_mult = mods_score_multiplier(int(getattr(meta, "mods", 0) or 0))
            fid = compute_candidates(meta, bm, osu_path, _mod_mult)
            _sim_final = int(getattr(sim, "_std_sim_final", 0) or 0)
            val, src = resolve_authoritative(fid, _sim_final)
            if val > 0:
                sim.repin_score(int(val))
                import dataclasses as _dc
                meta = _dc.replace(meta, score=int(val))
            fid.pop("legacy_attrs", None)
            fid.pop("osu_facts", None)
            fid.update({"score_v3": int(val), "source": src,
                        "sim_final": _sim_final,
                        "player": getattr(meta, "player_name", "")})
            fid["players"] = [dict(fid)]
            score_fid = fid
            print(f"[taiko-renderer] score fidelity: header="
                  f"{fid['header_score']:,} -> standardised {int(val):,} "
                  f"(source={src}, sim_final={_sim_final:,})",
                  file=sys.stderr, flush=True)
        except Exception as _sf_e:  # noqa: BLE001 — never break a render
            # Last-resort safety net: the sim guards degenerate inputs internally
            # (non-finite times/values), so a throw here is genuinely unexpected.
            # Fall back to the .osr header quietly — a single calm warning, not a
            # scary "FAILED" (the render itself is fine).
            print(f"[taiko-renderer] score fidelity unavailable, keeping .osr "
                  f"header score: {type(_sf_e).__name__}: {_sf_e}",
                  file=sys.stderr, flush=True)
            score_fid = None

    # preempt = the first object's actual on-screen travel time (scroll_time is
    # the SV=1 base; real notes scroll faster, so the visible time is
    # scroll_time / scroll_vel). Using the raw scroll_time left ~5s of empty
    # playfield before the first note entered.
    first = bm.objects[0].time_ms
    # End of the map = the latest object END (a trailing swell/drumroll runs
    # well past its start), not just the last object's start — otherwise the
    # gameplay tail (and the swell's clear animation) gets clipped. Matches
    # the same computation TaikoSim uses for its life-bar/fail gating.
    last = max((o.end_ms or o.time_ms) for o in bm.objects)
    first_sv = max(getattr(bm.objects[0], "scroll_vel", 1.0) or 1.0, 0.1)
    preempt = sim.geo.scroll_time / first_sv
    # skip_intro: start at the first object's approach; else render the full
    # intro from the song start.
    lead = min(cfg.lead_in_ms, 800)        # brief lead-in, not a long empty gap
    # Cap the first note's visible approach when skipping the intro: a low-SV /
    # low-BPM first note has a huge travel time (preempt), which otherwise made
    # the render open on several seconds of an almost-empty playfield while one
    # note slowly crawled in. Start closer so the intro is always tight (~2s).
    approach = min(preempt, 2000.0)
    if cfg.skip_intro:
        start_ms = int(first - approach - lead)
    else:
        start_ms = min(0, int(first - preempt - lead))
    # intro R3D splash window opens at the render's first frame (no seizure
    # card in taiko, so it begins immediately -- std offsets by the seizure
    # duration). The sim fades it out at the first note's scroll-in
    # (sim.first_spawn_ms), the same "first approach" the std/catch use.
    sim.logo_start_ms = start_ms if cfg.show_logo else None
    # A failed/quit replay stops recording at death; end the video there rather
    # than playing the rest of the map (sim flags this from the life-bar / where
    # the replay frames stop).
    if getattr(sim, "failed", False) and getattr(sim, "fail_time_ms", 0):
        gameplay_end_ms = int(sim.fail_time_ms + 700)
    else:
        gameplay_end_ms = int(last + cfg.tail_ms)
    # results outro (matches osu_renderer: 800ms gap, then the card) — on by default
    RESULTS_GAP_MS, FADE_MS = 800, 400
    if cfg.show_results:
        results_start_ms = gameplay_end_ms + RESULTS_GAP_MS
        total_end_ms = results_start_ms + cfg.results_ms
    else:
        results_start_ms = total_end_ms = gameplay_end_ms
    # DT/HT playback: the simulation lives on the map-time axis, but a DT play
    # should *look* 1.5x faster. So gameplay frames advance map-time by
    # frame_ms*rate per output frame (fewer frames at the same fps => sped up),
    # and the audio is atempo'd by the same rate. The results outro stays
    # real-time for cross-mode consistency with the mania renderer.
    rate = getattr(bm, "rate", 1.0) or 1.0
    frame_ms = 1000.0 / cfg.fps
    map_step = frame_ms * rate
    gameplay_frames = max(1, int((gameplay_end_ms - start_ms) / map_step))
    outro_frames = max(0, int((total_end_ms - gameplay_end_ms) / frame_ms)) if cfg.show_results else 0
    n_frames = gameplay_frames + outro_frames

    w, h = cfg.resolution
    renderer = SpriteRenderer(w, h)
    if skin is not None:
        for key, rgba in skin.textures.items():
            renderer.upload_texture(key, rgba)
    else:
        for key, rgba in build_textures(cfg.skin_dir).items():
            renderer.upload_texture(key, rgba)
    _mark("skin_textures_uploaded")
    from osu_taiko_renderer.skin.assets import bake_logo_tile, logo_glow_rgba
    renderer.upload_texture("logo_tile", bake_logo_tile())
    renderer.upload_texture("logo_glow", logo_glow_rgba())
    if bg is not None:
        _bg_tex = _bg_cover(bg, w, h, cfg.bg_blur)
        if _bg_tex is not None:
            renderer.upload_texture("bg", _bg_tex)

    # Storyboard renderer (phase 4/5): constructed only when --storyboard is on
    # (see below). While None, the frame loop takes the exact single-draw path
    # it always has, so live renders are byte-identical.
    _mark("bg_tex")
    storyboard = _build_storyboard(cfg, renderer, osu_path, w, h)
    _mark("storyboard")

    total_dur_s = n_frames / cfg.fps
    # Per-note hitsound dub (taiko previously rendered music-only). Built from
    # the sim's judged hits, aligned to the FINAL video timeline; amixed with
    # the song in _spawn_ffmpeg. Fail-soft: any problem just drops the dub.
    hitsound = None
    if audio is not None:
        try:
            from osu_taiko_renderer.render.hitsounds import build_taiko_hitsound_track
            _bdir = osu_path.parent if osu_path is not None else None
            hitsound = build_taiko_hitsound_track(
                notes=sim.notes, note_hit=sim.note_hit, beatmap=bm,
                sample_dirs=(([_bdir] if cfg.beatmap_hitsounds else [])
                             + [cfg.skin_dir, cfg.default_skin_dir]),
                output_wav=output_path.with_suffix(".hits.wav"),
                video_ms=total_dur_s * 1000.0, start_ms=start_ms, rate=rate,
                miss_hitsound=cfg.miss_hitsound,
                nightcore=cfg.nightcore_hitsounds,
                nc_mod=bool(int(getattr(meta, "mods", 0) or 0) & 512),  # NC mod
                gameplay_end_ms=gameplay_end_ms,   # beat overlays stop here (no results bleed)
                press_edges=sim.drum_press_edges(), ok_window_ms=sim.ok_w)
        except Exception as _e:  # noqa: BLE001 — hitsounds never break a render
            print(f"[taiko-renderer] hitsound build skipped: {_e}", file=sys.stderr)
            hitsound = None
    is_nc = bool(int(getattr(meta, "mods", 0) or 0) & 512)   # Nightcore bit
    _mark("hitsounds")
    _mark("assets_done")
    # INLINE PREVIEW (R3D_PREVIEW_INLINE=1, default OFF): have the SAME ffmpeg
    # that encodes the master also write the lean 720p30 preview embed as a
    # second output, so it is finished the moment the render is. Without it the
    # node re-encodes the finished master afterwards before anything can be
    # published. Flag unset -> the ffmpeg command is built exactly as before.
    preview_path = None
    if os.environ.get("R3D_PREVIEW_INLINE") == "1":
        preview_path = output_path.parent / (output_path.stem + ".embed.mp4")
        print(f"[taiko] inline preview -> {preview_path.name}",
              file=sys.stderr, flush=True)
    # STREAMABLE MASTER (R3D_STREAM_MASTER=1, default OFF): write the master
    # front to back (no +faststart) with the final loudness pass already
    # applied, so the contributor client can upload it WHILE it renders. The
    # marker file tells the client this engine honoured the flag; without it
    # the client keeps its post-render loudness pass.
    # INLINE DISCORD COPY (R3D_COMPACT_INLINE=1, default OFF; needs the inline
    # preview): a third output encoded to the compact plan the node/bot use
    # for `-embed-sm.mp4`, so nothing is left to encode after the render.
    compact_path = None
    if preview_path is not None and _compact_wanted(
            total_dur_s, cfg.resolution[0], cfg.resolution[1], cfg.fps, 0.5):
        compact_path = output_path.parent / (output_path.stem + ".embed-sm.mp4")
        print(f"[taiko] inline discord copy -> {compact_path.name}",
              file=sys.stderr, flush=True)
    stream_master = os.environ.get("R3D_STREAM_MASTER") == "1"
    if stream_master:
        import json as _json
        (output_path.parent / (output_path.stem + ".stream.json")).write_text(
            _json.dumps({"schema": 1, "faststart": False,
                         "loudnorm": _STREAM_LOUDNORM,
                         "compact": compact_path is not None}))
        print("[taiko] streamable master (no faststart, loudnorm in-engine)",
              file=sys.stderr, flush=True)
    proc = _spawn_ffmpeg(cfg, output_path, audio, start_ms, rate, total_dur_s,
                         hitsound=hitsound, is_nc=is_nc,
                         preview_path=preview_path,
                         stream_master=stream_master,
                         compact_path=compact_path)
    # HUD: legacy (true-to-skin) when the skin ships a score font, else Argon.
    from osu_taiko_renderer.argon.hud import ArgonHud
    from osu_taiko_renderer.hud.skin_hud import LegacyHud
    # ArgonHud is the single HUD: it renders the skin's digit font + scorebar HP
    # bar per-element (SkinDigitFont / SkinHealthBar) with Argon fallback, while
    # keeping Argon's progress bar / key counter / hit-error bar / title chrome
    # that the legacy HUD lacked. (Was: swap to LegacyHud iff the skin shipped a
    # score-N font, which discarded the HP bar + skin font for lazer skins.)
    hud = ArgonHud(cfg.resolution, meta, bm, first, last, sim, cfg=cfg)
    # lazer's BreakOverlay (countdown + progress bar + CURRENT PROGRESS info
    # + slide-in chevrons) — break_overlay.py, a 1:1 port of
    # osu.Game/Screens/Play/BreakOverlay.cs on this engine's CPU compositing
    # (the catch d8ccb60 rollout). lazer taiko shows the same screen-centre
    # overlay (a Player-level component), so it is composited over the full
    # frame on BOTH HUD variants from one wiring point in _emit_gameplay.
    # Fed the same map-time [Events] breaks that drive the dim envelope.
    from osu_taiko_renderer.hud.break_overlay import LazerBreakOverlay
    break_overlay = LazerBreakOverlay(
        w, h, getattr(bm, "breaks", []) or [],
        mods=int(getattr(meta, "mods", 0) or 0))
    # OsuModFlashlight (TaikoModFlashlight): darken each gameplay frame to a
    # circular spotlight fixed on the playfield hit target, combo-scaled with
    # an 800ms OutQuint fade. Composited OVER the playfield but UNDER the HUD
    # (both HUD variants) so score/combo/health stay visible. No-op without FL.
    from osu_taiko_renderer.render.flashlight import TaikoFlashlight
    _fl_mods = int(getattr(meta, "mods", 0) or 0)
    flashlight = TaikoFlashlight(
        sim.geo, getattr(sim, "_rt", None), getattr(sim, "_cum", None),
        _fl_mods)
    from osu_taiko_renderer.argon.compositor import ArgonEffects, bloom as _bloom
    effects = ArgonEffects(sim.geo, cfg.skin_dir)

    from osu_taiko_renderer.hud.hud import draw_results
    # results-screen map leaderboard (parity with std/catch): build + bake ONCE,
    # up front, so the outro just composites the pre-baked flank cards each
    # frame. Fully fail-soft — any problem leaves the plain results card (renders
    # unchanged). Attached to the HUD instance; both HUD variants composite it.
    baked_board = None
    if cfg.show_results and getattr(cfg, "show_leaderboard", True):
        try:
            from osu_taiko_renderer.hud.lb_cards import build_taiko_board
            baked_board = build_taiko_board(cfg, meta, bm, "")
        except Exception as e:  # noqa: BLE001 — a board must never break a render
            print(f"[taiko-renderer] leaderboard skipped: {e}", file=sys.stderr)
            baked_board = None
    # FEATURED results-card avatar (the current player's real osu! pfp PNG).
    # Missing/unreadable -> the results screen falls back to the procedural chip.
    feat_bytes = None
    _feat_png = getattr(cfg, "featured_avatar_png", None)
    if _feat_png is not None:
        try:
            feat_bytes = Path(_feat_png).read_bytes() or None
        except Exception:  # noqa: BLE001 — avatar wiring never breaks a render
            feat_bytes = None
    try:
        hud.board = baked_board
        hud.featured_avatar_bytes = feat_bytes
    except Exception:  # noqa: BLE001 — HUD attach never breaks a render
        pass
    # Ported osu!(lazer) results screen (catch's CatchLazerResults): the outro
    # draws THIS instead of hud.draw_results. Hand the featured player's real
    # osu! avatar to the module global it reads (same mechanism as catch); the
    # instance is built lazily on the first outro frame and cached here.
    if cfg.show_results:
        try:
            from osu_taiko_renderer.hud.lazer_results import set_featured_avatar_png
            set_featured_avatar_png(getattr(cfg, "featured_avatar_png", None))
        except Exception:  # noqa: BLE001 — avatar wiring never breaks a render
            pass
    _lazer_results_cache: dict = {}
    # perf: build the ported lazer results screen + pre-bake its deterministic
    # animation assets on a BACKGROUND thread while gameplay renders (the
    # build alone stalled the first outro frame ~320 ms, and the arc-sweep /
    # score-roll / stats-unfold bakes were ~16 ms/frame peaks). The schedule
    # below replicates the outro loop's (opacity, age_ms) sequence exactly.
    # _draw_lazer_results waits only on the build event; every pre-baked
    # asset is the same pure call the per-frame path makes, so frames are
    # bit-identical. Fully fail-soft: any problem falls back to the original
    # lazy inline build.
    if cfg.show_results and outro_frames > 0:
        _pre_evt = threading.Event()
        _lazer_results_cache["evt"] = _pre_evt
        # R3D_TAIKO_RESULTS_AHEAD: composite the results frames DURING GAMEPLAY on the
        # prebake thread, so the outro becomes a dequeue instead of ~11 ms of PIL per
        # frame (measured: 202 full composites, a FIXED ~2.2 s per render regardless of
        # map length).
        #
        # Safe because past the fade (op >= 0.999, i.e. 320 ms in) `render_frame` takes
        # its `wash_a >= 255` branch and NEVER READS the background frame -- it starts
        # from cached opaque black, so the output depends only on age_ms, which is
        # deterministic from the frame index. Frames before that still need the real
        # last-gameplay frame and stay inline.
        #
        # ONE producer, not N: render_frame mutates self._settled and self._black_base,
        # so concurrent callers would race. One is enough -- it produces at ~11 ms/frame
        # against a 16.7 ms/frame consumer.
        _RES_AHEAD = _sw.RESULTS_AHEAD
        _res_ready: dict = {}
        _res_cv = threading.Condition()
        _RES_MAX = int(os.environ.get("R3D_TAIKO_RESULTS_AHEAD_MAX", "32"))

        def _prebuild_results() -> None:
            try:
                from osu_taiko_renderer.hud.lazer_results import CatchLazerResults
                scr = CatchLazerResults((w, h), meta, bm, board=baked_board,
                                        osu_path=osu_path, sim=sim)
                _lazer_results_cache["pre"] = scr
                _pre_evt.set()           # ORIGINAL: publish first
                sched = []
                sched_idx = []
                for _i in range(gameplay_frames, n_frames):
                    _t = int(gameplay_end_ms + (_i - gameplay_frames) * frame_ms)
                    if _t >= results_start_ms:
                        _op = min(1.0, (_t - results_start_ms) / FADE_MS)
                        _age = float(_t - results_start_ms)
                        sched.append((_op, _age))
                        sched_idx.append((_i, _op, _age))
                # INTERLEAVED: compose each results frame the moment ITS assets are
                # baked, rather than waiting for the whole schedule. Producing after
                # prebake_anim returned capped the win at +5% purely because the
                # producer had almost no runway left on a short map.
                _shape = np.zeros((h, w, 3), np.uint8) if _RES_AHEAD else None

                def _on_ready(_k, _op, _age):
                    if not _RES_AHEAD or _op < 0.999:
                        return                # needs the real background; stays inline
                    _idx = sched_idx[_k][0]
                    _f = scr.render_frame(_shape, _op, _age)
                    with _res_cv:
                        while len(_res_ready) >= _RES_MAX:
                            _res_cv.wait(0.5)
                        _res_ready[_idx] = _f
                        _res_cv.notify_all()

                scr.prebake_anim(sched, on_ready=_on_ready if _RES_AHEAD else None)
            except Exception as _e:  # noqa: BLE001 — never break a render
                print(f"[taiko-renderer] results prebuild skipped: {_e}",
                      file=sys.stderr)
            finally:
                _pre_evt.set()

        threading.Thread(target=_prebuild_results, daemon=True,
                         name="results-prebake").start()
    last_gameplay = None
    # Async pipeline (ported from the std renderer's proven design):
    #   * GPU readback goes through a 3-deep PBO ring (read_rgb_async returns
    #     None while the ring fills; frames pop out ~2 frames late, in strict
    #     submission order; read_drain() flushes the tail).
    #   * The CPU-side compositing (Argon effects + HUD) is deferred until a
    #     frame's pixels pop out of the ring: everything it needs is captured
    #     at BUILD time — (scene, active_effects(t), drum_flashes(t)) — and
    #     queued alongside. All of it is a pure function of sim state
    #     precomputed in __init__ (build_scene mutates nothing), called once
    #     per frame in frame order, exactly as the synchronous path did.
    #   * The ffmpeg pipe write (tobytes + stdin.write) happens on a writer
    #     thread behind a small bounded queue (_FrameWriter).
    # Frame count, order and bytes are identical to the synchronous path.
    # R3D_TAIKO_GPU_FX: register the prebaked effect textures with GL once, so
    # the drum flashes / hit explosions can be drawn as additive sprites inside
    # the existing pass (before readback) instead of composited on the CPU after.
    # (the GPU switches depend on each other; render/envflag.py resolves them)
    _GPU_FX = _sw.GPU_FX
    _GPU_FL = _sw.GPU_FL
    # R3D_TAIKO_GPU_HUD: run the WHOLE HUD as GL sprites (hud.overlay_gl).
    # Requires GPU_FX (needs the sprite plumbing) and, when FL is on, GPU_FL.
    _GPU_HUD = _sw.GPU_HUD
    _GPU_FINISH = envflag("R3D_TAIKO_GPU_FINISH")
    # R3D_TAIKO_FL_EXACT: draw the flashlight by SAMPLING the CPU's own
    # (1 - alpha) ramp as an R32F texture instead of recomputing the smoothstep
    # in GLSL. Removes the last ~25 channel values of +-1 difference, making the
    # GPU flashlight BYTE-IDENTICAL to composite(). Needs GPU_FL.
    _FL_EXACT = _sw.FL_EXACT
    # R3D_TAIKO_GPU_BREAK: draw the break overlay's SHADOW as a GL sprite in the
    # main pass instead of compositing it on the CPU after readback. Only the
    # shadow: it is the overlay's bottom element, so moving it keeps the stacking
    # identical (HUD < shadow < rest-of-break) while the rest still composites on
    # the CPU on top. It is also 622k of the overlay's ~1.13M pixels at 1080p.
    #
    # Needs the GPU HUD, and needs it to have SUCCEEDED this frame: if overlay_gl
    # overflowed and the HUD fell back to the CPU, a GL shadow would land UNDER
    # that CPU HUD instead of over it.
    _GPU_BREAK = _sw.GPU_BREAK
    if _GPU_FX:
        for _k, _im in effects.gpu_fx_textures().items():
            renderer.upload_texture(_k, _im)
        # judgement ring/popup textures are baked lazily at their exact integer
        # sizes on first use, so the compositor needs the GL handle
        effects.bind_gl(renderer)
    _mark("setup_done/loop_start")
    _t_render0 = time.monotonic()
    writer = _FrameWriter(proc)
    pending = deque()   # (scene, exps, judges, drum_flashes) awaiting pixels
    # Per-frame score sidecar (#135, live overlay): one sample per GAMEPLAY frame
    # {t_ms (gameplay/map ms), score (ScoreV2), combo, acc 0..1}, read straight
    # off the re-pinned sim scene. Collected here, written after a successful
    # render. None -> feature disabled (no overhead on normal renders).
    _score_samples: list | None = (
        [] if getattr(cfg, "score_json_path", None) else None)

    _yuv_cpu_frames = 0
    _yuv_last_rgb = None    # the last gameplay frame's RGB, if the CPU finished it

    def _cpu_chain(raw, p_scene, p_exps, p_judges, p_drums, p_hud_gpu, p_brk_gpu):
        """Everything that is composited on the CPU after readback, in lazer's
        z-order. Each stage is a no-op when its switch moved it into the GL pass."""
        # hud_opacity 0 suppresses the GREAT/OK/MISS judgement-text popups
        # (p_judges) too — the YT overlay owns judgement display. Hit
        # explosions (p_exps) + drum flashes (p_drums) stay = gameplay.
        _c = time.perf_counter if _STAGE else None
        _a = _c() if _c else 0
        out = effects.composite(
            raw, p_exps, p_judges if cfg.hud_opacity > 0.0 else [], p_drums)
        if _c: _b = _c(); _acc("effects", _b - _a); _a = _b
        out = flashlight.composite(out, p_scene.time_ms)
        if _c: _b = _c(); _acc("flashlight", _b - _a); _a = _b
        if cfg.hud_opacity > 0.0 and not p_hud_gpu:
            out = hud.overlay(out, p_scene)
        if _c: _b = _c(); _acc("hud", _b - _a); _a = _b
        # lazer z-order: BreakOverlay is a LATER overlay-component child
        # than HUDOverlay (Player.createOverlayComponents) — composited
        # ABOVE every HUD element, both HUD variants. Live accuracy from
        # the sim's running scene value (bound like lazer's bindable).
        # Cheap no-op outside break windows (frame bytes untouched).
        if cfg.hud_opacity > 0.0:
            break_overlay.draw(out, p_scene.time_ms, p_scene.accuracy,
                               skip_shadow=p_brk_gpu)
        if _c: _b = _c(); _acc("break_overlay", _b - _a)
        return out

    def _yuv_needs_cpu(p_scene, p_exps, p_judges, p_drums, p_hud_gpu):
        """Under R3D_TAIKO_GPU_YUV: would the CPU chain change THIS frame? Then it
        cannot leave the GPU as finished yuv420p; it takes the slow road instead
        (read RGB, composite, convert that). Which frames those are depends on
        the other switches: with none of them, every frame; with all of them,
        only the frames of a break (the overlay's text and bar are CPU-drawn)
        and any frame whose HUD overflowed its reserved texture."""
        hud_on = cfg.hud_opacity > 0.0
        return (effects.cpu_work(p_exps, p_judges if hud_on else (), p_drums)
                or flashlight.cpu_work()
                or (hud_on and not p_hud_gpu)
                or (hud_on and break_overlay.will_draw(p_scene.time_ms)))

    def _emit_gameplay(raw):
        nonlocal last_gameplay
        (p_scene, p_exps, p_judges, p_drums, p_hud_gpu,
         p_brk_gpu) = pending.popleft()
        if _GPU_YUV_R:
            # `raw` is finished planar yuv420p: either straight off the GPU, or a
            # frame that needed the CPU chain and was composited BEFORE it was
            # converted and queued (see the frame loop). Nothing is left to do.
            if _DUMP is not None:
                _di = _DUMP[1].get(getattr(_emit_gameplay, "n", 0))
                _emit_gameplay.n = getattr(_emit_gameplay, "n", 0) + 1
                if _di is not None:
                    import numpy as _np
                    _np.save(f"{_DUMP[0]}/f{_emit_gameplay.n - 1:06d}.npy", raw)
            # last_gameplay stays None: the outro needs an RGB frame to composite
            # the results card onto, and falls back to renderer.read_rgb() for it.
            _pa = time.perf_counter() if _STAGE else 0
            writer.push(raw)
            if _STAGE:
                _acc("push", time.perf_counter() - _pa); _acc("n", 1)
            return
        _c = time.perf_counter if _STAGE else None
        out = _cpu_chain(raw, p_scene, p_exps, p_judges, p_drums, p_hud_gpu,
                         p_brk_gpu)
        _a = _c() if _c else 0
        # HARNESS: R3D_TAIKO_DUMP=dir,i0,i1,... writes the RAW composited frame
        # (pre-encode) so two runs can be compared exactly, without x264 noise.
        if _DUMP is not None:
            _di = _DUMP[1].get(_emit_gameplay.n if hasattr(_emit_gameplay, "n") else 0)
            _emit_gameplay.n = getattr(_emit_gameplay, "n", 0) + 1
            if _di is not None:
                import numpy as _np
                _np.save(f"{_DUMP[0]}/f{_emit_gameplay.n - 1:06d}.npy", out)
        last_gameplay = out
        writer.push(out)
        if _c: _acc("push", _c() - _a); _acc("n", 1)

    try:
        try:
            for i in range(n_frames):
                if i < gameplay_frames:
                    t = int(start_ms + i * map_step)
                    _c2 = time.perf_counter if _STAGE else None
                    _p = _c2() if _c2 else 0
                    scene = sim.build_scene(t)
                    if _c2: _q = _c2(); _acc("build_scene", _q - _p); _p = _q
                    if _score_samples is not None:
                        _score_samples.append({
                            "t_ms": int(t),
                            "score": int(scene.score),
                            "combo": int(scene.combo),
                            "acc": round(float(scene.accuracy), 6),
                        })
                    # GPU_FX needs exps/drums BEFORE the draw call. Both are
                    # pure functions of `t` over precomputed sim state, so
                    # hoisting them changes nothing but the call order.
                    _hud_gl_ok = False       # also read by the plain path below
                    if _GPU_FX:
                        exps, judges = sim.active_effects(t)
                        _drums = sim.drum_flashes(t)
                        # z-order, matching the CPU chain: playfield (and the
                        # storyboard overlay), then the additive hit explosions,
                        # then the judgement bursts (ring pieces additive, popup
                        # text straight alpha on top). They are drawn in their OWN
                        # ordered call after the playfield (draw_ordered below),
                        # not appended to scene.sprites: draw() is two-phase and
                        # must stay so for the drum press flashes in the scene.
                        _fx = effects.gpu_fx_sprites(exps, _drums)
                        _fx += effects.gpu_judge_sprites(judges)
                        # The HUD numbers are drawn in a SECOND batch, after the
                        # flashlight pass, because lazer's z-order is
                        # playfield -> effects -> FLASHLIGHT -> HUD: the
                        # spotlight darkens gameplay but must NOT darken the HUD.
                        # GUARD: moving the HUD numbers into the GL pass is only
                        # safe if the flashlight is in the pass too. lazer draws
                        # the HUD ABOVE the flashlight; with the numbers on the
                        # GPU but FL still on the CPU, the CPU pass runs after
                        # readback and BLACKS THE NUMBERS OUT. Measured: HUD
                        # pixels [255,255,255] -> [0,0,0] on an FL replay, while
                        # a NoMod fixture shows nothing wrong at all.
                        if _c2: _q = _c2(); _acc("sb_fx", _q - _p); _p = _q
                        _hud_sp = []
                        _w0 = _c2() if _SBPROF else 0
                        if cfg.hud_opacity > 0.0 and (_GPU_FL or not flashlight.on):
                            if _GPU_HUD:
                                # Whole HUD as GL sprites. Returns None if any
                                # element overflowed its reserved texture, in which
                                # case this frame falls back to the CPU overlay
                                # rather than dropping an element.
                                _s = hud.overlay_gl(scene, renderer)
                                if _s is not None:
                                    _hud_sp, _hud_gl_ok = _s, True
                            if not _hud_gl_ok:
                                _hud_sp = hud.prepare_numbers(scene, renderer)
                        if _SBPROF:
                            _w1 = _c2()
                            _SB["hud_gl"] += _w1 - _w0
                            _SB["n"] += 1
                        _fl_p = flashlight.gl_params(t) if _GPU_FL else None
                        _fl_xp = (flashlight.gl_keep_params(t)
                                  if _FL_EXACT else None)
                        if _SBPROF:
                            _SB["fl_params"] += _c2() - _w1
                    # break-overlay shadow -> GL. Decided per frame and carried in
                    # the pending tuple, because _emit_gameplay runs ~2 frames behind
                    # and a shared cell would apply this frame's answer to that one.
                    _brk_gpu = (_GPU_BREAK and _hud_gl_ok
                                and cfg.hud_opacity > 0.0)
                    _brk_sp = (break_overlay.gl_shadow_sprite(t, renderer)
                               if _brk_gpu else None)
                    if _c2: _q = _c2(); _acc("sprite_build", _q - _p); _p = _q
                    renderer.begin()
                    if _c2: _q = _c2(); _acc("gl_begin_clear", _q - _p); _p = _q
                    if storyboard is None:
                        # exact single-draw path (byte-identical to pre-SB)
                        renderer.draw(scene.sprites)
                    else:
                        # interleave the two storyboard z-slices around the
                        # playfield: bg image -> SB underlay (Background/Fail/
                        # Pass/Foreground) -> playfield sprites -> SB overlay
                        # (Overlay layer). taiko's HUD/effects are CPU-composited
                        # after readback, so the whole GL pass sits under them —
                        # the SB Overlay lands over gameplay, under the HUD, as
                        # in lazer. Both slices share the bg dim (sb_brightness).
                        n = scene.bg_split
                        b = scene.sb_brightness
                        if n:
                            renderer.draw(scene.sprites[:n])
                        storyboard.draw_underlay(t, b)
                        renderer.draw(scene.sprites[n:])
                        storyboard.draw_overlay(t, b)
                    if _GPU_FX:
                        # effects, above the playfield and the storyboard overlay
                        # (where the CPU chain composites them), in strict order
                        if _fx:
                            renderer.draw_ordered(_fx)
                        # FL between the two batches (see the z-order note above).
                        if _fl_xp is not None:
                            renderer.draw_flashlight_exact(*_fl_xp)
                        elif _fl_p is not None:
                            renderer.draw_flashlight(*_fl_p)
                        elif _GPU_FL and flashlight.on:
                            # radius collapsed: the CPU path returns an all-black
                            # frame, so reproduce that rather than skipping.
                            renderer.draw_flashlight(0.0, 0.0, 1, 0.0)
                        if _hud_sp:
                            renderer.draw(_hud_sp)
                        # ABOVE the HUD (lazer: BreakOverlay is a later
                        # overlay-component child than HUDOverlay), and in its own
                        # draw so the HUD run is not split.
                        if _brk_sp is not None:
                            renderer.draw([_brk_sp])
                    if _c2: _q = _c2(); _acc("gl_draw", _q - _p); _p = _q
                    # R3D_TAIKO_GPU_FINISH: block until the GPU has actually executed,
                    # so its time lands in its own bucket instead of hiding inside
                    # readback_block. DIAGNOSTIC ONLY — it serialises CPU and GPU, so
                    # total fps drops while it is on.
                    if _GPU_FINISH:
                        renderer.ctx.finish()
                        if _c2: _q = _c2(); _acc("gpu_execute", _q - _p); _p = _q
                    if not _GPU_FX:
                        exps, judges = sim.active_effects(t)
                        _drums = sim.drum_flashes(t)
                    if _c2: _q = _c2(); _acc("active_effects", _q - _p); _p = _q
                    pending.append((scene, exps, judges, _drums, _hud_gl_ok,
                                    _brk_gpu))
                    if _c2: _q = _c2(); _acc("drum_flashes", _q - _p); _p = _q
                    if not _GPU_YUV_R:
                        raw = renderer.read_rgb_async()
                    elif _yuv_needs_cpu(scene, exps, judges, _drums, _hud_gl_ok):
                        # read this frame back as RGB now, composite it, and put
                        # the result through the same GPU conversion and the same
                        # ring, so frame order holds. np.array: read_rgb hands
                        # back a read-only view and the chain writes in place.
                        _yuv_cpu_frames += 1
                        _yuv_last_rgb = _cpu_chain(
                            np.array(renderer.read_rgb()[..., :3]), scene, exps,
                            judges, _drums, _hud_gl_ok, _brk_gpu)
                        raw = renderer.yuv_from_rgb(_yuv_last_rgb)
                    else:
                        _yuv_last_rgb = None
                        # the break overlay's bar eases every frame, in and out of
                        # breaks; keep its clock running (it draws nothing here)
                        if cfg.hud_opacity > 0.0:
                            break_overlay.draw(None, scene.time_ms, scene.accuracy)
                        raw = renderer.read_yuv_async()
                    if _c2: _acc("readback_block", _c2() - _p)
                    if raw is not None:
                        _emit_gameplay(raw)
                else:
                    # gameplay -> outro boundary: flush the PBO ring first so
                    # last_gameplay is the true final gameplay frame and
                    # ordering is preserved across the boundary.
                    #
                    # GUARDED ON `pending`: this else-branch runs for EVERY outro
                    # frame, not just the first. That was harmless while outro
                    # conversion was synchronous (the ring was empty after the
                    # boundary), but the async outro path queues INTO this same ring,
                    # so an unguarded drain pulls outro frames back out and hands them
                    # to _emit_gameplay, which pops an empty `pending`. `pending` is
                    # non-empty only at the true boundary, so it is the exact latch.
                    if pending:
                        for raw in (renderer.read_yuv_drain() if _GPU_YUV_R
                                    else renderer.read_drain()):
                            _emit_gameplay(raw)
                        if _GPU_YUV_R and last_gameplay is None:
                            # Grab the frozen background ONCE. Under GPU_YUV the
                            # gameplay path never produces an RGB frame (it emits
                            # planar YUV), so last_gameplay stays None and the
                            # `else renderer.read_rgb()` below would fire for EVERY
                            # outro frame -- 318 full 6.2 MB readbacks of a frame
                            # that never changes. The scene fbo still holds the final
                            # gameplay frame at this point, which is exactly what the
                            # outro wants as its background.
                            # ...unless the CPU chain finished that last frame:
                            # then the fbo holds it WITHOUT what the CPU added.
                            last_gameplay = (_yuv_last_rgb
                                             if _yuv_last_rgb is not None
                                             else renderer.read_rgb())
                    # perf: materialise the frozen final gameplay frame ONCE
                    # (it was .copy()'d per outro frame). Nothing downstream
                    # mutates it — the results screen builds new arrays — so
                    # re-pushing the same array is byte-identical.
                    if last_gameplay is not None and \
                            not last_gameplay.flags.c_contiguous:
                        last_gameplay = np.ascontiguousarray(last_gameplay)
                    # outro: frozen final gameplay frame, then the results card
                    # fades in (consistent with the mania renderer). Real-time.
                    t = int(gameplay_end_ms + (i - gameplay_frames) * frame_ms)
                    rgb = last_gameplay if last_gameplay is not None else \
                        renderer.read_rgb()
                    if cfg.show_results and t >= results_start_ms:
                        op = min(1.0, (t - results_start_ms) / FADE_MS)
                        # age_ms drives the ported lazer results' two-stage
                        # animation (arc sweep / grade punch / score roll /
                        # flank slide-in, then the stage-2 stats panels
                        # unfolding from the right). osu_path lets it compute
                        # stars + pp (rosu); sim feeds the COMBO panel its
                        # per-object combo series. BYPASSES hud.draw_results.
                        _pre = None
                        if _RES_AHEAD:
                            with _res_cv:
                                # drop anything the inline path already overtook, so a
                                # producer that fell behind cannot pin the bound forever
                                for _k in [k for k in _res_ready if k < i]:
                                    del _res_ready[_k]
                                _pre = _res_ready.pop(i, None)
                                _res_cv.notify_all()
                        if _pre is not None:
                            rgb = _pre
                        else:
                            if _OUTRO_CPROF:
                                _OCP.enable()
                            rgb = _draw_lazer_results(
                                _lazer_results_cache, rgb, meta, bm, op,
                                age_ms=float(t - results_start_ms),
                                board=baked_board, osu_path=osu_path, sim=sim)
                            if _OUTRO_CPROF:
                                _OCP.disable()
                    # outro frames are CPU-composited RGB and never touch the
                    # scene texture, so they convert on the CPU -- bit-identical
                    # to the shader (both verified max|d|=0 vs swscale).
                    if _GPU_YUV_R:
                        # async now: None while the ring fills, drained after the loop
                        _y = renderer.yuv_from_rgb(rgb)
                        if _y is not None:
                            writer.push(_y)
                    else:
                        writer.push(rgb)
                if progress_callback and i % cfg.fps == 0:
                    progress_callback(int(i / n_frames * 100))

            # Map end: flush whatever is still in the ring. GAMEPLAY frames still have a
            # `pending` entry and must go through _emit_gameplay; OUTRO frames have none
            # and are already finished buffers, so they go straight to the writer.
            # Routing on `pending` keeps ONE drain correct for both -- draining an outro
            # frame through _emit_gameplay would popleft an empty deque.
            for raw in (renderer.read_yuv_drain() if _GPU_YUV_R
                        else renderer.read_drain()):
                if pending:
                    _emit_gameplay(raw)
                else:
                    writer.push(raw)
        except BrokenPipeError:
            pass               # ffmpeg died — surfaced via ret below
    finally:
        writer.close()
        if proc.stdin:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
        _mark("loop_end+drain")
        ret = proc.wait()
        _mark("ffmpeg_done")
        renderer.release()
        import sys as _rsys
        _wall = time.monotonic() - _t_render0
        if _GPU_YUV_R and _yuv_cpu_frames:
            print(f"[taiko-renderer] GPU colour conversion: {_yuv_cpu_frames} of "
                  f"{gameplay_frames} gameplay frames needed CPU compositing "
                  f"and took the slow path", file=sys.stderr)
        if _STAGE and _ST.get("n"):
            _nf = _ST.pop("n")
            print(f"[taiko-stage] per-frame ms over {int(_nf)} composited "
                  f"frames:", file=sys.stderr)
            for _k, _v in sorted(_ST.items(), key=lambda kv: -kv[1]):
                print(f"[taiko-stage]   {_k:<16s} {_v / _nf * 1e3:7.3f}"
                      f"   ({_v:6.1f}s total)", file=sys.stderr)
            print(f"[taiko-stage]   {'SUM':<16s} "
                  f"{sum(_ST.values()) / _nf * 1e3:7.3f}", file=sys.stderr)
        if _PERF and _MARKS:
            _t0 = _MARKS[0][1]
            print("[taiko-perf] phase table (s):", file=sys.stderr)
            _prev = _t0
            for _lbl, _ts in _MARKS[1:]:
                print(f"[taiko-perf]   {_lbl:<24s} +{_ts - _prev:7.2f}"
                      f"   (t={_ts - _t0:7.2f})", file=sys.stderr)
                _prev = _ts
            print(f"[taiko-perf]   {'TOTAL from entry':<24s} "
                  f" {_MARKS[-1][1] - _t0:7.2f}", file=sys.stderr)
        print(f"done: {n_frames} frames in {_wall:.1f}s "
              f"({(n_frames / _wall) if _wall else 0.0:.1f} fps) ret={ret}",
              file=_rsys.stderr, flush=True)
        if storyboard is not None:
            try:
                st = storyboard.stats()
                print(f"storyboard cache: {st['uploads']} uploads, "
                      f"{st['evictions']} evictions, {st['peak_mb']:.0f} MB "
                      f"peak, {st['resident']} resident",
                      file=_rsys.stderr, flush=True)
            except Exception:  # noqa: BLE001 — stats print never breaks a render
                pass

    if ret != 0:
        tail = ""
        errlog = getattr(proc, "_catch_errlog", None)
        if errlog and Path(errlog).exists():
            tail = Path(errlog).read_text(errors="replace")[-800:]
        # a hardware preview that failed must not fail the NEXT render too
        if _phw.note_preview_failure(list(getattr(proc, "args", []) or []),
                                     tail.encode("utf-8", "replace")):
            tail += ("\n[the preview's hardware encoder is now off for 24 h on "
                     "this node; the next render uses the CPU preview]")
        raise TaikoRenderError(f"ffmpeg exited {ret}\n{tail}")
    if not output_path.exists() or output_path.stat().st_size < 8_000:
        raise TaikoRenderError("output too small / missing — render likely failed")
    # score-fidelity sidecar: `<output>.mp4.score.json` next to the mp4 — the bot
    # (worker.py / cli/r3d_render.py) reads it into the completion marker so the
    # DB/website card store/display the SAME standardised total the in-video
    # counter ended on. Same filename + `score_v3` key the catch/mania engines
    # emit. Best-effort: a failed sidecar never fails a completed render.
    if score_fid is not None:
        try:
            import json as _json
            sidecar = Path(str(output_path) + ".score.json")
            # Gameplay-start anchor for the YT versus HUD (all-mode sync):
            # video-seconds into THIS panel where map-time 0 lands (frame 0 is
            # map-time start_ms), plus the rate-mods speed.
            _map0_video_s = round((0 - start_ms) / (rate * 1000.0), 6)
            sidecar.write_text(_json.dumps(
                {"schema": 1, "mode": 1,
                 "map0_video_s": _map0_video_s, "rate": float(rate),
                 **score_fid}, default=str))
        except Exception as _sc_e:  # noqa: BLE001 — sidecar is best-effort
            print(f"[taiko-renderer] score sidecar write failed: {_sc_e}",
                  file=sys.stderr, flush=True)
    # per-frame score.json for the live overlay compositor (#135) — best-effort.
    if _score_samples is not None:
        try:
            import json as _json
            _sjp = Path(cfg.score_json_path)
            if _sjp.parent and not _sjp.parent.exists():
                _sjp.parent.mkdir(parents=True, exist_ok=True)
            _sjp.write_text(_json.dumps(_score_samples, separators=(",", ":")))
            print(f"[taiko-renderer] wrote {len(_score_samples)} score.json "
                  f"samples -> {_sjp}", file=sys.stderr, flush=True)
        except Exception as _sj_e:  # noqa: BLE001 — never fail a done render
            print(f"[taiko-renderer] score.json write failed: {_sj_e}",
                  file=sys.stderr, flush=True)
    if progress_callback:
        progress_callback(100)
    return output_path


# --- ffmpeg -------------------------------------------------------------------

def _probe_encoder(cfg: RenderConfig) -> tuple[str, str | None]:
    if cfg.encoder != "auto":
        # vaapi always needs a device for the hwupload filter; default it.
        if cfg.encoder == "h264_vaapi":
            return cfg.encoder, cfg.encoder_device or "/dev/dri/renderD128"
        return cfg.encoder, cfg.encoder_device
    # nvenc FIRST: R3D renders on NVIDIA (2070S / 1070). The old vaapi-first
    # auto-probe silently won over the far-faster nvenc whenever R3D_ENCODER
    # was unset — a landmine if the worker env ever drops.
    if _ffmpeg_has("h264_nvenc"):
        return "h264_nvenc", None
    dev = cfg.encoder_device or "/dev/dri/renderD128"
    if Path(dev).exists() and _ffmpeg_has("h264_vaapi"):
        return "h264_vaapi", dev
    return "libx264", None


def _ffmpeg_has(name: str) -> bool:
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=15).stdout
    except Exception:  # noqa: BLE001
        return False
    return name in out


# ---- libx264 knobs behind env hooks (same names in all four engines) --------
# libx264 is the master encoder wherever a node has no hardware encoder (every
# Mac). The four engines ask it for different things (std crf 16 / faster,
# taiko crf 20 / veryfast, catch crf 23 / veryfast, mania 2500k / medium), so
# these make the choice settable per run without a code edit:
#   R3D_X264_PRESET   R3D_X264_CRF   R3D_X264_THREADS   R3D_X264_PARAMS
# THE DEFAULTS REPRODUCE THIS ENGINE'S CURRENT COMMAND EXACTLY (veryfast, crf 20, cores - 2 threads):
# with none of them set the ffmpeg argv is unchanged, argument for argument.
_X264_PRESET = os.environ.get("R3D_X264_PRESET", "").strip()
_X264_CRF = os.environ.get("R3D_X264_CRF", "").strip()
_X264_THREADS = os.environ.get("R3D_X264_THREADS", "").strip()
_X264_PARAMS = os.environ.get("R3D_X264_PARAMS", "").strip()


def nvenc_target_bps(w: int, h: int, fps: float) -> int:
    """Resolution-scaled NVENC bitrate ladder (R3D cross-engine policy, 2026-07).

    Replaces the flat per-engine bitrate: scale a 4 Mbps 720p30 reference
    by pixel rate with a perceptual exponent (0.70 -- deliberately NOT
    linear), clamped to [2.5, 16] Mbps.  Anchors: 720p30=4.0M,
    720p60=6.5M, 1080p30=7.1M, 1080p60=11.5M, 1440p60/1080p120+=16M cap.
    Callers pair the target with maxrate=1.5x / bufsize=2x for NVENC VBR.
    Same formula in all four engines (catch/taiko/std/mania v2).
    """
    ref = 1280.0 * 720.0 * 30.0
    target = 4_000_000.0 * ((float(w) * float(h) * float(fps)) / ref) ** 0.70
    return int(min(16_000_000.0, max(2_500_000.0, target)))


def _preview_video_bps(total_dur_s: "float | None") -> int:
    """Video bitrate of the lean preview embed. Mirrors the contributor
    client's makeEmbedVariant (and the bot's _transcode_embed_unbounded):
    ~1.4 Mbps, lowered on long maps so the file stays <= ~24 MiB, floor 500k.
    Same formula as the catch engine."""
    vbps = 1_400_000
    if total_dur_s and total_dur_s > 0:
        vbps = int(24 * 1024 * 1024 * 8 / total_dur_s) - 128_000
        vbps = max(500_000, min(1_400_000, vbps))
    return vbps


# The loudness pass the contributor client runs on a finished master; applied
# in-engine (on the mixed output) when R3D_STREAM_MASTER=1.
_STREAM_LOUDNORM = "loudnorm=I=-18:TP=-1.5:LRA=11"


def _compact_wanted(total_dur_s, w, h, fps, default_factor) -> bool:
    """Whether to write the inline Discord copy for this render.

    R3D_COMPACT_INLINE=1 asks for it. The copy costs a second software encode
    for the whole render, and it is only used when the master is too big to be
    the Discord file itself, so R3D_COMPACT_IF_OVER_BYTES=<n> limits it to
    renders whose master is EXPECTED to exceed n bytes: duration x bitrate,
    with the bitrate taken from R3D_COMPACT_EXPECT_BPS (what this node's
    masters of this kind have actually averaged, supplied by the client) or,
    lacking that, the encoder ladder times `default_factor`. Without a limit,
    or without a duration, the copy is always written."""
    if os.environ.get("R3D_COMPACT_INLINE") != "1":
        return False
    try:
        limit = int(os.environ.get("R3D_COMPACT_IF_OVER_BYTES", "0") or 0)
    except ValueError:
        limit = 0
    if limit <= 0 or not total_dur_s or total_dur_s <= 0:
        return True
    try:
        bps = float(os.environ.get("R3D_COMPACT_EXPECT_BPS", "0") or 0)
    except ValueError:
        bps = 0.0
    if bps <= 0:
        bps = nvenc_target_bps(int(w), int(h), float(fps)) * default_factor
    return total_dur_s * bps / 8.0 > 0.9 * limit


def _compact_plan(total_dur_s: "float | None") -> "tuple[int, int, int, int]":
    """(scale_h, maxrate_bps, audio_bps, fps) for the inline Discord copy
    (`-embed-sm.mp4`). Same plan as the contributor client's compactPlan and
    the bot's _compact_plan at the 56 MiB node budget: 1080p60 on short plays,
    720p60 on longer ones, 720p30 only when the budget is genuinely too small."""
    budget_bits = 56 * 1024 * 1024 * 8
    dur = float(total_dur_s or 0.0)
    if dur <= 1:
        return 1080, 8_000_000, 192_000, 60
    total_rate = int(budget_bits / dur) or 1
    pref = 192_000 if dur <= 240 else (128_000 if dur <= 600 else 96_000)
    audio = min(pref, max(32_000, total_rate // 4))
    maxrate = max(32_000, min(8_000_000, total_rate - audio))
    if maxrate >= 3_000_000:
        return 1080, maxrate, audio, 60
    if maxrate >= 500_000:
        return 720, maxrate, audio, 60
    return 720, maxrate, audio, 30


def _preview_sink_args(preview_path) -> list:
    """Output arguments for the inline preview.

    Default: one faststart mp4 at ``preview_path`` (unchanged).

    LIVE PREVIEW (R3D_PREVIEW_LIVE=1, default OFF): the same encode is written
    as 2 s self-contained fMP4 segments plus a growing playlist in
    ``<out stem>.live/`` (init.mp4, seg_00000.m4s ..., live.m3u8), so the
    contributor client can upload the preview WHILE the render runs and the
    site can play it before the render is done. No ``.embed.mp4`` is written in
    this mode; the client stitches one from the segments (a stream copy). A
    segment is renamed into place only when it is complete, and is listed in
    the playlist only after that."""
    if os.environ.get("R3D_PREVIEW_LIVE") != "1":
        return ["-movflags", "+faststart", str(preview_path)]
    live_dir = str(preview_path)[:-len(".embed.mp4")] + ".live"
    os.makedirs(live_dir, exist_ok=True)
    for _old in os.listdir(live_dir):       # a retry must not show stale segments
        try:
            os.remove(os.path.join(live_dir, _old))
        except OSError:
            pass
    return ["-f", "hls", "-hls_time", "2", "-hls_segment_type", "fmp4",
            "-hls_playlist_type", "event",
            "-hls_flags", "independent_segments+temp_file",
            "-hls_fmp4_init_filename", "init.mp4",
            "-hls_segment_filename", os.path.join(live_dir, "seg_%05d.m4s"),
            os.path.join(live_dir, "live.m3u8")]


def _spawn_ffmpeg(cfg: RenderConfig, output_path: Path, audio: Path | None,
                  start_ms: int, rate: float = 1.0, total_dur_s: float | None = None,
                  hitsound: Path | None = None, is_nc: bool = False,
                  preview_path: "Path | None" = None,
                  stream_master: bool = False,
                  compact_path: "Path | None" = None):
    w, h = cfg.resolution
    enc, dev = _probe_encoder(cfg)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if enc == "h264_vaapi" and dev:
        cmd += ["-vaapi_device", dev]
    # GPU_YUV hands ffmpeg finished yuv420p, so swscale does NO conversion.
    # Read the SAME flag object the readback path uses, not the env var again --
    # they must never disagree about the pixel format or ffmpeg gets mislabelled bytes.
    _pixfmt = "yuv420p" if _GPU_YUV_R else "rgb24"
    cmd += ["-f", "rawvideo", "-pix_fmt", _pixfmt, "-s", f"{w}x{h}", "-r", str(cfg.fps),
            "-i", "pipe:0"]
    # Resolve the music input + its live filter chain. With the loudnorm cache
    # on, a hit feeds ffmpeg the pre-loudnorm'd PCM and drops loudnorm (+ the
    # rate/trim baked into it) from the live chain; a miss builds the cache first;
    # disabled/failure falls back to the source + the full inline chain. All
    # three yield byte-identical muxed audio (loudnorm f64 -> pcm_f64le -> f64).
    music_chain = None
    if audio is not None:
        _pre, _post = _audio_parts(start_ms, rate, total_dur_s,
                                   music_volume=cfg.music_volume,
                                   general_volume=cfg.general_volume,
                                   audio_offset_ms=cfg.audio_offset_ms, is_nc=is_nc)
        audio_input, music_chain = _resolve_music_audio(audio, _pre, _post)
        cmd += ["-i", str(audio_input)]
    if audio is not None and hitsound is not None:
        cmd += ["-i", str(hitsound)]

    # video codec + pixel path (collected in `vc`; appended below)
    vc: list = []
    if enc == "h264_vaapi":
        _vb = str(cfg.video_bitrate) if cfg.video_bitrate else "8M"
        vc += ["-vf", "format=nv12,hwupload", "-c:v", "h264_vaapi", "-b:v", _vb]
    elif enc == "h264_nvenc":
        # Resolution-scaled bitrate ladder (was flat 8M) -- R3D cross-engine
        # NVENC policy; see nvenc_target_bps above.
        # p3 (was p4): measured 247 -> 346 fps consumer-side at 1080p60 on the
        # 2070S -- the raw-frame producer was backpressuring on the encoder.
        # Same bitrate ladder/VBR caps, so quality stays visually equivalent.
        _tgt = cfg.video_bitrate or nvenc_target_bps(w, h, cfg.fps)
        vc += ["-c:v", "h264_nvenc", "-preset", "p3", "-pix_fmt", "yuv420p",
               "-b:v", str(_tgt), "-maxrate", str(int(_tgt * 1.5)),
               "-bufsize", str(_tgt * 2)]
    else:
        # CPU-encode thread cap (R3D host-governance, 2026-09): leave >=2
        # logical cores free for the machine's owner. Uncapped, libx264 spawns
        # threads on EVERY core at normal priority and can freeze a
        # contributor's desktop (the rel/Stella "semi-crash"). Harmless on
        # dedicated render boxes: libx264 is only the no-HW-encoder fallback.
        # Same cap in all four engines (catch/taiko/std/mania v2).
        _thr = ["-threads", _X264_THREADS
                or str(max(2, (os.cpu_count() or 4) - 2))]
        if _X264_PARAMS:
            _thr += ["-x264-params", _X264_PARAMS]
        _preset = _X264_PRESET or "veryfast"
        if cfg.video_bitrate:
            _vb = int(cfg.video_bitrate)
            vc += ["-c:v", "libx264", "-preset", _preset, "-pix_fmt", "yuv420p",
                   "-b:v", str(_vb), "-maxrate", str(int(_vb * 1.5)),
                   "-bufsize", str(_vb * 2)] + _thr
        else:
            vc += ["-c:v", "libx264", "-preset", _preset, "-pix_fmt", "yuv420p",
                   "-crf", _X264_CRF or "20"] + _thr

    # The master's audio graph when a hitsound dub is mixed in (-> [aout]).
    # [1:a] song -> music chain; [2:a] hitsound dub scaled by the preset
    # effects (general) volume AND the -8 LU music-match gain (so the
    # hits drop with the music, not blast on top); amix without
    # auto-normalise so neither side is ducked.
    fc = None
    if audio is not None and hitsound is not None:
        hs_vol = max(0.0, cfg.general_volume / 100.0) * _HITSOUND_MUSIC_MATCH_GAIN
        mc = music_chain if music_chain else "anull"
        fc = ("[1:a]" + mc + "[m];"
              "[2:a]volume=" + f"{hs_vol:.3f}" + "[h];"
              "[m][h]amix=inputs=2:duration=longest:normalize=0:"
              "dropout_transition=0[aout]")
    acodec = ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    # stream_master: no +faststart (it rewrites the whole file at close), and
    # the master is the FINAL file, so it gets the 48 kHz the client's pass
    # would have forced (loudnorm emits 192 kHz).
    _mfast = [] if stream_master else ["-movflags", "+faststart"]
    if stream_master:
        acodec = ["-c:a", "aac", "-ar", "48000", "-b:a", "192k", "-shortest"]

    if preview_path is None:
        cmd += vc
        if audio is not None:
            if fc is not None:
                # Video (0) mapped explicitly.
                if stream_master:
                    cmd += ["-filter_complex",
                            fc + f";[aout]{_STREAM_LOUDNORM}[aoutn]",
                            "-map", "0:v", "-map", "[aoutn]"]
                else:
                    cmd += ["-filter_complex", fc, "-map", "0:v", "-map", "[aout]"]
            elif stream_master:
                cmd += ["-af", (music_chain + "," if music_chain else "")
                        + _STREAM_LOUDNORM]
            elif music_chain:
                cmd += ["-af", music_chain]
            cmd += acodec

        # web-streamable: move the moov atom to the front so browsers/iOS can
        # play before the whole file downloads (loudnorm re-adds this, but be
        # robust if that post-step is skipped/fails).
        cmd += _mfast + [str(output_path)]
    else:
        # TWO OUTPUTS FROM ONE PROCESS. The frame pipe is read once; `split`
        # hands the SAME rgb24 frames to the master encoder (unchanged
        # settings; its rgb24 -> yuv420p conversion is still the encoder-side
        # auto-inserted one, now after the split) and to a 720p30 libx264
        # preview. The audio graph is the master's own, then `asplit`; the
        # preview branch gets the loudness pass the contributor client would
        # otherwise apply before cutting its embed, so the preview needs no
        # post-processing at all. Taiko frames arrive top-down (no vflip on
        # the master), so the preview needs no flip either.
        graph = []
        pfps = min(30, int(round(float(cfg.fps))))
        # hardware preview (render/preview_hw.py; a Mac's media engine, where
        # the probe passes): the first frame repeated in front, cut off again
        # after encoding. "" leaves the graph as it was.
        # Only when the master is on a software encoder: then the preview's is
        # the one hardware session this process holds. A master that is itself
        # on a hardware encoder keeps the preview on x264 (a second session can
        # be refused, and one failed output kills the render).
        preview_hw = str(enc).startswith("lib") and _phw.preview_on_media_engine()
        if preview_hw and audio is not None and not (total_dur_s and total_dur_s > 0):
            preview_hw = False   # no known length to end the audio at: stay on x264
        _lead = ("," + _phw.vt_lead_in_filter(pfps)) if preview_hw else ""
        vm_tail = "null"
        if enc == "h264_vaapi":
            # the master's "-vf format=nv12,hwupload" moves into the graph
            vm_tail = "format=nv12,hwupload"
            vc = [x for i, x in enumerate(vc)
                  if not (x == "-vf" or (i and vc[i - 1] == "-vf"))]
        if compact_path is not None:
            # third branch: the Discord copy, to the compact plan (never
            # upscaled past the master, never above the master's frame rate)
            c_h, c_max, c_abps, c_fps = _compact_plan(total_dur_s)
            c_h = min(c_h, int(h))
            c_fps = min(c_fps, int(round(float(cfg.fps))))
            graph.append(f"[0:v]split=3[vm0][vp0][vc0];[vm0]{vm_tail}[vm];"
                         f"[vp0]fps={pfps},scale=-2:720{_lead}[vp];"
                         f"[vc0]fps={c_fps},scale=-2:{c_h}:flags=bilinear[vc]")
        else:
            graph.append(f"[0:v]split=2[vm0][vp0];[vm0]{vm_tail}[vm];"
                         f"[vp0]fps={pfps},scale=-2:720{_lead}[vp]")
        if audio is not None:
            if fc is not None:
                graph.append(fc)
            else:
                graph.append(f"[1:a]{music_chain or 'anull'}[aout]")
            _ac = "[ac]" if compact_path is not None else ""
            _an = 3 if compact_path is not None else 2
            if stream_master:
                # ONE loudness pass on the shared branch: the master, the
                # preview (and the Discord copy) carry the same normalised audio.
                graph.append(f"[aout]{_STREAM_LOUDNORM},"
                             f"aformat=sample_rates=48000,asplit={_an}[am][ap]{_ac}")
            elif compact_path is not None:
                # the Discord copy is cut from the FINAL (normalised) audio
                graph.append("[aout]asplit=2[am][ap0];"
                             "[ap0]loudnorm=I=-18:TP=-1.5:LRA=11,asplit=2[ap][ac]")
            else:
                graph.append("[aout]asplit=2[am][ap0];"
                             "[ap0]loudnorm=I=-18:TP=-1.5:LRA=11[ap]")
        if _lead and audio is not None:
            # `-shortest` measures the preview's video BEFORE the lead-in is cut
            # off, so it cannot be used on this output (see where it is left out
            # below). The video's length is known here: end the preview's audio
            # there, which is what `-shortest` does for the x264 preview.
            graph[-1] = graph[-1].replace("[ap]", "[ap_full]")
            graph.append(f"[ap_full]atrim=end={float(total_dur_s):.6f}[ap]")
        cmd += ["-filter_complex", ";".join(graph)]
        # output 1: the master, exactly as without the preview
        cmd += ["-map", "[vm]"] + (["-map", "[am]"] if audio is not None else [])
        cmd += vc + (acodec if audio is not None else [])
        cmd += _mfast + [str(output_path)]
        # output 2: the preview. libx264 unless a hardware session was proved
        # to open here (`preview_hw`): a second NVENC/VAAPI session can fail to
        # open (session limits), and one failed output kills the whole process
        # and with it the render. On a Mac the master is on x264, so the
        # preview's is the only hardware session this process holds.
        vbps = _preview_video_bps(total_dur_s)
        cmd += ["-map", "[vp]"] + (["-map", "[ap]"] if audio is not None else [])
        if preview_hw:
            cmd += _phw.hw_video_args(vbps, pfps)
        else:
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                    "-b:v", str(vbps), "-maxrate", str(int(vbps * 1.25)),
                    "-bufsize", str(vbps * 2), "-g", "30",
                    "-threads", str(max(2, min(4, (os.cpu_count() or 4) - 2)))]
        if audio is not None:
            cmd += ["-c:a", "aac", "-b:a", "128k", "-ar", "48000"]
            if not _lead:
                cmd += ["-shortest"]
            # with the lead-in `-shortest` is wrong in both directions: it
            # compares the streams BEFORE the lead-in is cut off, so it either
            # lets extra audio through or, once the audio is ended at the
            # video's length (the atrim above), cuts the last half second of
            # VIDEO. The audio is ended explicitly instead.
        cmd += _preview_sink_args(preview_path)
        if compact_path is not None:
            # output 3: the Discord copy. Same recipe as the node's own compact
            # encode (libx264 veryfast crf 21 + VBV at the plan's maxrate).
            cmd += ["-map", "[vc]"] + (["-map", "[ac]"] if audio is not None else [])
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                    "-crf", "21", "-maxrate", str(c_max),
                    "-bufsize", str(max(1, c_max // 2)), "-g", str(c_fps),
                    "-threads", str(max(2, min(6, (os.cpu_count() or 4) // 2)))]
            if audio is not None:
                cmd += ["-c:a", "aac", "-ar", "48000", "-b:a", str(c_abps), "-shortest"]
            cmd += _mfast + [str(compact_path)]
    import tempfile
    errf = tempfile.NamedTemporaryFile(
        prefix="catch_ffmpeg_", suffix=".log", delete=False, mode="w+",
    )
    # macOS: F_SETPIPE_SZ is Linux-only, so a Mac node pushes every frame through
    # the 64 KiB default pipe — ~95 kernel handoffs for one 6.22 MB RGB24 frame.
    # A unix socketpair CAN be grown (SO_SNDBUF/SO_RCVBUF). Catch measured this on
    # the same machine and the same ffmpeg line: pipe 256 fps -> socketpair
    # 406 fps, against a 419 fps file-fed ceiling. It matters far more now than it
    # did at 150 fps: with the composite chain on the GPU, `push` was measured at
    # 3.9 ms/frame — the pipe, not the renderer, had become the wall.
    # Bytes on the wire are unchanged, so output is byte-identical.
    if _sw.SOCKET_PIPE:
        import socket as _sock
        _par, _chi = _sock.socketpair(_sock.AF_UNIX, _sock.SOCK_STREAM)
        for _s, _opt in ((_par, _sock.SO_SNDBUF), (_chi, _sock.SO_RCVBUF)):
            try:
                _s.setsockopt(_sock.SOL_SOCKET, _opt, 1 << 20)
            except OSError:
                pass              # keep the default buffer; still correct
        proc = subprocess.Popen(cmd, stdin=_chi.fileno(), stderr=errf,
                                stdout=subprocess.DEVNULL, bufsize=0)
        _chi.close()

        class _SockStdin:
            """File-like shim over the socket. `sendall` is deliberate: a raw
            SocketIO.write may write PARTIALLY and return a short count, and
            _FrameWriter ignores write()'s return value — that would silently
            truncate a frame."""

            def __init__(self, sk):
                self._sk = sk

            def write(self, b):
                self._sk.sendall(b)
                return len(b)

            def flush(self):
                pass

            def fileno(self):
                return self._sk.fileno()

            def close(self):
                try:
                    self._sk.shutdown(_sock.SHUT_WR)
                except OSError:
                    pass
                self._sk.close()

        proc.stdin = _SockStdin(_par)
        proc._catch_errlog = errf.name  # type: ignore[attr-defined]
        return proc
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errf,
                            stdout=subprocess.DEVNULL, bufsize=0)
    try:
        import fcntl
        fcntl.fcntl(proc.stdin.fileno(), 1031, 1 << 20)   # F_SETPIPE_SZ (Linux)
    except (OSError, ImportError, AttributeError):
        pass                      # not Linux, or not permitted — default size
    proc._catch_errlog = errf.name  # type: ignore[attr-defined]
    return proc


# Single-pass loudnorm baseline (EBU R128) applied to every render's music so
# hot beatmap masters stop blasting. I=-18 LUFS: Red found every render WAY too
# loud (listens at ~5% volume); ALL four engines (std/catch/taiko/mania) share
# this one quieter integrated-loudness target so the whole site sounds uniform.
# Kept as one constant so the loudnorm cache key (which must capture everything
# determining the loudnorm OUTPUT) tracks any change to it automatically: the
# key hashes `pre` (which includes this string), so lowering the target
# auto-invalidates every cached PCM and rebuilds it — no stale (louder) audio.
_LOUDNORM_FILTER = "loudnorm=I=-18:TP=-1.5:LRA=11"

# Per-note hitsound track is amixed on TOP of the loudnorm'd music (normalize=0),
# so lowering only the music (I=-10 -> -18, an 8 LU drop) would leave the taiko
# drum hits blasting relative to the now-quieter song. Scale the hit track by the
# same 8 LU: 10^(-8/20) = 0.398 ~= 0.40, matching the std/catch/mania engines so
# the hit:music balance — and the overall render loudness — is uniform. Kept as
# its own constant + separate commit so it can be reverted to music-only easily.
_HITSOUND_MUSIC_MATCH_GAIN = 0.40


def _audio_parts(start_ms: int, rate: float = 1.0, total_dur_s: float | None = None,
                 music_volume: int = 100, general_volume: int = 100,
                 audio_offset_ms: int = 0, is_nc: bool = False) -> tuple[str, str]:
    """Build the music audio chain SPLIT at loudnorm -> (pre, post).

    pre  = rate/pitch (DT/HT atempo, NC resample), then align to `start_ms`
           (atrim/adelay), then loudnorm. This is the OUTPUT-DETERMINING,
           cacheable part: given the same source audio it is a pure function of
           this string, so it is what the loudnorm cache is keyed on and bakes.
    post = preset volume + apad (silence pad to the full video duration). These
           depend on per-render config / video length, so they always run live
           on top of the (possibly cached) loudnorm output.

    Joining pre + "," + post reproduces the previous single -af chain
    byte-for-byte (order: rate, trim, asetpts, loudnorm, volume, apad), so the
    no-cache / kill-switch path is unchanged."""
    pre = []
    if abs(rate - 1.0) > 1e-3:
        if is_nc:
            # Nightcore = a PURE RESAMPLE: speed AND pitch up together by the
            # rate, exactly like osu (and the mania v2 renderer). Reinterpreting
            # the samples at SR*rate then resampling back to SR is artifact-free.
            # The old atempo time-stretch + rubberband pitch-shift each added
            # audible smear vs in-game (reported distorted NC audio). Normalise to
            # 44100 first so a 48 kHz master still speeds by exactly `rate` — the
            # asetrate value is absolute, so the input rate must be known.
            pre.append("aresample=44100")
            pre.append(f"asetrate={int(round(44100 * rate))}")
            pre.append("aresample=44100")
        else:
            # Plain DT/HT: pitch-PRESERVING time-stretch (that IS DT/HT).
            pre.append(f"atempo={rate:.4f}")  # speed
    # start_ms is in MAP time; after the speed-up the song plays at map/rate, so the
    # real offset where video t=0 lands is start_ms/rate. audio_offset shifts the
    # song vs gameplay (negative = audio earlier).
    real_start = (start_ms - audio_offset_ms) / rate
    if real_start > 0:
        pre.append(f"atrim=start={real_start / 1000:.3f}")
        pre.append("asetpts=PTS-STARTPTS")
    elif real_start < 0:
        pre.append(f"adelay={int(-real_start)}:all=1")
    # Loudness-normalise to a consistent EBU R128 baseline (single-pass). loudnorm
    # runs here (after the trim) so its output is exactly what the volume trim
    # below is relative to. The cache boundary is right after this filter.
    pre.append(_LOUDNORM_FILTER)
    post = []
    vol = (general_volume / 100.0) * (music_volume / 100.0)
    if abs(vol - 1.0) > 1e-3:
        post.append(f"volume={max(0.0, vol):.3f}")
    # Pad with silence so the audio spans the full video (incl. the results
    # outro past the song's end). Bound the pad to the exact video duration —
    # an UNBOUNDED apad races the (slow) raw-video pipe and overflows the
    # filtergraph buffer (ffmpeg reports it as ENOSPC and dies).
    if total_dur_s and total_dur_s > 0:
        post.append(f"apad=whole_dur={total_dur_s:.3f}")
    else:
        post.append("apad")
    return ",".join(pre), ",".join(post)


def _audio_filter(start_ms: int, rate: float = 1.0, total_dur_s: float | None = None,
                  music_volume: int = 100, general_volume: int = 100,
                  audio_offset_ms: int = 0, is_nc: bool = False) -> str:
    """The full single-pass music chain (rate/trim/loudnorm/volume/apad) as one
    -af string — the no-cache path. Byte-identical to the previous impl."""
    pre, post = _audio_parts(start_ms, rate, total_dur_s,
                             music_volume=music_volume, general_volume=general_volume,
                             audio_offset_ms=audio_offset_ms, is_nc=is_nc)
    return pre + "," + post if post else pre


# --- loudnorm PCM cache -------------------------------------------------------
# The loudnorm pass reruns every render (~2-3 s) even for the same map. Cache its
# output PCM keyed on everything that determines it: the SOURCE audio bytes + the
# exact `pre` chain (rate/pitch mode + start alignment + loudnorm params). Stored
# as pcm_f64le WAV — loudnorm outputs f64 internally, so re-reading f64 and
# running the downstream volume/apad/amix + AAC yields BYTE-IDENTICAL audio to
# the inline chain (verified); f32/int would drift. Cache hit skips loudnorm.
_LOUDNORM_CACHE_DIR = os.environ.get(
    "R3D_TAIKO_LOUDNORM_CACHE_DIR", "/data/r3d/loudnorm-cache")


def _loudnorm_cache_enabled() -> bool:
    """Env kill-switch. R3D_TAIKO_LOUDNORM_CACHE=0 (or false/no/off) disables the
    cache entirely, restoring the inline loudnorm path."""
    return os.environ.get("R3D_TAIKO_LOUDNORM_CACHE", "1").strip().lower() \
        not in ("0", "false", "no", "off")


def _loudnorm_cache_key(audio_path: Path, pre: str) -> str:
    """sha256 of the source audio bytes + the exact loudnorm-determining chain
    (`pre`). Stable across re-downloads (hashes bytes, not mtime)."""
    h = hashlib.sha256()
    h.update(pre.encode("utf-8"))
    h.update(b"\x00")
    with open(audio_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _loudnorm_cache_valid(path: Path) -> bool:
    """A present cache entry is complete (writes are atomic via os.replace), but
    guard against truncation/corruption cheaply: RIFF/WAVE header + non-trivial
    size. Invalid -> treated as a miss (recompute)."""
    try:
        if os.path.getsize(path) < 1024:
            return False
        with open(path, "rb") as f:
            head = f.read(12)
        return head[:4] == b"RIFF" and head[8:12] == b"WAVE"
    except OSError:
        return False


def _build_loudnorm_cache(source: Path, pre: str, cache_path: Path) -> bool:
    """Run the `pre` chain (rate/pitch, trim, loudnorm) on `source` and write the
    result to `cache_path` as pcm_f64le WAV, atomically (temp + os.replace).
    Returns True on success; False (fail-soft) -> caller falls back to inline
    loudnorm."""
    import tempfile
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".ln_", suffix=".wav",
                                   dir=str(cache_path.parent))
        os.close(fd)
    except OSError as e:
        print(f"[taiko-renderer] loudnorm cache dir unusable ({e}); inline loudnorm",
              file=sys.stderr)
        return False
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(source), "-af", pre, "-c:a", "pcm_f64le", "-f", "wav", tmp],
            check=True, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if os.path.getsize(tmp) < 1024:
            raise RuntimeError("empty loudnorm output")
        os.replace(tmp, cache_path)
        return True
    except Exception as e:  # noqa: BLE001 — never let caching break a render
        print(f"[taiko-renderer] loudnorm cache build failed ({e}); inline loudnorm",
              file=sys.stderr)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return False


# --- loudness by ONE fixed gain (R3D_TAIKO_FIXED_GAIN=1 / R3D_FIXED_GAIN=1) ----
# Default OFF. The one-pass loudnorm filter is the whole cost of building a
# song's cache entry: 17.5 s for a 482 s song that takes 0.5 s to decode (it
# works at 192 kHz, on one thread). With the switch the build instead measures
# the integrated loudness (ffmpeg `ebur128`) at the point loudnorm stood, and
# applies ONE gain for the whole track to the same -18 LUFS, held back where it
# would put a sample above the same -1.5 dB: about a second. NOT the same
# sound: one-pass loudnorm moves its gain as the track goes (it lifts quiet
# passages); a fixed gain leaves the track's own dynamics alone. Hence a switch.
# Same numbers as the std / catch / mania engines. The entry is the same kind
# of file (pcm_f64le WAV) under its own key; everything downstream is untouched.
_FG_TARGET_LUFS = -18.0
_FG_PEAK_CEILING_DB = -1.5
_FG_EBUR128 = "ebur128=framelog=quiet"
_FG_PARAM = f"fixedgain:I={_FG_TARGET_LUFS:g}:P={_FG_PEAK_CEILING_DB:g}"
_FG_NOTHING_LUFS = -70.0   # what ebur128 reports when nothing passed its gate


def _fg_parse_lufs(stderr_text: str) -> "float | None":
    """The integrated loudness out of ffmpeg's `ebur128` summary; None when
    there is no summary to read. Silence reads -70.0."""
    import re
    m = re.findall(r"^\s*I:\s+(-?\d+(?:\.\d+)?) LUFS", stderr_text, re.M)
    return float(m[-1]) if m else None


def _fg_gain_db(integrated_lufs: "float | None", peak: float) -> float:
    """dB for the whole track: up or down to the target, never so far up that
    the loudest sample (`peak`, linear) passes the ceiling. Silence is left as
    it is."""
    import math
    if integrated_lufs is None or integrated_lufs <= _FG_NOTHING_LUFS:
        return 0.0
    gain = _FG_TARGET_LUFS - integrated_lufs
    if peak > 0.0:
        gain = min(gain, _FG_PEAK_CEILING_DB - 20.0 * math.log10(peak))
    return gain


_FG_PIN_192K = "aformat=sample_rates=192000"


def _fg_pre(pre: str) -> "str | None":
    """`pre` with the measurement where its loudnorm stood (it is always the
    last filter of `pre`); None if this is not a chain _audio_parts built.

    loudnorm only takes 192 kHz and hands 192 kHz on. The same rate is pinned
    in its place, for two reasons: (1) with it ffmpeg resamples BEFORE an
    `atempo` (DT/HT), so the time-stretch is exactly stock's (measured: same
    length to the sample, 0.999 correlation at lag 0; without the pin a
    different stretch, 0.37); (2) the cache entry keeps stock's sample rate, so
    everything downstream of it, the master's audio format included, is what
    it is with the stock entry."""
    if not pre.endswith(_LOUDNORM_FILTER):
        return None
    return pre[:-len(_LOUDNORM_FILTER)] + _FG_PIN_192K + "," + _FG_EBUR128


def _wav_data_span(path) -> "tuple[int, int]":
    """(offset, byte length) of the samples in a RIFF/WAVE file."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(12)
        if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise ValueError("not a wav")
        pos = 12
        while pos + 8 <= size:
            f.seek(pos)
            cid, clen = f.read(4), int.from_bytes(f.read(4), "little")
            if cid == b"data":
                return pos + 8, min(clen, size - pos - 8)
            pos += 8 + clen + (clen & 1)
    raise ValueError("no data chunk")


def _build_fixed_gain_cache(source: Path, pre: str, cache_path: Path) -> bool:
    """The fixed-gain build of a cache entry: run `pre` with the loudness
    measured in place of loudnorm, apply the one gain to the samples, publish
    atomically. False (fail-soft) when ffmpeg failed or printed no summary that
    can be read: the caller then builds the stock entry."""
    import tempfile
    pre_m = _fg_pre(pre)
    if pre_m is None:
        return False
    tmp = None
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".ln_", suffix=".wav",
                                   dir=str(cache_path.parent))
        os.close(fd)
        # `info` is the level ebur128 prints its summary at
        r = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-nostats", "-loglevel", "info",
             "-i", str(source), "-af", pre_m, "-c:a", "pcm_f64le", "-f", "wav", tmp],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE)
        lufs = _fg_parse_lufs((r.stderr or b"").decode(errors="replace"))
        if r.returncode != 0 or os.path.getsize(tmp) < 1024 or lufs is None:
            raise RuntimeError("no measurement")
        off, nbytes = _wav_data_span(tmp)
        pcm = np.memmap(tmp, dtype="<f8", mode="r+", offset=off,
                        shape=(nbytes // 8,))
        gain = _fg_gain_db(lufs, float(np.abs(pcm).max()) if pcm.size else 0.0)
        pcm *= 10.0 ** (gain / 20.0)
        pcm.flush()
        del pcm
        os.replace(tmp, cache_path)
        print("[taiko-renderer] song loudness: "
              + ("silent, left as it is" if lufs <= _FG_NOTHING_LUFS else
                 f"{lufs:.1f} LUFS, one gain of {gain:+.1f} dB"),
              file=sys.stderr, flush=True)
        return True
    except Exception as e:  # noqa: BLE001 — never let caching break a render
        print(f"[taiko-renderer] fixed-gain cache build failed ({e}); "
              f"the loudnorm filter", file=sys.stderr)
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return False


def _resolve_music_audio(audio: Path, pre: str, post: str,
                         fixed_gain: "bool | None" = None):
    """Resolve the music input + its live filter chain, using the loudnorm cache
    when enabled. Returns (audio_input, music_chain):
      * cache hit  -> (cache_wav, post)         loudnorm SKIPPED (baked in cache)
      * cache miss -> build cache, then as hit; on any failure fall back to
      * disabled / fallback -> (source, pre+','+post)   inline loudnorm

    music_chain is what runs live on the resolved input (post-only when the
    cache supplies the loudnorm'd PCM, else the full chain).
    `fixed_gain` None = follow the switch (render/envflag.py FIXED_GAIN)."""
    full = pre + "," + post if post else pre
    if not _loudnorm_cache_enabled():
        return audio, full
    if _sw.FIXED_GAIN if fixed_gain is None else fixed_gain:
        # one measured gain in place of the loudnorm pass (see above); if that
        # entry cannot be built here, the stock one below
        try:
            fg_path = Path(_LOUDNORM_CACHE_DIR) / (
                _loudnorm_cache_key(audio, (_fg_pre(pre) or pre) + "|" + _FG_PARAM)
                + ".wav")
            if _loudnorm_cache_valid(fg_path) \
                    or _build_fixed_gain_cache(audio, pre, fg_path):
                return fg_path, (post if post else "anull")
        except OSError:
            pass
    try:
        key = _loudnorm_cache_key(audio, pre)
    except OSError as e:
        print(f"[taiko-renderer] loudnorm cache key failed ({e}); inline loudnorm",
              file=sys.stderr)
        return audio, full
    cache_path = Path(_LOUDNORM_CACHE_DIR) / f"{key}.wav"
    if not _loudnorm_cache_valid(cache_path):
        if not _build_loudnorm_cache(audio, pre, cache_path):
            return audio, full
    return cache_path, (post if post else "anull")


def _bg_cover(path: Path, w: int, h: int, blur: int = 0) -> "np.ndarray | None":
    """Load the beatmap background and cover-crop it to WxH (no distortion).
    Returns None if the (user-supplied) background can't be decoded, so the
    caller skips the bg upload -- same as a map with no background."""
    try:
        im = Image.open(path).convert("RGB")
    except Exception as e:  # noqa: BLE001 -- a corrupt user bg must not crash the render
        log.warning("background image failed to decode, skipping: %s (%s)", path, e)
        return None
    scale = max(w / im.width, h / im.height)
    nw, nh = max(1, int(im.width * scale)), max(1, int(im.height * scale))
    im = im.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - w) // 2, (nh - h) // 2
    im = im.crop((left, top, left + w, top + h))
    if blur and blur > 0:
        from PIL import ImageFilter
        im = im.filter(ImageFilter.GaussianBlur(radius=float(blur)))
    return np.array(im)


def _find_osu(beatmap_dir: Path, md5: str) -> Path:
    osus = sorted(beatmap_dir.glob("*.osu"))
    if not osus:
        raise TaikoRenderError(f"no .osu in {beatmap_dir}")
    if md5:
        for p in osus:
            if hashlib.md5(p.read_bytes()).hexdigest() == md5:
                return p
    # DMCA/mirror-down recovery: the bot's manual-.osz upload path writes a
    # ".r3d_forced_osu" marker naming the difficulty it matched when the
    # replay's exact md5 is not in the archive (a pack shipping a different
    # version, or an unsubmitted map). Honour it before the mode/first
    # fallback so we render THAT diff -- and resolve ITS audio/bg -- instead
    # of the first same-mode one (which desyncs or renders silently).
    _forced_marker = beatmap_dir / ".r3d_forced_osu"
    if _forced_marker.is_file():
        try:
            _forced = beatmap_dir / pathlib.Path(
                _forced_marker.read_text(encoding="utf-8").strip()
            ).name
        except OSError:
            _forced = None
        if _forced is not None and _forced.is_file() \
                and _forced.suffix.lower() == ".osu":
            return _forced
    # fall back to a Mode:1 (taiko) beatmap, else the first
    for p in osus:
        head = p.read_text(encoding="utf-8", errors="replace")[:4000]
        if "Mode: 1" in head or "Mode:1" in head:
            return p
    return osus[0]


# --- HUD ----------------------------------------------------------------------

class _Hud:
    def __init__(self, w, h, meta, bm):
        self.w, self.h = w, h
        self.meta = meta
        self.bm = bm
        big = max(20, int(h * 0.07))
        med = max(16, int(h * 0.035))
        small = max(12, int(h * 0.025))
        self.f_combo = _font(big)
        self.f_score = _font(med)
        self.f_small = _font(small)

    def overlay(self, rgb: np.ndarray, scene) -> np.ndarray:
        img = Image.fromarray(rgb, "RGB")
        d = ImageDraw.Draw(img)
        # combo bottom-left
        if scene.combo > 0:
            d.text((int(self.w * 0.02), int(self.h * 0.86)), f"{scene.combo}x",
                   font=self.f_combo, fill=(255, 255, 255))
        # score top-right
        d.text((int(self.w * 0.98), int(self.h * 0.03)), f"{scene.score:,}",
               font=self.f_score, fill=(255, 255, 255), anchor="ra")
        # player + title top-left
        d.text((int(self.w * 0.02), int(self.h * 0.03)), self.meta.player_name,
               font=self.f_small, fill=(230, 230, 240))
        title = f"{self.bm.artist} - {self.bm.title} [{self.bm.version}]".strip(" -")
        d.text((int(self.w * 0.02), int(self.h * 0.065)), title,
               font=self.f_small, fill=(180, 180, 200))
        # hp bar top center
        bx, by, bw, bh = int(self.w * 0.30), int(self.h * 0.02), int(self.w * 0.40), 10
        d.rectangle([bx, by, bx + bw, by + bh], fill=(40, 40, 50))
        d.rectangle([bx, by, bx + int(bw * scene.hp), by + bh], fill=(120, 220, 140))
        return np.asarray(img)


from osu_taiko_renderer.skin.fonts import font as _font  # skin-aware, host-robust font resolver
