"""Phase 1 orchestrator: parse -> simulate -> per-frame GL draw + HUD -> ffmpeg.

Owns a small ffmpeg subprocess (raw rgb24 on stdin) so it stays decoupled
from osu_renderer's encode FIFO machinery. HUD text is composited on the CPU
with PIL after GL readback — cheap and avoids a GL text pass for Phase 1.
"""
from __future__ import annotations

import hashlib
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
from osu_taiko_renderer.beatmap.models import RenderConfig
from osu_taiko_renderer.beatmap.replay import parse_replay
from osu_taiko_renderer.render.scene import TaikoSim


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
    frames, meta = parse_replay(osr_path)
    osu_path = _find_osu(beatmap_dir, meta.beatmap_md5)
    bm = parse_beatmap(osu_path, mods=meta.mods)
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
            print(f"[taiko-renderer] score fidelity FAILED (header kept): "
                  f"{_sf_e}", file=sys.stderr, flush=True)
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
    from osu_taiko_renderer.skin.assets import bake_logo_tile, logo_glow_rgba
    renderer.upload_texture("logo_tile", bake_logo_tile())
    renderer.upload_texture("logo_glow", logo_glow_rgba())
    if bg is not None:
        renderer.upload_texture("bg", _bg_cover(bg, w, h, cfg.bg_blur))

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
    proc = _spawn_ffmpeg(cfg, output_path, audio, start_ms, rate, total_dur_s,
                         hitsound=hitsound, is_nc=is_nc)
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
    flashlight = TaikoFlashlight(
        sim.geo, getattr(sim, "_rt", None), getattr(sim, "_cum", None),
        int(getattr(meta, "mods", 0) or 0))
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

        def _prebuild_results() -> None:
            try:
                from osu_taiko_renderer.hud.lazer_results import CatchLazerResults
                scr = CatchLazerResults((w, h), meta, bm, board=baked_board,
                                        osu_path=osu_path, sim=sim)
                _lazer_results_cache["pre"] = scr
                _pre_evt.set()           # publish the instance first…
                sched = []
                for _i in range(gameplay_frames, n_frames):
                    _t = int(gameplay_end_ms + (_i - gameplay_frames) * frame_ms)
                    if _t >= results_start_ms:
                        _op = min(1.0, (_t - results_start_ms) / FADE_MS)
                        sched.append((_op, float(_t - results_start_ms)))
                scr.prebake_anim(sched)  # …then keep baking in the background
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
    _t_render0 = time.monotonic()
    writer = _FrameWriter(proc)
    pending = deque()   # (scene, exps, judges, drum_flashes) awaiting pixels

    def _emit_gameplay(raw):
        nonlocal last_gameplay
        p_scene, p_exps, p_judges, p_drums = pending.popleft()
        out = effects.composite(raw, p_exps, p_judges, p_drums)
        out = flashlight.composite(out, p_scene.time_ms)
        out = hud.overlay(out, p_scene)
        # lazer z-order: BreakOverlay is a LATER overlay-component child
        # than HUDOverlay (Player.createOverlayComponents) — composited
        # ABOVE every HUD element, both HUD variants. Live accuracy from
        # the sim's running scene value (bound like lazer's bindable).
        # Cheap no-op outside break windows (frame bytes untouched).
        break_overlay.draw(out, p_scene.time_ms, p_scene.accuracy)
        last_gameplay = out
        writer.push(out)

    try:
        try:
            for i in range(n_frames):
                if i < gameplay_frames:
                    t = int(start_ms + i * map_step)
                    scene = sim.build_scene(t)
                    renderer.begin()
                    renderer.draw(scene.sprites)
                    exps, judges = sim.active_effects(t)
                    pending.append((scene, exps, judges, sim.drum_flashes(t)))
                    raw = renderer.read_rgb_async()
                    if raw is not None:
                        _emit_gameplay(raw)
                else:
                    # gameplay -> outro boundary: flush the PBO ring first so
                    # last_gameplay is the true final gameplay frame and
                    # ordering is preserved across the boundary.
                    for raw in renderer.read_drain():
                        _emit_gameplay(raw)
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
                        rgb = _draw_lazer_results(
                            _lazer_results_cache, rgb, meta, bm, op,
                            age_ms=float(t - results_start_ms),
                            board=baked_board, osu_path=osu_path, sim=sim)
                    writer.push(rgb)
                if progress_callback and i % cfg.fps == 0:
                    progress_callback(int(i / n_frames * 100))
            # map end with no outro configured: flush the ring tail.
            for raw in renderer.read_drain():
                _emit_gameplay(raw)
        except BrokenPipeError:
            pass               # ffmpeg died — surfaced via ret below
    finally:
        writer.close()
        if proc.stdin:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
        ret = proc.wait()
        renderer.release()
        import sys as _rsys
        _wall = time.monotonic() - _t_render0
        print(f"done: {n_frames} frames in {_wall:.1f}s "
              f"({(n_frames / _wall) if _wall else 0.0:.1f} fps) ret={ret}",
              file=_rsys.stderr, flush=True)

    if ret != 0:
        tail = ""
        errlog = getattr(proc, "_catch_errlog", None)
        if errlog and Path(errlog).exists():
            tail = Path(errlog).read_text(errors="replace")[-800:]
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
            sidecar.write_text(_json.dumps(
                {"schema": 1, "mode": 1, **score_fid}, default=str))
        except Exception as _sc_e:  # noqa: BLE001 — sidecar is best-effort
            print(f"[taiko-renderer] score sidecar write failed: {_sc_e}",
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


def _spawn_ffmpeg(cfg: RenderConfig, output_path: Path, audio: Path | None,
                  start_ms: int, rate: float = 1.0, total_dur_s: float | None = None,
                  hitsound: Path | None = None, is_nc: bool = False):
    w, h = cfg.resolution
    enc, dev = _probe_encoder(cfg)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if enc == "h264_vaapi" and dev:
        cmd += ["-vaapi_device", dev]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(cfg.fps),
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

    # video codec + pixel path
    if enc == "h264_vaapi":
        cmd += ["-vf", "format=nv12,hwupload", "-c:v", "h264_vaapi", "-b:v", "8M"]
    elif enc == "h264_nvenc":
        # Resolution-scaled bitrate ladder (was flat 8M) -- R3D cross-engine
        # NVENC policy; see nvenc_target_bps above.
        # p3 (was p4): measured 247 -> 346 fps consumer-side at 1080p60 on the
        # 2070S -- the raw-frame producer was backpressuring on the encoder.
        # Same bitrate ladder/VBR caps, so quality stays visually equivalent.
        _tgt = nvenc_target_bps(w, h, cfg.fps)
        cmd += ["-c:v", "h264_nvenc", "-preset", "p3", "-pix_fmt", "yuv420p",
                "-b:v", str(_tgt), "-maxrate", str(int(_tgt * 1.5)),
                "-bufsize", str(_tgt * 2)]
    else:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-crf", "20"]

    if audio is not None:
        if hitsound is not None:
            # [1:a] song -> music chain; [2:a] hitsound dub scaled by the preset
            # effects (general) volume; amix without auto-normalise so neither
            # side is ducked. Video (input 0) mapped explicitly.
            hs_vol = max(0.0, cfg.general_volume / 100.0)
            mc = music_chain if music_chain else "anull"
            fc = ("[1:a]" + mc + "[m];"
                  "[2:a]volume=" + f"{hs_vol:.3f}" + "[h];"
                  "[m][h]amix=inputs=2:duration=longest:normalize=0:"
                  "dropout_transition=0[aout]")
            cmd += ["-filter_complex", fc, "-map", "0:v", "-map", "[aout]"]
        elif music_chain:
            cmd += ["-af", music_chain]
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]

    # web-streamable: move the moov atom to the front so browsers/iOS can
    # play before the whole file downloads (loudnorm re-adds this, but be
    # robust if that post-step is skipped/fails).
    cmd += ["-movflags", "+faststart", str(output_path)]
    import tempfile
    errf = tempfile.NamedTemporaryFile(
        prefix="catch_ffmpeg_", suffix=".log", delete=False, mode="w+",
    )
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errf, bufsize=0)
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


def _resolve_music_audio(audio: Path, pre: str, post: str):
    """Resolve the music input + its live filter chain, using the loudnorm cache
    when enabled. Returns (audio_input, music_chain):
      * cache hit  -> (cache_wav, post)         loudnorm SKIPPED (baked in cache)
      * cache miss -> build cache, then as hit; on any failure fall back to
      * disabled / fallback -> (source, pre+','+post)   inline loudnorm

    music_chain is what runs live on the resolved input (post-only when the
    cache supplies the loudnorm'd PCM, else the full chain)."""
    full = pre + "," + post if post else pre
    if not _loudnorm_cache_enabled():
        return audio, full
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


def _bg_cover(path: Path, w: int, h: int, blur: int = 0) -> "np.ndarray":
    """Load the beatmap background and cover-crop it to WxH (no distortion)."""
    im = Image.open(path).convert("RGB")
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
