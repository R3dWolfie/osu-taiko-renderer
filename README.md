# osu-taiko-renderer

An in-house osu!taiko replay → MP4 renderer. Parses a `.osr` replay against its beatmap, simulates and judges the play, then draws the right-to-left scrolling taiko playfield frame-by-frame and encodes it to video with ffmpeg.

## Part of the R3D Renderer

This is the **taiko engine** for [R3D Renderer](https://renderer.r3dwolfie.com), a self-hosted osu! replay→MP4 service (Discord bot + website, package `mania_ordr`) that dispatches each render to a per-mode engine repo. The core launches this engine as a fresh subprocess per render, so an edit-in-place deploys on the next render with no service restart. In production this engine runs on the **catch** engine's Python venv (shared interpreter; the code is its own repo).

> Note: parts of this repo were forked from the catch renderer — some module docstrings (e.g. `osu_taiko_renderer/__init__.py`, `render/gl.py`) still say "catch". The runtime code is taiko-specific. (verify: docstrings may be cleaned up later.)

## What it renders / fidelity

- **Taiko objects**: don (red centre), kat (blue rim), big/finish notes (both keys), drumrolls (+ ticks), and swells/denden (alternating-hit shakers). Four inputs are modelled: centre-left/right and rim-left/right.
- **Faithful in-order sim + judgment** (`render/scene.py`): each key-press consumes the frontmost still-hittable same-colour note inside the OD hit window (notelock, no press-stealing); a note whose window passes unhit is a MISS. This lands combo breaks on the notes the player actually missed.
- **Header reconcile** (`beatmap/score_fidelity.py`): the `.osr` header stays the count-authority — a reconcile pass snaps GREAT/OK/MISS totals and max-combo to the header, so counts/accuracy/score are header-exact while *which* notes broke combo is corrected from the sim.
- **Argon (lazer) HUD** by default (`argon/` — compositor, HUD, counters, procedural glyphs). No osu! game assets are bundled; skinless output uses procedurally-generated Argon art and the bundled Nunito font. A user skin can be supplied via `--skin`.
- **pp / SR**: live pp counter and star rating estimated via `rosu_pp_py`; `--pp` / `--sr` pin the results card (and pp-counter endpoint) to osu!'s official values while keeping the live curve's rosu shape.
- **Storyboards**: optional in-house storyboard engine (`--storyboard`, default off; parity with the std engine). When off, output is byte-identical to before.
- **Extras**: results-screen outro with a per-map render leaderboard (local render DB, or osu! global top scores via `--leaderboard-source osu`), background dim/blur, break letterboxing, beatmap/miss/nightcore hitsounds, watermark, and a per-frame score sidecar (`--score-json`) for the live-overlay compositor.

## Usage

Invoked as a module:

```
python -m osu_taiko_renderer REPLAY.osr BEATMAP_DIR -o out.mp4 \
    [--resolution 1920x1080] [--fps 60] [--encoder auto] [--skin DIR]
```

- `REPLAY.osr` — the replay file (positional).
- `BEATMAP_DIR` — directory containing the `.osu` + audio + background (positional).
- `-o, --output` — output path (**required**).

Selected options (all verified from `osu_taiko_renderer/cli.py`):

| Flag | Default | Purpose |
| --- | --- | --- |
| `--resolution WxH` | `1920x1080` | output resolution |
| `--fps N` | `60` | frame rate |
| `--encoder` | `auto` | `auto \| h264_vaapi \| h264_nvenc \| libx264` |
| `--encoder-device` | none | e.g. `/dev/dri/renderD128` |
| `--skin DIR` / `--default-skin DIR` | none | extracted skin dir / fallback |
| `--scroll-time MS` | `1600` | ms a 1.0×-SV note is visible (lower = faster) |
| `--pp FLOAT` / `--sr FLOAT` | none | pin official pp / star rating on the results card |
| `--storyboard` | off | render the map's storyboard (in-house engine) |
| `--[no-]results`, `--[no-]pp-counter`, `--[no-]hit-counter`, `--[no-]leaderboard` | on | HUD/outro toggles |
| `--leaderboard-source {r3d,osu}` | `r3d` | results flank-card source (`osu` reads `--leaderboard-json`) |
| `--watermark`, `--music-volume`, `--general-volume`, `--audio-offset`, `--bg-dim-*`, `--bg-blur` | — | audio/background tuning |

Boolean toggles use argparse `BooleanOptionalAction` (`--flag` / `--no-flag`). Unknown flags are accepted and ignored, so the shared render pipeline can pass mode-agnostic flags it also sends to other engines.

Relevant environment variables: `R3D_EGL_DEVICE_INDEX` (pin the EGL/GPU device for pool isolation), `R3D_FRAME_MD5` (hash raw frames for bit-identical-output proofs).

## Requirements

- **Python 3.10+** (uses `X | Y` type unions and `BooleanOptionalAction`).
- Third-party packages (imported directly; no `requirements.txt`/`pyproject.toml` is committed — deps come from the shared venv):
  - `numpy`
  - `Pillow` (PIL — CPU HUD compositing)
  - `moderngl` (standalone **EGL** context / offscreen RGBA framebuffer — headless GPU)
  - `osrparse` (replay parsing)
  - `rosu_pp_py` (pp / star-rating; optional — features degrade if absent)
  - `osu_renderer` — the shared R3D core package, reused for its ffmpeg encode path and `wiki_renderer.SkinPair` skinning
- **ffmpeg** on `PATH` (the renderer owns its own ffmpeg subprocess, raw rgb24 on stdin).

## Layout

```
osu_taiko_renderer/
  __main__.py, cli.py       CLI entry point (argparse) → render_taiko()
  beatmap/                  parsing + sim models
    beatmap.py, models.py, replay.py, sliderpath.py,
    storyboard.py, score_fidelity.py, legacy_random.py
  render/                   orchestration + GL + encode
    render.py               parse → simulate → per-frame GL draw + HUD → ffmpeg
    scene.py                taiko sim + per-frame scene builder
    gl.py                   moderngl/EGL sprite batch
    effects.py, flashlight.py, dim.py, hitsounds.py,
    storyboard_engine.py, storyboard_render.py
  argon/                    procedural Argon (lazer) HUD — compositor, hud,
                            counter, geometry, textures, font, glyphs/
  hud/                      HUD + results: hud.py, lazer_results.py,
                            leaderboard.py, lb_cards.py, break_overlay.py, …
  skin/                     skin loading: taiko_skin.py, lazer_skin.py,
                            assets.py, fonts.py
  assets/                   logo, Nunito font (OFL), default nightcore hitsounds
  fonts_data/               Torus font atlases
tests/                      storyboard + lead-in tests
```

## License

**GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)** — see `LICENSE`. Copyright (C) 2026 Cool Adults (see `COPYRIGHT`).

Attribution (per `COPYRIGHT`): gameplay, timing, judgment, scoring and HUD logic are behavioural ports from osu! / osu-framework by peppy / ppy (MIT); danser-go by Wieku (GPL-3.0) was studied as the behavioural reference. **No osu! game assets are bundled** — osu!'s skins/beatmaps/audio/art are CC BY-NC and are not included; this repo ships only original or procedurally-generated art. The bundled **Nunito** variable font is under the SIL Open Font License 1.1 (`assets/fonts/OFL.txt`).
