"""Standalone osu!catch (mode 2 / "fruits") replay renderer.

A quick, self-contained catch renderer that follows the VRender wiki-driven
philosophy (osu_renderer.wiki_renderer.SkinPair for skinning) and reuses
osu_renderer's ffmpeg encode path, but owns its own GL sprite pipeline so it
can be ripped out and replaced by the VRender branch later.

Pipeline: parse .osu -> generate catch objects -> parse .osr -> per-frame
scene (falling fruit + catcher from replay) -> GL draw -> ffmpeg encode.
"""

import warnings

from PIL import Image

Image.MAX_IMAGE_PIXELS = 50_000_000
warnings.simplefilter("error", Image.DecompressionBombWarning)

__all__ = ["__version__"]
__version__ = "0.1.0"
