import lzma
import tempfile
from pathlib import Path

from osu_taiko_renderer.security import (
    bounded_lzma_decompress,
    ffmpeg_file_input_args,
    safe_related_file,
)


def test_bounded_lzma_decompress_rejects_bomb():
    compressed = lzma.compress(b"x" * (2 * 1024 * 1024))
    try:
        bounded_lzma_decompress(compressed, max_output=1024 * 1024)
    except ValueError as exc:
        assert "output limit" in str(exc)
    else:
        raise AssertionError("oversized LZMA stream was accepted")


def test_safe_related_file_blocks_absolute_and_traversal():
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        target = tmp_path / "Assets" / "Song.MP3"
        target.parent.mkdir()
        target.write_bytes(b"x")
        assert safe_related_file(tmp_path, "assets/song.mp3") == target
        assert safe_related_file(tmp_path, "../Song.MP3") is None
        assert safe_related_file(tmp_path, "/etc/passwd") is None


def test_ffmpeg_file_input_args_forces_file_protocol():
    with tempfile.TemporaryDirectory() as directory:
        args = ffmpeg_file_input_args(Path(directory) / "hit.ogg")
        assert args[:4] == ["-protocol_whitelist", "file", "-f", "ogg"]
