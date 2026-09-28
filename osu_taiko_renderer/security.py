"""Fail-closed guards for untrusted renderer inputs."""
from __future__ import annotations

import lzma
import struct
from pathlib import Path, PureWindowsPath

MAX_REPLAY_DECOMPRESSED_BYTES = 32 * 1024 * 1024
MAX_REPLAY_FILE_BYTES = 64 * 1024 * 1024
_DECOMPRESS_CHUNK_BYTES = 1024 * 1024
_FFMPEG_DEMUXERS = {
    ".aac": "aac", ".flac": "flac", ".m4a": "mov", ".mkv": "matroska",
    ".mp3": "mp3", ".mp4": "mov", ".ogg": "ogg", ".opus": "ogg",
    ".wav": "wav", ".webm": "matroska",
}


def bounded_lzma_decompress(compressed: bytes, *, format: int = lzma.FORMAT_AUTO,
                            max_output: int = MAX_REPLAY_DECOMPRESSED_BYTES) -> bytes:
    """Decode a stream in bounded chunks, rejecting expansion past ``max_output``."""
    if max_output <= 0:
        raise ValueError("max_output must be positive")
    decoder = lzma.LZMADecompressor(format=format)
    chunks: list[bytes] = []
    total = 0
    source = compressed
    while True:
        chunk = decoder.decompress(
            source, max_length=min(_DECOMPRESS_CHUNK_BYTES, max_output - total + 1)
        )
        source = b""
        total += len(chunk)
        if total > max_output:
            raise ValueError("LZMA payload exceeds replay output limit")
        chunks.append(chunk)
        if decoder.eof:
            return b"".join(chunks)
        if decoder.needs_input:
            raise lzma.LZMAError("truncated LZMA payload")


def _skip_osr_string(data: bytes, offset: int) -> int:
    if offset >= len(data):
        raise ValueError("truncated replay header")
    marker = data[offset]
    offset += 1
    if marker == 0x00:
        return offset
    if marker != 0x0B:
        raise ValueError("invalid replay string marker")
    length = shift = 0
    while True:
        if offset >= len(data) or shift > 63:
            raise ValueError("truncated replay string length")
        byte = data[offset]
        offset += 1
        length |= (byte & 0x7F) << shift
        if not byte & 0x80:
            break
        shift += 7
    if length > len(data) - offset:
        raise ValueError("truncated replay string")
    return offset + length


def validate_replay_payload(path: Path) -> None:
    """Preflight the frame payload before the unbounded ``osrparse`` decode."""
    path = Path(path)
    if path.stat().st_size > MAX_REPLAY_FILE_BYTES:
        raise ValueError("replay file exceeds input size limit")
    data = path.read_bytes()
    offset = 1 + 4
    for _ in range(3):
        offset = _skip_osr_string(data, offset)
    fixed_header = 2 * 6 + 4 + 2 + 1 + 4
    if offset + fixed_header > len(data):
        raise ValueError("truncated replay header")
    offset += fixed_header
    offset = _skip_osr_string(data, offset)
    if offset + 8 + 4 > len(data):
        raise ValueError("truncated replay header")
    offset += 8
    (replay_length,) = struct.unpack_from("<i", data, offset)
    offset += 4
    if replay_length < 0 or replay_length > len(data) - offset:
        raise ValueError("invalid replay payload length")
    bounded_lzma_decompress(data[offset:offset + replay_length])


def safe_related_file(base_dir: Path, name: str | None) -> Path | None:
    """Return only a real beatmap asset contained by ``base_dir``."""
    if not name:
        return None
    raw = str(name).replace("\\", "/")
    if Path(raw).is_absolute() or PureWindowsPath(raw).drive:
        return None
    if any(part in (".", "..") for part in raw.split("/")):
        return None
    key = raw.rsplit("/", 1)[-1].lower()
    if not key:
        return None
    base = Path(base_dir)
    try:
        for candidate in base.rglob("*"):
            if not candidate.is_file() or candidate.name.lower() != key:
                continue
            try:
                candidate.resolve().relative_to(base.resolve())
            except (OSError, ValueError):
                continue
            return candidate
    except OSError:
        return None
    return None


def ffmpeg_file_input_args(path: Path) -> list[str]:
    """Arguments for a local uploaded-media input, never a network protocol."""
    args = ["-protocol_whitelist", "file"]
    demuxer = _FFMPEG_DEMUXERS.get(Path(path).suffix.lower())
    if demuxer:
        args += ["-f", demuxer]
    return [*args, "-i", str(path)]
