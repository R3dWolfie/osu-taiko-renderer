"""Port of osu!'s LegacyRandom (xorshift-128), used by CatchBeatmapProcessor
to position bananas and apply the +-20px tiny-droplet XOffset.

Ported from ppy/osu osu.Game/Utils/LegacyRandom.cs. All arithmetic is masked
to 32-bit unsigned to match C# uint semantics exactly.

NOTE: applying these offsets bit-exactly requires consuming the RNG in osu's
exact object order (banana: NextDouble + 3x Next; tiny droplet: Next(-20,20);
large droplet: Next), which in turn requires the banana/droplet GENERATION
counts to match osu exactly. See CatchBeatmapProcessor.ApplyPositionOffsets.
"""
from __future__ import annotations

_MASK = 0xFFFFFFFF
_INT_MASK = 0x7FFFFFFF
_INT_TO_REAL = 1.0 / (0x7FFFFFFF + 1.0)
RNG_SEED = 1337


class LegacyRandom:
    def __init__(self, seed: int = RNG_SEED):
        self.x = seed & _MASK
        self.y = 842502087
        self.z = 3579807591
        self.w = 273326509
        self._bit_buffer = 0
        self._bit_index = 32

    def next_uint(self) -> int:
        t = (self.x ^ ((self.x << 11) & _MASK)) & _MASK
        self.x, self.y, self.z = self.y, self.z, self.w
        self.w = (self.w ^ (self.w >> 19) ^ t ^ (t >> 8)) & _MASK
        return self.w

    def next(self) -> int:
        return _INT_MASK & self.next_uint()

    def next_double(self) -> float:
        return _INT_TO_REAL * self.next()

    def next_max(self, upper: float) -> int:
        return int(self.next_double() * upper)

    def next_range(self, lower: float, upper: float) -> int:
        return int(lower + self.next_double() * (upper - lower))

    def next_bool(self) -> bool:
        if self._bit_index == 32:
            self._bit_buffer = self.next_uint()
            self._bit_index = 1
            return (self._bit_buffer & 1) == 1
        self._bit_index += 1
        self._bit_buffer >>= 1
        return (self._bit_buffer & 1) == 1
