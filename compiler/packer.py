"""DMA payload serializer: pack PhotonicTiler beats into hardware-aligned
INT4 byte streams.

``DMAPacker`` walks a :class:`~compiler.tiler.PhotonicTiler`'s ``beats()``
generator and, for each beat, slices the real ``A`` (activations) and ``B``
(weights) matrices down to the exact ``GRID_DIM``-row / ``GRID_DIM``-column,
``K_MAX``-deep chunks ``gcb_tensor_sram_controller.sv``'s A row-band and
B_tile write ports expect, then bit-packs each length-``K_MAX`` INT4 vector
into the exact 8-byte word ``a_wr_data`` / ``b_wr_data`` carries.

Bit convention -- copied from ``gcb_dwdm_pkg::pack_int4`` (V1.0's own
packer, ``rtl/gcb_dwdm_pkg.sv``), not reinvented: lane ``l`` occupies bits
``[l*4 +: 4]`` of the word, little-lane-first, so lane 0 is the LOW nibble of
byte 0. Every ``VEC_W``-wide port in this repo's RTL (``a_wr_data``,
``b_wr_data``, the broadcast fabric's unpack, this repo's own testbenches'
``pack_vec()``) already uses this identical convention -- this module is the
Python-side instance of the same one bit layout, not a new one.

One beat's PE grid consumes ``GRID_DIM`` (16) distinct A row-vectors and
``GRID_DIM`` distinct B column-vectors (gcb-wiki/05_photonic_memory_dataflow.md
sec. 4b), so packing "the beat's A/B slice" means 16 separate 8-byte A words
and 16 separate 8-byte B words, not one. Deliberately out of scope here (see
the module docstring's closing note): A/B are actually resident once per
band/pass in the real hardware, not reloaded every beat that reuses them --
deduplicating the byte stream against that residency is a DMA burst-
scheduling concern, not a bit-packing one, and is left to a later task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from compiler.tiler import K_MAX, PhotonicBeat, PhotonicTiler

# INT4 two's-complement range this packer accepts (int4_t, gcb_dwdm_pkg.sv).
INT4_MIN, INT4_MAX = -8, 7


@dataclass(frozen=True)
class DMAPayload:
    """The two independent byte streams a real GDDR7 DMA engine pushes --
    A (row-band channel) and B (tile channel) are separate write ports in
    gcb_tensor_sram_controller.sv, never merged on the wire."""

    a_stream: bytes
    b_stream: bytes


def pack_vector(lanes: np.ndarray) -> bytes:
    """Pack K_MAX=16 signed INT4 lanes into an exact 8-byte little-endian
    word: lane ``l`` -> bits ``[l*4 +: 4]`` (lane 0 = low nibble of byte 0),
    matching ``gcb_dwdm_pkg::pack_int4`` bit-for-bit. K_MAX is a fixed
    cross-architecture hardware invariant (``gcb_dwdm_pkg.sv``: "K_MAX is a
    HARD cross-architecture invariant, fixed at 16 - not a tunable"), so
    this is hardcoded to exactly 16 lanes / 8 bytes, not parameterized.

    Pure ``numpy`` bitwise ops: mask each lane to its 4-bit two's-complement
    representation, shift each into its lane position, OR them together.
    """
    lanes = np.asarray(lanes)
    if lanes.shape != (K_MAX,):
        raise ValueError(f"expected exactly {K_MAX} INT4 lanes, got shape {lanes.shape}")
    if lanes.min() < INT4_MIN or lanes.max() > INT4_MAX:
        raise ValueError(
            f"lane value out of INT4 range [{INT4_MIN}, {INT4_MAX}]: "
            f"min={lanes.min()}, max={lanes.max()}"
        )

    # int64 first so a negative Python/NumPy int's `& 0xF` truncation is the
    # correct two's-complement low nibble (e.g. -1 & 0xF == 0xF), then widen
    # to uint64 for the shift/OR -- shifting a signed dtype by up to 60 bits
    # is undefined-adjacent territory this sidesteps entirely.
    nibbles = (lanes.astype(np.int64) & 0xF).astype(np.uint64)
    shifts = np.arange(K_MAX, dtype=np.uint64) * np.uint64(4)
    word = np.uint64(np.bitwise_or.reduce(nibbles << shifts))
    return np.array([word], dtype="<u8").tobytes()


def unpack_vector(data: bytes) -> np.ndarray:
    """Inverse of :func:`pack_vector` -- an 8-byte word back to 16 signed
    INT4 lanes. Used only to prove pack/unpack round-trips exactly; the real
    hardware-side inverse is ``gcb_dwdm_pkg::unpack_int4``."""
    if len(data) != 8:
        raise ValueError(f"expected an 8-byte word, got {len(data)} bytes")
    word = int(np.frombuffer(data, dtype="<u8")[0])
    # plain Python ints throughout the sign-extension arithmetic -- doing it
    # in a numpy uint8 dtype instead overflow-wraps n - 16 for n < 16 rather
    # than producing the negative two's-complement value.
    nibbles = [(word >> (4 * l)) & 0xF for l in range(K_MAX)]
    return np.array([n - 16 if n >= 8 else n for n in nibbles], dtype=np.int64)


class DMAPacker:
    """Serializes a :class:`PhotonicTiler`'s beats into hardware-aligned
    INT4 DMA byte streams.

    Parameters
    ----------
    tiler:
        The lowered dispatch plan for this GEMM (``compiler/tiler.py``).
    a:
        Activations, shape ``(M, K)``.
    b:
        Weights, shape ``(K, N)``.
    """

    def __init__(self, tiler: PhotonicTiler, a: np.ndarray, b: np.ndarray) -> None:
        self.tiler = tiler
        self.a = np.asarray(a)
        self.b = np.asarray(b)
        if self.a.shape != (tiler.m, tiler.k):
            raise ValueError(f"A shape {self.a.shape} != tiler (M,K) = ({tiler.m}, {tiler.k})")
        if self.b.shape != (tiler.k, tiler.n):
            raise ValueError(f"B shape {self.b.shape} != tiler (K,N) = ({tiler.k}, {tiler.n})")

    def pack_beat(self, beat: PhotonicBeat) -> tuple[bytes, bytes]:
        """Pack one beat's full GRID_DIM-row A slice and GRID_DIM-column B
        slice -- GRID_DIM separate 8-byte words each, concatenated in row
        (resp. column) order."""
        a_rows = self.a[beat.m_start:beat.m_end, beat.k_start:beat.k_end]
        b_cols = self.b[beat.k_start:beat.k_end, beat.n_start:beat.n_end]
        a_bytes = b"".join(pack_vector(a_rows[r, :]) for r in range(a_rows.shape[0]))
        b_bytes = b"".join(pack_vector(b_cols[:, c]) for c in range(b_cols.shape[1]))
        return a_bytes, b_bytes

    def pack_beats(self) -> Iterator[tuple[PhotonicBeat, bytes, bytes]]:
        """Lazily yield ``(beat, a_bytes, b_bytes)`` for every beat, in the
        exact order ``tiler.beats()`` produces them. Prefer this over
        :meth:`pack_stream` for anything beyond a small demo matrix -- a
        real 70B layer is hundreds of millions of beats."""
        for beat in self.tiler.beats():
            a_bytes, b_bytes = self.pack_beat(beat)
            yield beat, a_bytes, b_bytes

    def pack_stream(self) -> DMAPayload:
        """Materialize the full continuous A and B byte streams (every beat,
        concatenated). Only for demo-scale matrices -- see :meth:`pack_beats`."""
        a_chunks: list[bytes] = []
        b_chunks: list[bytes] = []
        for _beat, a_bytes, b_bytes in self.pack_beats():
            a_chunks.append(a_bytes)
            b_chunks.append(b_bytes)
        return DMAPayload(a_stream=b"".join(a_chunks), b_stream=b"".join(b_chunks))


# =============================================================================
# Verification block
# =============================================================================
if __name__ == "__main__":
    from compiler.tiler import GRID_DIM, K_MAX

    # Deliberately tiny: M=K=N=16=GRID_DIM=K_MAX collapses the 4D loop to
    # exactly ONE beat (M_BANDS=N_SUBTILES=K_STEPS=N_PASSES=1), so the whole
    # matmul's A/B payload streams print in full without flooding the console.
    M = K = N = GRID_DIM
    tiler = PhotonicTiler(m=M, k=K, n=N, n_tile=N)  # n_tile must divide N; N itself is legal

    rng = np.random.default_rng(0)
    a = rng.integers(INT4_MIN, INT4_MAX + 1, size=(M, K))
    b = rng.integers(INT4_MIN, INT4_MAX + 1, size=(K, N))

    packer = DMAPacker(tiler, a, b)
    plan = tiler.dispatch_plan()
    print(f"M=K=N={M} (=GRID_DIM=K_MAX)  ->  {plan.total_beats} beat(s), "
          f"{plan.m_bands} A row-write(s) x {GRID_DIM} rows, "
          f"{plan.n_passes} B tile-write(s) x {GRID_DIM} cols\n")

    payload = packer.pack_stream()
    print(f"A stream: {len(payload.a_stream)} bytes   "
          f"({len(payload.a_stream) // 8} words x 8 bytes)")
    print(f"B stream: {len(payload.b_stream)} bytes   "
          f"({len(payload.b_stream) // 8} words x 8 bytes)\n")

    print("first 4 beats' A words (row r's K=16-lane vector, hex, little-endian):")
    for r in range(4):
        word = payload.a_stream[r * 8:(r + 1) * 8]
        print(f"  A row {r:2d}:  {word.hex()}   lanes={a[r, :].tolist()}")

    print("\nfirst 4 beats' B words (col c's K=16-lane vector, hex, little-endian):")
    for c in range(4):
        word = payload.b_stream[c * 8:(c + 1) * 8]
        print(f"  B col {c:2d}:  {word.hex()}   lanes={b[:, c].tolist()}")

    # Round-trip every packed word back through unpack_vector and confirm it
    # reproduces the exact original INT4 lanes -- proof the bit-packing is
    # mathematically exact, not just plausible-looking hex.
    ok = True
    for r in range(M):
        word = payload.a_stream[r * 8:(r + 1) * 8]
        recovered = unpack_vector(word)
        if not np.array_equal(recovered, a[r, :]):
            ok = False
            print(f"MISMATCH at A row {r}: packed {word.hex()} -> {recovered}, want {a[r, :]}")
    for c in range(N):
        word = payload.b_stream[c * 8:(c + 1) * 8]
        recovered = unpack_vector(word)
        if not np.array_equal(recovered, b[:, c]):
            ok = False
            print(f"MISMATCH at B col {c}: packed {word.hex()} -> {recovered}, want {b[:, c]}")

    assert ok, "pack_vector/unpack_vector round-trip failed"
    print(f"\nverified: all {M} A rows and {N} B cols round-trip pack_vector -> "
          f"unpack_vector exactly (INT4 range [{INT4_MIN}, {INT4_MAX}])")
