"""Generate the binary DMA payloads for the RTL co-simulation loop.

Builds a small test GEMM, tiles and bit-packs it exactly as a real host DMA
engine would (``compiler.tiler.PhotonicTiler`` + ``compiler.packer.DMAPacker``),
computes the golden INT16 result independently in NumPy, and writes all three
as raw binary files into ``gcb-wdm-transceiver/sim/`` for
``tb/tb_dwdm_dualmode_top.sv``'s "Scenario C" to ``$fread`` and replay against
the real ``gcb_tensor_memory_engine`` RTL.

File formats (all little-endian, byte-for-byte what the RTL testbench reads):

    a_payload.bin
        ``DMAPacker.pack_host_streams()``'s A chunks, concatenated in the
        EXACT order ``gcb_tensor_sram_controller.sv`` will issue
        ``a_load_req`` (one chunk per (pass, m_band); here N_PASSES=1, so
        M_BANDS chunks). Each chunk is GRID_DIM rows x K_STEPS kstep-words,
        each word 8 bytes (K_MAX=16 packed INT4 lanes, ``compiler.packer``'s
        bit convention).
    b_payload.bin
        Same, for B: one chunk per pass (N_TILE columns x K_STEPS kstep-words
        each).
    golden_c.bin
        The exact ``C = A @ B`` result, INT16, little-endian, row-major
        (``M`` rows x ``N`` columns) -- what every ``qam_result[]`` cell,
        accumulated across its K_STEPS beats on the RTL testbench's own host
        side (gcb-wiki/05 sec. 4a's K-tiling doctrine: a reduction beyond
        K_MAX is tiled into <=16-wide steps and accumulated DIGITALLY, off
        the optical core -- exactly what ``scheduler.py``'s ``TensorTiler.
        matmul()`` already does for V1; the RTL testbench does the same for
        V2 in Scenario C), must bit-exactly equal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from compiler.packer import DMAPacker
from compiler.tiler import GRID_DIM, K_MAX, PhotonicTiler

# Small enough that the RTL testbench's file-driven replay (256 real optical
# cores, elaborated a second time alongside the existing dut) stays fast:
# M_BANDS=N_SUBTILES=K_STEPS=2, N_PASSES=1, TOTAL_BEATS=8.
M, K, N = 32, 32, 32
N_TILE = 32  # = N: one resident B_tile covers the whole output width (N_PASSES=1)

INT4_MIN, INT4_MAX = -8, 7
SEED = 0

OUT_DIR = Path(__file__).resolve().parents[2] / "gcb-wdm-transceiver" / "sim"


def main() -> None:
    if GRID_DIM != 16 or K_MAX != 16:
        raise AssertionError("compiler.tiler's hardware constants changed unexpectedly")

    rng = np.random.default_rng(SEED)
    a = rng.integers(INT4_MIN, INT4_MAX + 1, size=(M, K))
    b = rng.integers(INT4_MIN, INT4_MAX + 1, size=(K, N))

    tiler = PhotonicTiler(m=M, k=K, n=N, n_tile=N_TILE)
    plan = tiler.dispatch_plan()
    print(
        f"M={M} K={K} N={N}  N_TILE={N_TILE}  ->  "
        f"N_PASSES={plan.n_passes} M_BANDS={plan.m_bands} "
        f"N_SUBTILES={plan.n_subtiles} K_STEPS={plan.k_steps}  "
        f"(TOTAL_BEATS={plan.total_beats})"
    )

    packer = DMAPacker(tiler, a, b)
    a_chunks, b_chunks = packer.pack_host_streams()
    assert len(a_chunks) == plan.n_passes * plan.m_bands
    assert len(b_chunks) == plan.n_passes
    a_bytes = b"".join(a_chunks)
    b_bytes = b"".join(b_chunks)

    # Golden result: the exact, untiled C = A @ B. INT64 accumulation is pure
    # overkill for this shape (max |sum| = K * 8 * 8 = 2048, nowhere near
    # overflowing INT16) but matches scheduler.py's own established
    # convention (accumulate wide, narrow once at the end).
    golden = (a.astype(np.int64) @ b.astype(np.int64))
    if golden.min() < -32768 or golden.max() > 32767:
        raise OverflowError(f"golden C out of INT16 range: [{golden.min()}, {golden.max()}]")
    golden_i16 = golden.astype("<i2")
    golden_bytes = golden_i16.tobytes()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a_path = OUT_DIR / "a_payload.bin"
    b_path = OUT_DIR / "b_payload.bin"
    c_path = OUT_DIR / "golden_c.bin"
    a_path.write_bytes(a_bytes)
    b_path.write_bytes(b_bytes)
    c_path.write_bytes(golden_bytes)

    print(f"\nwrote {a_path}  ({len(a_bytes)} bytes, {len(a_chunks)} A-band chunk(s))")
    print(f"wrote {b_path}  ({len(b_bytes)} bytes, {len(b_chunks)} B-tile chunk(s))")
    print(f"wrote {c_path}  ({len(golden_bytes)} bytes, {M}x{N} INT16)")

    print("\nfirst A-band chunk, first 2 words (hex):")
    for i in range(2):
        word = a_bytes[i * 8:(i + 1) * 8]
        print(f"  {word.hex()}")
    print("first B-tile chunk, first 2 words (hex):")
    for i in range(2):
        word = b_bytes[i * 8:(i + 1) * 8]
        print(f"  {word.hex()}")
    print(f"\ngolden C[0][0:4] = {golden_i16[0, :4].tolist()}   "
          f"(bytes: {golden_bytes[:8].hex()})")
    print(f"golden C range: [{int(golden.min())}, {int(golden.max())}]  (INT16-safe)")


if __name__ == "__main__":
    main()
