"""Host-side tensor scheduler for the GCB V2.0 photonic QAM accelerator.

The optical datapath computes exact INT16 dot products, but the balanced-homodyne
phase-noise floor caps the contraction length at ``K <= 16`` per optical beat
(see ``gcb-photonic-qam-v2/intent.md``, "Software Tiling Constraint"). Any larger
contraction has to be split along ``K`` into <=16-wide chunks whose INT16 partial
products are accumulated on the host.

``TensorTiler`` turns a single LLM-layer matmul ``Y = A @ B`` -- ``A`` is
``M x K``, ``B`` is ``K x N`` -- into a stream of DMA-ready tile pairs sized
``(M_tile x k)`` and ``(k x N_tile)`` with ``k <= K_MAX_HW``, tagged with the
bookkeeping a host accumulator needs (output-block offsets and the
first/last-chunk flags that bracket an accumulation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

# Hardware contraction limit: the balanced-homodyne SNR floor (intent.md).
K_MAX_HW = 16


def ceil_div(numerator: int, denominator: int) -> int:
    """Integer ceiling division for positive arguments."""
    return -(-numerator // denominator)


@dataclass(frozen=True)
class Tile:
    """One DMA-ready tile pair plus its place in the output accumulation.

    Attributes
    ----------
    a, b:
        Operand sub-matrices, shapes ``(m_tile, k)`` and ``(k, n_tile)`` with
        ``k <= K_MAX_HW``. These are views into the parent matrices.
    row, col:
        Top-left offset of the ``M_tile x N_tile`` output block this tile
        contributes to, in the full ``M x N`` result.
    k_off:
        Contraction offset of this chunk within ``K``.
    first_k, last_k:
        ``first_k`` marks the chunk that initialises the output block's
        accumulator; ``last_k`` marks the chunk after which it can be flushed
        to host memory. For ``K <= K_MAX_HW`` both are true on the only tile.
    """

    a: np.ndarray
    b: np.ndarray
    row: int
    col: int
    k_off: int
    first_k: bool
    last_k: bool

    @property
    def m_tile(self) -> int:
        return self.a.shape[0]

    @property
    def n_tile(self) -> int:
        return self.b.shape[1]

    @property
    def k(self) -> int:
        return self.a.shape[1]


class TensorTiler:
    """Chunk an ``M x K`` by ``K x N`` matmul into hardware-legal tile pairs."""

    def __init__(
        self,
        m_tile: int = 32,
        n_tile: int = 32,
        k_max: int = K_MAX_HW,
    ) -> None:
        if min(m_tile, n_tile, k_max) < 1:
            raise ValueError("tile dimensions must be >= 1")
        if k_max > K_MAX_HW:
            raise ValueError(
                f"k_max={k_max} exceeds the hardware contraction limit "
                f"K_MAX_HW={K_MAX_HW}"
            )
        self.m_tile = m_tile
        self.n_tile = n_tile
        self.k_max = k_max

    def tile_count(self, m: int, k: int, n: int) -> int:
        """Number of tiles ``tiles()`` will yield for an ``m x k`` @ ``k x n``."""
        return (
            ceil_div(m, self.m_tile)
            * ceil_div(n, self.n_tile)
            * ceil_div(k, self.k_max)
        )

    def tiles(self, a: np.ndarray, b: np.ndarray) -> Iterator[Tile]:
        """Yield DMA-ready :class:`Tile` pairs in output-block, then k, order."""
        a = np.asarray(a)
        b = np.asarray(b)
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("a and b must be 2-D matrices")
        m, k = a.shape
        k_b, n = b.shape
        if k != k_b:
            raise ValueError(
                f"inner dimensions disagree: A is {a.shape}, B is {b.shape}"
            )

        k_starts = range(0, k, self.k_max)
        last_k_start = k_starts[-1] if len(k_starts) else 0

        for row in range(0, m, self.m_tile):
            r_end = min(row + self.m_tile, m)
            for col in range(0, n, self.n_tile):
                c_end = min(col + self.n_tile, n)
                for k_off in k_starts:
                    k_end = min(k_off + self.k_max, k)
                    yield Tile(
                        a=a[row:r_end, k_off:k_end],
                        b=b[k_off:k_end, col:c_end],
                        row=row,
                        col=col,
                        k_off=k_off,
                        first_k=(k_off == 0),
                        last_k=(k_off == last_k_start),
                    )

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Reference: run the tiled schedule on the host, accumulating in INT64.

        Mirrors what the orchestrator does with the accelerator -- stream each
        tile pair, add the returned partial into the output block, flush on
        ``last_k`` -- so a test can assert the schedule is loss-free.
        """
        a = np.asarray(a)
        b = np.asarray(b)
        out = np.zeros((a.shape[0], b.shape[1]), dtype=np.int64)
        for tile in self.tiles(a, b):
            partial = tile.a.astype(np.int64) @ tile.b.astype(np.int64)
            out[tile.row : tile.row + tile.m_tile,
                tile.col : tile.col + tile.n_tile] += partial
        return out


def _demo() -> None:
    rng = np.random.default_rng(0)

    # INT4 operands -- the datapath's native precision.
    a = rng.integers(-8, 8, size=(64, 64), dtype=np.int64)
    b = rng.integers(-8, 8, size=(64, 64), dtype=np.int64)

    tiler = TensorTiler(m_tile=32, n_tile=32)
    m, k = a.shape
    n = b.shape[1]

    print(f"A {a.shape}  @  B {b.shape}     (K_MAX_HW = {K_MAX_HW})")
    print(
        f"schedule: {tiler.tile_count(m, k, n)} tiles"
        f"  =  {ceil_div(m, tiler.m_tile)} x {ceil_div(n, tiler.n_tile)}"
        f" output blocks  x  {ceil_div(k, tiler.k_max)} k-chunks\n"
    )

    for idx, tile in enumerate(tiler.tiles(a, b)):
        print(
            f"tile {idx:2d}:  A{tile.a.shape} @ B{tile.b.shape}"
            f"  ->  Y[{tile.row}:{tile.row + tile.m_tile},"
            f" {tile.col}:{tile.col + tile.n_tile}]"
            f"   k_off={tile.k_off:2d}  first_k={tile.first_k:d}  last_k={tile.last_k:d}"
        )

    got = tiler.matmul(a, b)
    ref = a @ b
    assert np.array_equal(got, ref), "tiled schedule does not reproduce A @ B"
    print(
        f"\nverified: tiled result == dense A @ B"
        f"   (shape {ref.shape}, max |Y| = {int(np.abs(ref).max())}, fits INT16)"
    )


if __name__ == "__main__":
    _demo()
