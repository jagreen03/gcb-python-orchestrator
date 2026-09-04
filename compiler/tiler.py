"""Hardware-aware tiling: lower a PHOTONIC nn.Linear node into SRAM controller
loop bounds.

``PhotonicTiler`` takes the global ``M x K -> M x N`` shape of one
PHOTONIC-tagged ``nn.Linear`` node (``HybridPartitioner`` in
``compiler/partitioner.py``) and derives the exact 4D loop nest
``gcb_tensor_sram_controller.sv`` sequences over -- ``pass_idx / m_band_idx /
n_sub_idx / k_step``, outer to inner -- plus the same per-beat matrix
addressing that RTL's own header comment documents. Every formula here is
copied verbatim from that module's derived-parameter block and elaboration
guards (rtl/gcb_tensor_sram_controller.sv), not re-derived, so a
:class:`TileDispatchPlan` is a drop-in match for the RTL counters it lowers
onto:

    M_TOTAL, N_TOTAL, K_TOTAL, N_TILE, GRID_DIM, K_MAX  (given)
    K_STEPS    = K_TOTAL / K_MAX
    N_SUBTILES = N_TILE  / GRID_DIM
    M_BANDS    = M_TOTAL / GRID_DIM
    N_PASSES   = N_TOTAL / N_TILE
    KSW/NSW/MBW/NPW/COLW = the same (<=1) ? 1 : $clog2(...) counter widths

Deliberately shallow, matching ``partitioner.py``'s own scope: this computes
loop bounds and per-beat addresses, it does not schedule DMA bursts, model
GDDR7 timing, or lower to an executable IR. Bandwidth/SRAM sizing is
``gcb-wiki/05_photonic_memory_dataflow.md``'s job, already done; this module
answers a narrower question -- given one Linear layer's shape, what does the
already-built RTL's loop nest look like.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterator

import torch.fx as fx
import torch.nn as nn

# Hardware constants (gcb_dwdm_pkg::K_MAX; gcb-wiki/05 sec. 4's 16x16 grid).
K_MAX = 16
GRID_DIM = 16
N_TILE_DEFAULT = 1024  # gcb-wiki/05 sec. 5's selected design point


def _counter_width(count: int) -> int:
    """SystemVerilog ``(count <= 1) ? 1 : $clog2(count)`` -- the exact
    expression gcb_tensor_sram_controller.sv uses for KSW/NSW/MBW/NPW/COLW."""
    if count <= 1:
        return 1
    return math.ceil(math.log2(count))


@dataclass(frozen=True)
class TileDispatchPlan:
    """The 4D loop nest gcb_tensor_sram_controller.sv sequences, outer to
    inner: ``pass_idx -> m_band_idx -> n_sub_idx -> k_step``.

    Attributes
    ----------
    n_passes, m_bands, n_subtiles, k_steps:
        Loop bound for each nested index -- direct counterparts of the RTL's
        ``N_PASSES`` / ``M_BANDS`` / ``N_SUBTILES`` / ``K_STEPS`` parameters.
    total_beats:
        ``k_steps * n_subtiles * m_bands * n_passes`` -- the number of
        ``sram_valid`` beats one full pass set takes, matching
        ``TOTAL_BEATS`` in this repo's RTL testbenches exactly.
    ksw, nsw, mbw, npw, colw:
        Register widths (bits) the RTL sizes ``k_step`` / ``n_sub_idx`` /
        ``m_band_idx`` / ``pass_idx`` / a B-tile column index at -- the same
        ``KSW`` / ``NSW`` / ``MBW`` / ``NPW`` / ``COLW`` derived parameters,
        for a host-side DMA engine's own counters to agree with.
    """

    n_passes: int
    m_bands: int
    n_subtiles: int
    k_steps: int
    total_beats: int
    ksw: int
    nsw: int
    mbw: int
    npw: int
    colw: int


@dataclass(frozen=True)
class PhotonicBeat:
    """One ``sram_valid`` beat's exact address range in the original
    ``M x K`` / ``K x N`` / ``M x N`` matrices -- gcb_tensor_sram_controller.sv's
    own header-documented addressing:

        A row r, K-step k of band a_load_band -> matrix row a_load_band*GRID_DIM+r,
                                                   K offset k*K_MAX
        B col c, K-step k of pass b_load_pass  -> matrix col b_load_pass*N_TILE+c,
                                                   K offset k*K_MAX
    """

    pass_idx: int
    m_band_idx: int
    n_sub_idx: int
    k_step: int
    m_start: int
    m_end: int  # rows [m_start:m_end) of A / Y this beat's PE grid covers
    n_start: int
    n_end: int  # cols [n_start:n_end) of B / Y this beat's PE grid covers
    k_start: int
    k_end: int  # the K_MAX-wide reduction slice this beat contracts


class PhotonicTiler:
    """Lowers one PHOTONIC ``nn.Linear`` node's global ``M x K -> M x N``
    GEMM shape onto ``gcb_tensor_sram_controller.sv``'s loop bounds.

    Parameters
    ----------
    m, k, n:
        Global matrix dimensions: ``M`` = batch/token count, ``K`` =
        input_dim (``nn.Linear.in_features``), ``N`` = output_dim
        (``nn.Linear.out_features``) -- ``Y[M,N] = X[M,K] @ W[K,N]``.
    k_max, grid_dim:
        Hardware constants; override only to model a different Tensor
        Complex build (defaults match the shipped RTL exactly).
    n_tile:
        Resident weight-stationary SRAM tile width (must divide ``n`` and be
        a multiple of ``grid_dim``) -- gcb-wiki/05 sec. 5's design-point
        knob; default matches the selected 1024.
    """

    def __init__(
        self,
        m: int,
        k: int,
        n: int,
        *,
        k_max: int = K_MAX,
        grid_dim: int = GRID_DIM,
        n_tile: int = N_TILE_DEFAULT,
    ) -> None:
        self.m, self.k, self.n = m, k, n
        self.k_max, self.grid_dim, self.n_tile = k_max, grid_dim, n_tile
        self._validate()

    @classmethod
    def from_linear_node(
        cls, node: fx.Node, traced: fx.GraphModule, m: int, **kwargs: int
    ) -> "PhotonicTiler":
        """Build a tiler from a PHOTONIC-tagged ``call_module`` FX node.

        ``K`` and ``N`` come straight off the ``nn.Linear`` submodule's own
        shape (``in_features`` / ``out_features``); ``M`` (the batch/token
        count) is NOT part of a Linear layer's definition -- it is a
        property of whatever input tensor flows through the graph at run
        time, so the caller supplies it explicitly (e.g. from a shape-
        propagation pass, or the serving batch size).
        """
        if node.op != "call_module":
            raise TypeError(f"expected a call_module node, got node.op={node.op!r}")
        submodule = traced.get_submodule(str(node.target))
        if not isinstance(submodule, nn.Linear):
            raise TypeError(
                f"node {node.name!r} targets {type(submodule).__name__}, not nn.Linear"
            )
        return cls(m=m, k=submodule.in_features, n=submodule.out_features, **kwargs)

    # ---- validation: identical constraints to gcb_tensor_sram_controller.sv's
    #      own `initial begin : g_assert` block -----------------------------
    def _validate(self) -> None:
        errors: list[str] = []
        if self.k % self.k_max != 0:
            errors.append(f"K ({self.k}) must be a multiple of K_MAX ({self.k_max})")
        if self.n_tile % self.grid_dim != 0:
            errors.append(f"N_TILE ({self.n_tile}) must be a multiple of GRID_DIM ({self.grid_dim})")
        if self.m % self.grid_dim != 0:
            errors.append(f"M ({self.m}) must be a multiple of GRID_DIM ({self.grid_dim})")
        if self.n % self.n_tile != 0:
            errors.append(f"N ({self.n}) must be a multiple of N_TILE ({self.n_tile})")
        if self.n_tile > self.n:
            errors.append(f"N_TILE ({self.n_tile}) must not exceed N ({self.n})")
        if errors:
            raise ValueError(
                "PhotonicTiler: shape is not legal for gcb_tensor_sram_controller.sv "
                "(would trip its own elaboration guards) -- " + "; ".join(errors)
            )

    # ---- derived loop bounds, formulas copied verbatim from the RTL -------
    @property
    def k_steps(self) -> int:
        return self.k // self.k_max

    @property
    def n_subtiles(self) -> int:
        return self.n_tile // self.grid_dim

    @property
    def m_bands(self) -> int:
        return self.m // self.grid_dim

    @property
    def n_passes(self) -> int:
        return self.n // self.n_tile

    def dispatch_plan(self) -> TileDispatchPlan:
        """The 4D loop bounds (N_PASSES, M_BANDS, N_SUBTILES, K_STEPS) the
        SRAM controller's FSM will sequence, plus their RTL counter widths."""
        k_steps, n_subtiles, m_bands, n_passes = (
            self.k_steps, self.n_subtiles, self.m_bands, self.n_passes
        )
        return TileDispatchPlan(
            n_passes=n_passes,
            m_bands=m_bands,
            n_subtiles=n_subtiles,
            k_steps=k_steps,
            total_beats=k_steps * n_subtiles * m_bands * n_passes,
            ksw=_counter_width(k_steps),
            nsw=_counter_width(n_subtiles),
            mbw=_counter_width(m_bands),
            npw=_counter_width(n_passes),
            colw=_counter_width(self.n_tile),
        )

    def beats(self) -> Iterator[PhotonicBeat]:
        """Yield every beat's resolved matrix addresses, in the EXACT nested
        order gcb_tensor_sram_controller.sv's FSM produces them: pass_idx outer,
        then m_band_idx, then n_sub_idx, then k_step innermost. A generator,
        not a list -- a real 70B layer is hundreds of millions of beats
        (see the verification block); iterate lazily."""
        for pass_idx in range(self.n_passes):
            n_pass_start = pass_idx * self.n_tile
            for m_band_idx in range(self.m_bands):
                m_start = m_band_idx * self.grid_dim
                for n_sub_idx in range(self.n_subtiles):
                    n_start = n_pass_start + n_sub_idx * self.grid_dim
                    for k_step in range(self.k_steps):
                        k_start = k_step * self.k_max
                        yield PhotonicBeat(
                            pass_idx=pass_idx,
                            m_band_idx=m_band_idx,
                            n_sub_idx=n_sub_idx,
                            k_step=k_step,
                            m_start=m_start,
                            m_end=m_start + self.grid_dim,
                            n_start=n_start,
                            n_end=n_start + self.grid_dim,
                            k_start=k_start,
                            k_end=k_start + self.k_max,
                        )


def print_dispatch_plan(tiler: PhotonicTiler, *, label: str = "") -> None:
    """Console-print a tiler's shape, 4D loop bounds, and RTL counter widths."""
    plan = tiler.dispatch_plan()
    if label:
        print(f"=== {label} ===")
    print(
        f"M={tiler.m}  K={tiler.k}  N={tiler.n}   "
        f"(K_MAX={tiler.k_max}, GRID_DIM={tiler.grid_dim}, N_TILE={tiler.n_tile})\n"
    )
    print("4D loop bounds (gcb_tensor_sram_controller.sv FSM, outer -> inner):")
    print(f"  N_PASSES   = N / N_TILE       = {tiler.n} / {tiler.n_tile:<6} = {plan.n_passes}")
    print(f"  M_BANDS    = M / GRID_DIM     = {tiler.m} / {tiler.grid_dim:<6} = {plan.m_bands}")
    print(f"  N_SUBTILES = N_TILE / GRID_DIM= {tiler.n_tile} / {tiler.grid_dim:<6} = {plan.n_subtiles}")
    print(f"  K_STEPS    = K / K_MAX        = {tiler.k} / {tiler.k_max:<6} = {plan.k_steps}")
    print(
        f"\n  TOTAL_BEATS = K_STEPS * N_SUBTILES * M_BANDS * N_PASSES "
        f"= {plan.k_steps} * {plan.n_subtiles} * {plan.m_bands} * {plan.n_passes} "
        f"= {plan.total_beats:,}"
    )
    print(
        f"\n  RTL counter widths: KSW={plan.ksw}  NSW={plan.nsw}  "
        f"MBW={plan.mbw}  NPW={plan.npw}  COLW={plan.colw}  (bits)"
    )


# =============================================================================
# Verification block
# =============================================================================
if __name__ == "__main__":
    # LLaMA-class 70B FFN up-projection (gcb-wiki/06_70B_transformer_workload_
    # trace.md sec. 3-4): M = 4096 tokens (prefill batch), K = 8192 (hidden,
    # in_features), N = 28672 (FFN intermediate, out_features).
    M, K, N = 4096, 8192, 28672

    tiler = PhotonicTiler(m=M, k=K, n=N)
    print_dispatch_plan(tiler, label="LLaMA-70B FFN up-projection (gate/up_proj shape)")

    plan = tiler.dispatch_plan()
    # Cross-check against gcb-wiki/06 sec. 4's independently-derived numbers,
    # not just this module's own arithmetic repeated back at itself.
    assert plan.n_passes == 28, plan
    assert plan.m_bands == 256, plan
    assert plan.n_subtiles == 64, plan
    assert plan.k_steps == 512, plan
    assert plan.total_beats == 234_881_024, plan
    print(
        "\nverified: matches gcb-wiki/06_70B_transformer_workload_trace.md sec. 4 exactly "
        "(234,881,024 beats = 117.440512 ms compute-bound at 2 GHz)."
    )

    # A first few beats, to show beats() resolves real matrix addresses, not
    # just bounds -- NOT materialized as a list (234M entries).
    print("\nfirst 3 beats (pass_idx, m_band_idx, n_sub_idx, k_step -> M/N/K address ranges):")
    for beat in itertools.islice(tiler.beats(), 3):
        print(
            f"  (p={beat.pass_idx}, m={beat.m_band_idx}, n={beat.n_sub_idx}, k={beat.k_step})"
            f"  ->  Y[{beat.m_start}:{beat.m_end}, {beat.n_start}:{beat.n_end}]"
            f"   K[{beat.k_start}:{beat.k_end}]"
        )

    # A deliberately small, illegal shape, to prove the tiler refuses to lower
    # anything the RTL's own elaboration guards would trip on rather than
    # silently emitting a plan the hardware can't run.
    print("\nrejecting a shape the RTL's own elaboration guards would trip on:")
    try:
        PhotonicTiler(m=100, k=8192, n=28672)  # 100 is not a multiple of GRID_DIM=16
    except ValueError as exc:
        print(f"  ValueError: {exc}")
