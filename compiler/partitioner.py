"""torch.fx-based hybrid partitioner: Photonic vs. Vortex node tagging.

Ingests a standard ``torch.nn.Module`` (a Transformer block), traces it with
``torch.fx``, and tags every node in the resulting graph with the GCB engine
that owns it, per the hybrid partitioning doctrine
(``gcb-wiki/06_70B_transformer_workload_trace.md`` sec. 2):

    Photonic Tensor Complex  -- static-weight dense linear algebra: every
        nn.Linear (QKVO projections, FFN gate/up/down), and any raw matmul
        whose weight operand is a stored parameter/buffer.
    Digital Vortex           -- dynamic, data-dependent, or non-linear:
        Softmax, RMSNorm/LayerNorm, SiLU/GELU, elementwise ops between two
        activations (residual adds, the SwiGLU gating multiply), and
        attention's own matmuls (Q @ K^T, Attn @ V) -- structurally the same
        ``torch.matmul`` primitive as a Linear layer's GEMM, but BOTH
        operands are activations, not a stored weight, so it cannot be
        weight-stationary and must route differently. Disambiguating this
        is the one place this module can't just dispatch on op identity --
        see ``HybridPartitioner._has_static_operand``.
    Host                      -- everything structural: graph I/O, parameter/
        buffer access, and shape/indexing ops (view, reshape, transpose,
        permute, ...). Neither engine executes these; they are not compute.

Deliberately shallow: this tags the graph, it does not lower it. No new IR,
no cross-node fusion or scheduling decisions -- those are later tasks.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F


class ExecutionTarget(Enum):
    """Which GCB execution engine a graph node's operation is routed to."""

    PHOTONIC = auto()  # static-weight dense GEMM
    VORTEX = auto()  # dynamic / data-dependent / non-linear
    HOST = auto()  # structural -- not a compute op


# ---- dispatch tables --------------------------------------------------
# Leaf modules that are always a static-weight dense GEMM.
_PHOTONIC_MODULE_TYPES: tuple[type, ...] = (nn.Linear,)

# Leaf module CLASS NAMES (substring, case-insensitive) that are dynamic /
# non-linear. Substring match rather than isinstance() because LLaMA-class
# models almost always define their own RMSNorm class rather than using a
# torch built-in (nn.RMSNorm only exists from PyTorch 2.4+, and even then
# most LLM codebases still ship a custom one) -- matching by name is the
# robust choice against whatever concrete class a caller's model uses.
_VORTEX_MODULE_NAME_KEYWORDS: tuple[str, ...] = ("norm", "softmax")

# call_function targets that are unconditionally a static-weight GEMM: the
# weight argument is what F.linear(input, weight, bias) MEANS, regardless of
# whether that tensor happens to trace to a get_attr node in THIS graph.
_STATIC_MATMUL_FUNCS: frozenset[Callable[..., Any]] = frozenset({F.linear})

# call_function / call_method targets that are matmul but NOT inherently
# weight-stationary -- route per-node via _has_static_operand.
_MATMUL_FUNCS: frozenset[Callable[..., Any]] = frozenset(
    {torch.matmul, torch.mm, torch.bmm, operator.matmul}
)
_MATMUL_METHODS: frozenset[str] = frozenset({"matmul"})

# Non-linear / dynamic elementwise and reduction ops. operator.add/mul cover
# residual adds and the SwiGLU gating multiply -- both are elementwise
# between two activations, never weight-stationary, matching sec. 2's
# "residual adds" -> Vortex line item directly.
_VORTEX_FUNCS: frozenset[Callable[..., Any]] = frozenset(
    {
        F.softmax,
        torch.softmax,
        F.silu,
        F.gelu,
        F.relu,
        F.sigmoid,
        torch.sigmoid,
        operator.add,
        operator.mul,
        operator.sub,
    }
)
_VORTEX_METHODS: frozenset[str] = frozenset({"softmax", "silu"})

# Pure bookkeeping / shape ops. Neither engine "runs" these.
_STRUCTURAL_METHODS: frozenset[str] = frozenset(
    {
        "view",
        "reshape",
        "transpose",
        "permute",
        "contiguous",
        "flatten",
        "unsqueeze",
        "squeeze",
        "size",
        "to",
        "type_as",
        "expand",
    }
)


def _callable_name(fn: Callable[..., Any]) -> str:
    return getattr(fn, "__name__", str(fn))


@dataclass(frozen=True)
class PartitionResult:
    """One graph node's routing decision.

    Attributes
    ----------
    node_name:
        The FX node's name (unique within the graph).
    op:
        The FX node kind (``placeholder``, ``call_module``, ``call_function``,
        ``call_method``, ``get_attr``, ``output``).
    target:
        Human-readable form of ``node.target`` (module type, function, or
        method name).
    execution_target:
        The assigned :class:`ExecutionTarget`.
    reason:
        One-line justification, so the routing decision is auditable without
        re-deriving it.
    """

    node_name: str
    op: str
    target: str
    execution_target: ExecutionTarget
    reason: str


class HybridPartitioner:
    """Traces a module with ``torch.fx`` and tags every node Photonic /
    Vortex / Host.

    Parameters
    ----------
    module:
        A standard ``torch.nn.Module`` -- typically one Transformer block
        (attention or FFN). Must be traceable by ``torch.fx.symbolic_trace``
        (no data-dependent Python control flow on tensor values).
    """

    def __init__(self, module: nn.Module) -> None:
        self.module = module
        self.traced: fx.GraphModule = fx.symbolic_trace(module)

    def partition(self) -> list[PartitionResult]:
        """Classify every node in the traced graph, in graph (source) order."""
        return [self._classify(node) for node in self.traced.graph.nodes]

    # ---- per-node classification ---------------------------------------

    def _classify(self, node: fx.Node) -> PartitionResult:
        target_str = self._target_str(node)

        if node.op in ("placeholder", "output", "get_attr"):
            return self._result(node, target_str, ExecutionTarget.HOST, f"{node.op} (structural)")

        if node.op == "call_module":
            return self._classify_module(node, target_str)

        if node.op == "call_function":
            return self._classify_function(node, target_str)

        if node.op == "call_method":
            return self._classify_method(node, target_str)

        # torch.fx defines exactly the five node.op kinds handled above; this
        # is unreachable against any real traced graph, but fail loudly
        # rather than silently misroute an op a future FX version adds.
        raise ValueError(f"unhandled FX node op {node.op!r} on node {node.name!r}")

    def _classify_module(self, node: fx.Node, target_str: str) -> PartitionResult:
        submodule = self.traced.get_submodule(str(node.target))
        mod_type = type(submodule)

        if isinstance(submodule, _PHOTONIC_MODULE_TYPES):
            return self._result(
                node, target_str, ExecutionTarget.PHOTONIC,
                f"{mod_type.__name__} (static-weight dense GEMM)",
            )

        name_lower = mod_type.__name__.lower()
        if any(keyword in name_lower for keyword in _VORTEX_MODULE_NAME_KEYWORDS):
            return self._result(
                node, target_str, ExecutionTarget.VORTEX,
                f"{mod_type.__name__} (normalization / non-linear)",
            )

        # Not a recognized static-weight or normalization leaf module --
        # default to Host rather than guessing which compute engine owns it.
        return self._result(
            node, target_str, ExecutionTarget.HOST,
            f"{mod_type.__name__} (unrecognized module type -- defaults to host)",
        )

    def _classify_function(self, node: fx.Node, target_str: str) -> PartitionResult:
        fn = node.target

        if fn in _STATIC_MATMUL_FUNCS:
            return self._result(
                node, target_str, ExecutionTarget.PHOTONIC,
                f"{_callable_name(fn)} (static-weight dense GEMM)",
            )

        if fn in _MATMUL_FUNCS:
            return self._matmul_result(node, target_str)

        if fn in _VORTEX_FUNCS:
            return self._result(
                node, target_str, ExecutionTarget.VORTEX,
                f"{_callable_name(fn)} (non-linear / dynamic elementwise)",
            )

        return self._result(
            node, target_str, ExecutionTarget.HOST,
            f"{_callable_name(fn)} (unrecognized function -- defaults to host)",
        )

    def _classify_method(self, node: fx.Node, target_str: str) -> PartitionResult:
        method = str(node.target)

        if method in _MATMUL_METHODS:
            return self._matmul_result(node, target_str)

        if method in _VORTEX_METHODS:
            return self._result(
                node, target_str, ExecutionTarget.VORTEX,
                f"{method}() (non-linear / dynamic elementwise)",
            )

        if method in _STRUCTURAL_METHODS:
            return self._result(node, target_str, ExecutionTarget.HOST, f"{method}() (structural)")

        return self._result(
            node, target_str, ExecutionTarget.HOST,
            f"{method}() (unrecognized method -- defaults to host)",
        )

    def _matmul_result(self, node: fx.Node, target_str: str) -> PartitionResult:
        """Route a matmul-family node: PHOTONIC if it contracts against a
        stored parameter/buffer (weight-stationary), VORTEX if both operands
        are activations (e.g. attention's Q @ K^T or Attn @ V -- structurally
        identical op, but no static operand to hold resident)."""
        name = _callable_name(node.target) if callable(node.target) else str(node.target)
        if self._has_static_operand(node):
            return self._result(
                node, target_str, ExecutionTarget.PHOTONIC,
                f"{name} against a parameter/buffer (static-weight GEMM)",
            )
        return self._result(
            node, target_str, ExecutionTarget.VORTEX,
            f"{name} between two activations (dynamic attention)",
        )

    @staticmethod
    def _has_static_operand(node: fx.Node) -> bool:
        """True if any argument is a ``get_attr`` node -- a parameter or
        buffer read directly out of the module, i.e. a static weight."""
        candidates = list(node.args) + list(node.kwargs.values())
        return any(isinstance(arg, fx.Node) and arg.op == "get_attr" for arg in candidates)

    @staticmethod
    def _target_str(node: fx.Node) -> str:
        if node.op == "call_module":
            return str(node.target)
        if callable(node.target):
            return _callable_name(node.target)
        return str(node.target)

    @staticmethod
    def _result(
        node: fx.Node, target_str: str, execution_target: ExecutionTarget, reason: str
    ) -> PartitionResult:
        return PartitionResult(node.name, node.op, target_str, execution_target, reason)


def print_partition(results: list[PartitionResult]) -> None:
    """Console-print a partition report: one row per node, in graph order."""
    name_w = max((len(r.node_name) for r in results), default=4)
    op_w = max((len(r.op) for r in results), default=2)
    target_w = max((len(r.target) for r in results), default=6)

    header = f"{'node':<{name_w}}  {'op':<{op_w}}  {'target':<{target_w}}  engine    reason"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.node_name:<{name_w}}  {r.op:<{op_w}}  {r.target:<{target_w}}  "
            f"{r.execution_target.name:<8}  {r.reason}"
        )

    counts = {t: 0 for t in ExecutionTarget}
    for r in results:
        counts[r.execution_target] += 1
    print("-" * len(header))
    print(
        "totals: "
        + "  ".join(f"{t.name}={counts[t]}" for t in ExecutionTarget)
        + f"  (nodes={len(results)})"
    )


# =============================================================================
# Verification block
# =============================================================================
if __name__ == "__main__":
    # Real 70B-class dims (gcb-wiki/06_70B_transformer_workload_trace.md) are
    # 8192 / 28672; kept small here so weight allocation and tracing are
    # instant -- the graph SHAPE this partitioner reasons about is identical
    # either way, only the tensor sizes would differ.
    DIM, HIDDEN_DIM = 64, 128

    class LlamaFFN(nn.Module):
        """LLaMA-style SwiGLU FFN: gate + up projections, SiLU gate, elementwise
        multiply, down projection. Every projection is a static-weight
        nn.Linear -> Photonic; SiLU and the gating multiply are dynamic
        elementwise -> Vortex."""

        def __init__(self, dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
            self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
            self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    class ToyAttentionScore(nn.Module):
        """Q @ K^T then softmax -- the other half of the routing story the
        FFN block above can't exercise on its own: torch.matmul between two
        DYNAMIC activations (no stored weight anywhere) must route to
        Vortex, even though it's the identical primitive a Linear layer's
        matmul against a static weight routes to Photonic."""

        def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
            scores = torch.matmul(q, k.transpose(-2, -1))
            return F.softmax(scores, dim=-1)

    print(f"=== LlamaFFN (dim={DIM}, hidden_dim={HIDDEN_DIM}) ===\n")
    ffn = LlamaFFN(DIM, HIDDEN_DIM)
    ffn_results = HybridPartitioner(ffn).partition()
    print_partition(ffn_results)

    print("\n=== ToyAttentionScore (Q @ K^T -> softmax) ===\n")
    attn = ToyAttentionScore()
    attn_results = HybridPartitioner(attn).partition()
    print_partition(attn_results)

    # Sanity: the two matmul-family call sites in these two graphs MUST
    # disagree on engine, despite being the same op family -- that disagreement
    # is the entire point of _has_static_operand. If this ever fails, the
    # partitioner has regressed to routing matmul on identity alone.
    ffn_photonic = {r.node_name for r in ffn_results if r.execution_target is ExecutionTarget.PHOTONIC}
    attn_vortex = {r.node_name for r in attn_results if r.execution_target is ExecutionTarget.VORTEX}
    assert len(ffn_photonic) == 3, f"expected 3 Photonic nodes (gate/up/down), got {ffn_photonic}"
    assert any("matmul" in r.target for r in attn_results if r.execution_target is ExecutionTarget.VORTEX), (
        "expected the Q @ K^T matmul to route Vortex"
    )
    print("\nverified: static-weight matmul (Linear) -> Photonic, "
          "dynamic activation-activation matmul (attention) -> Vortex")
