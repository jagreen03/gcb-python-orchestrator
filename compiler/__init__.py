"""GCB PyTorch compiler backend.

Maps a standard ``torch.nn.Module`` (a Transformer block) onto the GCB hybrid
architecture: the Photonic Tensor Complex (static-weight dense linear
algebra) and the digital Vortex cores (dynamic / data-dependent / non-linear
compute) -- see ``gcb-wiki/06_70B_transformer_workload_trace.md`` sec. 2 for
the partitioning doctrine this package implements.
"""

from __future__ import annotations

from .partitioner import ExecutionTarget, HybridPartitioner, PartitionResult

__all__ = ["ExecutionTarget", "HybridPartitioner", "PartitionResult"]
