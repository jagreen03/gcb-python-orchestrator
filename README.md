# gcb-python-orchestrator

Host-side Python control plane for GreenCoyote-Bay: compiles workloads onto
the Photonic/Vortex hybrid fabric, and owns the durable engineering-telemetry
persistence layer. Two largely independent roles sharing one repo — neither
is optional scaffolding, both are live.

## Role 1 — Compiler / scheduler (`compiler/`, `scheduler.py`)

Lowers a PyTorch workload onto the Photonic + Vortex hybrid engine:

| File | Role |
|---|---|
| `compiler/partitioner.py` | `torch.fx`-based `HybridPartitioner` — traces a `nn.Module`, tags every node PHOTONIC vs. VORTEX per the hybrid-partitioning doctrine (`gcb-wiki/06`) |
| `compiler/tiler.py` | `PhotonicTiler` — lowers one PHOTONIC `nn.Linear` node's `M x K -> M x N` shape into SRAM-controller loop bounds |
| `compiler/packer.py` | `DMAPacker` — serializes `PhotonicTiler` beats into hardware-aligned INT4 byte streams for DMA |
| `scheduler.py` | `TensorTiler` — splits a contraction longer than the optical beat's `K <= 16` phase-noise-limited width into host-accumulated chunks (see `gcb-photonic-qam-v2/intent.md`, "Software Tiling Constraint") |

## Role 2 — Cyclopter: the PBG telemetry persistence layer (`deploy/cyclopter/`, `scripts/`)

The durable, cross-session, queryable record of every ticket's PBG outcome
(`gcb-wiki/09_pbg_telemetry_vector_space.md`) — a local PostgreSQL +
pgvector Docker deployment. **Control-plane state; per `gcb-wiki/04`'s
boundary invariant, it lives here, never in `gcb-rtl` /
`gcb-wdm-transceiver` / `gcb-host`.** See `deploy/cyclopter/README.md` for
bring-up. Briefly:

| File | Role |
|---|---|
| `deploy/cyclopter/docker-compose.yml` | `pgvector/pgvector:pg16`, host port 5433 |
| `deploy/cyclopter/migrations/001_pbg_telemetry.sql` | the `pbg_telemetry` table + HNSW cosine/L2 indexes, per Doc 09 §5, verbatim |
| `scripts/parse_pbg_telemetry.py` | scrapes `gcb-workspaces/backlog/*.md` frontmatter into JSONL (TKT-007) |
| `scripts/load_telemetry_pg.py` | migrates + bulk-upserts the JSONL + runs a KNN demo |
| `scripts/cyclopter_scavenger.py` | restart-safe daemon: watches the backlog, auto-reconciles the DB on ticket closure (TKT-013) |
| `scripts/kg_gap_analysis_spike.py` | knowledge-graph gap-analysis spike (TKT-014) |

`GCB-039` (infrastructure/MLOps, `gcb-workspaces/backlog`) proposes
*extending* this same deployment with a semantic-retrieval layer (document
chunks + text embeddings) for architectural Q&A, explicitly additive to
(never replacing) `pbg_telemetry` and explicitly never authoritative over
the repo files it indexes — see that ticket for the full architecture
(proposed, not yet implemented).

## Other contents

- `scripts/gen_dma_params.py`, `scripts/dma_params.yaml`, `scripts/generate_cosim_payload.py` — GDDR7 tensor-DMA parameter generation / co-sim payload tooling (EPIC `TKT-017`/`TKT-018`).
- `scripts/templates/` — Jinja2 templates for generated RTL/config (the project's standing no-Chisel/no-Amaranth generation convention, `gcb-wiki/10`).

(Created 2026-09-11, GCB-039; this repo previously had no README.)
