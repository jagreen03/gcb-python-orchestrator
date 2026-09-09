# Cyclopter persistence layer (PostgreSQL + pgvector)

The durable, cross-session record of every ticket's PBG outcome
(`gcb-wiki/09_pbg_telemetry_vector_space.md`). Orchestration queries it to ask
"what past work is nearest, in outcome shape, to what I'm about to attempt".

Control-plane state — it lives here, **never** in `gcb-rtl` /
`gcb-wdm-transceiver` / `gcb-host` (`gcb-wiki/04` boundary invariant).

## Layout

| Path | Role |
|---|---|
| `docker-compose.yml` | `pgvector/pgvector:pg16`, host port **5433** (avoids the Windows PG17 on 5432), password `cyclopter` |
| `migrations/001_pbg_telemetry.sql` | Doc 09 §5.1 table + §5.2 HNSW cosine/L2 indexes, verbatim; idempotent |
| `.env.example` | copy to `.env` (git-ignored); sets `CYCLOPTER_DSN` |
| `../../scripts/load_telemetry_pg.py` | applies the migration, bulk-upserts the JSONL, runs the KNN demo |

## Bring-up

```bash
# from the repo root (gcb-python-orchestrator/)

# 1. database
docker compose -f deploy/cyclopter/docker-compose.yml up -d

# 2. scrape the backlog -> JSONL  (TKT-007)
python scripts/parse_pbg_telemetry.py \
    --backlog ../gcb-workspaces/backlog --out telemetry_export.jsonl

# 3. migrate + load + demonstrate KNN  (idempotent; re-run any time)
python scripts/load_telemetry_pg.py

# teardown, keep data:   docker compose -f deploy/cyclopter/docker-compose.yml down
# teardown, wipe data:   docker compose -f deploy/cyclopter/docker-compose.yml down -v
```

`load_telemetry_pg.py` resolves its DSN from `--dsn`, then `$CYCLOPTER_DSN`,
then `deploy/cyclopter/.env`, then
`postgresql://postgres:cyclopter@localhost:5433/postgres`.

## Query it directly

```sql
-- flavour: 3 nearest outcomes to a target vector, pending excluded
SELECT ticket_id, title, telemetry_color, vec_norm
FROM   pbg_telemetry
WHERE  telemetry_color <> 'pending'
ORDER  BY telemetry_vec <=> '[1,0.5,0.5]'      -- <=> cosine distance
LIMIT  3;

-- flavour + intensity (origin included)
SELECT ticket_id, telemetry_color, vec_norm
FROM   pbg_telemetry
ORDER  BY telemetry_vec <-> '[1,1,0]'          -- <-> L2 distance
LIMIT  3;
```

## Notes

- Schema is **instantiated, not redesigned** — column set, `vector(3)`
  dimension, and distance ops come straight from Doc 09 §5.
- `vec_norm` is a `GENERATED ALWAYS ... STORED` column; the loader never writes
  it. The scraper emits a `norm` field purely for cross-checking.
- No Docker on this machine? Any PostgreSQL 16+ with the `vector` extension
  works — point `CYCLOPTER_DSN` at it. The verification run for this ticket
  used an ephemeral `pgserver` (bundled PG16 + pgvector) because the workstation
  has neither Docker nor host pgvector; results are identical to the compose
  image.
