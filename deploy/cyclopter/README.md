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

### One-click (recommended)

```bash
# from anywhere - starts the container, waits for healthy, applies the
# COMMITTED migration verbatim via `docker exec ... psql`, verifies the schema.
deploy/cyclopter/db_up.sh            # start + migrate + verify
deploy/cyclopter/db_up.sh --load    # + scrape the backlog and load telemetry + KNN demo
deploy/cyclopter/db_up.sh --down    # stop, keep the data volume
deploy/cyclopter/db_up.sh --wipe    # stop and delete the data volume
```

`db_up.sh` is idempotent and repeatable; it deliberately applies the
unmodified `migrations/001_pbg_telemetry.sql` (the `pgvector/pgvector:pg16`
image has full pgvector, so the Doc 09 `vec_norm` STORED expression runs
as checked in — no ephemeral workaround).

### Manual

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

## Scavenger agent (TKT-013)

`../../scripts/cyclopter_scavenger.py` is a restart-safe daemon that removes the
manual scrape+load ritual: it polls `../gcb-workspaces/backlog/*.md`
(stdlib `stat` + content hash — no `watchdog` dependency) and, when a ticket
**transitions into a graded, closed state** (`status: done` **and**
`telemetry_color != pending`) or an already-closed ticket's file changes, it
debounces, takes a PID lock, and runs `parse_pbg_telemetry.py` +
`load_telemetry_pg.py` unchanged so the store converges with no human step.
Per-ticket `(hash, status, color)` is persisted to `.scavenger_state.json`, so a
kill+restart re-reads state, finds the DB already converged, and does not
double-process; `SIGINT`/`SIGTERM` finish the current pass, save state, drop the
lock, exit 0.

```bash
# one reconcile pass then exit (CI / cron)
python scripts/cyclopter_scavenger.py --once

# long-running watch, 5 s poll, emit the diversification artifact on each closure
python scripts/cyclopter_scavenger.py --interval 5 --noise 5

# preview only — detect + log, never scrape / upsert / write
python scripts/cyclopter_scavenger.py --once --dry-run
```

### Stochastic-noise artifact (retrieval diversification)

`--noise K` makes the agent write, **after a successful upsert**, a
timestamped, provenance-tagged JSON file under `deploy/cyclopter/noise/`
(git-ignored). It is **retrieval diversification, not perturbation**: a
read-only `SELECT` sample of `pbg_telemetry` meant to widen what a downstream
RAG planner sees. It is **never** written back into `pbg_telemetry` and it never
alters a stored vector.

Contents:

| part | how it is chosen |
|---|---|
| **K cosine neighbours** | exact `telemetry_vec <=> query` nearest non-`pending` rows. Query vector = the trigger ticket's own row (default), `--noise-vector "[x,y,z]"`, or the most-recently-updated non-`pending` row when there is no trigger. The trigger row itself is excluded. |
| **M random rows** (`--noise-random`, default 3) | a `random.Random(seed)` uniform draw from the remaining non-`pending` rows; `seed` (default: UTC `YYYYMMDD`, or `--seed`) is recorded in the file. |

Each entry carries a `provenance` tag (`cosine_neighbor` + rank + `cos_dist`, or
`random_sample` + `seed` + `pick_index`). The header records the query vector and
its source, K, M, the seed, the DB host (never credentials), and the
`gcb-workspaces` git HEAD. Schema id: `cyclopter.scavenger.noise/v1`.

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
