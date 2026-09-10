#!/usr/bin/env bash
# =============================================================================
# Cyclopter persistence layer - one-click bring-up  (TKT-020 finding Y2.3 gate)
# =============================================================================
# Repeatable, documented start for the pgvector store that backs the Cyclopter
# knowledge graph. Replaces the ad-hoc ephemeral workaround used during
# bring-up (a `/tmp` pgserver whose migration had the Doc 09 STORED `vec_norm`
# expression hand-patched to `vector_norm()` because that pgserver build lacked
# `vector[i]` subscripting). This script instead:
#
#   1. starts the committed docker-compose service (pgvector/pgvector:pg16),
#   2. waits for it to report healthy,
#   3. applies the COMMITTED, UNMODIFIED migration
#      migrations/001_pbg_telemetry.sql via `docker exec ... psql` - the pg16
#      image has full pgvector, so the Doc 09 schema runs exactly as checked in,
#   4. verifies the extension / table / generated column / indexes exist.
#
# Idempotent - safe to re-run (compose `up -d` is a no-op if already up; the
# migration is all `CREATE ... IF NOT EXISTS`).
#
# Usage:
#   deploy/cyclopter/db_up.sh            # start + migrate + verify
#   deploy/cyclopter/db_up.sh --load    # + parse the backlog and load telemetry + KNN demo
#   deploy/cyclopter/db_up.sh --down     # stop the container, keep the volume
#   deploy/cyclopter/db_up.sh --wipe     # stop the container and delete the volume
#
# No Docker on this machine? Point CYCLOPTER_DSN at any PostgreSQL 16+ that has
# the `vector` extension and run:  python scripts/load_telemetry_pg.py
# (it applies the same committed migration before upserting).
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
COMPOSE_FILE="$HERE/docker-compose.yml"
MIGRATION="$HERE/migrations/001_pbg_telemetry.sql"
ENV_FILE="$HERE/.env"
ENV_EXAMPLE="$HERE/.env.example"

CONTAINER="cyclopter-db"                 # == docker-compose.yml container_name
PGUSER="postgres"
PGDB="postgres"
HOST_PORT="5433"                         # == docker-compose.yml ports mapping
DSN="postgresql://postgres:cyclopter@localhost:${HOST_PORT}/${PGDB}"
HEALTH_TIMEOUT=90

die()  { printf '\n[db_up] ERROR: %s\n' "$*" >&2; exit 1; }
info() { printf '[db_up] %s\n' "$*"; }

# --- resolve a Compose driver -----------------------------------------------
compose() {
  if docker compose version >/dev/null 2>&1; then
    docker compose "$@"
  elif command -v docker-compose >/dev/null 2>&1; then
    docker-compose "$@"
  else
    die "need Docker with the Compose plugin ('docker compose') or 'docker-compose' on PATH. Install Docker Desktop / Engine, or use the no-Docker path in this script's header."
  fi
}

# --- optional teardown modes ----------------------------------------------------
case "${1:-}" in
  --down)  command -v docker >/dev/null 2>&1 || die "docker CLI not on PATH"
           info "stopping ${CONTAINER} (keeping the data volume) ..."
           compose -f "$COMPOSE_FILE" down
           exit 0 ;;
  --wipe)  command -v docker >/dev/null 2>&1 || die "docker CLI not on PATH"
           info "stopping ${CONTAINER} and DELETING the data volume ..."
           compose -f "$COMPOSE_FILE" down -v
           exit 0 ;;
esac

# --- 0. preflight ------------------------------------------------------------
command -v docker >/dev/null 2>&1 || die "docker CLI not on PATH"
[ -f "$COMPOSE_FILE" ] || die "compose file not found: $COMPOSE_FILE"
[ -f "$MIGRATION" ]    || die "migration not found: $MIGRATION"

# keep load_telemetry_pg.py's DSN resolution in sync (it reads .env)
if [ ! -f "$ENV_FILE" ] && [ -f "$ENV_EXAMPLE" ]; then
  cp "$ENV_EXAMPLE" "$ENV_FILE"
  info "created $(basename "$ENV_FILE") from $(basename "$ENV_EXAMPLE")"
fi

# --- 1. start the container ------------------------------------------------------
info "starting '${CONTAINER}' (pgvector/pgvector:pg16) on host port ${HOST_PORT} ..."
compose -f "$COMPOSE_FILE" up -d

# --- 2. wait for the compose healthcheck (pg_isready) --------------------------
info "waiting up to ${HEALTH_TIMEOUT}s for '${CONTAINER}' to become healthy ..."
deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
while :; do
  status="$(docker inspect -f '{{.State.Health.Status}}' "$CONTAINER" 2>/dev/null || echo missing)"
  case "$status" in
    healthy) info "'${CONTAINER}' is healthy"; break ;;
    missing) die "'${CONTAINER}' is not running - inspect: compose -f '$COMPOSE_FILE' logs ${CONTAINER}" ;;
  esac
  [ "$(date +%s)" -lt "$deadline" ] || die "timed out after ${HEALTH_TIMEOUT}s (last health status: ${status}). Inspect: compose -f '$COMPOSE_FILE' logs ${CONTAINER}"
  sleep 2
done

# --- 3. apply the COMMITTED migration verbatim --------------------------------
# pgvector/pgvector:pg16 has full pgvector -> the Doc 09 STORED vec_norm
# expression (telemetry_vec[1]*... subscripting) in 001_pbg_telemetry.sql runs
# as checked in. No patched workaround.
info "applying $(basename "$MIGRATION") via 'docker exec ... psql' ..."
docker exec -i "$CONTAINER" \
  psql -v ON_ERROR_STOP=1 -U "$PGUSER" -d "$PGDB" -q -f - < "$MIGRATION" \
  || die "migration failed - see psql output above"

# --- 4. verify the schema instantiated --------------------------------------
info "verifying schema ..."
report="$(docker exec -i "$CONTAINER" \
  psql -v ON_ERROR_STOP=1 -U "$PGUSER" -d "$PGDB" -qtA <<'SQL'
SELECT 'extension_vector=' || count(*) FROM pg_extension WHERE extname = 'vector';
SELECT 'table_pbg_telemetry=' || coalesce(to_regclass('public.pbg_telemetry')::text, 'MISSING');
SELECT 'vec_norm_generated=' || count(*) FROM information_schema.columns
       WHERE table_name = 'pbg_telemetry' AND column_name = 'vec_norm' AND is_generated = 'ALWAYS';
SELECT 'hnsw_indexes=' || count(*) FROM pg_indexes
       WHERE tablename = 'pbg_telemetry' AND indexname IN ('pbg_telemetry_cos', 'pbg_telemetry_l2');
SELECT 'row_count=' || count(*) FROM pbg_telemetry;
SQL
)"
printf '%s\n' "$report" | sed 's/^/[db_up]   /'
printf '%s\n' "$report" | grep -q 'table_pbg_telemetry=MISSING' && die "verification failed: pbg_telemetry not created"
printf '%s\n' "$report" | grep -q 'vec_norm_generated=1'        || die "verification failed: generated vec_norm column absent (committed migration did not apply cleanly)"
printf '%s\n' "$report" | grep -q 'hnsw_indexes=2'              || die "verification failed: expected both HNSW indexes (cosine + L2)"

info "pgvector is up and migrated with the committed schema (no workaround)."
printf '[db_up]   DSN: %s\n' "$DSN"

# --- 5. optional: parse the backlog + load telemetry + KNN demo --------------
if [ "${1:-}" = "--load" ]; then
  BACKLOG="$REPO_ROOT/../gcb-workspaces/backlog"
  JSONL="$REPO_ROOT/telemetry_export.jsonl"
  PY=""
  for cand in "$REPO_ROOT/.venv/Scripts/python.exe" "$REPO_ROOT/.venv/bin/python" python3 python; do
    if command -v "$cand" >/dev/null 2>&1 || [ -x "$cand" ]; then PY="$cand"; break; fi
  done
  [ -n "$PY" ] || die "--load: no python interpreter found (tried .venv, python3, python)"
  [ -d "$BACKLOG" ] || die "--load: backlog dir not found: $BACKLOG"
  info "--load: scraping backlog -> $(basename "$JSONL") ..."
  "$PY" "$REPO_ROOT/scripts/parse_pbg_telemetry.py" --backlog "$BACKLOG" --out "$JSONL"
  info "--load: migrate (again, idempotent) + upsert + KNN demo ..."
  CYCLOPTER_DSN="$DSN" "$PY" "$REPO_ROOT/scripts/load_telemetry_pg.py" --dsn "$DSN" --jsonl "$JSONL"
fi

cat <<EOF

[db_up] ready.
        DSN                : $DSN
        stop (keep data)   : deploy/cyclopter/db_up.sh --down
        stop (wipe volume) : deploy/cyclopter/db_up.sh --wipe

        load telemetry now :
          python scripts/parse_pbg_telemetry.py --backlog ../gcb-workspaces/backlog --out telemetry_export.jsonl
          python scripts/load_telemetry_pg.py
EOF
