#!/usr/bin/env python3
"""Load PBG telemetry JSONL into the Cyclopter pgvector store and run KNN.

Instantiates gcb-wiki/09_pbg_telemetry_vector_space.md section 5 (the schema is
in deploy/cyclopter/migrations/001_pbg_telemetry.sql - this script applies it,
then bulk-upserts). Idempotent: re-running after a fresh scrape converges, no
duplicates (ON CONFLICT (ticket_id) DO UPDATE).

The store is control-plane state and lives here (gcb-python-orchestrator), never
in gcb-rtl / gcb-wdm-transceiver / gcb-host - gcb-wiki/04 boundary invariant.

Typical flow
------------
    docker compose -f deploy/cyclopter/docker-compose.yml up -d
    python scripts/parse_pbg_telemetry.py --backlog ../gcb-workspaces/backlog \
        --out telemetry_export.jsonl
    python scripts/load_telemetry_pg.py            # migrate + upsert + KNN demo

DSN resolution order: --dsn  >  $CYCLOPTER_DSN  >  deploy/cyclopter/.env
(CYCLOPTER_DSN=...)  >  postgresql://postgres:cyclopter@localhost:5433/postgres
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import psycopg
except ImportError:  # pragma: no cover
    sys.exit("psycopg (v3) required:  pip install 'psycopg[binary]'")

REPO_ROOT = Path(__file__).resolve().parents[1]
MIGRATION = REPO_ROOT / "deploy" / "cyclopter" / "migrations" / "001_pbg_telemetry.sql"
ENV_FILE = REPO_ROOT / "deploy" / "cyclopter" / ".env"
JSONL_DEFAULT = REPO_ROOT / "telemetry_export.jsonl"
DSN_FALLBACK = "postgresql://postgres:cyclopter@localhost:5433/postgres"

UPSERT = """
INSERT INTO pbg_telemetry (ticket_id, title, status, telemetry_vec, telemetry_color, updated_at)
VALUES (%(ticket_id)s, %(title)s, %(status)s, %(vec)s::vector, %(telemetry_color)s, %(updated_at)s)
ON CONFLICT (ticket_id) DO UPDATE SET
    title           = EXCLUDED.title,
    status          = EXCLUDED.status,
    telemetry_vec   = EXCLUDED.telemetry_vec,
    telemetry_color = EXCLUDED.telemetry_color,
    updated_at      = EXCLUDED.updated_at;
"""

KNN_COSINE = """
SELECT ticket_id, title, telemetry_color, vec_norm,
       telemetry_vec <=> %(t)s::vector AS cos_dist
FROM   pbg_telemetry
WHERE  telemetry_color <> 'pending'
ORDER  BY telemetry_vec <=> %(t)s::vector
LIMIT  %(k)s;
"""

KNN_L2 = """
SELECT ticket_id, telemetry_color, vec_norm,
       telemetry_vec <-> %(t)s::vector AS l2_dist
FROM   pbg_telemetry
ORDER  BY telemetry_vec <-> %(t)s::vector
LIMIT  %(k)s;
"""


def resolve_dsn(cli_dsn: str | None) -> str:
    if cli_dsn:
        return cli_dsn
    if os.environ.get("CYCLOPTER_DSN"):
        return os.environ["CYCLOPTER_DSN"]
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("CYCLOPTER_DSN=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip()
    return DSN_FALLBACK


def vec_literal(v) -> str:
    """[x, y, z] -> '[x,y,z]' as pgvector expects."""
    return "[" + ",".join(repr(float(c)) for c in v) + "]"


def sql_statements(text: str) -> list[str]:
    """Split a migration script into statements. psycopg3 uses the extended
    protocol, which rejects multi-statement strings; the migration has no
    string literals or dollar-quoting, so splitting on ';' outside comments is
    safe here."""
    out, buf = [], []
    for line in text.splitlines():
        if line.lstrip().startswith("--"):
            continue
        buf.append(line)
        if line.rstrip().endswith(";"):
            stmt = "\n".join(buf).strip().rstrip(";").strip()
            if stmt:
                out.append(stmt)
            buf = []
    tail = "\n".join(buf).strip().rstrip(";").strip()
    if tail:
        out.append(tail)
    return out


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"JSONL not found: {path}\n"
                 f"run:  python scripts/parse_pbg_telemetry.py --backlog "
                 f"../gcb-workspaces/backlog --out {path.name}")
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            rows.append({
                "ticket_id": r["ticket_id"],
                "title": r["title"],
                "status": r["status"],
                "vec": vec_literal(r["telemetry_vec"]),
                "telemetry_color": r["telemetry_color"],
                "updated_at": r["updated_at"],
            })
        except (json.JSONDecodeError, KeyError) as exc:
            sys.exit(f"{path}:{n}: bad row: {exc}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", help="PostgreSQL DSN (overrides $CYCLOPTER_DSN and .env)")
    ap.add_argument("--jsonl", type=Path, default=JSONL_DEFAULT,
                    help=f"telemetry JSONL to load (default: {JSONL_DEFAULT.name})")
    ap.add_argument("--migration", type=Path, default=MIGRATION,
                    help="migration SQL to apply (default: deploy/cyclopter/migrations/001_pbg_telemetry.sql)")
    ap.add_argument("--target", default="[1,0.5,0.5]",
                    help="KNN query vector (default: [1,0.5,0.5] - Doc 09 sec.5.2 example)")
    ap.add_argument("--k", type=int, default=3, help="KNN neighbours to return (default: 3)")
    ap.add_argument("--no-load", action="store_true",
                    help="skip migrate + upsert; just run the KNN queries")
    args = ap.parse_args()

    dsn = resolve_dsn(args.dsn)
    safe = dsn.split("@")[-1] if "@" in dsn else dsn
    print(f"connect: ...@{safe}")

    try:
        conn = psycopg.connect(dsn, autocommit=True, connect_timeout=10)
    except psycopg.OperationalError as exc:
        sys.exit(f"cannot connect ({safe}): {exc}\n"
                 f"is the container up?  docker compose -f "
                 f"deploy/cyclopter/docker-compose.yml up -d")

    with conn:
        if not args.no_load:
            for stmt in sql_statements(args.migration.read_text(encoding="utf-8")):
                conn.execute(stmt)
            try:
                shown = args.migration.relative_to(REPO_ROOT)
            except ValueError:
                shown = args.migration
            print(f"migrated: {shown}")

            rows = read_rows(args.jsonl)
            with conn.cursor() as cur:
                cur.executemany(UPSERT, rows)
            n = conn.execute("SELECT count(*) FROM pbg_telemetry").fetchone()[0]
            print(f"upserted {len(rows)} row(s) from {args.jsonl.name}; "
                  f"pbg_telemetry now holds {n} row(s)")

        t = args.target.replace(" ", "")
        print(f"\ncosine KNN  (telemetry_vec <=> '{t}', pending excluded, k={args.k}):")
        print(f"  {'ticket':<9} {'color':<7} {'norm':>6} {'cos_dist':>10}  title")
        for tid, title, color, norm, dist in conn.execute(
                KNN_COSINE, {"t": t, "k": args.k}).fetchall():
            print(f"  {tid:<9} {color:<7} {norm:>6.3f} {dist:>10.6f}  {title[:56]}")

        print(f"\nL2 KNN  (telemetry_vec <-> '{t}', origin included, k={args.k}):")
        print(f"  {'ticket':<9} {'color':<7} {'norm':>6} {'l2_dist':>10}")
        for tid, color, norm, dist in conn.execute(
                KNN_L2, {"t": t, "k": args.k}).fetchall():
            print(f"  {tid:<9} {color:<7} {norm:>6.3f} {dist:>10.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
