#!/usr/bin/env python3
"""Cyclopter scavenger agent (TKT-013).

The first *active* Cyclopter agent: a restart-safe daemon that watches the PBG
backlog and keeps the ``pgvector`` store converged with no human step.

What it does
------------
1. **Auto-ingest on closure.** It polls ``../gcb-workspaces/backlog/*.md`` on an
   interval (stdlib ``stat`` + content hash - no ``watchdog`` dependency, and
   robust on WSL/Windows mounts where ``inotify`` is unreliable). When a ticket
   *transitions* into a **graded, closed** state (frontmatter ``status: done``
   **and** ``telemetry_color`` != ``pending``), or an already-closed ticket's
   file content changes (a re-grade), it debounces, takes a lock, and invokes
   the existing loader chain unchanged::

       python scripts/parse_pbg_telemetry.py --backlog <dir> --out telemetry_export.jsonl
       python scripts/load_telemetry_pg.py   --jsonl telemetry_export.jsonl --dsn <dsn>

   The upsert is ``ON CONFLICT (ticket_id) DO UPDATE`` (TKT-011), so re-running
   converges - no duplicate rows.

2. **Bounded retrieval-diversification artifact ("stochastic noise").**
   *Opt-in* (``--noise K``). After a successful upsert it writes a timestamped,
   provenance-tagged JSON file under ``deploy/cyclopter/noise/`` containing:

     * **K cosine neighbours** - the exact ``telemetry_vec <=> query`` nearest
       non-``pending`` rows to a query vector (the trigger ticket's own vector
       by default; ``--noise-vector`` to override; the most-recently-updated
       non-``pending`` row when there is no trigger).
     * **M uniformly-random non-``pending`` rows** (``--noise-random M``, default
       3) drawn with a recorded ``random.Random(seed)`` so the file is
       reproducible.

   Every entry is tagged with how it was selected (``cosine_neighbor`` + rank +
   distance, or ``random_sample`` + seed + pick index). The header records the
   query vector and its source, K, M, the seed, the DB host (never credentials),
   and the ``gcb-workspaces`` git HEAD.

   This is **retrieval diversification only** - a read-only SELECT sample of the
   corpus, meant to widen what a downstream RAG planner sees. It is an *output
   file*. It is **never** an ``INSERT``/``UPDATE`` into ``pbg_telemetry`` and it
   never perturbs a stored vector. The planner that consumes it is a separate
   ticket.

Safety / robustness
-------------------
* **Restart-safe.** Per-ticket ``(hash, status, color, last_hash_processed)`` is
  persisted to ``deploy/cyclopter/.scavenger_state.json``. On start it
  reconciles the whole corpus once (idempotent), then only *new* transitions
  fire. Kill it mid-watch and restart: it re-reads state, the DB is already
  converged, nothing double-processes; a transition that happened while it was
  down is caught on the next poll.
* **Single-run lock.** A PID lockfile (``deploy/cyclopter/.scavenger.lock``)
  refuses a second instance whose predecessor is still alive (``--force`` to
  override a stale lock). An in-process lock guards the scrape/upsert/noise
  critical section against signal re-entrancy.
* **Debounce.** A burst of editor writes to one file collapses into a single
  run once the file's hash has been stable for ``--debounce`` seconds.
* **Clean shutdown.** ``SIGINT`` / ``SIGTERM`` finish the current iteration,
  persist state, drop the lock, exit 0.
* **Structured logs.** One ``event=<name> key=value ...`` line per event
  (``startup`` / ``reconcile`` / ``detected`` / ``scraped`` / ``upserted`` /
  ``noise-emitted`` / ``dry-run`` / ``shutdown`` ...).

Boundary: pure control-plane, ``gcb-python-orchestrator`` only (``gcb-wiki/04``).
It does not modify the Doc 09 schema, ``parse_pbg_telemetry.py``, or
``load_telemetry_pg.py``'s contract - it only calls them.

CLI
---
    --once              one reconcile pass (+ optional --noise), then exit (CI)
    --interval N        poll period, seconds (default 5)
    --debounce N        quiet-hash seconds before a change fires (default 2.0)
    --backlog DIR       backlog dir (default ../gcb-workspaces/backlog)
    --dsn DSN           overrides $CYCLOPTER_DSN / .env / built-in default
    --migration PATH    alternate migration forwarded to load_telemetry_pg.py
    --dry-run           detect + log, but never scrape / upsert / write noise
    --noise K           emit the diversification artifact with K cosine neighbours
    --noise-random M    random non-pending rows in the artifact (default 3)
    --noise-vector V    query vector for the artifact, e.g. "[1,0.5,0.5]"
    --seed N            RNG seed for the random sample (default: real OS
                        entropy, different each run - TKT-020 R3; pass a
                        fixed value here to reproduce a specific draw)
    --force             steal a stale lock
    --verbose           DEBUG logging
"""

from __future__ import annotations

import argparse
import atexit
import errno
import hashlib
import json
import logging
import os
import random
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

# Reuse the existing tooling - never re-implement the frontmatter parser
# (TKT-007 Yellow reject) or the DSN / vector helpers (TKT-011 contract).
from parse_pbg_telemetry import (  # noqa: E402
    _load_frontmatter,
    normalize_color,
    TKT_RE,
)
from load_telemetry_pg import (  # noqa: E402
    resolve_dsn,
    vec_literal,
    KNN_COSINE,
)

DEFAULT_BACKLOG = REPO_ROOT.parent / "gcb-workspaces" / "backlog"
CYCLOPTER_DIR = REPO_ROOT / "deploy" / "cyclopter"
STATE_FILE = CYCLOPTER_DIR / ".scavenger_state.json"
LOCK_FILE = CYCLOPTER_DIR / ".scavenger.lock"
NOISE_DIR = CYCLOPTER_DIR / "noise"
EXPORT_JSONL = REPO_ROOT / "telemetry_export.jsonl"
PARSER_SCRIPT = SCRIPTS_DIR / "parse_pbg_telemetry.py"
LOADER_SCRIPT = SCRIPTS_DIR / "load_telemetry_pg.py"

TOOL = "cyclopter_scavenger.py"
VERSION = "1.0"
NOISE_SCHEMA = "cyclopter.scavenger.noise/v1"

log = logging.getLogger("cyclopter.scavenger")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "posix":
        try:
            os.kill(pid, 0)
        except OSError as exc:
            return exc.errno == errno.EPERM
        return True
    # best-effort Windows check
    try:
        import ctypes

        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if h:
            ctypes.windll.kernel32.CloseHandle(h)
            return True
        return False
    except Exception:
        return False


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ticket_fields(path: Path) -> tuple[str, str, str]:
    """(ticket_id, status, normalized_color) from a backlog .md's frontmatter."""
    try:
        meta = _load_frontmatter(path) or {}
    except Exception as exc:  # malformed YAML -> treat as unknown, log once
        log.warning("event=parse-error file=%s err=%s", path.name, exc)
        meta = {}
    tid = str(meta.get("id") or "").strip().upper()
    if not tid:
        m = TKT_RE.search(path.name)
        tid = m.group(1).upper() if m else path.stem.upper()
    status = str(meta.get("status") or "").strip().lower()
    color = normalize_color(meta.get("telemetry_color")) or "pending"
    return tid, status, color


def _is_closed_graded(status: str, color: str) -> bool:
    return status == "done" and color not in ("", "pending")


def _vec_to_list(v: Any) -> list[float]:
    """pgvector returns its text form '[1,2,3]'; make it a JSON list."""
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    try:
        return [float(x) for x in json.loads(str(v))]
    except Exception:
        return []


def _git_head(repo_dir: Path) -> str | None:
    try:
        r = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False, timeout=5,
        )
        return r.stdout.strip() or None if r.returncode == 0 else None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# agent
# --------------------------------------------------------------------------- #
class Scavenger:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.backlog: Path = args.backlog.resolve()
        self.dsn = resolve_dsn(args.dsn)
        self.dsn_host = self.dsn.split("@")[-1] if "@" in self.dsn else self.dsn
        self.migration: Path | None = args.migration.resolve() if args.migration else None
        self.interval = max(1, int(args.interval))
        self.debounce = max(0.0, float(args.debounce))
        self.dry_run = bool(args.dry_run)
        self.noise_k = int(args.noise) if args.noise is not None else 0
        self.noise_m = int(args.noise_random)
        self._stop = False
        self._have_lock = False
        self._pending: dict[str, dict[str, Any]] = {}   # tid -> {"fields":..., "hash":..., "deadline":ts}
        self.state: dict[str, Any] = {"tickets": {}, "updated_at": None}

    # ---- lock -------------------------------------------------------------
    def acquire_lock(self) -> None:
        CYCLOPTER_DIR.mkdir(parents=True, exist_ok=True)
        if LOCK_FILE.exists():
            try:
                old = int(LOCK_FILE.read_text(encoding="utf-8").split()[0])
            except Exception:
                old = -1
            if old > 0 and _pid_alive(old) and not self.args.force:
                log.error("event=lock-held pid=%d file=%s (use --force to steal)",
                          old, LOCK_FILE)
                raise SystemExit(3)
            log.warning("event=lock-stale pid=%s taking over", old)
        LOCK_FILE.write_text(f"{os.getpid()} {_now_iso()}\n", encoding="utf-8")
        self._have_lock = True
        atexit.register(self.release_lock)

    def release_lock(self) -> None:
        if not self._have_lock:
            return
        try:
            if LOCK_FILE.exists():
                owner = int(LOCK_FILE.read_text(encoding="utf-8").split()[0])
                if owner == os.getpid():
                    LOCK_FILE.unlink()
        except Exception:
            pass
        self._have_lock = False

    # ---- state ----------------------------------------------------------
    def load_state(self) -> None:
        if STATE_FILE.exists():
            try:
                self.state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
                self.state.setdefault("tickets", {})
            except Exception as exc:
                log.warning("event=state-unreadable file=%s err=%s (starting fresh)",
                            STATE_FILE.name, exc)
                self.state = {"tickets": {}, "updated_at": None}

    def save_state(self) -> None:
        if self.dry_run:                      # --dry-run never mutates on-disk state
            return
        self.state["updated_at"] = _now_iso()
        tmp = STATE_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(STATE_FILE)

    # ---- scan ---------------------------------------------------------
    def scan(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for p in sorted(self.backlog.glob("*.md")):
            tid, status, color = _ticket_fields(p)
            out[tid] = {
                "path": str(p),
                "hash": _hash_file(p),
                "status": status,
                "color": color,
                "closed_graded": _is_closed_graded(status, color),
            }
        return out

    def _record(self, tid: str, cur: dict[str, Any], *, processed: bool) -> None:
        entry = {
            "hash": cur["hash"],
            "status": cur["status"],
            "color": cur["color"],
            "closed_graded": cur["closed_graded"],
        }
        prev = self.state["tickets"].get(tid, {})
        entry["last_hash_processed"] = (
            cur["hash"] if processed else prev.get("last_hash_processed")
        )
        entry["last_processed"] = _now_iso() if processed else prev.get("last_processed")
        self.state["tickets"][tid] = entry

    # ---- subprocess steps ------------------------------------------
    @staticmethod
    def _run(cmd: list[str]) -> subprocess.CompletedProcess:
        log.debug("event=exec cmd=%s", " ".join(cmd))
        return subprocess.run(cmd, capture_output=True, text=True, check=False)

    def scrape(self) -> int:
        r = self._run([sys.executable, str(PARSER_SCRIPT),
                       "--backlog", str(self.backlog), "--out", str(EXPORT_JSONL)])
        if r.returncode != 0:
            log.error("event=scrape-failed rc=%d err=%s", r.returncode,
                      (r.stderr or "").strip()[:400])
            raise RuntimeError("scrape failed")
        rows = sum(1 for ln in EXPORT_JSONL.read_text(encoding="utf-8").splitlines()
                   if ln.strip())
        log.info("event=scraped rows=%d out=%s", rows, EXPORT_JSONL.name)
        return rows

    def upsert(self) -> int:
        cmd = [sys.executable, str(LOADER_SCRIPT),
               "--jsonl", str(EXPORT_JSONL), "--dsn", self.dsn]
        if self.migration:
            cmd += ["--migration", str(self.migration)]
        r = self._run(cmd)
        if r.returncode != 0:
            log.error("event=upsert-failed rc=%d err=%s", r.returncode,
                      (r.stderr or "").strip()[:400])
            raise RuntimeError("upsert failed")
        m = re.search(r"holds (\d+) row", r.stdout or "")
        db_rows = int(m.group(1)) if m else -1
        log.info("event=upserted db_rows=%d", db_rows)
        return db_rows

    # ---- noise artifact -------------------------------------------
    def emit_noise(self, trigger: dict[str, Any] | None) -> Path | None:
        if self.noise_k <= 0 and self.noise_m <= 0:
            return None
        try:
            import psycopg
        except ImportError:
            log.error("event=noise-skipped reason=no-psycopg")
            return None

        # TKT-020 R3 containment: the prior default (bare UTC YYYYMMDD) made
        # the "random" sample identical for the full 24h day - a fixed bias
        # wearing a random_sample label (R3 failure scenario (b)). Per R3's
        # own stated alternative ("draw without a fixed seed and record the
        # drawn ids"): an explicit --seed still gives full reproducibility;
        # absent that, seed from real OS entropy (random.Random(None) uses
        # os.urandom under the hood) so two runs on the same day draw
        # different samples. The actual seed used is still recorded below
        # (rand_meta["seed"] / the emitted artifact's header) - if you need
        # to reproduce a specific draw later, read that value and pass it
        # back via --seed.
        seed = (int(self.args.seed) if self.args.seed is not None
                else random.Random().randrange(1, 2**31))
        try:
            with psycopg.connect(self.dsn, autocommit=True, connect_timeout=10) as conn:
                # -- query vector + its provenance --
                if self.args.noise_vector:
                    qvec = _vec_to_list(self.args.noise_vector)
                    vsrc = "explicit --noise-vector"
                elif trigger is not None:
                    row = conn.execute(
                        "SELECT telemetry_vec FROM pbg_telemetry WHERE ticket_id = %s",
                        (trigger["ticket_id"],)).fetchone()
                    qvec = _vec_to_list(row[0]) if row else []
                    vsrc = f"trigger ticket {trigger['ticket_id']} pbg_telemetry row"
                else:
                    row = conn.execute(
                        "SELECT ticket_id, telemetry_vec FROM pbg_telemetry "
                        "WHERE telemetry_color <> 'pending' "
                        "ORDER BY updated_at DESC LIMIT 1").fetchone()
                    qvec = _vec_to_list(row[1]) if row else []
                    vsrc = (f"most-recently-updated non-pending row ({row[0]})"
                            if row else "none (empty store)")
                if not qvec:
                    log.error("event=noise-skipped reason=no-query-vector")
                    return None

                entries: list[dict[str, Any]] = []
                picked: set[str] = set()
                if trigger is not None:
                    picked.add(trigger["ticket_id"])

                # -- K exact cosine neighbours (reuse the loader's query) --
                # over-fetch by 1 so the trigger row itself can be dropped and
                # the artifact still carries K distinct *other* rows.
                if self.noise_k > 0:
                    rows = conn.execute(
                        KNN_COSINE,
                        {"t": vec_literal(qvec), "k": self.noise_k + (1 if trigger else 0)}
                    ).fetchall()
                    if trigger is not None:
                        rows = [r for r in rows if r[0] != trigger["ticket_id"]]
                    rows = rows[: self.noise_k]
                    for rank, (tid, title, color, norm, cdist) in enumerate(rows, 1):
                        picked.add(tid)
                        entries.append({
                            "provenance": "cosine_neighbor",
                            "rank": rank,
                            "cos_dist": round(float(cdist), 9),
                            "ticket_id": tid,
                            "title": title,
                            "telemetry_color": color,
                            "vec_norm": round(float(norm), 9),
                        })

                # -- M seeded-uniform random non-pending rows --
                rand_meta: dict[str, Any] = {"seed": seed, "count": 0}
                if self.noise_m > 0:
                    pool = conn.execute(
                        "SELECT ticket_id, title, telemetry_color, telemetry_vec, vec_norm "
                        "FROM pbg_telemetry WHERE telemetry_color <> 'pending' "
                        "ORDER BY ticket_id").fetchall()
                    cand = [r for r in pool if r[0] not in picked]
                    rnd = random.Random(seed)
                    draw = rnd.sample(cand, min(self.noise_m, len(cand))) if cand else []
                    rand_meta["count"] = len(draw)
                    for i, (tid, title, color, vec, norm) in enumerate(draw):
                        entries.append({
                            "provenance": "random_sample",
                            "seed": seed,
                            "pick_index": i,
                            "ticket_id": tid,
                            "title": title,
                            "telemetry_color": color,
                            "telemetry_vec": _vec_to_list(vec),
                            "vec_norm": round(float(norm), 9),
                        })
        except Exception as exc:
            log.error("event=noise-failed err=%s", exc)
            return None

        artifact = {
            "schema": NOISE_SCHEMA,
            "generated_at": _now_iso(),
            "generator": {"tool": TOOL, "version": VERSION, "pid": os.getpid()},
            "trigger": (
                {
                    "ticket_id": trigger["ticket_id"],
                    "status": trigger["status"],
                    "telemetry_color": trigger["color"],
                    "reason": "transition into graded+closed observed by watch",
                }
                if trigger is not None else
                {"ticket_id": None, "reason": "manual / --once noise pass"}
            ),
            "backlog_git_head": _git_head(self.backlog.parent),
            "query": {
                "vector": qvec,
                "vector_source": vsrc,
                "metric": "cosine distance (pgvector <=>)",
                "pending_excluded": True,
            },
            "sampling": {
                "k_cosine": self.noise_k,
                "m_random": rand_meta["count"],
                "rng_seed": rand_meta["seed"],
                "db_host": self.dsn_host,
            },
            "entries": entries,
            "notes": (
                "Retrieval-diversification sample of pbg_telemetry for RAG "
                "planning context. READ-ONLY: this file is never written back "
                "into pbg_telemetry and no stored vector is perturbed. "
                "cosine_neighbor entries are exact telemetry_vec <=> query "
                "nearest non-pending rows; random_sample entries are a "
                "seeded-uniform draw (seed recorded) from non-pending rows."
            ),
        }

        NOISE_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        tag = trigger["ticket_id"] if trigger is not None else "manual"
        out = NOISE_DIR / f"scavenger_noise_{ts}_{tag}.json"
        out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        latest = NOISE_DIR / "scavenger_noise.latest.json"
        latest.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        log.info("event=noise-emitted path=%s k=%d m=%d seed=%d",
                 out.relative_to(REPO_ROOT), self.noise_k, rand_meta["count"], seed)
        return out

    # ---- units of work -------------------------------------------
    def reconcile(self) -> None:
        if self.dry_run:
            log.info("event=dry-run would=reconcile (scrape + upsert whole corpus)")
            return
        log.info("event=reconcile backlog=%s dsn_host=%s", self.backlog, self.dsn_host)
        self.scrape()
        self.upsert()

    def process(self, tid: str, cur: dict[str, Any]) -> None:
        log.info("event=detected ticket=%s status=%s color=%s",
                 tid, cur["status"], cur["color"])
        if self.dry_run:
            log.info("event=dry-run ticket=%s would=scrape,upsert%s", tid,
                     ",noise" if (self.noise_k or self.noise_m) else "")
            return
        self.scrape()
        self.upsert()
        self.emit_noise(trigger={"ticket_id": tid, **cur})
        self._record(tid, cur, processed=True)
        self.save_state()

    # ---- poll pass -------------------------------------------------
    def _fire_candidate(self, tid: str, cur: dict[str, Any]) -> bool:
        prev = self.state["tickets"].get(tid)
        if not cur["closed_graded"]:
            return False
        if prev is None or not prev.get("closed_graded"):
            return True                                   # transition into graded+closed
        if prev.get("hash") != cur["hash"]:
            return prev.get("last_hash_processed") != cur["hash"]   # re-grade of a closed ticket
        return False

    def poll_pass(self) -> None:
        cur_all = self.scan()
        now = time.monotonic()

        for tid, cur in cur_all.items():
            if self._fire_candidate(tid, cur):
                pend = self._pending.get(tid)
                if pend is None or pend["hash"] != cur["hash"]:
                    self._pending[tid] = {"fields": cur, "hash": cur["hash"],
                                          "deadline": now + self.debounce}
                    log.info("event=queued ticket=%s debounce=%.1fs", tid, self.debounce)
            else:
                # keep state fresh for non-firing files (status/color drift, edits)
                if self.state["tickets"].get(tid, {}).get("hash") != cur["hash"]:
                    self._record(tid, cur, processed=False)

        # fire debounced, hash-stable candidates
        for tid in list(self._pending):
            pend = self._pending[tid]
            live = cur_all.get(tid)
            if live is None:
                self._pending.pop(tid, None)
                continue
            if live["hash"] != pend["hash"]:
                pend["fields"], pend["hash"] = live, live["hash"]
                pend["deadline"] = now + self.debounce
                continue
            if now >= pend["deadline"]:
                self._pending.pop(tid, None)
                try:
                    self.process(tid, pend["fields"])
                except Exception as exc:
                    log.error("event=process-failed ticket=%s err=%s", tid, exc)

        self.save_state()

    # ---- run ----------------------------------------------------------
    def _install_signals(self) -> None:
        def handler(signum, _frame):
            self._stop = True
            log.info("event=shutdown signal=%s", signal.Signals(signum).name)
        for s in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(s, handler)
            except (ValueError, OSError):
                pass

    def run(self) -> int:
        self.acquire_lock()
        self._install_signals()
        self.load_state()
        if not self.backlog.is_dir():
            log.error("event=fatal reason=backlog-not-found path=%s", self.backlog)
            return 2

        log.info("event=startup tool=%s v%s pid=%d mode=%s backlog=%s dsn_host=%s "
                 "interval=%ds debounce=%.1fs noise_k=%d noise_m=%d dry_run=%s",
                 TOOL, VERSION, os.getpid(), "once" if self.args.once else "daemon",
                 self.backlog, self.dsn_host, self.interval, self.debounce,
                 self.noise_k, self.noise_m, self.dry_run)

        # baseline: one reconcile, record current state without per-ticket noise
        try:
            self.reconcile()
        except Exception as exc:
            log.error("event=reconcile-failed err=%s", exc)
            if self.args.once:
                return 1
        for tid, cur in self.scan().items():
            self._record(tid, cur, processed=cur["closed_graded"])
        self.save_state()

        if self.args.once:
            if self.noise_k or self.noise_m:
                if self.dry_run:
                    log.info("event=dry-run would=noise")
                else:
                    self.emit_noise(trigger=None)
            log.info("event=once-complete tickets=%d", len(self.state["tickets"]))
            return 0

        log.info("event=watching dir=%s", self.backlog)
        while not self._stop:
            try:
                self.poll_pass()
            except Exception as exc:                        # never let one pass kill the daemon
                log.error("event=poll-error err=%s", exc)
            # interruptible sleep
            for _ in range(self.interval * 10):
                if self._stop:
                    break
                time.sleep(0.1)

        self.save_state()
        self.release_lock()
        log.info("event=stopped")
        return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=TOOL, description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--once", action="store_true",
                   help="single reconcile pass (+ optional --noise), then exit")
    p.add_argument("--interval", type=int, default=5, help="poll period seconds (default 5)")
    p.add_argument("--debounce", type=float, default=2.0,
                   help="seconds a file hash must be stable before firing (default 2.0)")
    p.add_argument("--backlog", type=Path, default=DEFAULT_BACKLOG,
                   help=f"backlog dir (default {DEFAULT_BACKLOG})")
    p.add_argument("--dsn", help="PostgreSQL DSN (overrides $CYCLOPTER_DSN and .env)")
    p.add_argument("--migration", type=Path,
                   help="alternate migration SQL forwarded to load_telemetry_pg.py")
    p.add_argument("--dry-run", action="store_true",
                   help="detect + log only; no scrape / upsert / noise write")
    p.add_argument("--noise", type=int, metavar="K",
                   help="emit the diversification artifact with K cosine neighbours")
    p.add_argument("--noise-random", type=int, default=3, metavar="M",
                   help="random non-pending rows in the artifact (default 3)")
    p.add_argument("--noise-vector", metavar="V",
                   help='query vector for the artifact, e.g. "[1,0.5,0.5]"')
    p.add_argument("--seed", type=int, help="RNG seed for the random sample "
                   "(default: real OS entropy, different each run - TKT-020 R3)")
    p.add_argument("--force", action="store_true", help="steal a stale lock")
    p.add_argument("--verbose", action="store_true", help="DEBUG logging")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z", stream=sys.stderr)
    try:
        return Scavenger(args).run()
    except SystemExit:
        raise
    except KeyboardInterrupt:
        log.info("event=shutdown signal=KeyboardInterrupt")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
