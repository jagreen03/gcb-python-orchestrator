#!/usr/bin/env python3
"""PBG telemetry scraper (TKT-007).

Crawls the backlog Markdown files in ``gcb-workspaces/backlog/``, reads each
file's YAML frontmatter, and emits one JSONL row per ticket for the ``pgvector``
store defined in ``gcb-wiki/09_pbg_telemetry_vector_space.md``.

Vector contract (doc 09 sec. 5.3):

* Primary: a ``telemetry:`` block of ``{spec, quality, emergence}`` floats in
  ``[-1, 1]`` is the source of the vector ``T = (x, y, z)``.
* ``telemetry_color`` is a *derived* dominant-spoke label this script
  *validates* against the vector, not the source.
* Legacy / migration: a file with ``telemetry_color`` but no ``telemetry:``
  block yields the pure spoke vector for that colour, with a warning.

Axis mapping (doc 09 sec. 2):

    X  Specification   green  = +1   red        = -1
    Y  Quality         blue   = +1   yellow     = -1
    Z  Emergence       purple = +1   light-gray = -1

Magnitude: ``||T||_2 = sqrt(x^2 + y^2 + z^2)`` in ``[0, sqrt(3)]``.

Output object (aligned with the ``pbg_telemetry`` columns, minus the
DB-generated ``vec_norm`` which is emitted here as ``norm`` for validation)::

    {"ticket_id", "title", "status", "telemetry_vec": [x, y, z],
     "telemetry_color", "norm", "updated_at"}

This script never touches a database - it only writes JSONL.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("pbg.telemetry")

# --------------------------------------------------------------------------- #
# Axis / spoke model (doc 09 sec. 2)
# --------------------------------------------------------------------------- #
SPOKES: dict[str, tuple[float, float, float]] = {
    "green":      (1.0, 0.0, 0.0),
    "red":        (-1.0, 0.0, 0.0),
    "blue":       (0.0, 1.0, 0.0),
    "yellow":     (0.0, -1.0, 0.0),
    "purple":     (0.0, 0.0, 1.0),
    "light-gray": (0.0, 0.0, -1.0),
    "pending":    (0.0, 0.0, 0.0),
}
_AXIS_POLE_TO_SPOKE = {
    ("x", 1): "green", ("x", -1): "red",
    ("y", 1): "blue",  ("y", -1): "yellow",
    ("z", 1): "purple", ("z", -1): "light-gray",
}
TKT_RE = re.compile(r"(TKT-\d+)", re.IGNORECASE)


def normalize_color(raw: Any) -> str | None:
    """Canonicalise a colour string, or return None if it is not a known spoke."""
    if not isinstance(raw, str):
        return None
    c = raw.strip().lower().replace(" ", "-").replace("_", "-")
    if c == "lightgray" or c == "grey" or c == "light-grey":
        c = "light-gray"
    return c if c in SPOKES else None


def dominant_spoke(x: float, y: float, z: float) -> str:
    """Derived ``telemetry_color`` per doc 09 sec. 3.4.

    argmax of |axis| with tie-break order X > Y > Z; the sign of that axis
    picks the pole. The all-zero vector is ``pending``.
    """
    if x == 0.0 and y == 0.0 and z == 0.0:
        return "pending"
    # max() returns the first maximal element -> X beats Y beats Z on ties.
    axis, val = max((("x", x), ("y", y), ("z", z)), key=lambda kv: abs(kv[1]))
    return _AXIS_POLE_TO_SPOKE[(axis, 1 if val > 0 else -1)]


def l2_norm(x: float, y: float, z: float) -> float:
    return math.sqrt(x * x + y * y + z * z)


# --------------------------------------------------------------------------- #
# Frontmatter loading - prefer python-frontmatter, then PyYAML, then a small
# stdlib fallback for the flat-scalar + one-level `telemetry:` block our
# tickets use. The fallback is a last resort, not the primary path (TKT-007
# Yellow reject == making a regex parser THE parser).
# --------------------------------------------------------------------------- #
def _load_frontmatter(path: Path) -> dict[str, Any] | None:
    text = path.read_text(encoding="utf-8")
    if not text.lstrip().startswith("---"):
        return None

    try:
        import frontmatter  # type: ignore
        meta = frontmatter.loads(text).metadata
        return dict(meta) if isinstance(meta, dict) else None
    except ModuleNotFoundError:
        pass
    except Exception as exc:  # malformed YAML etc.
        raise ValueError(f"frontmatter parse error: {exc}") from exc

    # split the leading --- ... --- block
    body = text.lstrip()
    end = body.find("\n---", 3)
    if end == -1:
        return None
    fm_text = body[3:end].lstrip("\n")

    try:
        import yaml  # type: ignore
        data = yaml.safe_load(fm_text)
        return data if isinstance(data, dict) else None
    except ModuleNotFoundError:
        pass
    except Exception as exc:
        raise ValueError(f"YAML parse error: {exc}") from exc

    return _fallback_yaml(fm_text)


def _fallback_yaml(fm_text: str) -> dict[str, Any]:
    """Tiny YAML-subset reader: ``key: value`` scalars plus a single nested
    ``telemetry:`` mapping indented by two spaces. Sufficient for the ticket
    template, and only ever reached if neither library is installed."""
    out: dict[str, Any] = {}
    cur_map: dict[str, Any] | None = None
    for line in fm_text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith("  ") and cur_map is not None:
            k, _, v = line.strip().partition(":")
            cur_map[k.strip()] = _coerce(v.strip())
            continue
        cur_map = None
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip()
        if v == "":
            cur_map = {}
            out[k] = cur_map
        else:
            out[k] = _coerce(v)
    return out


def _coerce(v: str) -> Any:
    v = v.strip().strip('"').strip("'")
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


# --------------------------------------------------------------------------- #
# Per-file processing
# --------------------------------------------------------------------------- #
class Skip(Exception):
    """Raised to skip a file with a logged reason."""


def _vector_from_block(block: Any) -> tuple[float, float, float]:
    if not isinstance(block, dict):
        raise Skip("`telemetry:` present but not a mapping")
    try:
        x = float(block["spec"])
        y = float(block["quality"])
        z = float(block["emergence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Skip(f"`telemetry:` block missing/!numeric spec|quality|emergence ({exc})")
    for name, val in (("spec", x), ("quality", y), ("emergence", z)):
        if not -1.0 <= val <= 1.0:
            raise Skip(f"telemetry.{name}={val} out of range [-1, 1]")
    return x, y, z


def process_file(path: Path, *, include_pending: bool) -> tuple[dict[str, Any], int, int]:
    """Return (row, warning_count, mismatch_count). Raises Skip to drop the file."""
    try:
        meta = _load_frontmatter(path)
    except ValueError as exc:
        raise Skip(str(exc)) from exc
    if meta is None:
        raise Skip("no YAML frontmatter")

    # ticket id: frontmatter `id`, else derived from the filename
    ticket_id = str(meta.get("id") or "").strip()
    if not ticket_id:
        m = TKT_RE.search(path.name)
        if not m:
            raise Skip("no `id` and filename has no TKT-NNN")
        ticket_id = m.group(1).upper()
        log.warning("%s: no `id` in frontmatter, using %s from filename", path.name, ticket_id)

    title = str(meta.get("title") or "").strip().strip('"')
    if not title:
        raise Skip("no `title`")
    status = str(meta.get("status") or "").strip() or "unknown"

    warnings = 0
    mismatches = 0
    written_color = normalize_color(meta.get("telemetry_color"))
    raw_color = meta.get("telemetry_color")
    block = meta.get("telemetry")

    if block is not None:
        # primary path - the vector comes from the block
        x, y, z = _vector_from_block(block)
        derived = dominant_spoke(x, y, z)
        if written_color is None and raw_color not in (None, ""):
            log.warning("%s: unknown telemetry_color %r; using derived %r",
                        path.name, raw_color, derived)
            warnings += 1
        elif written_color is not None and written_color != derived:
            log.warning("%s: telemetry_color=%r disagrees with derived %r from %s",
                        path.name, written_color, derived, [x, y, z])
            warnings += 1
            mismatches += 1
        color_out = derived
        if derived == "pending" and not include_pending:
            raise Skip("telemetry vector is the origin (pending)")
    else:
        # legacy / migration path - synthesise the pure spoke from the colour
        if written_color is None:
            if raw_color in (None, ""):
                raise Skip("no telemetry_color and no telemetry: block")
            raise Skip(f"unknown telemetry_color {raw_color!r}")
        if written_color == "pending" and not include_pending:
            raise Skip("telemetry_color: pending")
        x, y, z = SPOKES[written_color]
        color_out = written_color
        log.warning("%s: legacy scalar telemetry (telemetry_color only, no "
                    "telemetry: block) -> %s", path.name, [x, y, z])
        warnings += 1

    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    row = {
        "ticket_id": ticket_id,
        "title": title,
        "status": status,
        "telemetry_vec": [round(x, 6), round(y, 6), round(z, 6)],
        "telemetry_color": color_out,
        "norm": round(l2_norm(x, y, z), 9),
        "updated_at": mtime.isoformat(),
    }
    return row, warnings, mismatches


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _default_backlog() -> Path:
    # .../gcb-python-orchestrator/scripts/parse_pbg_telemetry.py
    #  parents[2] == the directory holding all the gcb-* repos
    return Path(__file__).resolve().parents[2] / "gcb-workspaces" / "backlog"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="parse_pbg_telemetry.py",
        description="Scrape PBG telemetry from backlog frontmatter into pgvector JSONL "
                    "(see gcb-wiki/09_pbg_telemetry_vector_space.md).",
    )
    p.add_argument("--backlog", type=Path, default=_default_backlog(),
                   help="directory of TKT-*.md files (default: ../../gcb-workspaces/backlog)")
    p.add_argument("--emit", choices=["jsonl"], default="jsonl",
                   help="output format (only 'jsonl' for now)")
    p.add_argument("--out", default="-",
                   help="output path, or '-' for stdout (e.g. telemetry_export.jsonl)")
    p.add_argument("--include-pending", action="store_true",
                   help="also emit rows whose vector is the origin / colour is pending "
                        "(doc 09 keeps them for the L2 index); off by default")
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero if any file was skipped or any telemetry_color "
                        "disagreed with the derived label")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s",
                        stream=sys.stderr)
    args = build_parser().parse_args(argv)

    backlog: Path = args.backlog
    if not backlog.is_dir():
        log.error("backlog dir not found: %s", backlog)
        return 2

    files = sorted(backlog.glob("*.md"))
    if not files:
        log.error("no .md files under %s", backlog)
        return 2

    rows: list[dict[str, Any]] = []
    skipped = 0
    mismatches = 0
    warnings = 0
    for path in files:
        try:
            row, w, mm = process_file(path, include_pending=args.include_pending)
        except Skip as exc:
            log.info("skip %s: %s", path.name, exc)
            skipped += 1
            continue
        except Exception as exc:  # noqa: BLE001 - never let one file kill the run
            log.error("skip %s: unexpected %s: %s", path.name, type(exc).__name__, exc)
            skipped += 1
            continue
        rows.append(row)
        warnings += w
        mismatches += mm

    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    if args.out == "-":
        # force UTF-8 even when stdout is a pipe on Windows (default there is
        # the ANSI code page); the JSONL is destined for a UTF-8 pg bulk load.
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
        sys.stdout.write(payload)
    else:
        Path(args.out).write_text(payload, encoding="utf-8")
        log.info("wrote %d rows -> %s", len(rows), args.out)

    log.info("done: %d emitted, %d skipped, %d warning(s), %d colour mismatch(es)",
             len(rows), skipped, warnings, mismatches)

    if args.strict and (skipped or mismatches):
        log.error("--strict: %d skipped, %d mismatched", skipped, mismatches)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
