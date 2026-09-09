#!/usr/bin/env python3
"""TKT-014 research spike - SystemVerilog <-> Markdown knowledge-graph gap analysis.

Time-boxed spike. Embeds the RTL modules of ``gcb-wdm-transceiver/rtl/*.sv`` and
the doctrine sections of ``gcb-wiki/*.md`` into ONE vector space, then uses
cross-type cosine distance to surface:

  * **Gaps** - an RTL module whose nearest doctrine chunk is past a stated
    cosine-distance threshold tau (undocumented silicon), or a doctrine chunk
    whose nearest RTL module is past tau (doctrine with no silicon).
  * **Serendipitous correlations** - an ``rtl <-> doc`` pair that scores highly
    similar under IDF weighting yet shares almost no *surface* vocabulary, i.e.
    the link rides on a few rare shared terms rather than boilerplate.

Embedding: a deterministic **TF-IDF** bag-of-identifiers - no model weights, no
network, fixed by construction (see ``embed()``). Identifiers are split on
``snake_case`` / ``camelCase`` so ``gcb_tensor_dma_arbiter`` contributes
``tensor``, ``dma``, ``arbiter``; SV keywords and English/Markdown stopwords are
dropped so the vector is carried by domain terms (``gddr7``, ``photonic``,
``int4``, ``telemetry`` ...). This is a *documented deterministic fallback*, not
a semantic model - conclusions are about shared terminology, not meaning
(see the FINDINGS "does not prove" section).

Storage: in-memory NumPy by default. ``--pg-dsn DSN`` additionally round-trips a
**throwaway** ``doc_rtl_embed`` ``vector(N)`` table with an HNSW cosine index,
cross-checks one KNN query against the NumPy result, and DROPs the table. It
never touches ``pbg_telemetry``.

Deterministic: same inputs -> identical output. ``--seed`` only orders exact-tie
score groups.

Boundary (gcb-wiki/04): this spike lives in ``gcb-python-orchestrator``; it
reads the RTL / wiki repos but writes nothing into them.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover
    sys.exit("numpy required:  pip install numpy")

REPO_ROOT = Path(__file__).resolve().parents[1]
SIBLINGS = REPO_ROOT.parent
DEFAULT_RTL_GLOB = str(SIBLINGS / "gcb-wdm-transceiver" / "rtl" / "*.sv")
DEFAULT_MD_GLOB = str(SIBLINGS / "gcb-wiki" / "*.md")

# --------------------------------------------------------------------------- #
# tokenisation
# --------------------------------------------------------------------------- #
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z0-9]+|[A-Z]+")
_SV_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_SV_LINE_COMMENT = re.compile(r"//[^\n]*")

STOP = frozenset("""
module endmodule input output inout logic wire reg parameter localparam begin end
assign always always_ff always_comb always_latch posedge negedge if else case
endcase default for while do genvar generate endgenerate typedef enum struct union
packed unsigned signed int bit byte integer real time string clk clock rst reset
rst_n resetn initial final function endfunction task endtask return void automatic
timescale ifndef ifdef elsif endif define include package import export unique
unique0 priority extern virtual pure const ref static this super new null
parameter_t width depth addr data valid ready the and for with this that from are
was were not but per its one two three four all any each out via use uses used see
sec doc docs fig figure table note notes will has have had can may must should
into than then when where which who whom whose here there over under above below
also more most less least such only same other another both few many much some
non pre post sub non_ pkg tb sim rtl top gcb wdm dwdm
""".split())

_MD_STRIP = re.compile(r"[#>*`|_~\[\]()!]+")


def _subtokens(ident: str) -> list[str]:
    parts: list[str] = []
    for piece in ident.split("_"):
        if not piece:
            continue
        parts.extend(_CAMEL.findall(piece) or [piece])
    return parts


def tokenize(text: str, *, is_rtl: bool) -> list[str]:
    if is_rtl:
        text = _SV_BLOCK_COMMENT.sub(" ", text)
        text = _SV_LINE_COMMENT.sub(" ", text)
    else:
        text = _MD_STRIP.sub(" ", text)
    out: list[str] = []
    for m in _IDENT.finditer(text):
        for tok in _subtokens(m.group(0)):
            t = tok.lower()
            if len(t) < 3 or t.isdigit() or t in STOP:
                continue
            if re.fullmatch(r"[a-z]\d+|\d+[a-z]", t):   # h1, x2, 4b ...
                continue
            out.append(t)
    return out


# --------------------------------------------------------------------------- #
# chunking
# --------------------------------------------------------------------------- #
_MODULE_RE = re.compile(r"^[ \t]*(?:module|package)[ \t]+([A-Za-z_]\w*)", re.MULTILINE)
_HEADING_RE = re.compile(r"^(#{1,3})[ \t]+(.+?)[ \t]*$", re.MULTILINE)
MIN_CHUNK_TOKENS = 8


class Chunk:
    __slots__ = ("cid", "kind", "source", "heading", "tokens", "tokset")

    def __init__(self, cid: str, kind: str, source: str, heading: str, tokens: list[str]):
        self.cid = cid
        self.kind = kind          # 'rtl' | 'doc'
        self.source = source      # file name
        self.heading = heading    # module name / section title
        self.tokens = tokens
        self.tokset = frozenset(tokens)


def chunk_rtl(path: Path) -> list[Chunk]:
    text = path.read_text(encoding="utf-8", errors="replace")
    hits = list(_MODULE_RE.finditer(text))
    chunks: list[Chunk] = []
    if not hits:
        toks = tokenize(text, is_rtl=True)
        if len(toks) >= MIN_CHUNK_TOKENS:
            chunks.append(Chunk(f"rtl:{path.name}", "rtl", path.name, path.stem, toks))
        return chunks
    for i, h in enumerate(hits):
        start = h.start()
        endp = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        name = h.group(1)
        toks = tokenize(text[start:endp], is_rtl=True)
        if len(toks) >= MIN_CHUNK_TOKENS:
            chunks.append(Chunk(f"rtl:{path.name}::{name}", "rtl", path.name, name, toks))
    return chunks


def chunk_md(path: Path) -> list[Chunk]:
    text = path.read_text(encoding="utf-8", errors="replace")
    hits = list(_HEADING_RE.finditer(text))
    chunks: list[Chunk] = []
    if not hits:
        toks = tokenize(text, is_rtl=False)
        if len(toks) >= MIN_CHUNK_TOKENS:
            chunks.append(Chunk(f"doc:{path.name}", "doc", path.name, path.stem, toks))
        return chunks
    # preamble before the first heading
    pre = text[: hits[0].start()]
    segs: list[tuple[str, str]] = []
    if tokenize(pre, is_rtl=False):
        segs.append((path.stem + " (preamble)", pre))
    for i, h in enumerate(hits):
        start = h.start()
        endp = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        segs.append((h.group(2).strip(), text[start:endp]))
    for title, body in segs:
        toks = tokenize(body, is_rtl=False)
        if len(toks) >= MIN_CHUNK_TOKENS:
            slug = re.sub(r"\W+", "-", title.lower()).strip("-")[:40]
            chunks.append(Chunk(f"doc:{path.name}::{slug}", "doc", path.name, title, toks))
    return chunks


# --------------------------------------------------------------------------- #
# TF-IDF embedding
# --------------------------------------------------------------------------- #
def embed(chunks: list[Chunk]) -> tuple[np.ndarray, list[str], np.ndarray]:
    vocab = sorted({t for c in chunks for t in c.tokset})
    vidx = {t: i for i, t in enumerate(vocab)}
    n = len(chunks)
    df = np.zeros(len(vocab))
    for c in chunks:
        for t in c.tokset:
            df[vidx[t]] += 1.0
    idf = np.log((1.0 + n) / (1.0 + df)) + 1.0

    mat = np.zeros((n, len(vocab)), dtype=np.float64)
    for r, c in enumerate(chunks):
        if not c.tokens:
            continue
        counts: dict[int, int] = {}
        for t in c.tokens:
            counts[vidx[t]] = counts.get(vidx[t], 0) + 1
        inv = 1.0 / len(c.tokens)
        for j, k in counts.items():
            mat[r, j] = k * inv * idf[j]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat /= norms
    return mat, vocab, idf


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def analyse(chunks: list[Chunk], mat: np.ndarray, vocab: list[str], idf: np.ndarray,
            *, tau: float, sim_hi: float, top: int, seed: int,
            min_shared: int = 1) -> dict[str, Any]:
    rnd = random.Random(seed)
    rtl = [i for i, c in enumerate(chunks) if c.kind == "rtl"]
    doc = [i for i, c in enumerate(chunks) if c.kind == "doc"]
    if not rtl or not doc:
        sys.exit("need at least one RTL chunk and one doc chunk")

    R = mat[rtl]                       # (nr, V)
    D = mat[doc]                       # (nd, V)
    sim = R @ D.T                      # cosine similarity, (nr, nd)
    dist = 1.0 - sim

    def _rank_lonely(rows_idx: list[int], axis_dist: np.ndarray, other_idx: list[int]):
        order = []
        for local, gi in enumerate(rows_idx):
            j = int(np.argmin(axis_dist[local]))
            order.append({
                "cid": chunks[gi].cid,
                "source": chunks[gi].source,
                "heading": chunks[gi].heading,
                "nearest": chunks[other_idx[j]].cid,
                "nearest_source": chunks[other_idx[j]].source,
                "nearest_heading": chunks[other_idx[j]].heading,
                "cos_dist": round(float(axis_dist[local, j]), 6),
                "cos_sim": round(float(1.0 - axis_dist[local, j]), 6),
            })
        # tie-safe deterministic sort: (dist, cid), with a seeded shuffle first
        rnd.shuffle(order)
        order.sort(key=lambda d: (-d["cos_dist"], d["cid"]))
        return order

    rtl_lonely = _rank_lonely(rtl, dist, doc)
    doc_lonely = _rank_lonely(doc, dist.T, rtl)

    rtl_gaps = [d for d in rtl_lonely if d["cos_dist"] > tau]
    doc_gaps = [d for d in doc_lonely if d["cos_dist"] > tau]

    # ---- file-level aggregate: a file's best cross-type tie is the min over
    #      its chunks. The worst such value is the headline gap.
    def _by_file(rows: list[dict[str, Any]]):
        best: dict[str, dict[str, Any]] = {}
        for d in rows:
            cur = best.get(d["source"])
            if cur is None or d["cos_dist"] < cur["cos_dist"]:
                best[d["source"]] = d
        return sorted(best.values(), key=lambda d: (-d["cos_dist"], d["source"]))

    rtl_by_file = _by_file(rtl_lonely)
    doc_by_file = _by_file(doc_lonely)

    # ---- serendipity: meaningful IDF-cosine, low surface Jaccard --------
    # "surprise" = cosine / surface-overlap; a tie-break toward pairs whose
    # *least-rare* shared term is still rare (idf) favours a genuine
    # cross-domain vocabulary collision over a common-word coincidence.
    vpos = {t: i for i, t in enumerate(vocab)}
    cand = []
    for a_local, gi in enumerate(rtl):
        for b_local, gj in enumerate(doc):
            s = float(sim[a_local, b_local])
            if s < sim_hi:
                continue
            shared = chunks[gi].tokset & chunks[gj].tokset
            if len(shared) < min_shared:
                continue
            jac = jaccard(chunks[gi].tokset, chunks[gj].tokset)
            shared_by_idf = sorted(shared, key=lambda t: -idf[vpos[t]])
            rarest_common = float(min(idf[vpos[t]] for t in shared))
            cand.append({
                "rtl": chunks[gi].cid, "rtl_heading": chunks[gi].heading,
                "rtl_source": chunks[gi].source,
                "doc": chunks[gj].cid, "doc_heading": chunks[gj].heading,
                "doc_source": chunks[gj].source,
                "cos_sim": round(s, 6),
                "surface_jaccard": round(jac, 6),
                "surprise": round(s / (jac + 1e-3), 3),
                "rarest_common_idf": round(rarest_common, 3),
                "shared_terms_rare_first": shared_by_idf[:10],
                "n_shared": len(shared),
            })
    rnd.shuffle(cand)
    cand.sort(key=lambda d: (-d["surprise"] * d["rarest_common_idf"],
                             -d["cos_sim"], d["rtl"], d["doc"]))

    return {
        "rtl_chunks": len(rtl), "doc_chunks": len(doc), "vocab": len(vocab),
        "tau": tau, "sim_hi": sim_hi,
        "rtl_lonely": rtl_lonely[:top],
        "doc_lonely": doc_lonely[:top],
        "rtl_by_file": rtl_by_file,
        "doc_by_file": doc_by_file,
        "rtl_gaps": rtl_gaps, "doc_gaps": doc_gaps,
        "headline_rtl_gap": rtl_by_file[0] if rtl_by_file else None,
        "headline_doc_gap": doc_by_file[0] if doc_by_file else None,
        "serendipity_top": cand[:top],
    }


# --------------------------------------------------------------------------- #
# optional pgvector round-trip
# --------------------------------------------------------------------------- #
def pg_roundtrip(dsn: str, chunks: list[Chunk], mat: np.ndarray, vocab: list[str],
                 idf: np.ndarray, *, dim: int, seed: int) -> dict[str, Any]:
    import psycopg

    # reduce to a fixed dim for an HNSW-indexable vector: top tokens by total
    # tf-idf mass across all chunks (deterministic).
    mass = mat.sum(axis=0)
    keep = np.argsort(-mass)[: min(dim, len(vocab))]
    keep.sort()
    sub = mat[:, keep]
    sub = sub / np.clip(np.linalg.norm(sub, axis=1, keepdims=True), 1e-12, None)
    N = sub.shape[1]

    def lit(v) -> str:
        return "[" + ",".join(f"{x:.6g}" for x in v) + "]"

    out: dict[str, Any] = {"dim": N}
    with psycopg.connect(dsn, autocommit=True, connect_timeout=10) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("DROP TABLE IF EXISTS doc_rtl_embed")
        conn.execute(f"""
            CREATE TABLE doc_rtl_embed (
                id        text PRIMARY KEY,
                kind      text NOT NULL CHECK (kind IN ('rtl','doc')),
                source    text NOT NULL,
                chunk     text NOT NULL,
                embedding vector({N}) NOT NULL
            )""")
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO doc_rtl_embed (id, kind, source, chunk, embedding) "
                "VALUES (%s,%s,%s,%s,%s::vector)",
                [(c.cid, c.kind, c.source, c.heading, lit(sub[i]))
                 for i, c in enumerate(chunks)])
        conn.execute("CREATE INDEX doc_rtl_embed_cos ON doc_rtl_embed "
                     "USING hnsw (embedding vector_cosine_ops)")
        n_rows = conn.execute("SELECT count(*) FROM doc_rtl_embed").fetchone()[0]

        # cross-check: nearest doc chunk to one probe RTL chunk, SQL vs NumPy
        rtl_ids = [i for i, c in enumerate(chunks) if c.kind == "rtl"]
        probe = random.Random(seed).choice(rtl_ids)
        sql = conn.execute(
            "SELECT id, embedding <=> %s::vector AS d FROM doc_rtl_embed "
            "WHERE kind = 'doc' ORDER BY d LIMIT 3", (lit(sub[probe]),)).fetchall()
        doc_ids = [i for i, c in enumerate(chunks) if c.kind == "doc"]
        npd = sorted(((chunks[j].cid, 1.0 - float(sub[probe] @ sub[j])) for j in doc_ids),
                     key=lambda kv: kv[1])[:3]
        conn.execute("DROP TABLE doc_rtl_embed")

    out.update({
        "rows_inserted": n_rows,
        "probe": chunks[probe].cid,
        "sql_top3": [[r[0], round(float(r[1]), 6)] for r in sql],
        "numpy_top3": [[cid, round(d, 6)] for cid, d in npd],
        "agree": [r[0] for r in sql] == [c for c, _ in npd],
        "table_dropped": True,
    })
    return out


# --------------------------------------------------------------------------- #
# CLI / report
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kg_gap_analysis_spike.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rtl-glob", default=DEFAULT_RTL_GLOB, help=f"(default {DEFAULT_RTL_GLOB})")
    p.add_argument("--md-glob", default=DEFAULT_MD_GLOB, help=f"(default {DEFAULT_MD_GLOB})")
    p.add_argument("--tau", type=float, default=0.92,
                   help="cosine-distance gap threshold: nearest cross-type dist > tau "
                        "=> gap (default 0.92, i.e. best cross-type cosine sim < 0.08)")
    p.add_argument("--sim-hi", type=float, default=0.10,
                   help="min cosine sim for a pair to enter the serendipity ranking "
                        "(default 0.10)")
    p.add_argument("--min-shared", type=int, default=1,
                   help="min shared terms for a serendipity pair (default 1)")
    p.add_argument("--top", type=int, default=8, help="rows per table (default 8)")
    p.add_argument("--seed", type=int, default=1402, help="tie-break RNG seed")
    p.add_argument("--pg-dsn", help="also round-trip a throwaway doc_rtl_embed "
                                    "vector(N) table through this PostgreSQL DSN")
    p.add_argument("--pg-dim", type=int, default=512, help="reduced dim for --pg-dsn (default 512)")
    p.add_argument("--json", type=Path, help="also write the full result as JSON here")
    return p


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass
    args = build_parser().parse_args(argv)

    rtl_files = sorted(Path(p) for p in glob.glob(args.rtl_glob))
    md_files = sorted(Path(p) for p in glob.glob(args.md_glob))
    if not rtl_files or not md_files:
        sys.exit(f"no inputs (rtl={len(rtl_files)} md={len(md_files)}) - check --rtl-glob/--md-glob")

    chunks: list[Chunk] = []
    for f in rtl_files:
        chunks.extend(chunk_rtl(f))
    for f in md_files:
        chunks.extend(chunk_md(f))

    mat, vocab, idf = embed(chunks)
    res = analyse(chunks, mat, vocab, idf,
                  tau=args.tau, sim_hi=args.sim_hi, top=args.top, seed=args.seed,
                  min_shared=args.min_shared)

    pr = print
    pr("=" * 78)
    pr("TKT-014 spike : SV <-> MD knowledge-graph gap analysis  (TF-IDF, cosine)")
    pr("=" * 78)
    pr(f"corpus : {len(rtl_files)} .sv files -> {res['rtl_chunks']} module chunks | "
       f"{len(md_files)} .md files -> {res['doc_chunks']} section chunks | "
       f"vocab {res['vocab']} terms")
    pr(f"tau = {args.tau}  (gap := nearest cross-type cosine distance > tau)")

    pr("\n-- RTL modules, loneliest first (nearest doctrine chunk) " + "-" * 22)
    pr(f"   {'cos_dist':>8}  {'module':<32} -> nearest doc section")
    for d in res["rtl_lonely"]:
        pr(f"   {d['cos_dist']:>8.4f}  {d['heading']:<32} -> {d['nearest_source']} :: {d['nearest_heading'][:32]}")

    pr("\n-- doctrine sections, loneliest first (nearest RTL module) " + "-" * 20)
    pr(f"   {'cos_dist':>8}  {'section':<44} -> nearest module")
    for d in res["doc_lonely"]:
        pr(f"   {d['cos_dist']:>8.4f}  {(d['source']+' :: '+d['heading'])[:44]:<44} -> {d['nearest_heading'][:28]}")

    pr(f"\n>>> HEADLINE GAPS  (per-file worst cross-type tie, tau={args.tau}) <<<")
    hr, hd = res["headline_rtl_gap"], res["headline_doc_gap"]
    if hr:
        flag = "GAP" if hr["cos_dist"] > args.tau else "closest-to-gap"
        pr(f"   [undocumented silicon / {flag}] {hr['source']}  "
           f"(best tie: {hr['nearest_source']} :: {hr['nearest_heading']})  "
           f"cos_dist {hr['cos_dist']:.4f} / cos_sim {hr['cos_sim']:.4f}")
    if hd:
        flag = "GAP" if hd["cos_dist"] > args.tau else "closest-to-gap"
        pr(f"   [doctrine without silicon / {flag}] {hd['source']} :: {hd['heading']}  "
           f"(best tie: {hd['nearest_heading']})  "
           f"cos_dist {hd['cos_dist']:.4f} / cos_sim {hd['cos_sim']:.4f}")

    pr(f"\n-- all doc files by worst cross-type tie " + "-" * 37)
    pr(f"   {'cos_dist':>8}  {'md file':<40} best module tie")
    for d in res["doc_by_file"][: args.top]:
        pr(f"   {d['cos_dist']:>8.4f}  {d['source']:<40} {d['nearest_heading'][:24]}")

    pr("\n-- serendipity : meaningful IDF-cosine, low surface overlap " + "-" * 19)
    pr(f"   {'surprise':>8} {'cos_sim':>7} {'jac':>6} {'rareIDF':>7}  rtl <-> doc")
    for d in res["serendipity_top"]:
        pr(f"   {d['surprise']:>8.2f} {d['cos_sim']:>7.4f} {d['surface_jaccard']:>6.3f} "
           f"{d['rarest_common_idf']:>7.2f}  {d['rtl_heading']} <-> {d['doc_source']} :: {d['doc_heading'][:26]}")
        pr(f"            shared[{d['n_shared']}]: {', '.join(d['shared_terms_rare_first'])}")

    out: dict[str, Any] = {"corpus": {
        "rtl_files": [f.name for f in rtl_files], "md_files": [f.name for f in md_files]},
        **res}

    if args.pg_dsn:
        pr("\n-- pgvector round-trip (throwaway doc_rtl_embed) " + "-" * 30)
        pg = pg_roundtrip(args.pg_dsn, chunks, mat, vocab, idf,
                          dim=args.pg_dim, seed=args.seed)
        out["pg"] = pg
        pr(f"   table doc_rtl_embed(vector({pg['dim']})) + HNSW cosine index : "
           f"{pg['rows_inserted']} rows inserted, then DROPPED")
        pr(f"   KNN cross-check for {pg['probe']}:")
        pr(f"     SQL  <=> top3 : {pg['sql_top3']}")
        pr(f"     NumPy    top3 : {pg['numpy_top3']}")
        pr(f"     agree        : {pg['agree']}")

    if args.json:
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        pr(f"\n[json] {args.json}")

    pr("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
