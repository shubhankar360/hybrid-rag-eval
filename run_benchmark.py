#!/usr/bin/env python3
"""Regenerate results/benchmark.md from the labelled query set.

    python run_benchmark.py
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from hybrid_rag.benchmark import run, to_markdown
from hybrid_rag.corpus import load_corpus, load_queries
from hybrid_rag.dense import DenseRetriever

RESULTS = Path(__file__).resolve().parent / "results" / "benchmark.md"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS)
    parser.add_argument("--stdout", action="store_true", help="print only, do not write")
    args = parser.parse_args()

    passages = load_corpus()
    queries = load_queries()
    backend = DenseRetriever(passages).embedder.name

    results = run(passages, queries)
    table = to_markdown(results, n_passages=len(passages), n_queries=len(queries))

    report = (
        f"# Retrieval benchmark\n\n"
        f"Generated {date.today().isoformat()} · embedding backend: `{backend}`\n\n"
        f"{table}\n\n"
        "Regenerate with `python run_benchmark.py`.\n"
    )

    print(report)
    if not args.stdout:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
