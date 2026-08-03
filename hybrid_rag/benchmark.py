"""Running every pipeline over the labelled query set and tabulating results."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field

from .corpus import Passage, Query
from .metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from .pipeline import RetrievalPipeline, build_pipelines


@dataclass
class Result:
    pipeline: str
    description: str
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    precision_at_3: float
    mrr_at_10: float
    ndcg_at_10: float
    median_latency_ms: float
    p95_latency_ms: float
    by_query_type: dict[str, float] = field(default_factory=dict)


def evaluate(pipeline: RetrievalPipeline, queries: Sequence[Query], *, k: int = 10) -> Result:
    r1, r3, r5, p3, rr, nd, latencies = [], [], [], [], [], [], []
    per_type: dict[str, list[float]] = {}

    for query in queries:
        results, elapsed = pipeline.timed_search(query.query, k)
        latencies.append(elapsed)
        r1.append(recall_at_k(results, query.relevant, 1))
        r3.append(recall_at_k(results, query.relevant, 3))
        r5.append(recall_at_k(results, query.relevant, 5))
        p3.append(precision_at_k(results, query.relevant, 3))
        rr.append(mrr(results, query.relevant, k))
        nd.append(ndcg_at_k(results, query.relevant, k))
        per_type.setdefault(query.type, []).append(recall_at_k(results, query.relevant, 3))

    latencies.sort()
    p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]

    return Result(
        pipeline=pipeline.name,
        description=pipeline.description,
        recall_at_1=statistics.fmean(r1),
        recall_at_3=statistics.fmean(r3),
        recall_at_5=statistics.fmean(r5),
        precision_at_3=statistics.fmean(p3),
        mrr_at_10=statistics.fmean(rr),
        ndcg_at_10=statistics.fmean(nd),
        median_latency_ms=statistics.median(latencies),
        p95_latency_ms=p95,
        by_query_type={t: statistics.fmean(v) for t, v in sorted(per_type.items())},
    )


def run(passages: Sequence[Passage], queries: Sequence[Query]) -> list[Result]:
    return [evaluate(p, queries) for p in build_pipelines(passages)]


def to_markdown(results: Sequence[Result], *, n_passages: int, n_queries: int) -> str:
    lines = [
        f"Corpus: **{n_passages} passages** · Query set: **{n_queries} labelled queries**",
        "",
        "| Pipeline | Recall@1 | Recall@3 | Recall@5 | P@3 | MRR@10 | nDCG@10 | median ms | p95 ms |",
        "| --- | --: | --: | --: | --: | --: | --: | --: | --: |",
    ]
    for r in results:
        lines.append(
            f"| `{r.pipeline}` | {r.recall_at_1:.3f} | {r.recall_at_3:.3f} | {r.recall_at_5:.3f} | "
            f"{r.precision_at_3:.3f} | {r.mrr_at_10:.3f} | {r.ndcg_at_10:.3f} | "
            f"{r.median_latency_ms:.2f} | {r.p95_latency_ms:.2f} |"
        )

    types = sorted({t for r in results for t in r.by_query_type})
    if types:
        lines += [
            "",
            "Recall@3 split by how the query is worded:",
            "",
            "| Pipeline | " + " | ".join(types) + " |",
            "| --- | " + " | ".join("--:" for _ in types) + " |",
        ]
        for r in results:
            cells = " | ".join(f"{r.by_query_type.get(t, 0.0):.3f}" for t in types)
            lines.append(f"| `{r.pipeline}` | {cells} |")

    return "\n".join(lines)
