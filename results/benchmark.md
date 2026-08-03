# Retrieval benchmark

Generated 2026-08-03 · embedding backend: `lsa`

Corpus: **56 passages** · Query set: **30 labelled queries**

| Pipeline | Recall@1 | Recall@3 | Recall@5 | P@3 | MRR@10 | nDCG@10 | median ms | p95 ms |
| --- | --: | --: | --: | --: | --: | --: | --: | --: |
| `bm25` | 0.600 | 0.917 | 0.933 | 0.333 | 0.756 | 0.801 | 0.12 | 0.20 |
| `dense` | 0.650 | 0.900 | 0.933 | 0.333 | 0.793 | 0.840 | 4.89 | 6.33 |
| `hybrid_rrf` | 0.617 | 0.900 | 0.933 | 0.333 | 0.783 | 0.825 | 4.52 | 6.08 |
| `hybrid_rerank` | 0.667 | 0.800 | 0.867 | 0.289 | 0.757 | 0.796 | 4.04 | 5.60 |

Recall@3 split by how the query is worded:

| Pipeline | lexical | paraphrase |
| --- | --: | --: |
| `bm25` | 1.000 | 0.853 |
| `dense` | 1.000 | 0.824 |
| `hybrid_rrf` | 1.000 | 0.824 |
| `hybrid_rerank` | 1.000 | 0.647 |

Regenerate with `python run_benchmark.py`.
