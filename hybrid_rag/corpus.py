"""Loading and tokenising the evaluation corpus."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Deliberately small: a long stoplist hides the vocabulary-mismatch problem that
# hybrid retrieval exists to solve, which would flatter the benchmark.
STOPWORDS = frozenset(
    "a an and are as at be by for from how i in is it of on or that the to was what when where which with you your".split()
)

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Lowercase, split on non-alphanumerics, optionally drop stopwords."""
    tokens = _TOKEN.findall(text.lower())
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


@dataclass(frozen=True)
class Passage:
    id: str
    title: str
    text: str

    @property
    def searchable(self) -> str:
        """Title is repeated so it carries more weight than a single mention."""
        return f"{self.title}. {self.title}. {self.text}"


@dataclass(frozen=True)
class Query:
    id: str
    query: str
    relevant: frozenset[str] = field(default_factory=frozenset)
    type: str = "unknown"


def load_corpus(path: Path | str | None = None) -> list[Passage]:
    path = Path(path) if path else DATA_DIR / "corpus.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Passage(id=r["id"], title=r["title"], text=r["text"]) for r in raw]


def load_queries(path: Path | str | None = None) -> list[Query]:
    path = Path(path) if path else DATA_DIR / "queries.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        Query(
            id=r["id"],
            query=r["query"],
            relevant=frozenset(r["relevant"]),
            type=r.get("type", "unknown"),
        )
        for r in raw
    ]
