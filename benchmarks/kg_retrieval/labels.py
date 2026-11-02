"""Loader for the labelled queries of the kg_graph_search retrieval eval.

One JSON object per line: {"id": ..., "query": ..., "relevant": [page ids]},
plus an optional "stress": true marking queries whose target page has a long
overview (the ones the reranker prompt cap can clip).
A bad file is an error, never a skipped line: a silently dropped query would
shift every metric without a trace.
"""

import json
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LabeledQuery:
    id: str
    query: str
    relevant: tuple[str, ...]
    stress: bool = False


def load_labels(
    path: str | Path,
    known_page_ids: Collection[str],
) -> list[LabeledQuery]:
    """Read the labelled queries, checking each id against the corpus."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"label file not found: {path}")
    queries = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        queries.append(_parse_line(f"{path}:{number}", line, known_page_ids))
    return queries


def _parse_line(
    where: str,
    line: str,
    known_page_ids: Collection[str],
) -> LabeledQuery:
    record = json.loads(line)
    if not isinstance(record, dict):
        raise ValueError(f"{where}: expected a JSON object, got {record!r}")
    for field in ("id", "query"):
        value = record.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{where}: {field!r} must be a non-empty string")
    relevant = record.get("relevant")
    if not isinstance(relevant, list) or not relevant:
        raise ValueError(f"{where}: 'relevant' must be a non-empty list")
    if not all(isinstance(page_id, str) for page_id in relevant):
        raise ValueError(f"{where}: 'relevant' must hold page id strings")
    stress = record.get("stress", False)
    if not isinstance(stress, bool):
        raise ValueError(f"{where}: 'stress' must be true or false")
    unknown = [i for i in relevant if i not in known_page_ids]
    if unknown:
        raise ValueError(f"{where}: unknown page ids {unknown}")
    return LabeledQuery(record["id"], record["query"], tuple(relevant), stress)
