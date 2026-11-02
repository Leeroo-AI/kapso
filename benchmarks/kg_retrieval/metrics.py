"""Retrieval metrics for the kg_graph_search eval: recall@k and MRR.

Pure functions over page-id rankings; nothing here reads files or calls a
search backend. A ranking lists page ids best first. An id that appears more
than once counts at its first position only (a re-indexed page can come back
twice), so a duplicate never inflates a score or pushes a later page down.
"""

from collections.abc import Collection, Sequence


def _first_occurrences(ranking: Sequence[str]) -> list[str]:
    """The ranking with repeated ids dropped, keeping each id's first place."""
    return list(dict.fromkeys(ranking))


def _check_relevant(relevant: Collection[str]) -> None:
    if not relevant:
        raise ValueError("relevant must name at least one page id")


def recall_at_k(
    ranking: Sequence[str],
    relevant: Collection[str],
    k: int,
) -> float:
    """Share of the relevant pages found in the first k distinct ids."""
    _check_relevant(relevant)
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    top = _first_occurrences(ranking)[:k]
    return len(set(top) & set(relevant)) / len(set(relevant))


def reciprocal_rank(
    ranking: Sequence[str],
    relevant: Collection[str],
) -> float:
    """1 / rank of the first relevant page, or 0.0 when none is ranked."""
    _check_relevant(relevant)
    for rank, page_id in enumerate(_first_occurrences(ranking), start=1):
        if page_id in relevant:
            return 1.0 / rank
    return 0.0


def aggregate(
    rankings: Sequence[Sequence[str]],
    relevants: Sequence[Collection[str]],
    ks: Sequence[int],
) -> dict[str, float]:
    """Mean recall@k for each k in ks, and MRR, over paired queries."""
    if not rankings:
        raise ValueError("aggregate needs at least one query")
    if len(rankings) != len(relevants):
        raise ValueError(
            f"got {len(rankings)} rankings but {len(relevants)} relevant sets"
        )
    pairs = list(zip(rankings, relevants))
    scores = {}
    for k in ks:
        recalls = [recall_at_k(r, rel, k) for r, rel in pairs]
        scores[f"recall@{k}"] = sum(recalls) / len(pairs)
    ranks = [reciprocal_rank(r, rel) for r, rel in pairs]
    scores["mrr"] = sum(ranks) / len(pairs)
    return scores
