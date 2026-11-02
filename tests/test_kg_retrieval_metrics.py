"""Metric arithmetic and label validation for the kg_graph_search eval.

Every expected value is worked out by hand in the comment above it, so a
change to the formulas shows up as a real disagreement, not a re-derived one.
"""

import json

import pytest

from benchmarks.kg_retrieval.labels import LabeledQuery, load_labels
from benchmarks.kg_retrieval.metrics import (
    aggregate,
    recall_at_k,
    reciprocal_rank,
)

KNOWN = {"A", "B", "C", "D"}


def test_reciprocal_rank_is_one_over_the_first_relevant_rank():
    # relevant at rank 1 -> 1.0
    assert reciprocal_rank(["A", "B"], {"A"}) == 1.0
    # first relevant is C at rank 3 -> 1/3; the later hit on A is ignored
    ranking = ["B", "D", "C", "A"]
    assert reciprocal_rank(ranking, {"A", "C"}) == pytest.approx(1 / 3)
    # no relevant page ranked at all -> 0
    assert reciprocal_rank(["B", "D"], {"A"}) == 0.0


def test_recall_counts_each_relevant_page_found_within_k():
    ranking = ["A", "B", "C", "D"]
    relevant = {"A", "C", "D"}
    # top 2 = A,B -> 1 of 3; top 3 = A,B,C -> 2 of 3; top 4 -> 3 of 3
    assert recall_at_k(ranking, relevant, 2) == pytest.approx(1 / 3)
    assert recall_at_k(ranking, relevant, 3) == pytest.approx(2 / 3)
    assert recall_at_k(ranking, relevant, 4) == 1.0
    # no relevant page in the top k -> 0
    assert recall_at_k(["B", "D"], {"A"}, 2) == 0.0


def test_k_larger_than_the_ranking_uses_what_is_there():
    # only two results came back; k=10 sees both: 1 of 2 relevant found
    assert recall_at_k(["A", "B"], {"A", "C"}, 10) == 0.5


def test_a_duplicated_id_counts_once_and_takes_one_rank():
    # A repeated: distinct ranking is A,B,C. Recall is 1 of 2, not 2 of 2,
    # and C sits at rank 3, not 4, so RR = 1/3.
    ranking = ["A", "A", "B", "C"]
    assert recall_at_k(ranking, {"A", "C"}, 2) == 0.5
    assert reciprocal_rank(ranking, {"C"}) == pytest.approx(1 / 3)


def test_aggregate_averages_recall_per_k_and_mrr():
    rankings = [["A", "B"], ["B", "C"], ["D"]]
    relevants = [{"A"}, {"C"}, {"A"}]
    # recall@1: 1, 0, 0 -> 1/3.  recall@2: 1, 1, 0 -> 2/3.
    # RR: 1, 1/2, 0 -> MRR = 1.5 / 3 = 0.5
    scores = aggregate(rankings, relevants, [1, 2])
    assert scores == {
        "recall@1": pytest.approx(1 / 3),
        "recall@2": pytest.approx(2 / 3),
        "mrr": pytest.approx(0.5),
    }


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda: recall_at_k(["A"], set(), 1), "at least one page id"),
        (lambda: reciprocal_rank(["A"], set()), "at least one page id"),
        (lambda: recall_at_k(["A"], {"A"}, 0), "positive integer"),
        (lambda: aggregate([], [], [1]), "at least one query"),
        (lambda: aggregate([["A"], ["B"]], [{"A"}], [1]), "2 rankings but 1"),
    ],
)
def test_metrics_refuse_inputs_that_would_score_silently(call, message):
    with pytest.raises(ValueError, match=message):
        call()


STRESSED = {"relevant": ["C"], "stress": True}


def write_labels(tmp_path, *lines):
    path = tmp_path / "queries.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_load_labels_reads_well_formed_lines(tmp_path):
    path = write_labels(
        tmp_path,
        json.dumps({"id": "q1", "query": "how to X", "relevant": ["A", "B"]}),
        json.dumps({"id": "q2", "query": "what is Y", **STRESSED}),
    )

    # stress is optional and defaults to False
    assert load_labels(path, KNOWN) == [
        LabeledQuery("q1", "how to X", ("A", "B"), stress=False),
        LabeledQuery("q2", "what is Y", ("C",), stress=True),
    ]


def test_a_malformed_label_line_raises_instead_of_being_skipped(tmp_path):
    good = json.dumps({"id": "q1", "query": "how to X", "relevant": ["A"]})
    path = write_labels(tmp_path, good, '{"id": "q2", "query": "oops"')

    with pytest.raises(ValueError):
        load_labels(path, KNOWN)


@pytest.mark.parametrize(
    "record, message",
    [
        ({"id": "q1", "query": "x", "relevant": []}, "non-empty list"),
        ({"id": "q1", "query": "x", "relevant": ["Z"]}, "unknown page ids"),
        ({"id": "q1", "query": "", "relevant": ["A"]}, "'query' must be"),
        (
            {"id": "q1", "query": "x", "relevant": ["A"], "stress": "yes"},
            "'stress' must be true or false",
        ),
    ],
)
def test_load_labels_names_the_line_and_the_problem(tmp_path, record, message):
    path = write_labels(tmp_path, json.dumps(record))

    with pytest.raises(ValueError, match=f"queries.jsonl:1: .*{message}"):
        load_labels(path, KNOWN)
