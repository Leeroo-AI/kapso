"""The retrieval eval's arm loop, scored against stand-ins with no services.

A stand-in retriever returns scripted pools and rerank orders, so the
expected metrics below are worked out by hand. The reranker-failure test
drives the real KGGraphSearch._rerank_results, which swallows backend
errors, to show the runner still aborts.
"""

import pytest

from benchmarks.kg_retrieval.labels import LabeledQuery
from benchmarks.kg_retrieval.runner import (
    KGRetriever,
    RerankFailed,
    check_isolated,
    evaluate,
    load_layers,
)
from kapso.core.config import ConfigError
from kapso.knowledge_base.search.base import KGResultItem
from kapso.knowledge_base.search.kg_graph_search import KGGraphSearch


def item(page_id, overview=""):
    return KGResultItem(
        id=page_id,
        score=1.0,
        page_type="Principle",
        overview=overview,
    )


class StandInRetriever:
    """Pool A,B,C per query. The reranker puts the last candidate first,
    or follows `orders` (pool positions, one entry per call) when given."""

    def __init__(self, orders=None):
        self.orders = iter(orders) if orders else None

    def top(self, query, top_k):
        return [item("B"), item("A")][:top_k]

    def pool(self, query, pool_size):
        overviews = [10, 400, 500]
        return [item(i, "x" * n) for i, n in zip("ABC", overviews)]

    def rerank(self, query, pool, top_k, overview_cap):
        order = next(self.orders) if self.orders else [2, 0]
        return [pool[i] for i in order][:top_k]


ARMS = [
    {"name": "off", "rerank": False},
    {"name": "capped", "rerank": True, "reranker_overview_max_chars": 300},
    {"name": "nocap", "rerank": True, "reranker_overview_max_chars": None},
]


def test_each_arm_is_scored_overall_and_on_the_stress_subset():
    queries = [
        LabeledQuery("q1", "first", ("C",)),
        LabeledQuery("q2", "second", ("A",), stress=True),
    ]

    retriever = StandInRetriever()
    report = evaluate(queries, ARMS, retriever, [1, 2], 2, 3, rerank_runs=1)

    mean = {
        (arm, subset): report["arms"][arm][subset]["mean"]
        for arm in report["arms"]
        for subset in ("overall", "stress")
    }
    assert report["queries"] == {"overall": 2, "stress": 1}
    # off returns B,A: q1 misses C (recall 0, RR 0); q2 finds A at rank 2
    assert mean["off", "overall"]["recall@1"] == 0.0
    assert mean["off", "overall"]["recall@2"] == 0.5
    assert mean["off", "overall"]["mrr"] == pytest.approx(0.25)
    assert mean["off", "stress"]["mrr"] == pytest.approx(0.5)
    # reranked order is C,A: q1 hits at rank 1, q2 at rank 2
    assert mean["capped", "overall"]["recall@1"] == 0.5
    assert mean["capped", "overall"]["mrr"] == pytest.approx(0.75)
    assert mean["capped", "stress"]["recall@1"] == 0.0
    # a 300 cap cuts 2 of 3 overviews per query; no cap cuts none; off none
    assert mean["capped", "overall"]["clipped_share"] == pytest.approx(2 / 3)
    assert mean["nocap", "overall"]["clipped_share"] == 0.0
    assert "clipped_share" not in mean["off", "overall"]
    # the pool holds every target, so the ceiling is 1.0
    assert report["pool_ceiling"]["overall"]["recall@3"] == 1.0


def test_reranker_runs_report_a_range_and_the_queries_that_moved():
    queries = [LabeledQuery("q1", "first", ("C",), stress=True)]
    arms = ARMS[1:]
    # capped: run 1 puts C first, run 2 puts it second; nocap is steady
    orders = [[2, 0], [0, 2], [2, 0], [2, 0]]

    retriever = StandInRetriever(orders)
    report = evaluate(queries, arms, retriever, [1], 2, 3, rerank_runs=2)

    capped = report["arms"]["capped"]["overall"]
    assert [run["recall@1"] for run in capped["runs"]] == [1.0, 0.0]
    assert capped["mean"]["recall@1"] == 0.5
    assert (capped["min"]["recall@1"], capped["max"]["recall@1"]) == (0.0, 1.0)
    nocap = report["arms"]["nocap"]["overall"]
    assert nocap["min"]["mrr"] == nocap["max"]["mrr"] == 1.0
    assert report["target_ranks"]["capped"] == {"q1": [1, 2]}
    assert report["unstable"] == ["q1"]


class RaisingBackend:
    def llm_completion_with_system_prompt(self, **kwargs):
        raise RuntimeError("provider down")


def reranking_search(backend):
    search = KGGraphSearch.__new__(KGGraphSearch)
    search._llm_backend = backend
    search.reranker_overview_max_chars = None
    search._neo4j_driver = search._weaviate_client = search._embedder = None
    return search


def test_a_swallowed_rerank_error_aborts_the_run():
    pool = [item("A", "first"), item("B", "second")]

    # KGGraphSearch hides the failure behind the original order ...
    silent = reranking_search(RaisingBackend())._rerank_results("q", pool, 2)
    assert [result.id for result in silent] == ["A", "B"]

    # ... and the retriever turns it into an error.
    retriever = KGRetriever(reranking_search(RaisingBackend()))
    with pytest.raises(RerankFailed):
        retriever.rerank("q", pool, 2, 300)


def test_the_eval_refuses_the_shipped_stores():
    own = {
        "weaviate_collection": "EvalOnly",
        "neo4j_uri": "bolt://localhost:7688",
    }
    check_isolated(own)

    with pytest.raises(ValueError, match="weaviate_collection"):
        check_isolated({**own, "weaviate_collection": "KapsoKG"})
    with pytest.raises(ValueError, match="neo4j_uri"):
        check_isolated({**own, "neo4j_uri": "bolt://localhost:7687"})


def test_a_local_config_overrides_only_what_it_names(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(
        "benchmark:\n  top_k: 10\n  search: {collection: Eval, port: 1}\n"
    )
    local = tmp_path / "local.yaml"
    local.write_text(
        "benchmark:\n  search: {port: 2, embedding_model: bge-m3}\n"
        "inference: {roles: {kg_rerank: {model: m}}}\n"
    )

    merged = load_layers([str(base), str(local)])

    search = {"collection": "Eval", "port": 2, "embedding_model": "bge-m3"}
    assert merged["benchmark"] == {"top_k": 10, "search": search}
    assert merged["inference"]["roles"]["kg_rerank"]["model"] == "m"
    # Each file is a kapso override: a key the engine does not know is refused.
    local.write_text("retrieval_eval: {top_k: 5}\n")
    with pytest.raises(ConfigError, match="unknown key"):
        load_layers([str(base), str(local)])
