"""The reranker the gates and the workflow search ask for must run, or the search must raise."""

from pathlib import Path

import pytest

from kapso.core.config import load_config
from kapso.experience.ingest import search_for, search_from_refs
from kapso.experience.store import IndexRecord
from kapso.knowledge_base.search import kg_graph_search as module
from kapso.knowledge_base.search.base import KGResultItem, KGSearchFilters
from kapso.knowledge_base.search.factory import KnowledgeSearchFactory
from kapso.gated_mcp.presets import get_mcp_config
from kapso.knowledge_base.search.workflow_search import WorkflowRepoSearch

pytest.importorskip("weaviate")

INDEX_REFS = {"corpus": "index-id", "weaviate_collection": "KapsoKG_org_x"}


@pytest.fixture
def rerank_calls(monkeypatch):
    """The stores, the embedder and the rerank session are stood in for."""
    calls = []
    pool = [KGResultItem(id="a", score=0.9, page_type="Workflow", overview="o", content="c")]
    search = module.KGGraphSearch
    monkeypatch.setattr(search, "_initialize_neo4j", lambda self: None)
    monkeypatch.setattr(search, "_initialize_weaviate", lambda self: None)
    monkeypatch.setattr(search, "_semantic_search", lambda self, query, filters: list(pool))
    monkeypatch.setattr(search, "_rerank_results", lambda self, query, results, top_k: calls.append(top_k) or results)
    monkeypatch.setattr(module, "make_embedder", lambda params: None)
    return calls


def test_a_backend_built_from_an_index_record_alone_reranks_when_asked(rerank_calls):
    backend = KnowledgeSearchFactory.create("kg_graph_search", params=INDEX_REFS)
    output = backend.search("q", KGSearchFilters(top_k=3), use_llm_reranker=True)
    assert rerank_calls == [3]
    assert output.search_metadata["reranked"] is True


def test_asking_for_a_rerank_the_backend_cannot_run_raises(rerank_calls):
    backend = KnowledgeSearchFactory.create("kg_graph_search", params=INDEX_REFS, preset="FAST")
    with pytest.raises(ValueError, match="use_llm_reranker=False"):
        backend.search("q", KGSearchFilters(top_k=3), use_llm_reranker=True)
    output = backend.search("q", KGSearchFilters(top_k=3))
    assert rerank_calls == []
    assert output.search_metadata["reranked"] is False


def test_a_misspelled_preset_raises(rerank_calls):
    with pytest.raises(ValueError, match="unknown preset 'FSAT'"):
        KnowledgeSearchFactory.create("kg_graph_search", params=INDEX_REFS, preset="FSAT")


@pytest.mark.parametrize("preset, reranks", [("DEFAULT", True), ("RERANK_ONLY", True), ("FAST", False)])
def test_the_configs_preset_reaches_the_backend_a_build_connects(rerank_calls, tmp_path, preset, reranks):
    config = load_config(None)
    config["knowledge_search"]["preset"] = preset
    record = IndexRecord(
        name="org-x", id="index-id", created_at="t", source="s", search_backend="kg_graph_search",
        backend_refs=dict(INDEX_REFS), page_count=1, path=Path(tmp_path),
    )
    backend = search_for(record, config)
    assert (backend.use_llm_reranker, backend._llm_backend is not None) == (reranks, reranks)


def test_the_workflow_search_follows_its_backends_reranker(rerank_calls):
    fast = KnowledgeSearchFactory.create("kg_graph_search", params=INDEX_REFS, preset="FAST")
    WorkflowRepoSearch(kg_search=fast).search("goal")
    full = KnowledgeSearchFactory.create("kg_graph_search", params=INDEX_REFS)
    WorkflowRepoSearch(kg_search=full).search("goal")
    assert len(rerank_calls) == 1


def test_the_gates_backend_is_built_from_the_campaigns_config(rerank_calls):
    config = load_config(None)
    config["inference"]["default"]["cli"] = "claude_code"
    backend = search_from_refs("kg_graph_search", INDEX_REFS, config)
    assert backend._llm_backend._inference == config["inference"]

    config["knowledge_search"]["preset"] = "FAST"
    assert search_from_refs("kg_graph_search", INDEX_REFS, config)._llm_backend is None


def test_a_campaign_sessions_gate_server_is_launched_with_the_campaigns_config(tmp_path):
    servers, _tools = get_mcp_config(gates=["idea"], kg_index_path=str(tmp_path / "index.json"), config_path="/c/kapso.yaml")
    assert servers["gated-knowledge"]["args"][-2:] == ["--config", "/c/kapso.yaml"]


def test_a_failed_rerank_raises_instead_of_returning_the_original_order():
    class Down:
        def llm_completion_with_system_prompt(self, **kwargs):
            raise RuntimeError("provider down")

    search = module.KGGraphSearch.__new__(module.KGGraphSearch)
    search._llm_backend = Down()
    pool = [KGResultItem(id="a", score=0.9, page_type="Workflow", overview="o")]
    with pytest.raises(RuntimeError, match="provider down"):
        search._rerank_results("q", pool, 1)
