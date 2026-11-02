"""Ingesting a corpus into an index of the experience home: a repository,
or a research question answered on the web, becomes wiki pages through the
knowledge pipeline; the pages are indexed into the graph backend under the
index's own namespace (a Weaviate collection and a Neo4j corpus named
after the index), and `index.json` pins the index at its id. An index is
never written again after `add`: a name that exists is refused, `remove`
deletes the namespace with the record. Design:
docs/plans/redesign/wp3-experience.md (change 7).
"""

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from kapso.experience.reference import RESERVED, valid_index_name
from kapso.experience.store import ExperienceStore, IndexRecord
from kapso.knowledge_base.learners import KnowledgePipeline, Source
from kapso.knowledge_base.learners.defaults import learner_defaults
from kapso.knowledge_base.search import KGIndexInput, KnowledgeSearchFactory
from kapso.researcher import Researcher

GRAPH_SEARCH = "kg_graph_search"
WIKI_DIR = "wiki"
_NOT_ALNUM = re.compile(r"[^A-Za-z0-9]+")


class IngestError(ValueError):
    """A corpus that cannot be ingested, with the line that would make it."""


def collection_name(name: str, index_id: str) -> str:
    """The Weaviate collection an index owns: a class name (letters and
    digits, capitalised) from the index's name and id, so a name removed
    and added again never meets its old pages."""
    words = "".join(part.capitalize() for part in _NOT_ALNUM.split(name) if part)
    return f"Kapso{words}{index_id}"


def backend_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """The config's `knowledge_search.params` with the models, retry and
    inference blocks the backend needs."""
    params = dict(config["knowledge_search"]["params"])
    params.setdefault("models", config["models"])
    params.setdefault("retry", config["retry"])
    params.setdefault("inference", config["inference"])
    return params


def search_params(config: Dict[str, Any], name: str, index_id: str) -> Dict[str, Any]:
    """The graph backend's parameters for one index: the backend's, and the
    index's own namespace over them."""
    params = backend_params(config)
    params["weaviate_collection"] = collection_name(name, index_id)
    params["corpus"] = index_id
    return params


def search_from_refs(search_backend: str, backend_refs: Dict[str, Any], config: Dict[str, Any]):
    """The search backend an index names, connected: its backend references
    over the config's parameters and preset, the way the index was built."""
    return KnowledgeSearchFactory.create(
        search_type=search_backend,
        params={**backend_params(config), **backend_refs},
        preset=config["knowledge_search"]["preset"],
    )


def search_for(record: IndexRecord, config: Dict[str, Any]):
    return search_from_refs(record.search_backend, record.backend_refs, config)


def check_name(name: Optional[str], store: ExperienceStore) -> str:
    if not name:
        raise IngestError("an index needs a name: `--name <n>` (`kapso experience list` shows the names taken)")
    if name in RESERVED:
        raise IngestError(f"`{name}` is a reserved reference; name the index something else")
    if not valid_index_name(name):
        raise IngestError(f"`{name}` is not an index name: letters, digits and dashes, starting with a letter or digit")
    if (store.index_dir(name) / "index.json").is_file():
        raise IngestError(
            f"the index `{name}` exists (`kapso experience list`); an index is never written again after `add` — "
            f"`kapso experience remove {name}` first, or pick another name"
        )
    return name


def research_sources(question: str, settings: Dict[str, Any], config_path: Optional[str]) -> List[Any]:
    """The web researched for a question, in the modes and at the depth
    the config's `experience` block names: the typed sources inside the
    researcher's findings (its ideas, its implementations, its report
    when one was asked for), the pipeline's input."""
    findings = Researcher(config_path=config_path).research(
        question, mode=list(settings["research_modes"]), depth=settings["research_depth"],
    )
    sources: List[Any] = list(findings.ideas) + list(findings.implementations)
    if findings.report is not None:
        sources.append(findings.report)
    if not sources:
        raise IngestError(f"the research on {question!r} returned nothing to ingest")
    return sources


def ingest(
    sources: List[Any],
    *,
    name: str,
    origin: str,
    store: ExperienceStore,
    config: Dict[str, Any],
    config_path: Optional[str],
    status,
    pipeline_factory: Optional[Callable[..., Any]] = None,
):
    """Run the knowledge pipeline into a new index `name`: the pages under
    `<index>/wiki/`, the backend namespace built by the pipeline's index
    builder, the record written when the pages are indexed. `origin` is
    what the record says the index was made from. Returns the record and
    the pipeline's result."""
    index_id = uuid.uuid4().hex[:12]
    directory = store.index_dir(name)
    wiki_dir = directory / WIKI_DIR
    learner = config["learner"]
    ingestor_params = dict(learner.get("ingestor") or {})
    ingestor_params.setdefault("publish_workflows", learner_defaults("ingestor")["publish_workflows"])
    written: Dict[str, IndexRecord] = {}

    def build_index(wiki_dir: str, save_to: str) -> str:
        """The pipeline's index builder: the pages of `wiki_dir` into the
        index's namespace, then the record; `save_to` is the merger's
        suggestion, the record lives at the index's own place."""
        search = KnowledgeSearchFactory.create(search_type=GRAPH_SEARCH, params=search_params(config, name, index_id))
        search.index(KGIndexInput(wiki_dir=wiki_dir))
        record = IndexRecord(
            name=name, id=index_id, created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), source=origin,
            search_backend=GRAPH_SEARCH, backend_refs=search.get_backend_refs(), page_count=search.get_indexed_count(),
            path=directory,
        )
        search.close()
        written["record"] = record
        return str(store.write_index_record(record))

    pipeline = (pipeline_factory or KnowledgePipeline)(
        wiki_dir=wiki_dir, ingestor_params=ingestor_params, merger_params=dict(learner.get("merger") or {}),
        index_builder=build_index,
    )
    result = pipeline.run(*sources, status=status)
    if "record" not in written:
        raise IngestError(
            f"the pipeline ingested nothing from {origin}: {result.total_pages_extracted} pages extracted, "
            f"{len(result.errors)} errors — no index was written"
        )
    return written["record"], result


def repository_source(url: str) -> List[Any]:
    return [Source.Repo(url)]
