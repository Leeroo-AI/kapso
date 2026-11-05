"""Looking inside an ingested index: `search` runs a query over the
index's stores the way a build's search gates do, `show` reads one page
from the index's wiki. Both are read-only: nothing is written to the
index, the experience home or the stores."""

from typing import Any, Dict, List

from kapso.experience.ingest import WIKI_DIR, search_for
from kapso.experience.sources import ExperienceError
from kapso.experience.store import IndexRecord
from kapso.knowledge_base.search import KGOutput, WikiPage
from kapso.knowledge_base.search.kg_graph_search import parse_wiki_directory


def open_search(record: IndexRecord, config: Dict[str, Any]):
    """The search backend an index names, connected, with its pages
    proven present. A backend whose stores are down or were wiped answers
    every query with nothing, so an empty store is refused here rather
    than read as "no match"."""
    search = search_for(record, config)
    if not search.validate_backend_data():
        search.close()
        raise ExperienceError(
            f"the index `{record.name}` has no pages in its stores (collection "
            f"{record.backend_refs.get('weaviate_collection')} at "
            f"{record.backend_refs.get('weaviate_url')}); the stores are down or were wiped since it was "
            f"added — `kapso experience remove {record.name}` and add it again"
        )
    return search


def index_page(record: IndexRecord, page_id: str) -> WikiPage:
    """One page of an index, from the wiki files the index was built from
    (the text its stores hold), by id: `Principle/Paged_Optimizers`."""
    pages = {page.id: page for page in parse_wiki_directory(record.path / WIKI_DIR)}
    if page_id not in pages:
        raise ExperienceError(
            f"the index `{record.name}` has no page `{page_id}` "
            f"(`kapso experience search {record.name} \"<query>\"` prints the ids it has)"
        )
    return pages[page_id]


def render_hits(output: KGOutput) -> str:
    """The ranked pages of a search, each with its id, type, score and
    overview."""
    if output.is_empty:
        return f"no pages matched {output.query!r}"
    lines: List[str] = []
    for rank, item in enumerate(output.results, 1):
        lines.append(f"[{rank}] {item.id}   {item.page_type}   score {item.score:.2f}")
        lines.append(item.overview)
        lines.append("")
    return "\n".join(lines).rstrip()


def render_page(page: WikiPage) -> str:
    """One page whole: its identity, its text, its sources and the pages
    it links to."""
    lines = [f"{page.id}   {page.page_type}"]
    if page.domains:
        lines.append(f"domains: {', '.join(page.domains)}")
    if page.last_updated:
        lines.append(f"last updated: {page.last_updated}")
    lines += ["", page.content.rstrip()]
    if page.sources:
        lines += ["", "sources"]
        lines += [f"  {source.get('type', 'Link')}: {source.get('title', 'Reference')} {source.get('url', '')}".rstrip() for source in page.sources]
    if page.outgoing_links:
        lines += ["", "links"]
        lines += [f"  {link['edge_type']} -> {link['target_type']}: {link['target_id']}" for link in page.outgoing_links]
    return "\n".join(lines)
