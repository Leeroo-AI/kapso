## Incremental KG Index Updates for `Tinkerer.learn()`

### Problem statement
`Tinkerer.learn()` writes new knowledge into a persistent wiki directory (`wiki_dir`, usually `data/wikis`). Many KGs are large, so **re-indexing the entire KG after every learn is too expensive**.

We need a way for `learn()` to **incrementally update the active KG index** for pages that were:
- **added**
- **edited**
- **deleted**

This must work across different search/index backends in `src/knowledge/search/` (e.g., `kg_graph_search`, `kg_llm_navigation`), each of which has different storage and indexing behavior.

### Current behavior (what exists today)
- **Source of truth**: wiki files on disk (e.g., `data/wikis/...`) are treated as the KG ground truth in most docs and code paths.
- **Index creation**: `Tinkerer.index_kg(...)` builds the index once, and writes a `.index` “pointer file” containing:
  - backend type (`search_backend`)
  - source location (`data_source`)
  - backend references (`backend_refs`) like Weaviate collection names or Neo4j URIs
  - `page_count`
- **Index loading**: `Tinkerer(kg_index="...")` loads the `.index` and constructs `self.knowledge_search` for later retrieval during `evolve()`.
- **Learn path today**:
  - `learn()` runs `KnowledgePipeline(...).run(...)`.
  - The pipeline extracts `WikiPage` objects (Stage 1), then optionally “merges” (Stage 2).
  - There is **no explicit “incremental index sync” step** that updates the loaded index in a deterministic way.
  - There is **no supported delete-by-page-id API** across backends.

### Requirements
- **R1: Simple API**. If an index exists (`kg_index` was provided or loaded), `learn()` updates it. No additional “mode” flags.
- **R2: Incremental sync**. Only touched pages should be re-indexed.
- **R3: Deletes**. Removed pages must be removed from the index (when the backend supports it).
- **R4: Backend-aware**. The design must support different indexing strategies per backend.
- **R5: Single source of truth**. The wiki directory should remain the ground truth for wiki-based KGs. Indexes are derived and rebuildable.
- **R6: Safe failure mode**. If index sync fails, the wiki files must still be correct. Users can rebuild the index later.

### Non-goals (v1)
- **No live filesystem watch service**. We will not run a daemon to continuously sync changes.
- **No remote KG targets** (e.g., URL-based KGs) in v1.
- **No automatic “semantic merging” improvements**. This design focuses on index sync, not on better knowledge extraction.

---

## Design (v1)

### Key idea
Make indexing updates **a deterministic sync step** that compares the wiki directory state to the previously indexed state, and then applies a **delta** to the backend.

This makes index updates correct even when:
- the pipeline rewrites files
- the user manually edits pages
- pages are deleted

### v1: New concepts

#### 1) `KGIndexState` manifest
Alongside the `.index` file, store a small “last synced snapshot” of wiki pages, e.g.:

- **Path**: `data/indexes/<name>.state.json` (or `save_to + ".state.json"`)
- **Contents** (per page):
  - `page_id`
  - `file_path`
  - `mtime_ns`
  - `size_bytes`
  - `content_sha256` (only for changed files; stored to avoid re-hashing unchanged ones)

This state file is the authoritative reference for computing deltas.

#### 2) `KGIndexDelta`
Compute:
- **added**: page_ids present now, absent in state
- **updated**: page_ids present in both, but file changed
- **deleted**: page_ids absent now, present in state

#### 3) Backend incremental operations (new interface)
Extend `KnowledgeSearch` with optional incremental methods:

```python
class KnowledgeSearch:
    def upsert_pages(self, pages: list[WikiPage]) -> None: ...
    def delete_pages(self, page_ids: list[str]) -> None: ...
```

- Default implementations can raise `NotImplementedError`.
- Backends implement what they can.

### v1: `learn()` API (final proposal)
No backward-compat constraints. Keep the API “one-switch”: **if `kg_index` exists, we sync it**.

```python
def learn(
    self,
    *sources,
    wiki_dir: str,
    kg_index: str | None = None,
) -> PipelineResult:
    ...
```

**Behavior**
- If `kg_index` is passed, `learn()` updates that index.
- Else if the `Tinkerer` was initialized with `kg_index=...`, `learn()` updates that loaded index.
- Else, `learn()` only updates `wiki_dir` on disk (no index sync).

**Safety check**
- If an index is present and it declares a `data_source` that is a wiki directory, require `data_source == wiki_dir` (to prevent updating the wrong index).

### v1: Index sync flow

#### Step A: Learn writes wiki changes (source-of-truth)
`learn()` runs the pipeline and ensures updated wiki files are present in `wiki_dir`.

#### Step B: Compute delta (fast)
Scan wiki files in `wiki_dir` and compare to the `KGIndexState` manifest.
- Read file content only if `mtime` or size changed.

#### Step C: Apply delta (backend-specific)
Parse changed files into `WikiPage` objects, then:
- **added + updated** → `knowledge_search.upsert_pages(pages)`
- **deleted** → `knowledge_search.delete_pages(page_ids)`

#### Step D: Update metadata
- Update the `.index` file’s `page_count`.
- Write updated `KGIndexState` manifest.

### Backend-specific behavior (v1)

#### `kg_graph_search` (Weaviate + Neo4j)
- **Upsert**:
  - Neo4j: `MERGE` page node by `page_id`, then rebuild edges for that page.
  - Weaviate: use a stable identifier to avoid duplicates.
    - **Preferred**: use deterministic UUID = `uuid5(NAMESPACE, page_id)` and always upsert by UUID.
    - **Fallback**: query by `page_id`, update if found else insert.
- **Delete**:
  - Neo4j: `MATCH (p:WikiPage {id: $page_id}) DETACH DELETE p`
  - Weaviate: delete objects where `page_id == $page_id`

#### `kg_llm_navigation` (Neo4j only, Node/edge dict)
Two valid options:
- **Option 1 (recommended for v1)**: declare `learn()` index sync unsupported for `kg_llm_navigation` indexes. If the active index backend is `kg_llm_navigation`, `learn()` raises a clear error telling the user to use a wiki-backed index (`kg_graph_search`) or to implement Option 2.
- **Option 2 (more ambitious)**: add a “wiki → node/edge” adapter:
  - Node: `{id: page.id, name: page.page_title, type: page.page_type, overview, content, domains}`
  - Edge: `{source: page.id, target: f"{t}/{id}", relationship: edge_type.upper()}`

### Failure handling
- **If sync fails**:
  - Keep wiki files as the source-of-truth.
  - Return the pipeline result plus an index-sync error list (the index is now known-stale).
- **If infrastructure is down**:
  - Same behavior: wiki files are correct; users can rebuild with `index_kg(...)`.

### Why this is the “best” design
- **Correctness-first**: delta is computed from the source-of-truth, not from fragile intermediate results.
- **Works with manual edits**: any file changes are picked up.
- **Supports deletes**: state-based diff naturally captures deletions.
- **Backend abstraction is explicit**: backends implement incremental primitives where they can.

### Critique (what’s still not perfect)
- **Directory scan cost**: scanning a very large wiki directory is still \(O(N)\) in number of files. We reduce the heavy work by hashing only changed files.
- **Requires new backend APIs**: `delete_pages` and `upsert_pages` need implementation work.
- **Weaviate uniqueness**: deterministic UUIDs are the clean fix, but it is a schema/behavior change.
- **Two sources of truth risk (if not careful)**: we must be strict that wiki files are canonical and the index is derived.

### Refinement (v1.1 tweaks)
- **Use deterministic UUIDs in Weaviate** for `kg_graph_search`.
  - Eliminates duplicates.
  - Makes upserts cheap and reliable.
- **Store `overview_sha256` in state** to avoid re-embedding when only non-overview content changes (optional).
- **Add a public method** `Tinkerer.sync_kg_index(...)` so users can sync without re-learning.

---

## Recommended rollout plan
- **Phase 1**: Implement `sync_kg_index(wiki_dir, kg_index)` (state + delta computation + apply delta).
- **Phase 2**: Implement `kg_graph_search.upsert_pages` + `delete_pages` (deterministic Weaviate identity).
- **Phase 3**: Implement `kg_llm_navigation.upsert_pages` + `delete_pages` (wiki→Node/edge adapter) or explicitly document it as unsupported.

## Test plan (high value)
- **Unit**:
  - delta detection: add/update/delete files → correct delta
  - state file round-trip
- **Integration (kg_graph_search)**:
  - index small wiki_dir, sync incremental, verify:
    - new page searchable
    - updated page has updated overview embedding and Neo4j edges
    - deleted page no longer searchable / no Neo4j node
