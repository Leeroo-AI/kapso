## v0 Design Draft: Make `Tinkerer.learn()` index-aware

### Problem statement (v0)
`Tinkerer.learn()` writes new knowledge into a target KG directory (usually a wiki directory like `data/wikis`). KGs can be large, so re-indexing the entire KG after every `learn()` is too expensive.

We need `learn()` to be able to update an existing KG index (when present), instead of requiring a full rebuild.

### Current behavior (relevant bits)
- `Tinkerer.index_kg(...)` builds an index once and writes a `.index` file (backend type + backend refs + page_count).
- `Tinkerer(kg_index="...")` loads the index and constructs `self.knowledge_search`.
- `learn()` ingests sources into wiki pages and optionally merges them, but has no explicit “incremental index sync” contract.

### v0 proposal: update the `learn()` signature
Add keyword-only args so callers can opt into index updates:

```python
def learn(
    self,
    *sources,
    wiki_dir: str = "data/wikis",
    skip_merge: bool = False,
    *,
    kg_index: str | None = None,
    index_update: str = "auto",  # "auto" | "incremental" | "full" | "none"
) -> PipelineResult:
    ...
```

### v0 incremental algorithm (naive)
Use `PipelineResult.merge_result` to infer which pages changed:
- **Added** = `merge_result.created`
- **Edited** = merge targets from `merge_result.merged` (tuple `(proposed_id, target_id)`)
- **Deleted** = not handled

Then:
- For “added”: call backend `index()` on those pages only.
- For “edited”: call backend `edit()` on those pages only.

### Next step
This is intentionally the first-pass draft. It has correctness gaps (notably deletes and robustness).

See `design.md` for the critique and the refined design.
