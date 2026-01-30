# Task: Enhance Experiment Memory with Insight Extraction

## Status: COMPLETED

## Overview

Add LLM-based insight extraction to experiment memory, enabling agents to learn generalized lessons from past experiments instead of raw error messages.

## Tasks

### Phase 1: Create Insight Extractor

- [ ] **1.1** Create `src/execution/memories/experiment_memory/insight_extractor.py`
  - Define `ExtractedInsight` dataclass:
    - `lesson: str` - Generalized principle
    - `trigger_conditions: str` - When this applies
    - `suggested_fix: str` - Actionable steps
    - `confidence: float` - 0.0-1.0
    - `tags: List[str]` - Keywords for retrieval
  - Define `InsightType` enum:
    - `CRITICAL_ERROR` - Mistakes to avoid
    - `BEST_PRACTICE` - Patterns that work
  - Implement `InsightExtractor` class:
    - `extract_from_error(error_message, goal, solution?)` -> `ExtractedInsight`
    - `extract_from_success(feedback, score, goal, solution?)` -> `ExtractedInsight`
  - Use lazy LLM initialization (avoid import at module load)

- [ ] **1.2** Create prompt templates
  - `src/execution/memories/experiment_memory/prompts/extract_error_insight.md`
    - Copy from `archive/memory/prompts/extract_error_insight.md`
    - Adapt variables: `{context}`, `{error_message}`
  - `src/execution/memories/experiment_memory/prompts/extract_success_insight.md`
    - Copy from `archive/memory/prompts/extract_success_insight.md`
    - Adapt variables: `{context}`, `{feedback}`

### Phase 2: Enhance ExperimentRecord

- [ ] **2.1** Update `src/execution/memories/experiment_memory/store.py`
  - Add new fields to `ExperimentRecord`:
    ```python
    insight: Optional[str] = None
    insight_type: Optional[str] = None  # "critical_error" or "best_practice"
    insight_confidence: Optional[float] = None
    insight_tags: List[str] = field(default_factory=list)
    ```
  - Ensure backward compatibility (new fields are optional)

### Phase 3: Integrate Insight Extraction into Store

- [ ] **3.1** Update `ExperimentHistoryStore.__init__`
  - Add optional `goal` parameter (needed for insight extraction context)
  - Add optional `enable_insights` parameter (default: True)
  - Lazy-initialize `InsightExtractor` when needed

- [ ] **3.2** Update `ExperimentHistoryStore.add_experiment`
  - After creating `ExperimentRecord`, extract insight:
    ```python
    if self.enable_insights:
        if record.had_error and record.error_message:
            insight = self._extract_error_insight(record)
            record.insight = insight.lesson
            record.insight_type = "critical_error"
            record.insight_confidence = insight.confidence
            record.insight_tags = insight.tags
        elif record.score and record.score > 0.7:
            insight = self._extract_success_insight(record)
            record.insight = insight.lesson
            record.insight_type = "best_practice"
            record.insight_confidence = insight.confidence
            record.insight_tags = insight.tags
    ```
  - Add helper methods: `_extract_error_insight`, `_extract_success_insight`

- [ ] **3.3** Update Weaviate indexing
  - Add `insight`, `insight_type`, `insight_confidence` to Weaviate schema
  - Index insight text for semantic search (in addition to solution+feedback)

### Phase 4: Add Duplicate Detection

- [ ] **4.1** Add duplicate detection to store
  - Add `DUPLICATE_THRESHOLD = 0.95` constant
  - Implement `_is_duplicate_insight(insight_text)` method:
    - If Weaviate available: use cosine similarity
    - Fallback: exact string match in local list
  - Skip storing if duplicate detected

### Phase 5: Update MCP Gate

- [ ] **5.1** Update `src/gated_mcp/gates/experiment_history_gate.py`
  - Update `_format_experiments` to include insight if available:
    ```python
    if exp.insight:
        lines.append(f"**Insight ({exp.insight_type}):**")
        lines.append(f"{exp.insight}")
    ```

- [ ] **5.2** Add new tool: `get_insights`
  - Returns only experiments with extracted insights
  - Filters by insight_type if specified
  - Sorted by insight_confidence

- [ ] **5.3** Update `src/gated_mcp/presets.py`
  - Add `get_insights` to `experiment_history` gate tools list

### Phase 6: Update Orchestrator

- [ ] **6.1** Update `src/execution/orchestrator.py`
  - Pass `goal` to `ExperimentHistoryStore` constructor:
    ```python
    self.experiment_store = ExperimentHistoryStore(
        json_path=experiment_history_path,
        weaviate_url=os.environ.get("WEAVIATE_URL"),
        goal=self.goal,  # NEW
    )
    ```

### Phase 7: Update Exports

- [ ] **7.1** Update `src/execution/memories/experiment_memory/__init__.py`
  - Export `InsightExtractor`, `ExtractedInsight`, `InsightType`

### Phase 8: Testing

- [ ] **8.1** Create `tests/test_insight_extractor.py`
  - Test error insight extraction
  - Test success insight extraction
  - Test fallback when LLM fails

- [ ] **8.2** Update existing experiment memory tests
  - Verify backward compatibility (old records without insights still work)
  - Test insight storage and retrieval

## Notes

- Insight extraction is async-friendly but runs synchronously (LLM call)
- Keep extraction optional to avoid breaking existing usage
- Use `gpt-4o-mini` as default model for cost efficiency
- Prompts should produce JSON output for reliable parsing

## Dependencies

- Existing: `openai` (for LLM calls)
- No new dependencies required
