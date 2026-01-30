# Enhance Experiment Memory

## Goal

Leverage concepts from the archived `archive/memory` module to enhance the current `src/execution/memories/experiment_memory` system.

## Current State

The current `ExperimentHistoryStore` is a simple storage system:
- Stores raw experiment records (solution, score, feedback, error)
- Provides basic retrieval (top by score, recent, semantic search via Weaviate)
- Accessed via MCP tools by agents during ideation

## Archived Memory System Features

The archived `archive/memory` module had a sophisticated cognitive memory architecture with several valuable concepts:

### 1. Insight Extraction (LLM-based generalization)

**Source**: `archive/memory/insight_extractor.py`

Instead of storing raw errors/feedback, the system used LLM to extract **generalized, reusable insights**:

```python
@dataclass
class ExtractedInsight:
    lesson: str           # "The 'peft' library must be installed for LoRA operations"
    trigger_conditions: str  # "When using LoraConfig, get_peft_model, or PEFT classes"
    suggested_fix: str    # "Run 'pip install peft' before running the script"
    confidence: float
    tags: List[str]
```

**Key benefit**: Raw errors like `ModuleNotFoundError: No module named 'peft'` become actionable lessons that transfer to future problems.

**Prompts used**:
- `extract_error_insight.md` - Generalizes errors into lessons
- `extract_success_insight.md` - Extracts best practices from successes

### 2. Insight Types and Confidence

**Source**: `archive/memory/types.py`

```python
class InsightType(Enum):
    CRITICAL_ERROR = "critical_error"      # Mistakes to avoid
    BEST_PRACTICE = "best_practice"        # Patterns that work
    DOMAIN_KNOWLEDGE = "domain_knowledge"  # Facts about the problem domain
```

**Key benefit**: Categorized insights enable better filtering and prioritization.

### 3. LLM-Governed Episodic Retrieval

**Source**: `archive/memory/episodic_retriever.py`

Instead of simple semantic search, the system used LLM to:
1. **Formulate smart queries** based on current context (goal, step, error)
2. **Rank retrieved insights** for actual relevance
3. **Filter** to only directly applicable insights

```python
@dataclass
class RankedInsight:
    content: str
    relevance_score: float
    applicability: str      # "How this insight applies to current situation"
    should_use: bool        # LLM decides if it's actually useful
```

**Key benefit**: Avoids dumping irrelevant past errors into context.

### 4. Duplicate Detection

**Source**: `archive/memory/episodic.py`

```python
DUPLICATE_THRESHOLD = 0.95  # Similarity threshold

def _is_duplicate(self, embedding: List[float]) -> bool:
    """Check if a similar insight already exists in Weaviate."""
    # Uses cosine similarity to detect near-duplicates
```

**Key benefit**: Prevents storing the same lesson multiple times.

### 5. Confidence-Based Filtering

**Source**: `archive/memory/episodic.py`

```python
min_confidence = 0.5  # Configurable threshold

# Low-confidence insights stored in JSON only (not Weaviate)
if insight.confidence < self.min_confidence:
    # JSON only - won't pollute semantic search
```

**Key benefit**: High-quality insights get priority in retrieval.

## Proposed Enhancements

### Phase 1: Add Insight Extraction

Add LLM-based insight extraction to `ExperimentHistoryStore`:

```python
# New file: src/execution/memories/experiment_memory/insight_extractor.py

class InsightExtractor:
    """Extract generalized insights from experiment results."""
    
    def extract_from_error(self, error_message: str, goal: str) -> ExtractedInsight:
        """Generalize error into reusable lesson."""
        
    def extract_from_success(self, feedback: str, score: float, goal: str) -> ExtractedInsight:
        """Extract best practice from success."""
```

**Changes to store.py**:
```python
def add_experiment(self, node: Any) -> None:
    # ... existing code ...
    
    # NEW: Extract generalized insight
    if node.had_error and node.error_message:
        insight = self.insight_extractor.extract_from_error(
            error_message=node.error_message,
            goal=self.goal,
        )
        self._store_insight(insight)
    elif node.score and node.score > 0.7:  # Good result
        insight = self.insight_extractor.extract_from_success(
            feedback=node.feedback,
            score=node.score,
            goal=self.goal,
        )
        self._store_insight(insight)
```

### Phase 2: Add Insight Types

Enhance `ExperimentRecord` with insight categorization:

```python
@dataclass
class ExperimentRecord:
    # ... existing fields ...
    
    # NEW: Extracted insight (if any)
    insight: Optional[str] = None
    insight_type: Optional[str] = None  # "critical_error", "best_practice"
    insight_confidence: Optional[float] = None
    insight_tags: List[str] = field(default_factory=list)
```

### Phase 3: Add Smart Retrieval

Add LLM-governed retrieval to `ExperimentHistoryGate`:

```python
# New tool: search_relevant_insights
Tool(
    name="search_relevant_insights",
    description=(
        "Search for relevant insights from past experiments. "
        "Uses LLM to find and rank insights that are actually applicable "
        "to your current situation."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "goal": {"type": "string", "description": "Current goal"},
            "current_error": {"type": "string", "description": "Current error (if any)"},
            "k": {"type": "integer", "default": 5},
        },
        "required": ["goal"],
    },
)
```

### Phase 4: Add Duplicate Detection

Prevent storing duplicate insights:

```python
def _is_duplicate_insight(self, insight_text: str) -> bool:
    """Check if similar insight already exists."""
    if not self.weaviate:
        # Fallback: exact string match
        return any(e.insight == insight_text for e in self.experiments)
    
    # Semantic similarity check
    embedding = self._generate_embedding(insight_text)
    results = self.weaviate.query.near_vector(embedding, limit=1)
    if results and results[0].distance < 0.1:  # Very similar
        return True
    return False
```

## Implementation Priority

1. **High Priority**: Insight extraction from errors
   - Most impactful for learning from failures
   - Directly improves ideation quality

2. **Medium Priority**: Smart retrieval with LLM ranking
   - Reduces noise in context
   - Requires more infrastructure

3. **Lower Priority**: Success insight extraction
   - Useful but less critical than error learning

4. **Lower Priority**: Duplicate detection
   - Nice to have, prevents bloat

## Files to Create/Modify

### New Files
- `src/execution/memories/experiment_memory/insight_extractor.py`
- `src/execution/memories/experiment_memory/prompts/extract_error_insight.md`
- `src/execution/memories/experiment_memory/prompts/extract_success_insight.md`

### Modified Files
- `src/execution/memories/experiment_memory/store.py` - Add insight extraction
- `src/execution/memories/experiment_memory/__init__.py` - Export new types
- `src/gated_mcp/gates/experiment_history_gate.py` - Add smart retrieval tool
- `src/gated_mcp/presets.py` - Add new tool to gate definition

## Migration Notes

- The archived `archive/memory` module has dependencies on `src.memory.*` imports that no longer exist
- The `CognitiveController` and `DecisionMaker` are tightly coupled to the old context manager system
- We should extract only the insight extraction and retrieval logic, not the full cognitive architecture
- The prompts in `archive/memory/prompts/` can be reused directly

## Testing

- Unit tests for `InsightExtractor`
- Integration tests for insight storage and retrieval
- E2E test: run experiment loop and verify insights are extracted and retrievable
