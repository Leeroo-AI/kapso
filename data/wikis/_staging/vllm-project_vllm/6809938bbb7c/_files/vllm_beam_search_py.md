# File: `vllm/beam_search.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 88 |
| Classes | `BeamSearchSequence`, `BeamSearchOutput`, `BeamSearchInstance` |
| Functions | `get_beam_search_score`, `create_sort_beams_key_function` |
| Imports | dataclasses, typing, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Beam search algorithm implementation for text generation.

**Mechanism:** Implements beam search decoding for exploring multiple hypothesis sequences in parallel. The `BeamSearchSequence` dataclass tracks individual sequence candidates with their IDs, tokens, and cumulative scores. `BeamSearchOutput` holds the final results. `BeamSearchInstance` manages the search state for a request, maintaining beams and tracking completion. The `get_beam_search_score()` function computes length-normalized scores, while `create_sort_beams_key_function()` creates sorting functions for ranking beams. Beam search explores the top-k most promising sequences at each step, pruning less likely paths.

**Significance:** Provides an alternative to greedy or sampling-based decoding. Beam search is useful for tasks requiring high-quality, diverse outputs like translation or summarization. It balances exploration vs exploitation by maintaining multiple hypotheses. While more expensive than greedy decoding, it often produces better results for structured generation tasks.
