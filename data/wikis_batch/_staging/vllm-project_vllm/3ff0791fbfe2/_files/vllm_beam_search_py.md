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

**Purpose:** Beam search algorithm implementation

**Mechanism:** Implements beam search decoding through three main classes: BeamSearchSequence (tracks individual beam candidates with tokens and cumulative probability), BeamSearchOutput (groups sequences by request), and BeamSearchInstance (manages beam search state per request). Includes scoring function (get_beam_search_score) with length penalty support and beam sorting utilities. Tracks parent-child relationships between beams for path reconstruction.

**Significance:** Provides alternative decoding strategy to greedy/sampling for improved generation quality in specific use cases. Beam search explores multiple hypotheses in parallel, useful for tasks requiring high-quality outputs like translation or summarization. Complements vLLM's primary sampling-based generation.
