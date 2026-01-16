# File: `WebAgent/WebWalker/src/prompts.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 65 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines prompt templates for the WebWalker agent system, including the main exploration prompt and critic prompts for information extraction and answer generation.

**Mechanism:** Contains three main prompts: (1) `SYSTEM_EXPLORER` - the ReAct-style prompt that instructs the agent to dig through buttons to find information, with format specifications for Thought/Action/Action Input/Observation cycles (up to 20 iterations), emphasizing that the agent must always take action and explore recursively; (2) `STSTEM_CRITIIC_INFORMATION` - instructs an LLM to evaluate if an observation contains useful information for the query, returning JSON with usefulness boolean and extracted information; (3) `STSTEM_CRITIIC_ANSWER` - instructs an LLM to judge if accumulated information is sufficient to answer the query, returning JSON with judge boolean and generated answer.

**Significance:** Configuration file that defines WebWalker's exploration behavior and two-stage critic system. The prompts enforce persistent exploration (never giving up) and structured information accumulation until a satisfactory answer can be generated.
