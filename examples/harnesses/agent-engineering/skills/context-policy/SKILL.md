---
name: context-policy
description: What the scaffold keeps in the model's window and how, with the attention budget, the cache-stable append-only prefix kept until an explicit compaction boundary, just-in-time retrieval, data kept out of the model, shaped tool results, the working state in files, compaction that preserves decision evidence verbatim, the durable log outside the window, subagents as context firewalls, when to keep errors and when to clear them, and budgets chosen from the tested serving limit. Use when building the loop, when episodes loop, overflow or cost more than the budget, and when the cache hit rate drops.
---
# Context policy

## The attention budget

Performance degrades with input length on every model tested, even on trivial tasks and
even when the padding is whitespace; a single distractor costs accuracy; a focused
300-token prompt beat a 113,000-token full prompt on every model in one study; beyond
about 100,000 tokens one agent repeated actions from its history instead of planning.
Choose context by what the next decision requires, and measure the cumulative prompt
tokens of an episode separately from its largest single request: a window that never
overflows can still cost a fortune through repeated prefill of a long history.

## The prefix is a cache

Input to output runs about 100 to 1 on agent workloads, and a cache hit is far cheaper
than a prefill, so the cache hit rate is the first production metric of an agent; measure
cache reuse and prefill cost on the supplied endpoint, since hosted price ratios do not
predict local latency. Static content first (system prompt, stable tool definitions,
instruction file), dynamic content last, updates appended rather than edited,
deterministic ordering of tools and keys, no timestamp in the system prompt, and
compaction forked from the same prefix. Deferred and discovered tool definitions follow
tool-layer; cache reuse never overrides tool availability. On a local engine the same
rules hold for prefix caching, which stops at the first differing block.

## Just in time, and data out of the model

Keep identifiers (paths, queries, links) and load the material when needed with head and
tail rather than whole objects; let the folder structure carry meaning. Filter intermediate
data in code before the model sees it (150,000 tokens to 2,000 in one deployment; "five
rows instead of 10,000"); shape tool results with a cap, a preview and a breadcrumb to
re-fetch (tool-layer). Agentic search over the files outperformed retrieval indexes for
one shipped harness; a trained retriever gained 12.5 percent for another on its own
codebase; grep stays in the loop either way (grounding).

## The working state

One structured block, rewritten by the loop after every tool result and kept at the end
of the context, closest to the next generation:

```
task:        what is asked, in the task's words; the constraints
party:       who the user or requester is and what they have said (verified or not, where the task verifies identity)
constraints: runtime permissions, unresolved ambiguities and their sources
items:       records, files, accounts, URLs the task named, with ids and paths from tool outputs
plans:       one compiled plan per operation (action-review), with a status per step
sources:     documents, files, schemas read, with the quoted lines that filled a field
pending:     questions to ask, lookups to run, checks to run, handoffs awaited
done:        actions made, in order, with the ids or outputs they returned
```

Memory helps only when curated and read as a phase: a curated memory lifted one browsing
benchmark from 60.4 to 67.2, uncurated notes scored below having none, and a 14,000-step
run left 31 near-duplicate files of them. A JSON ledger is overwritten less readily than prose. Keep
the always-loaded instructions to the decisions the model cannot recover from available
sources, and measure instruction conflicts and context cost rather than trusting a length.

## Compaction preserves decision evidence

Keep the active message history append-only until an explicit compaction boundary. At
compaction, keep tool-call and tool-result pairs together; copy into the working state,
verbatim, the identifiers, paths, amounts, dates, enumerated values, tool signatures,
error messages, the requester's own messages, and the conditions, exceptions and source
spans that support pending decisions; summarise only document prose, long listings, long
logs and the model's own earlier reasoning. Keep the complete permitted trace outside the
prompt with pointers for recovery, and re-read a source span before acting when a
decision needs detail that was discarded. Summaries lose artifacts and decisions
(artifact tracking is the weakest dimension for every vendor measured), so durable state
lives in files and an append-only session log; re-inject the instruction file, the plan
and the recently touched files after any compaction, and compact on a schedule with
recall maximised rather than when the window is already full.

## Errors, kept or cleared

Within a task, keep the failed action and its error so the model updates its beliefs
(error recovery is agentic behaviour). After repeated corrections without state progress,
preserve the failed action and its error, write the lesson into the working state, and
hand over to the bounded recovery policy (deterministic-guards); do not erase evidence
only because a retry count was reached.

## Subagents as context firewalls

A subagent that consumes tens of thousands of tokens and returns a 1,000 to 2,000-token
summary pays for itself when its reduced main-context cost exceeds its extra generation
and coordination cost; measure both. It buys isolation or a fresh pair of eyes, not a
smarter answer, at 3 to 10 times the tokens. Parallel writers act on conflicting
assumptions; read-only investigations are safe.

## Budgets

A hard cap on steps per episode with a graceful close two steps before it. Reserve room
for the next tool result, the reasoning and the final output from the tested serving
limit, and compact before that reserve is exhausted; choose the threshold from measured
prefill, cache reuse, decode latency and recovery errors together. Tokens per episode are
measured before and after every change and written into the cost line.
