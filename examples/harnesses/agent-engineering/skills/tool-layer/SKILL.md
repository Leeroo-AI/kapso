---
name: tool-layer
description: Design of the tools a scaffold gives its model, with few distinct tools tested for confusability, descriptions with examples, a shell tool plus an exact-match editor, results shaped and restorable rather than truncated, deferred and discovered tools through a registry without breaking the prefix, native function calling, the round-trip test of the actual adapter, and the parse-as-untrusted rule on local engines. Use when adding or changing a tool, when the triage shows malformed calls, redundant calls or wrong-parameter errors, and when tool results flood the context.
---
# The tool layer

## Few, distinct, documented

- Distinguish tools by effect and precondition; each earns its slot by doing something no
  other tool does. Test confusable tool pairs on the target checkpoint before adding more
  descriptions or another selector: in the measurements, many distinct tools worked where
  few overlapping ones failed.
- A description says what the tool does, when to use it, and what it returns; one to five
  usage examples raised parameter accuracy from 72 to 90 percent in one study, and a
  tool-testing agent that rewrote descriptions from its own failures made task completion
  40 percent faster in another.
- Test the checkpoint's supported tool protocol with representative payloads; native
  function calling beat text protocols where measured, and code inside a JSON string
  still needs escaping, so a tool that takes code takes it as a plain string field or a
  file path and the round trip is checked.

## Deferred and discovered tools

Keep the stable tool definitions in a deterministic prefix. For tools loaded on demand or
discovered at run time, keep an authoritative registry of name, schema, provenance and
current permission; load definitions through the runtime's supported discovery mechanism;
dispatch only calls whose schema and permission are currently valid; preserve earlier
calls and results when availability changes; and accept the cache miss when correctness
requires a definition the prefix did not hold. Deferred loading cut context 85 percent
and raised accuracy from 49 to 74 on one evaluation; a mention in a document is not a
permission, and the unlock state is cleared between episodes.

## The shell and the editor

- A shell tool is the universal fallback; it is also where command-text rules fail, so the
  sandbox, not the tool description, is the boundary (deterministic-guards).
- Use an editor contract the target checkpoint can satisfy: an `old_str` to `new_str`
  replacement that must match exactly once, the file read first, absolute paths, a stale
  match forcing a re-read. Measure stale matches, ambiguous edits and rejected edits;
  earlier models needed unified diffs or fuzzy matching, current ones do not, and the
  strict contract catches more mistakes.
- Run cheap syntax or type checks on edit when they can localise the failure to the edit,
  and return the error as the tool result: a mistake caught at the write costs one step;
  the same mistake found by the test suite costs ten.

## Results shaped, not truncated

- Cap a result (one shipped harness caps tool responses at 25,000 tokens and spills shell
  output past 30,000 characters to a file with a preview); keep a breadcrumb so the full
  result can be re-fetched; pipe noisy commands through a filter so a test run returns
  hundreds of tokens instead of tens of thousands.
- Keep data out of the model: a filter written in code turned 150,000 tokens into 2,000,
  "five rows instead of 10,000", and letting the model filter its own tool output raised
  one browsing benchmark from 45.3 to 61.6. Loops, error handling and personal-data
  tokenisation happen in code, not in the context.
- Repeated invalid-parameter errors are a description problem; redundant calls are a
  pagination or result-shape problem.

## The round trip, tested on the actual adapter

Before scoring, exercise the scaffold's own adapter with a synthetic tool that takes a
string containing quotes and a newline, an enum and a number. Complete the cycle
assistant call, tool result, assistant reply; verify the call ids, the argument types and
the result association. Exercise a tool-only assistant message, several calls in one
turn, a truncated call and a rejected call. Never execute partially parsed arguments.

## On a local engine

Tool calls are extracted from text by a parser that the engine itself calls best-effort;
a malformed call comes back as content. Validate the parser end to end on the exact engine
version; constrain the arguments with a grammar, never the whole response (a
whole-response schema beside tools drove tool invocation to zero on every open-weight
model tested); re-ask once with the schema on a malformed call; log every repair. The
local-model-serving skill has the engine facts.

## What to measure

Tool error rate, redundant calls per episode, tokens per call, the share of episodes with
a repaired call, and the result-size distribution; a tool whose results are always cut is
a tool that returns the wrong shape.
