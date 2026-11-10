---
name: local-model-serving
description: What a locally served open-weight model does to an agent loop, with the scaffold-sensitive regime, tool calls as untrusted text, grammar constraints on arguments only, reasoning that shares the completion budget unless the engine offers a separate control, the engine's real context limit and the KV cache at concurrency, prefix caching, quantization, loop detection, sampling per checkpoint, non-reproducibility at temperature zero, and a smoke probe that measures what it can and says what it cannot. Use in setup to probe the endpoint, in implementation when a failure comes from the endpoint rather than the scaffold, and before a deployment.
---
# Local model serving

A locally served 7B to 70B model lives in the regime frontier models have left: the prompt,
the serialization of tool results and the loop guards are worth tens of points, not single
digits (a system prompt moved one 4B model's multi-turn tool calling from 22.9 to 37.5;
one open model spread 25.8 to 36.2 across four scaffolds on a terminal benchmark). On a
local stack the serving engine is part of the harness: an object to probe, pin and
budget.

## The smoke probe

```
python3 $KAPSO_HARNESS_ROOT/skills/local-model-serving/scripts/probe_endpoint.py \
  --base-url http://localhost:8000/v1 --model <name> [--api-key-env NAME] \
  [--sampling '{"temperature": 0.6, "top_p": 0.95}'] [--concurrency 6]
```

Prints one JSON object: the models the server lists and the context window it advertises;
time to first token and tokens per second for one stream; whether a reply carries
reasoning content; whether a schema-checked synthetic tool call comes back and whether
its round trip (call, tool result, final reply) completes; what a reply looks like when
the completion budget is smaller than the model's reasoning; and throughput at the given
concurrency with the number of streams that completed. The sampling settings are the
task's, passed through as given; the probe imposes none. It is a short-request smoke
test: it does not establish usable context capacity or the runner's own tool round trip,
which are tested through the actual adapter with representative long histories at the
evaluator's concurrency. The key, when one is needed, is read from the named environment
variable and never printed.

## What the measurements say

- **Tool calls are text, parsed best-effort.** Engines extract calls from raw text per model
  family and say arguments "may occasionally be malformed"; a named tool choice guarantees
  a parsable call, not a good one. Validate the parser end to end on the exact engine
  version: on hybrid-thinking models a call emitted inside the thinking block is swallowed
  and `tool_calls` comes back empty; one speculative-decoding mode corrupted arguments
  while the reasoning showed the right command. Parse as untrusted, re-ask once with the
  schema, log repairs.
- **Constrain arguments, never the response.** Grammar-constrained decoding lifted small
  models' correct-call rates by 27 to 45 points with full schema validity, so a constrained
  3B beat an unconstrained 70B; but a whole-response schema beside tools drove tool
  invocation to zero on every open-weight model tested, and constraint worsens abstention.
- **Thinking consumes the output budget.** Record the checkpoint, the chat template, the
  parser and the serving version, and test whether that combination supports a separate
  reasoning budget or a non-thinking mode; otherwise reasoning shares the completion
  limit, a small budget on a tool turn yields an empty message, and runners count a few
  consecutive empty replies as a failed episode. Distinguish a valid tool-only response
  from reasoning-only truncation and from an empty response by the finish reason and the
  usage. Use the controls the task statement permits; sampling settings differ per
  checkpoint (one family forbids greedy decoding in thinking mode), so read the card.
- **Read the engine's real context limit.** One engine takes a model's default from its
  config and reserves part of it for output; another defaults to a few thousand tokens.
  Effective context is the minimum of the window, the engine default, the KV memory at the
  chosen concurrency and the slot share; past it the engine preempts and recomputes.
  Measure long-request latency and preemption at the task's concurrency; a short
  single-stream probe does not establish usable capacity.
- **Prefix caching rewards stable prompts** and stops at the first differing block; hybrid
  attention models re-prefill in full without checkpoints. Static first, append-only.
- **Quantization**: 8-bit KV cache is near-lossless; 4-bit substantially degrades tool
  calling by the engine's own warning.
- **Small models loop.** Repeated action sequences predicted failure at 89 percent in one
  study; untuned students were stuck in loops 21 to 40 percent of the time and took fewer
  steps than they needed. Loop detection, a re-prompt with the lesson, and generous step
  budgets belong in the loop (deterministic-guards).
- **Temperature zero is not reproducible.** A thousand greedy completions of one model gave
  80 unique outputs; seeds alone move multi-turn tool calling by about 3 points on small
  models. Estimate pass^k from seeded samples at the card's temperature, with intervals.
- **Throughput is a budget line.** Measure tokens per second on the supplied checkpoint
  and hardware at one stream and at the evaluator's concurrency, and budget the turns
  from the measured latency distribution; the model stays fixed.

## Endpoint settings a scaffold may change, when the task statement allows it

Max tokens per step; the reasoning budget or switch per phase; stop sequences; the request
timeout and retry count; the tool-call grammar. Never the model, the weights, the
quantization or the context length: those are the task's.
