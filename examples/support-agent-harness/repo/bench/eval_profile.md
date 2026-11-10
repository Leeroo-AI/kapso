# Evaluation profile

Measured 2026-11-03 and 2026-11-04 on the recorded runs. Inherit and extend this; do not re-measure what is here.

## Mechanics (read from the code)
- `evaluate.py` runs `run_dev.py` unchanged. A full run prints `pass_1` last; `--fast` prints `fast_pass_1` and records no score.
- An episode passes when the runner's reward is at least 1.0. The bar is above Fable 5.1's 10 of 20, so at least 11 of 20; 10 of 20 is exactly 0.50 and does not clear it.
- One run on 20 conversations has a task-level SE of about 0.11, so about 2 conversations is noise. Read a paired flip table against the references, not the headline alone.
- The runner sends `model`, `seed`, `temperature` (0.0), `tool_choice` (`auto`), `messages`, `tools`. It sends no `max_tokens`.
- It sends 17 tools. The source has 44 discoverable tools.
- A reply with both text and tool calls, or an empty reply, sets AGENT_ERROR and ends the episode. Unparsable outer tool JSON raises in the runner's `generate`.
- Discoverable tool calls log to the DB only when state-mutating or allowlisted, so extra reads do not change the score. Numbers are normalised to float.

## Endpoint limits (probed 2026-11-03)
- A `system` message anywhere but first returns 400 `System message must be at the beginning.` This holds for a final system message after a user or a tool message.
- `reasoning_content` on an assistant message is consumed: prompt tokens rose 328 to 389 on a tool round.
- `chat_template_kwargs.reasoning_effort` accepts only `xhigh` (the template's default), `medium` and `low`; `high` returns 400.
- Tool-call ids look like `chatcmpl-tool-<hex>` and are unique per call.
- Qwen emitted text with tool calls and two calls at once in thinking mode at medium effort, seen in a live check.

## Input distribution (reference episodes on the 20, metadata only)
- Scoring basis: ACTION 2, DB 17, DB+NL_ASSERTION 1.
- Opus 5.5: steps 8 to 47 (median 22), tool calls 3 to 53 (median 20), largest prompt 11K to 68K tokens (median 41K), 24 to 278 s.
- All 20 reference episodes ended `user_stop`.

## Reference outcomes on these 20 (one run each, 2026-11-02)
- Fable 5.1, the current agent: 10 of 20 (`bench/reference/fable-5.1/`). Opus 5.5: 8 of 20 (`bench/reference/opus-5.5/`).
- Per-conversation outcomes are in each reference's `episodes.jsonl`; the conversations the two references disagree on are where a run's score moves.

## Coverage axes
- Scoring basis: DB, ACTION, DB+NL_ASSERTION.
- Operation family: account open/close/transfer, debit disputes and card logistics, credit disputes, cash-back handoffs, credit limits, referrals, verification-only and transfer.
- Length: steps and largest prompt.
- Reference outcome: passed or failed under Fable 5.1 and under Opus 5.5.

## Replication noise
- Measured on full runs of 40 in the campaign of 2026-11-03 and 2026-11-04: builds of identical behaviour scored 20 and 18, then 23 and 22, flipping 7 to 10 of 40 conversations. On 20 conversations expect 3 to 5 flips between identical runs, a spread of about 2 in the score.
- A single run cannot resolve a change of one or two conversations. Compare paired flips against the references and against the previous run of the same build.
