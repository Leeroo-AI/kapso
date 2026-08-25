# One inference path — every LLM call becomes a coding-agent CLI session

**Status:** DESIGN for review. Converts the platform's remaining direct
`LLMBackend` completions to coding-agent CLI sessions, defaulting to
**codex / gpt-5.6-sol / xhigh**. Embeddings are explicitly OUT of scope
(experiment-memory vectors keep the API path). Written against the
shipped code at 91a5cf8f.

**Decision driver (user, 2026-08-25):** convert them all; codex CLI with
gpt-5.6-sol at xhigh reasoning is the default for the converted calls.

---

## 1. What is actually being converted

Direct-completion call sites today (embeddings excluded by decision):

| # | Call site | Shape | Notes |
|---|---|---|---|
| 1 | feedback judge scoring (`feedback_generator.py`) | prompt → tagged text (`<score>`, `<stop>`, `<feedback>`, `cards_load_bearing`) | highest-stakes; tool-locked and card-blind by design (§5.3) |
| 2 | researcher (`researcher/researcher.py`) | prompt + **web search** → structured findings | needs a live web tool, not just text |
| 3 | KG rerank (`kg_graph_search.py:1101`) | system+user → ranked ids | per-query, latency-sensitive |
| 4 | KG navigation (`kg_llm_navigation_search.py:394`) | prompt → next-hop choice | called in a traversal loop |
| 5 | repo-memory builders (`builders.py:369,420`) | prompt → section prose | already file-adjacent work |
| 6 | commit-message generator | diff → one line | trivial, high frequency |
| 7 | parallel completions (`llm_multiple_completions*`) | fan-out of the above | ensemble/fan-out helper |

Everything else already runs as a CLI session (evolve's ideation /
implementation / lens planner, all learning crews, knowledge ingestion,
agentic KG search).

## 2. The design: one adapter-backed `Inference` seam

Rather than rewriting seven call sites against `CodingAgentFactory`
directly (seven new session-lifecycle bugs), introduce ONE seam with the
same method names the callers already use:

```python
class CliInference:
    """LLMBackend's completion surface, served by coding-agent CLI
    sessions. Same methods, same returns — callers do not change."""

    def __init__(self, config: Dict[str, Any], role_defaults: Dict[str, Any]): ...

    def llm_completion(self, model=None, messages=..., **kw) -> str: ...
    def llm_completion_with_system_prompt(self, system, user, **kw) -> str: ...
    def llm_multiple_completions(self, models, messages, **kw) -> List[str]: ...
    def llm_completion_with_web_search(self, model, messages,
                                       search_context_size="medium") -> str: ...
```

Mechanics of one call:
1. **Materialize a scratch session dir** (`tempfile.mkdtemp` under the
   configured run root) — CLI agents need a cwd; nothing else lives there.
2. **Render the prompt**: messages flatten to one prompt string, system
   content first (the CLI has no separate system channel).
3. **Build `CodingAgentConfig`** from the resolved role spec: `agent_type`,
   `model`, `debug_model`, and `agent_specific` carrying `effort`,
   `sandbox: "read-only"` for codex (these calls MUST NOT write) or
   `auth_mode` for claude_code, plus `timeout`.
4. **Run** `agent.generate_code(prompt, timeout_seconds=…)`, then
   `agent.cleanup()`; delete the scratch dir.
5. **Return `result.output`** — the same `str` the direct call returned.
   `success=False` or empty output RAISES (Rule 2): a silent empty
   completion is exactly how the research gate poisoned the E2E.

Two properties this buys: every converted call gets the CLI's own
retry/streaming/deadline handling, and the platform gains **one** place
where inference concurrency, timeouts, and auth are reasoned about.

## 3. Config (Rule 1) — roles keep their names, gain CLI specs

The existing `models:` role map (utility / reasoning / web_search) is
replaced for these call sites by an `inference:` block whose defaults are
the user's decision:

```yaml
inference:
  # Every non-embedding completion runs as a coding-agent CLI session.
  default: &cli_default
    cli: codex
    model: "gpt-5.6-sol"
    effort: "xhigh"
    sandbox: "read-only"          # these calls never write
    timeout_seconds: 900
  roles:
    judge:        {<<: *cli_default}
    research:     {<<: *cli_default, web_search: true, timeout_seconds: 1800}
    kg_rerank:    {<<: *cli_default, effort: "low", timeout_seconds: 300}
    kg_navigate:  {<<: *cli_default, effort: "low", timeout_seconds: 300}
    repo_memory:  {<<: *cli_default}
    commit_message: {<<: *cli_default, effort: "low", timeout_seconds: 180}
  # Embeddings are NOT converted (decision 2026-08-25): the experiment
  # store's vectors stay on the API path under models.embedding.
```

Per-role overrides exist because latency profiles genuinely differ — a
commit message at xhigh is waste, and the KG rerank sits inside a query
loop. The defaults are codex/gpt-5.6-sol/xhigh exactly as specified;
roles only lower effort where the call is mechanical.

## 4. The two call sites that need more than a rename

**Research (web search).** A CLI session cannot take our
`search_context_size`; codex has its own `--search` flag. The role spec
sets `web_search: true`, the adapter passes `--search`, and the prompt
carries the depth instruction in prose ("survey broadly / verify each
claim"). This also removes the entire Responses-API wrapper and the
model-route threading the E2E just fixed — one fewer bespoke path.

**KG rerank / navigation.** These run inside a retrieval loop; a CLI
session per hop is 10-100x the latency of a completion. Two honest
options:
- (a) convert as specified and accept the latency (correct per the
  decision, and these paths are already the slowest part of KG search);
- (b) batch the loop into ONE session that reasons over all hops.
I recommend (b) as the *implementation* of (a): same CLI-only contract,
one session per query instead of per hop. It changes the prompt shape,
not the semantics.

## 5. What gets deleted (Rule 7 — no dual paths)

- `LLMBackend.llm_completion`, `_with_system_prompt`,
  `llm_multiple_completions`, `llm_completion_with_web_search`,
  `llm_multiple_completions_with_web_search` and their retry plumbing.
- `DEFAULT_MODEL_ROUTES` entries for utility / reasoning / web_search
  (embedding stays).
- The Responses-API web-search wrapper and `RESEARCH_WEB_SEARCH_MODEL`
  threading through the gate subprocess.
- `Researcher`'s model parameter and its litellm dependency.

`LLMBackend` survives as the **embedding** backend only, and the config's
`models:` block shrinks to `embedding:` plus its retry policy.

## 6. Consequences to accept, stated plainly

1. **Cost and quota move from the API key to CLI accounts.** Every judge
   scoring, every KG rerank, every commit message now draws on the codex
   subscription. Campaign quota planning changes shape: today a probe of
   Claude/codex limits tells half the story; after this, it tells nearly
   all of it. This is the single biggest operational consequence.
2. **Latency rises per call** (process spawn + session setup ≈ seconds vs
   a sub-second completion). It matters most for the judge (once per
   iteration — fine), commit messages (frequent, hence low effort), and
   KG rerank (hence §4's batching).
3. **Parallel fan-out becomes process fan-out.** `llm_multiple_completions`
   currently uses asyncio; the CLI version needs a bounded
   `ThreadPoolExecutor` (the ensemble path already does this) with an
   explicit concurrency cap in config.
4. **Structured output gets less reliable, and that is the real risk.**
   A completion returns exactly the model's text; a CLI session returns a
   transcript the adapter extracts from. The judge's `<score>` tags and
   the researcher's parsers must tolerate surrounding narration — the
   parsers already do (they regex tags), but this deserves a test each.
5. **Nothing runs without a CLI.** Today a bare `pip install` + API key
   can score, research, and rerank; after this, codex (or claude) must be
   installed and authenticated. Document it as a hard requirement.

## 7. Implementation plan

1. `src/kapso/core/cli_inference.py`: `CliInference` (§2), role
   resolution, scratch-dir lifecycle, fail-loud on empty output.
2. Config: `inference:` block (§3); delete the retired `models:` roles.
3. Swap construction at the seams — `OrchestratorAgent`, `Researcher`,
   both KG search backends, repo-memory builders, commit-message
   generator — passing `CliInference` where `LLMBackend` went. Call sites
   keep their method calls.
4. Batch the KG loops into single sessions (§4b).
5. Delete the retired `LLMBackend` methods + web-search wrapper (§5).
6. Tests: one per converted call site asserting a CLI session was
   constructed with the resolved role spec (codex/gpt-5.6-sol/xhigh
   unless overridden) and that empty/failed output raises; parser
   robustness tests for judge tags and research findings under
   narration; a config test that `inference.roles` defaults resolve.
7. Live smoke: one short evolve (judge + commit messages + repo memory on
   CLI) and one `learn_knowledge` (research on CLI), then compare the
   E2E's stage timings against the API-path baseline recorded in
   learning/e2e-facade/.

## 8. Open questions — RESOLVED (user, 2026-08-25)

1. **Judge account contention** — RESOLVED: keep codex as the judge's
   default, sharing the account with the codex implementor. Accepted
   consequence: judge scoring and implementation draw on one quota
   window, so a starved account degrades scoring and building together.
   Operational note for the runbook: a codex-quota probe now covers the
   judge too, and if contention ever bites, the fix is a second codex
   account in the judge's role spec — a config change, not a redesign.
2. **KG rerank / navigation latency** — RESOLVED: convert them like
   everything else; the CLI is the inference path, no API exception.
   §4b's batching (one session per query rather than per hop) is the
   implementation, so the contract stays CLI-only while the loop cost
   stays bounded.
3. **CLI-unavailable fallback** — RESOLVED: fail loud (Rule 2). No
   emergency API path, no config flag to re-enable one: a missing or
   unauthenticated CLI raises at construction with the fix named. The
   API-path code is deleted rather than left dormant (Rule 7). Revisit
   only if a real deployment need appears.
