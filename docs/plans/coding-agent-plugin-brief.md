# Kapso × Coding Agents — integration research brief

*2026-08-28. Research base: three parallel investigations — host extension
surfaces (Claude Code / Codex / Cursor, with adoption case studies),
daily-workflow pain evidence + the memory/learning product landscape, and
an inventory of Kapso's own pluggable surfaces. Sources cited inline in
the underlying reports; headline citations kept here.*

## Thesis

Engineers' coding-agent workflows have a crowded **input side** (rules
files, memories, docs injection, ticket context) and an **empty output
side**: the moment a session ends or a PR merges — when the trajectory,
the diff, and the outcome all exist — no incumbent product touches. That
post-run moment is where Kapso's loop (harvest → mine → evidence-priced
lessons → serve next session) plugs in, and the mid-2026 host platforms
have converged on exactly the primitives needed to build it as one
plugin shipped to all three hosts: MCP for tools, SKILL.md for
procedures, hooks for capture, `plugin.json` for packaging.

The differentiator is not "memory" — hosts nativized or killed generic
memory (Claude auto-memory; Cursor Memories removed; Windsurf Memories
dead). It is **evidence**: every surviving knowledge mechanism is
unverified text, and the only controlled study of the dominant one
(AGENTS.md/CLAUDE.md context files) found it *does not improve success
rates while adding ~20% inference cost* (ETH Zurich/LogicStar,
arxiv.org/abs/2602.11988). Positioning line: **"Your host remembers
your preferences. Kapso proves what works."**

## 1. Where engineers need Kapso — the moments map

Ranked by (evidence of pain) × (Kapso's right to win):

| # | Moment | Pain evidence | Who's there today | Kapso's angle |
|---|--------|---------------|-------------------|---------------|
| M1 | **Post-run: session ends / PR merges** | Session amnesia is the #2 documented pain (claude-mem at 92.5k stars exists solely for it; GitHub/Anthropic both shipped native memory); but ALL capture today is untested one-line notes | **Nobody** mines finished trajectories into tested lessons — Devin auto-*suggests*, Copilot Memory auto-*expires* (28d TTL = decay management, not verification) | `learn()` — mine the finished session/campaign into evidence-priced cards; the whole crew machinery exists |
| M2 | **The optimization-shaped task** ("make it faster / more accurate / pass the suite / tune this prompt") | DORA 2025: AI throughput up AND instability up; METR: 39-point gap between perceived and real speedup — outcomes are unmeasured | **Nobody** — every product injects context and generates once; no campaigns, no arms, no scoreboard | `evolve()` as a delegated tool: hand off a goal + repo, get back a *scored* solution. Kapso's unique muscle |
| M3 | **Session start** | Re-explaining context every session; rules sprawl (60k+ repos of hand-tended AGENTS.md that measurably don't help) | Crowded: rules, CLAUDE.md, auto-memory, Devin Knowledge | A ~30-line evidence-priced brief (bank `compile_intro`), not another wall of unverified text |
| M4 | **Mid-task unknowns** (unfamiliar library, "how do we do X here") | Atlassian 2025: information-finding is the #1 developer friction; 10 hrs/wk saved by AI, 10 lost to friction | Context7 owns *public* docs (61.3k stars, free); org knowledge is unserved | KG search / wiki pages / experiment history / repo memory / bank cards as MCP pull tools — the org's own knowledge, already built |
| M5 | **The PR gate** | The only place learned team preferences demonstrably monetize (CodeRabbit ~$40M ARR, Greptile $30/seat) | CodeRabbit/Greptile learn style/norms, not engineering lessons | Later phase: bank-informed review; lessons with measured deltas beat style preferences |
| M6 | **The rules-file audit** | ETH study + practitioner consensus ("rules made from scratch are usually not followed"); Anthropic's own <200-line warnings | Nobody measures whether a CLAUDE.md helps | Wedge campaign: "measure your agent files" — Kapso's A/B arms + gauntlet machinery pointed at a team's existing rules |

## 2. The landscape in one paragraph

Personal-and-automatic memory (Cursor Memories, Windsurf, Copilot
Memory, claude-mem) never becomes team knowledge by design; shared-and-
manual knowledge (rules, AGENTS.md, Devin Knowledge, Skills) is human-
curated trust-me text; review bots (CodeRabbit, Greptile) learn
preferences only at the PR gate; memory infra (mem0, Zep, Letta) serves
agent *builders*, not the IDE loop; Factory.ai sells the org-learning
story at a $1.5B valuation **without a published mechanism** — proof the
story sells, and that the mechanism is the open slot. Four white spaces,
all Kapso-shaped: (1) lessons carrying measured evidence, (2) automatic
learning from *finished* sessions, (3) a promotion path where a personal
lesson graduates to team-shared because its evidence cleared a bar,
(4) experiments instead of one-shot generation.

## 3. What the hosts let us build (mid-2026)

The stack converged: **MCP** (universal tool layer, Linux Foundation),
**SKILL.md** (agentskills.io — loaded natively by all three hosts),
**AGENTS.md** (shared repo instructions), **Agent Plugins 1.0**
(vendor-neutral packaging; Cursor/OpenAI/Microsoft on the TSC; Claude
Code keeps its own near-identical `.claude-plugin` format). One source
tree + two thin manifests covers everything.

Per host, what matters most for us:

- **Claude Code** — richest surface: plugins bundle skills + agents +
  **hooks (~30 events incl. SessionStart/Stop/SessionEnd)** + MCP + LSP
  + **monitors** (background watchers streaming into the session) +
  `bin/`. Distribution: official marketplace (curated, no form),
  community marketplace (automated review), any-git-repo marketplaces,
  and `extraKnownMarketplaces` in a repo's `.claude/settings.json` —
  the repo itself auto-offers the team plugin. The **plugin-hint
  protocol** lets the `kapso` CLI print a hint that makes Claude Code
  offer our plugin to the user — a nearly unexploited growth loop
  (official-marketplace plugins only).
- **Codex** — MCP via `config.toml` (shared across CLI/IDE/ChatGPT
  desktop), skills in the vendor-neutral `.agents/skills`, plugins in a
  directory shared with ChatGPT's user base (curated tab).
- **Cursor** — marketplace (manual review, **open-source required** —
  fits us), MCP deeplinks ("Add to Cursor" one-click), hooks (~20
  events), Team Rules pushed from admin. Cursor *removed* Memories and
  is sunsetting @Docs — two fresh gaps.

Adoption lessons from the winners (Superpowers ~279k stars, Context7
~918k weekly downloads, Playwright MCP 5.9M weekly, Sentry/Linear/
GitHub hosted-OAuth MCPs, claude-mem 92.5k stars):

1. **Own a moment, not a category** — inject at the moment of need, not
   a session-start context wall.
2. **Skills auto-trigger; that's what made Superpowers self-invoking**
   rather than a manual chore.
3. **Install must be one command** (`npx ctx7 setup` energy), and the
   repo is the viral unit — skills/rules/bank committed in-repo install
   themselves for every teammate who clones.
4. **Context budget is product surface** — hosts now show per-plugin
   token cost at install and flag unused plugins; a tiny always-on
   brief + everything else behind on-demand tools.
5. **Hooks-based learning plugins reach mass adoption** (claude-mem) —
   the capture pattern is proven; nobody tests what they capture.

## 4. What Kapso already has (inventory → effort)

| Surface | State | Effort to expose |
|---|---|---|
| `gated-knowledge` MCP server — kg search/wiki pages, idea/code search, research, experiment history, repo memory | **Works today** (stdio, env-configured; config builder already emits Claude-shaped `mcpServers` JSON) | Near-zero: add a `kapso-mcp` console script + `.mcp.json` snippets |
| Bank tools (`bank_index` / `bank_get_card` / `bank_get_card_with_evidence`) | Work inside campaigns | Small adapter: stage a bank checkout for an *external* session (generalize `serving_launch`), pick task family + pull-log home |
| Session-start brief | `compile_intro` exists | Same adapter; inject via SessionStart hook / CLAUDE.md include |
| `evolve` / `learn` / `research` / `learn_knowledge` as host-callable ops | Library works; `moltbook_bot` FastAPI is a working async precedent | Wrap as start/poll MCP tools over `on_status` + `Kapso.status()` |
| Progress into the session | `kapso watch --json`, `Kapso.status()` (staticmethod) | One read-only tool; Claude Code monitors can stream it |
| Harvest a **host** session into `learn()` | Design only — store contract needs `experiment_history.json` + final report; facade already synthesizes the rest | **The biggest build**: transcript→experiment-history translator + a scoring story (tests/CI/review outcomes as settlement signals) |
| `codify` (card → runnable procedure) | Works, YAML-driven | Adapter → "apply this lesson here" verb; sleeper feature |

## 5. The ideal integration — an engineer's day with Kapso

Morning: open Claude Code (or Cursor/Codex) in the repo. A SessionStart
hook injects a ~30-line **bank brief** — the evidence-priced index for
this repo's task family, not a 400-line CLAUDE.md. Work proceeds
normally. When the agent hits a repeated failure or an unfamiliar
subsystem, an auto-triggering **skill** tells it to consult the lesson
bank / KG / experiment history via **MCP pull tools** — the Context7
moment, but for the org's own knowledge. When the task is
optimization-shaped ("cut p95 30%", "get accuracy over 0.8", "make the
suite pass"), the engineer — or the agent itself — delegates:
`/kapso:evolve` starts a measured campaign (start/poll tools, a monitor
streaming iteration scores into the session) and comes back with a
scored, explained solution instead of a vibe-checked one-shot. When the
session ends, a Stop/SessionEnd **hook enqueues the trajectory** (diff,
transcript, test outcomes); `learn` mines it — on-demand or nightly —
and the cards land as **reviewable markdown in the repo's bank**,
carrying provenance and measured deltas. A card whose evidence clears
the bar is promoted team-wide in the next bank PR; every teammate's
clone serves it automatically. The team watches one number: the bank's
learning curve.

Capture with hooks → decide with skills → retrieve with MCP →
experiment with evolve → consolidate with learn → distribute through
the repo. Each piece maps onto a host primitive that already exists.

## 6. Phased build

- **P0 (days): "Kapso knowledge in your agent."** `kapso-mcp` console
  script; `claude mcp add` / `codex mcp add` / Cursor-deeplink
  one-liners in the README; mounts kg + research + experiment-history +
  repo-memory gates in any host. Submit to the Claude community
  marketplace + cursor.directory. Zero new machinery.
- **P1 (1–2 wks): the plugin.** One skeleton, two manifests
  (`.claude-plugin/` + Agent Plugins 1.0). Skills (consult-the-bank,
  delegate-to-evolve), SessionStart brief hook, bank staging adapter so
  external sessions get served cards. `kapso setup` one-liner. Cursor
  marketplace submission (we're open source — eligible).
- **P2 (2–4 wks): the delegation loop.** `evolve`/`research`/`learn`
  as start/poll MCP tools (moltbook pattern), status tool + Claude Code
  monitor streaming progress. This is the demo that no competitor can
  copy: a measured campaign from inside your daily agent.
- **P3 (the moat): close the loop on host sessions.** Transcript →
  trajectory translator; settlement from tests/CI/review outcomes; a
  **light mining tier** for daily sessions (full crew is ~4.5h/campaign
  today — fine nightly for campaigns, too heavy per session); the
  personal→team promotion path; the "measure your CLAUDE.md" audit
  campaign as the marketing wedge. Pursue the official Anthropic
  marketplace listing to unlock the plugin-hint growth loop.

## 7. Honesty box (claim safety)

- Our own A/B evidence for served lessons is **within noise** so far —
  the pitch is auditability and measurement machinery ("measurably
  knows more"), not proven uplift ("measurably better") until design-
  partner learning curves exist. The plugin is itself the instrument
  that generates that evidence.
- `learn()` at production quality is hours, not seconds — daily-session
  mining needs the light tier before M1 is honest at session
  granularity.
- Host sessions have no scores today; without the settlement story
  (tests/CI), cards mined from them would be weak — sequence P3 after
  P2 so evolve-produced trajectories (which ARE scored) carry the early
  bank.
- Native encroachment is real (Anthropic auto-memory + Auto Dream);
  the defensible ground is evidence + team promotion + experiments,
  not capture.
