# P2 — Mining crew over the corpus

**Goal:** every imported trajectory carries its `mined/` view; the corpus
becomes "simultaneously curriculum, benchmark, and first real input"
(MD§8.2). **Design sources:** MD§3.4.1 (the mined view), MP in full (the
mined-format contract, lead launch prompt, flow-writer + critic agent defs,
frame policies), MD§4.2. **Depends on:** P1. **Doubts:** D4 (models before
the first run), D7.

## Deliverables

`crews/` home for instruction material (`src/kapso/learning/crews/mining/`:
the mined-format contract doc, lead prompt template, `.claude/agents/`
flow-writer + critic definitions — all lifted from MP, which is the source of
truth; the plan adds no prompt content); `learner.py`'s first slice: the
mining frame (stage → launch lead session → check → commit); `kapso learn
mine <trajectory-id|--all>`.

## Work items

1. **Frame staging** (MP): stage a workspace per trajectory — read-only
   bundle mount, `.claude/agents/` definitions, the mined-format contract;
   launch the Claude-led lead session (self-organization needs the CLI's
   native subagents, MD§4.2) via the existing coding-agent adapter (prompt
   via stdin, `--append-system-prompt` — the adapter facts MP records).
2. **Frame checks** (MP; MD§4.2): mined-format schema; **coverage arithmetic
   on stable identities** (every ledger node accounted for); quote re-grep;
   raw immutability via manifest hashes; one repair loop then fail loud.
   Mined view written into the store as a derived layer (regenerable, marked
   derived in the manifest — MD§3.4.1).
3. **Batch driver**: `--all` mines every un-mined imported trajectory
   (per-trajectory, idempotent, bank-blind — parallelizable per MD§4.1).
4. **Mining report** per run (MP's mining-report step) — the acceptance
   surface for the human checkpoint.

## Tests

- Format validator accepts a golden mined-view fixture; rejects a fixture
  with a fabricated quote (re-grep miss) and one with an unaccounted ledger
  node (coverage arithmetic).
- Immutability check trips when a staged bundle file is doctored (hash
  mismatch).
- Derived-layer rule: re-mining replaces `mined/` without touching raw
  prefixes.

## Done gate

Mined views exist for the whole imported corpus; coverage arithmetic green
everywhere; **human review of the first mined views** (MD§8.2) signed off.
