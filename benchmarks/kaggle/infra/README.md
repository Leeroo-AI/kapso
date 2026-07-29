# Kaggle benchmark — box infra

The competition path is `URL → preflight → runner → public score`. These
scripts get a GPU box to where that path can run, without the 20–40 min
dependency install landing on the competition clock.

## The two "builds", kept separate
- **Dependencies** (torch, kapso, kaggle CLI, node+codex) — task-agnostic, slow,
  stable. **Bake** them into a golden image once (`setup_box.sh` + snapshot).
- **Task substrate** (e.g. Task 1's frozen-encoder embedding cache) —
  per-competition; **not** pre-baked, built in round 1 on the box. See
  `../K_RAMP_SPEC.md` for paying it exactly once under k=8.

## Scripts
| script | where | when | does |
|---|---|---|---|
| `setup_box.sh` | fresh DLVM | once (image) | installs everything; snapshot the disk after → golden image |
| `provision.sh` | dev box | per run | `gcloud` creates the 8×L4 box (golden image, or base DLVM) |
| `bootstrap.sh` | on the box | per boot | pulls code to the target commit, injects **only** the safe secrets, smoke-tests |
| `run_competition.sh` | on the box | per run | `preflight(URL)` → `runner(root)` |

## Golden-image flow (one-time)
1. `provision.sh` a base-DLVM box (leave `IMAGE_FAMILY` unset).
2. `GITHUB_PAT=… bash setup_box.sh` on it.
3. Snapshot the disk into an image family; set `IMAGE_FAMILY` in `provision.sh`.

## Per-competition flow
1. `provision.sh` (from the golden image).
2. Inject secrets + `bash bootstrap.sh`. **Never** ship the Bedrock trio
   (`AWS_BEARER_TOKEN_BEDROCK` / `CLAUDE_CODE_USE_BEDROCK` / `ANTHROPIC_MODEL`) —
   they hijack Claude Code. Ship only `OPENAI_API_KEY` + `CLAUDE_CODE_OAUTH_TOKEN`
   + `HF_TOKEN`, plus the codex/kaggle auth files.
3. `bash run_competition.sh <competition-URL>`.
4. Salvage artifacts; `gcloud compute instances delete …`.

## Credential + safety notes
- `~/.codex/auth.json` (ChatGPT token) **rotates** — always inject a fresh copy.
- `CLAUDE_CODE_OAUTH_TOKEN` is a shared Max token with 5h + 7d rolling caps;
  probe utilization before a long run.
- Secrets are **injected per boot, never baked** into the image (rotation +
  leakage).
