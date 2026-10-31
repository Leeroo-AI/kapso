---
name: kapso-setup
description: Set Kapso up on this machine from nothing to a green doctor, in one pass — detect what is there, install the leeroo-kapso package and the coding-agent CLIs it drives, run the logins, write the keys to .env, verify with kapso doctor, and orient the user on the harness and experience to start from. Use when the user says set up kapso, install kapso, get kapso working, runs /kapso:setup, or when any kapso command fails on a missing package, CLI, login or key. Do not use for a package problem unrelated to Kapso, or inside a Kapso campaign session.
---

# Set Kapso up

From a bare machine to `kapso doctor build` green, in under five minutes, with at
most three questions and two links to click. Detect first, in silence; ask only what
the environment cannot answer; never print a key or a token.

## 1. Detect before asking

Run these and read them, without narrating each one:

```bash
node --version; python3 --version; git --version
command -v uv uvx pipx kapso kapso-mcp claude codex
claude auth status 2>&1 | head -3
codex login status 2>&1 | head -3
ls -a .env kapso-config.yaml harness.yaml 2>/dev/null
```

What you learn: the runtimes, which install method the machine has (`uv` first,
`pipx` second, `pip` into the active environment last, never the global Python),
whether `kapso` and `kapso-mcp` are on PATH and from one install (`kapso --version`
and `kapso-mcp --version` agree), which CLIs exist and are logged in, which keys
`.env` names (read the names only, with `grep -o '^[A-Z_]*='`; a
`CLAUDE_CODE_OAUTH_TOKEN` there is Claude Code's login as far as Kapso's sessions
are concerned, whatever `claude auth status` says), whether the project has a
`kapso-config.yaml` or a harness directory.

## 2. Install what is missing

The package, pinned to this plugin's version:

```bash
uv tool install "leeroo-kapso==0.8.0"      # else: pipx install "leeroo-kapso==0.8.0"; else, in the active venv: pip install "leeroo-kapso==0.8.0"
```

- The index has no such version (a development build, until the release that
  carries it): say so, and install the source the plugin came from, at the same
  version: on Claude Code `uv tool install "${CLAUDE_PLUGIN_ROOT}/.."` (the
  marketplace clone is the repository); on Codex `uv tool install "leeroo-kapso @
  git+https://github.com/Leeroo-AI/kapso@redesign"`. Then `kapso --version` must
  print the plugin's version.
- Neither `uv` nor `pipx` present: install `uv` the way its site says
  (`curl -LsSf https://astral.sh/uv/install.sh | sh`), then the line above.
- The PyPI package named `kapso` is an unrelated tool that shadows the command: if
  `pip show kapso` finds it, `pip uninstall -y kapso` first.
- Say what was installed and where, and the PATH line the shell needs when
  `~/.local/bin` is not on it (`export PATH="$HOME/.local/bin:$PATH"`).
- The coding-agent CLIs, when absent: `npm install -g @anthropic-ai/claude-code
  @openai/codex` (Node 18+; a global install that needs root goes to
  `--prefix "$HOME/.local"` instead). Kapso sends every model call through them;
  there is no API-key fallback for generation.

## 3. Logins

Run the doctor line of step 6 first: a CLI whose row is green needs no login (a token
in `.env` counts). For each CLI the doctor reports as not logged in, give the user the
command to run in their own terminal, `claude auth login` or `codex login`
(`codex login --device-auth` on a machine without a browser), and wait for their word:
a login started from a tool call ends with the call, and its link stops working.
Re-check both statuses before going on.

## 4. The session's tools

On Codex, add Kapso's MCP server to the user's config in this session, whether or
not the doctor's rows are green — a green install says nothing about the server,
and a session without it has no build, status or bank tools and does not say why.
Run both lines, then `codex mcp list` must show a `kapso` row; a list without one
is a setup that is not done, said to the user as such. Give the server two minutes to
start: Codex drops a server that has not answered within ten seconds, and
`kapso-mcp` imports the package first, which a busy machine can stretch past that.

```bash
codex mcp add kapso -- kapso-mcp
python3 -c 'import os, pathlib; p = pathlib.Path(os.environ.get("CODEX_HOME", pathlib.Path.home() / ".codex")) / "config.toml"; t = p.read_text(); h = "[mcp_servers.kapso]\n"; p.write_text(t.replace(h, h + "startup_timeout_sec = 120\n", 1)) if "startup_timeout_sec" not in t.split(h, 1)[1].split("\n[", 1)[0] else None'
```

On Claude Code the plugin's own entry runs `kapso-mcp` and needs nothing. A host
starts its MCP servers once per session, so the tools installed in this step reach
this session after a reconnect (`/mcp` on Claude Code) or in the next session (Codex).

## 5. Keys

Ask only for what the user's next step needs, write each to `.env` in the project
root with `printf 'NAME=value\n' >> .env`, and never print a value back:

- `LEEROOPEDIA_API_KEY` (recommended): a hosted knowledge base of ML and AI
  frameworks the campaigns consult; a new account at
  https://app.leeroopedia.com/dashboard comes with $20 of free credit.
- `OPENAI_API_KEY` (optional): embeddings only, for an ingested corpus and semantic
  search over past experiments; generation never uses it.

The user has neither: continue, and say which doctor row would report it.

## 6. Verify

```bash
kapso doctor build --json --config kapso-config.yaml
```

(Without the flag when the project has no `kapso-config.yaml` yet: the packaged
config is the default.) Read the JSON: every row with `"required": true` must be `"ok": true`; fix what a
failing row's `fix` says and run it again. Rows with `"required": false` are optional
and name what they would enable; leave them unless asked. Use the next verb's rows
(`build` first), never the union: `kapso doctor` alone asks for the stores an ingest
needs. Offer `kapso doctor build --models --config kapso-config.yaml`, which spends a
few tokens to confirm each configured model answers.

## 7. Orientation

```bash
kapso harness list --config kapso-config.yaml
```

One line per bundled harness; recommend one from what the repository looks like
(its dependencies, data files, notebooks) or none when nothing fits. Offer
`kapso experience add mine --config kapso-config.yaml` (the bank a learn fills), and,
when no Leeroopedia key is present, the Leeroopedia plugin's install line from its
own README (https://github.com/Leeroo-AI/leeroopedia-mcp).

## 8. Finish

Three lines, then stop: what to run next on this repository (`/kapso:build <what you
want>` on Claude Code, `$kapso-build <what you want>` on Codex; the tools this setup
installed reach a session after `/mcp` reconnects on Claude Code, or in a new session
on Codex), where the record will go (`<campaign>/.kapso/`, the campaign a sibling
directory outside the repository), how to watch (`kapso watch <campaign> --follow`).
Do not start a campaign.

## Rules

- Never a question the environment can answer; one summary at the end.
- A failure names the `doctor` row and its fix, and the skill retries after the fix.
- A key is written where it goes and never echoed, pasted into a reply, or committed.
- Nothing beyond these steps: no campaign, no harness written, no config edited
  except what a fix line says.
