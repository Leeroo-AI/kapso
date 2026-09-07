# Skills for coding agents

One folder per coding agent, holding everything that agent needs. The
`kapso` skill teaches an agent to operate Kapso from a session: verify the
install, launch and follow a campaign, answer the inbox, resume, learn,
ingest knowledge, deploy.

| Folder | Agent | Discovered from |
|---|---|---|
| `claude-code/kapso/` | Claude Code | `.claude/skills/kapso` (project) or `~/.claude/skills/kapso` (personal) |
| `codex/kapso/` | Codex CLI | `.agents/skills/kapso` (repo, walking up to the git root) or `~/.agents/skills/kapso` (user); `agents/openai.yaml` decorates the `/skills` list |

The copies are deliberately separate files, not symlinks: they are identical
today, and each can diverge as its agent's conventions do. Their test record
is `docs/plans/coding-agent-skill-findings.md`.

## Install

From a checkout, into the project you want the agent to use it in:

```bash
# Claude Code
mkdir -p .claude/skills && ln -s /path/to/kapso/skills/claude-code/kapso .claude/skills/kapso

# Codex
mkdir -p .agents/skills && ln -s /path/to/kapso/skills/codex/kapso .agents/skills/kapso
```

OpenCode reads both `.claude/skills` and `.agents/skills`, so either symlink
serves it.
