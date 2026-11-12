"""Staging a harness into one session: the module's pages as the "Domain
guidance" block, its skills where the host discovers them, its servers
beside the engine's gates, its delegates as the host's sub-agents carrying
the same guidance, its hooks merged into the file the host reads, the
environment, the prepared resources and the task statement, and a record
of what was staged. One function, called at every site that constructs a
coding-agent session. Design: docs/research/harness-design.md sections 3
and 5; the plan: docs/plans/redesign/wp2-harness.md ("Staging").

Idempotent (a second call on the same directory and host replaces what it
wrote) and safe under concurrency (each host's files have fixed names and
are written whole). Refused before anything is written when the session
directory already tracks a path it would write, and when the harness
changed since it was loaded.
"""

import json
import os
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import git

from kapso.core.agent_manifest import load_agent_manifest
from kapso.execution.coding_agents.adapters.codex_agent import agent_role_overrides, mcp_config_overrides, toml_string
from kapso.execution.coding_agents.session_record import write_whole
from kapso.core.front_matter import read_front_matter
from kapso.harness.hash import describe_change, harness_files, harness_hash, short
from kapso.harness.hooks import ENGINE_HOOKS, engine_hooks, expand_plugin_root, merge_hooks
from kapso.harness.load import Harness, HarnessError
from kapso.harness.manifest import MODULES

HARNESS_ROOT_VARIABLE = "KAPSO_HARNESS_ROOT"
TASK_STATEMENT_FILE = "kapso_task.md"
PREPARED_DIR = "prepared"
# Per session, under its record directory: the record the staging writes
# and the adapters complete (one run entry per host invocation), and the
# host's event stream.
SESSION_RECORD_FILE = "session.json"
STREAM_FILE = "stream.jsonl"
# Where a campaign's session records live, under the campaign directory.
SESSIONS_DIR = Path(".kapso") / "sessions"
CLAUDE_HOST = "claude_code"
CODEX_HOST = "codex"
# The staging each adapter's sessions get, from the registry's `host`
# (kapso.core.roles.build_hosts reads the same key): the Claude Code
# layout for the Claude adapters, Codex's for codex.
HOST_OF_AGENT = {
    name: entry["host"] for name, entry in load_agent_manifest()["agents"].items() if entry.get("host")
}
_SKILLS_DIR = {CLAUDE_HOST: Path(".claude") / "skills", CODEX_HOST: Path(".agents") / "skills"}
_HOOKS_FILE = {CLAUDE_HOST: Path(".claude") / "harness-hooks.json", CODEX_HOST: Path(".codex") / "hooks.json"}
_AGENTS_PATH = {CLAUDE_HOST: Path(".claude") / "harness-agents.json", CODEX_HOST: Path(".codex") / "harness-agents"}
# Everything above lands in `.claude`, `.agents` or `.codex`: the names a
# harvest leaves out of a bundle (kapso.learning.trajectory_store.STAGING_DIRECTORIES).
_OPEN_SKILL = {CLAUDE_HOST: "/{name}", CODEX_HOST: "${name}"}
_ASK_AGENT = {
    CLAUDE_HOST: 'the Agent tool with subagent_type "{name}"',
    CODEX_HOST: 'spawn the agent role "{name}" and wait for its reply',
}
GUIDANCE_HEADING = "## Domain guidance"


def skills_dir(session_dir: Path, host: str) -> Path:
    """Where a session on this host finds its staged skills."""
    if host not in _SKILLS_DIR:
        raise HarnessError(f"unknown host {host!r}; the hosts are {', '.join(_SKILLS_DIR)}")
    return Path(session_dir) / _SKILLS_DIR[host]


def open_skill_instruction(host: str, name: str) -> str:
    """How a session on this host opens a staged skill by name: `/name`
    on Claude Code, `$name` on Codex. A prompt that lists skills of its
    own (the deployment session's targets) renders each line with it."""
    if host not in _OPEN_SKILL:
        raise HarnessError(f"unknown host {host!r}; the hosts are {', '.join(_SKILLS_DIR)}")
    return _OPEN_SKILL[host].format(name=name)


def host_for(agent_type: str) -> str:
    if agent_type not in HOST_OF_AGENT:
        raise HarnessError(
            f"a harness stages into {', '.join(sorted(set(HOST_OF_AGENT.values())))} sessions; "
            f"coding agent {agent_type!r} has no staging"
        )
    return HOST_OF_AGENT[agent_type]


@dataclass
class StagedHarness:
    module: str
    host: str
    session_dir: Path
    session_id: str
    guidance: str                                   # "" without a harness
    agent_specific: Dict[str, Any] = field(default_factory=dict)   # keys for CodingAgentConfig.agent_specific
    mcp_servers: Dict[str, Any] = field(default_factory=dict)      # the harness servers, host-shaped, beside the gates
    allowed_tools: List[str] = field(default_factory=list)         # `mcp__<server>` per mounted server
    env: Dict[str, str] = field(default_factory=dict)              # KAPSO_HARNESS_ROOT
    record: Dict[str, Any] = field(default_factory=dict)
    record_path: Optional[Path] = None                             # <campaign>/.kapso/sessions/<id>/session.json

    def merged_agent_specific(self, own: Dict[str, Any]) -> Dict[str, Any]:
        """The session's own keys over the staged ones; `env_overrides` and
        `mcp_servers` are merged, not replaced. A session that streams
        elsewhere (ideation, beside its artifacts) has the record say so."""
        merged = {**self.agent_specific, **own}
        # A key appears only when something is there to merge, so a session
        # without a harness keeps exactly the keys it had.
        if self.env or own.get("env_overrides"):
            merged["env_overrides"] = {**self.env, **(own.get("env_overrides") or {})}
        if self.mcp_servers or own.get("mcp_servers"):
            merged["mcp_servers"] = {**(own.get("mcp_servers") or {}), **self.mcp_servers}
        stream = own.get("stream_artifact_path")
        if self.record_path is not None and stream and stream != self.record.get("stream"):
            self.record["stream"] = str(stream)
            write_whole(self.record_path, json.dumps(self.record, indent=2) + "\n")
        return merged

    def codex_exec_args(self) -> List[str]:
        """The `codex exec` arguments a raw codex runner adds for this
        staging: the servers, the agent roles, the hook trust bypass (the
        adapter builds the same from the keys)."""
        args = list(mcp_config_overrides(self.mcp_servers))
        args += agent_role_overrides(self.agent_specific.get("agents") or {})
        if self.agent_specific.get("hooks_trust_bypass"):
            args.append("--dangerously-bypass-hook-trust")
        return args


def record_without_harness(
    module: str, host: str, session_dir: Path, session_id: str, records_dir: Path,
    engine_skills: Sequence[Path] = (),
) -> StagedHarness:
    """What a session gets without a harness: the engine's own skills when
    it has any (the deployment session's targets), nothing else staged,
    and the record written all the same (`harness: null`), so every
    session of a campaign has one for its runs to land in."""
    if engine_skills and host not in _SKILLS_DIR:
        raise HarnessError(f"unknown host {host!r}; the hosts are {', '.join(_SKILLS_DIR)}")
    for skill in engine_skills:
        shutil.copytree(skill, session_dir / _SKILLS_DIR[host] / Path(skill).name, dirs_exist_ok=True)
    record_dir = records_dir / session_id
    staged = StagedHarness(
        module=module, host=host, session_dir=session_dir, session_id=session_id, guidance="",
        agent_specific={
            "stream_artifact_path": str(record_dir / STREAM_FILE),
            "session_record_path": str(record_dir / SESSION_RECORD_FILE),
        },
    )
    staged.record = {
        "module": module,
        "host": host,
        "harness": None,
        "session_dir": str(session_dir),
        "pages": [],
        "skills": [],
        "engine_skills": [Path(skill).name for skill in engine_skills],
        "mcp_servers": [],
        "agents": [],
        "hooks": [],
        "prepared": [],
        "stream": staged.agent_specific["stream_artifact_path"],
        "runs": existing_runs(record_dir / SESSION_RECORD_FILE),
    }
    staged.record_path = record_dir / SESSION_RECORD_FILE
    write_whole(staged.record_path, json.dumps(staged.record, indent=2) + "\n")
    return staged


# The engine's own inference sessions (the repo-memory bootstrap, the
# commit messages) are recorded under `engine/<role>/<n>/`.
ENGINE_SESSIONS = "engine"


def engine_session_record(records_dir: Path, role: Optional[str], host: str, session_dir: Path) -> Dict[str, str]:
    """The record of one of the engine's own inference sessions,
    `engine/<role>/<n>` numbered in order under the campaign's records:
    the keys the adapter writes its runs and stream into."""
    label = f"{ENGINE_SESSIONS}/{role or 'default'}"
    earlier = records_dir / label
    # Parallel lanes mint their commit messages at the same time: the
    # number and the record are taken under the one lock, so two sessions
    # never share a record.
    with _STAGING_LOCK:
        number = 1 + (sum(1 for child in earlier.iterdir() if child.is_dir()) if earlier.is_dir() else 0)
        return dict(record_without_harness("engine", host, session_dir, f"{label}/{number}", records_dir).agent_specific)


def existing_runs(record_path: Path) -> list:
    """The run entries a record at this path already holds: a session
    staged again under the same label (a phase re-run after a crash or a
    resume) keeps the invocations before it, so the record says how the
    first one ended."""
    if not record_path.is_file():
        return []
    return list(json.loads(record_path.read_text(encoding="utf-8"))["runs"])


def check_harness_unchanged(harness: Harness) -> None:
    """The harness on disk is the one loaded, by its hash; named
    otherwise: a session's file-tool write is denied by the engine's
    protect-harness hook, so what changed came through a shell or by
    hand, and this check is what catches it."""
    current = harness_hash(harness.path)
    if current != harness.hash:
        raise HarnessError(
            f"the harness changed since launch: {harness.label()} recorded, now {short(current)} "
            f"({describe_change(harness.files, harness_files(harness.path))}); {harness.restore_hint()}"
        )


def stage_harness(
    harness: Optional[Harness],
    module: str,
    *,
    session_dir: Path,
    campaign_dir: Path,
    session_id: str,
    host: str,
    records_dir: Optional[Path] = None,
    engine_skills: Sequence[Path] = (),
) -> StagedHarness:
    """Stage one module of the harness into one session directory. The
    session's record and stream go under `<campaign>/.kapso/sessions`, or
    under `records_dir` when the campaign directory takes no writes (an
    archived bundle a learning crew reads). `engine_skills` are the
    engine's own skill directories (the deployment session's targets),
    staged beside the module's on either host, with or without a harness.
    Without a harness nothing else is staged and the record is written
    all the same."""
    session_dir = Path(session_dir)
    campaign_dir = Path(campaign_dir)
    records = Path(records_dir) if records_dir is not None else campaign_dir / SESSIONS_DIR
    if harness is None:
        return record_without_harness(module, host, session_dir, session_id, records, engine_skills)
    if module not in MODULES:
        raise ValueError(f"unknown module {module!r}; the modules are {', '.join(MODULES)}")
    if host not in _SKILLS_DIR:
        raise HarnessError(f"unknown host {host!r}; the hosts are {', '.join(_SKILLS_DIR)}")
    check_harness_unchanged(harness)
    stager = _Stager(harness, module, session_dir, campaign_dir, session_id, host, records, engine_skills)
    # Ensemble members stage into one directory from several threads; the
    # files are written one staging at a time.
    with _STAGING_LOCK:
        stager.refuse_tracked_collisions()
        stager.mount_campaign_files()
        stager.skills()
        stager.servers()
        stager.hooks()
        stager.agents()
        stager.record()
    return stager.staged


_STAGING_LOCK = threading.Lock()


class _Stager:
    def __init__(
        self, harness: Harness, module: str, session_dir: Path, campaign_dir: Path,
        session_id: str, host: str, records_dir: Path, engine_skills: Sequence[Path] = (),
    ):
        self.harness = harness
        self.module = module
        self.attachments = harness.module(module)
        self.engine_skills = [Path(skill) for skill in engine_skills]
        self.session_dir = session_dir
        self.campaign_dir = campaign_dir
        self.session_id = session_id
        self.host = host
        self.records_dir = records_dir
        self.prepared = self.prepared_names()
        self.staged = StagedHarness(
            module=module, host=host, session_dir=session_dir, session_id=session_id,
            guidance=self.guidance(),
            env={HARNESS_ROOT_VARIABLE: str(harness.path)},
            agent_specific={
                "stream_artifact_path": str(self.session_record_dir() / STREAM_FILE),
                "session_record_path": str(self.session_record_dir() / SESSION_RECORD_FILE),
            },
        )

    # ------------------------------------------------------------- paths

    def session_record_dir(self) -> Path:
        return self.records_dir / self.session_id

    def writes(self) -> List[Path]:
        """Every path under the session directory this staging writes."""
        paths = [self.session_dir / _SKILLS_DIR[self.host] / name for name in self.attachments.skills]
        paths.extend(self.session_dir / _SKILLS_DIR[self.host] / skill.name for skill in self.engine_skills)
        paths.append(self.session_dir / _HOOKS_FILE[self.host])   # always: the engine's own hooks
        if self.attachments.agents:
            paths.append(self.session_dir / _AGENTS_PATH[self.host])
        if not self.at_campaign_root():
            paths.append(self.session_dir / PREPARED_DIR)
            paths.append(self.session_dir / TASK_STATEMENT_FILE)
        return paths

    def at_campaign_root(self) -> bool:
        return self.session_dir.resolve() == self.campaign_dir.resolve()

    def prepared_names(self) -> Tuple[str, ...]:
        prepared = self.campaign_dir / PREPARED_DIR
        if not prepared.is_dir():
            return ()
        return tuple(sorted(p.name for p in prepared.iterdir() if p.is_dir()))

    # ----------------------------------------------------------- refusals

    def refuse_tracked_collisions(self) -> None:
        """A seed repository that tracks a path the staging would write
        (its own skill of the same name, its own .codex/hooks.json) is
        refused by name: staging never overwrites the person's files and a
        session's closing commit never carries the harness's."""
        if not (self.session_dir / ".git").exists():
            return
        repo = git.Repo(self.session_dir)
        for path in self.writes():
            relative = path.relative_to(self.session_dir).as_posix()
            if repo.git.ls_files(relative).strip():
                raise HarnessError(
                    f"harness {self.harness.name} cannot be staged into {self.session_dir}: the repository "
                    f"tracks {relative}, which the harness would write; move or rename it in the repository"
                )

    # ------------------------------------------------------------ mounting

    def mount_campaign_files(self) -> None:
        """The task statement (copied) and the prepared resources (linked)
        reach every session directory that is not the campaign root."""
        if self.at_campaign_root():
            return
        statement = self.campaign_dir / TASK_STATEMENT_FILE
        if statement.is_file():
            shutil.copy2(statement, self.session_dir / TASK_STATEMENT_FILE)
        prepared = self.campaign_dir / PREPARED_DIR
        link = self.session_dir / PREPARED_DIR
        if prepared.is_dir() and not link.exists() and not link.is_symlink():
            os.symlink(prepared.resolve(), link)

    def skills(self) -> None:
        root = self.session_dir / _SKILLS_DIR[self.host]
        # Exactly this module's skills: one left here by another module's
        # staging of the same directory goes, since nothing is inherited.
        for name in self.harness.manifest.listed("skills"):
            if name not in self.attachments.skills and (root / name).is_dir():
                shutil.rmtree(root / name)
        for name in self.attachments.skills:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(self.harness.path / "skills" / name, target, dirs_exist_ok=True)
        # The engine's own skills beside the module's; not the manifest's,
        # so another module's staging of the directory leaves them.
        for skill in self.engine_skills:
            shutil.copytree(skill, root / skill.name, dirs_exist_ok=True)

    def servers(self) -> None:
        if not self.attachments.mcp_servers:
            return
        defined = json.loads((self.harness.path / "mcp.json").read_text(encoding="utf-8"))["mcpServers"]
        for name in self.attachments.mcp_servers:
            entry = expand_plugin_root(defined[name], self.harness.path)
            if "command" in entry:
                entry.setdefault("cwd", str(self.session_dir))
                env = dict(entry.get("env") or {})
                env[HARNESS_ROOT_VARIABLE] = str(self.harness.path)
                if self.host == CODEX_HOST:
                    # Forwarded by name; the codex adapter writes env_vars.
                    references = [v[2:-1] for v in env.values() if is_reference(v)]
                    env = {k: v for k, v in env.items() if not is_reference(v)}
                    if references:
                        entry["env_vars"] = references
                entry["env"] = env
            else:
                entry.setdefault("timeout", _http_timeout_ms(self.harness))
            if self.host == CODEX_HOST:
                entry["approval_mode"] = "approve"
            self.staged.mcp_servers[name] = entry
            self.staged.allowed_tools.append(f"mcp__{name}")

    def hooks(self) -> None:
        """The engine's own hooks first (the harness directory is read-only
        in every session that mounts a harness), then the module's in
        manifest order; every harness session has the file, so a Codex
        session with a harness carries the hook trust bypass."""
        file = self.session_dir / _HOOKS_FILE[self.host]
        merged = merge_hooks(self.harness.path, self.attachments.hooks)
        for event, entries in engine_hooks()["hooks"].items():
            merged["hooks"][event] = entries + merged["hooks"].get(event, [])
        write_whole(file, json.dumps(merged, indent=2) + "\n")
        if self.host == CLAUDE_HOST:
            self.staged.agent_specific["settings_path"] = str(file)
        else:
            self.staged.agent_specific["hooks_trust_bypass"] = True

    def agents(self) -> None:
        target = self.session_dir / _AGENTS_PATH[self.host]
        if not self.attachments.agents:
            if target.is_file():
                target.unlink()   # another module's, staged here before
            elif target.is_dir():
                shutil.rmtree(target)
            return
        prompts = {}
        for name in self.attachments.agents:
            data, body = read_front_matter(self.harness.path / "agents" / f"{name}.md")
            prompts[name] = (str(data["description"]), self.staged.guidance + "\n\n" + body.strip() + "\n")
        if self.host == CLAUDE_HOST:
            file = self.session_dir / _AGENTS_PATH[self.host]
            write_whole(file, json.dumps(
                {name: {"description": description, "prompt": prompt} for name, (description, prompt) in prompts.items()},
                indent=2,
            ) + "\n")
            self.staged.agent_specific["agents_path"] = str(file)
        else:
            directory = self.session_dir / _AGENTS_PATH[self.host]
            roles = {}
            for name, (description, prompt) in prompts.items():
                file = directory / f"{name}.toml"
                write_whole(file, f"developer_instructions = {toml_string(prompt)}\n")
                roles[name] = {"description": description, "config_file": str(file)}
            self.staged.agent_specific["agents"] = roles

    def record(self) -> None:
        self.staged.record = {
            "module": self.module,
            "host": self.host,
            "harness": self.harness.ref_dict(),
            "session_dir": str(self.session_dir),
            "pages": self.page_names(),
            "skills": list(self.attachments.skills),
            "engine_skills": [skill.name for skill in self.engine_skills],
            "mcp_servers": list(self.attachments.mcp_servers),
            "agents": list(self.attachments.agents),
            "hooks": [*ENGINE_HOOKS, *self.attachments.hooks],
            "prepared": list(self.prepared),
            "stream": self.staged.agent_specific["stream_artifact_path"],
            # The host invocations the adapter appends, one entry each; a
            # session staged again keeps the entries already there.
            "runs": existing_runs(self.session_record_dir() / SESSION_RECORD_FILE),
        }
        self.staged.record_path = self.session_record_dir() / SESSION_RECORD_FILE
        write_whole(self.staged.record_path, json.dumps(self.staged.record, indent=2) + "\n")

    # ------------------------------------------------------------ guidance

    def page_names(self) -> List[str]:
        names = []
        for page in ("guardrails.md", f"{self.module}.md"):
            if (self.harness.path / "instructions" / page).is_file():
                names.append(page)
        return names

    def guidance(self) -> str:
        """The block the module's prompt carries: both pages whole, then the
        attachments by name and description, then the paths."""
        lines = [f"{GUIDANCE_HEADING} (harness {self.harness.name} {self.harness.version})", ""]
        for page in self.page_names():
            lines.append((self.harness.path / "instructions" / page).read_text(encoding="utf-8").rstrip())
            lines.append("")
        if self.attachments.skills:
            lines.append("### Skills (open one when its situation arises)")
            for name in self.attachments.skills:
                data, _ = read_front_matter(self.harness.path / "skills" / name / "SKILL.md")
                lines.append(f"- {name} — {data['description']} (open with `{_OPEN_SKILL[self.host].format(name=name)}`)")
            lines.append("")
        if self.attachments.agents:
            lines.append("### Delegates (ask by name; each reads this guidance and the task statement)")
            for name in self.attachments.agents:
                data, _ = read_front_matter(self.harness.path / "agents" / f"{name}.md")
                lines.append(f"- {name} — {data['description']} (ask through {_ASK_AGENT[self.host].format(name=name)})")
            lines.append("")
        if self.attachments.mcp_servers:
            lines.append("### Tool servers")
            for name in self.attachments.mcp_servers:
                lines.append(f"- {name} — its tools are named mcp__{name}__<tool>")
            lines.append("")
        lines.append("### Paths")
        # The statement is named when the campaign has one: a build writes
        # it before its sessions; a learning crew's run directory has none,
        # and an archived campaign has the one its build wrote.
        if (self.campaign_dir / TASK_STATEMENT_FILE).is_file():
            statement = self.session_dir / TASK_STATEMENT_FILE if not self.at_campaign_root() else self.campaign_dir / TASK_STATEMENT_FILE
            lines.append(f"- Task statement: {statement} (read it before anything else)")
        lines.append(f"- Harness root: ${HARNESS_ROOT_VARIABLE} = {self.harness.path}")
        if self.prepared:
            lines.append("- Prepared resources: " + ", ".join(f"{PREPARED_DIR}/{name}" for name in self.prepared))
        return "\n".join(lines).rstrip() + "\n"


def _http_timeout_ms(harness: Harness) -> int:
    """A streamable-http server's per-call timeout where the host takes one
    (codex), from the config the harness was loaded with."""
    return int(harness.settings["server_tool_timeout_seconds"]) * 1000


def is_reference(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("${") and value.endswith("}") and len(value) > 3


