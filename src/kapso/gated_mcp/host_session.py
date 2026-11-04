# The MCP server for a HOST session — the engineer's own Claude Code or
# Codex session — as opposed to the sessions Kapso spawns inside a campaign.
#
# A campaign session gets its gate configuration built per session by
# get_mcp_config(): the interpreter running Kapso, the campaign's paths, the
# loaded experience. A host's MCP file is static and read at session start,
# so it can name none of these. This entry point is what it names instead:
# a console script (`kapso-mcp`, on the interpreter Kapso is installed in)
# that derives the same configuration from where the session runs and from
# what `--experience` loads, stages the serving for the session, and serves
# the bundled gates on stdio.
#
#   claude mcp add kapso -- kapso-mcp --experience mine
#   codex mcp add kapso -- kapso-mcp --experience mine,org-x
#
# What rides by what the directory holds: web research (the codex CLI), the
# repository's own memory when a campaign has built one, the experiment
# history when the session runs inside a campaign directory. What rides by
# what is loaded (docs/plans/redesign/wp3-experience.md): the bank gate over
# every loaded bank, the search gates over a loaded index. What rides always
# (wp7-plugin.md): the campaign gate, the verbs as this session's tools.
# `--experience` absent, the project's config lists the sources
# (`experience.session_sources`); the config is `--config`, else the
# project root's `kapso-config.yaml`. Each gate left out is named on
# stderr and in the server's instructions with the reason; the server
# connects whatever is off; stdout is the MCP transport.

import argparse
import asyncio
import atexit
import os
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from kapso import __version__
from kapso.cli.errors import install_error_hook
from kapso.core.config import load_config
from kapso.experience.sources import ExperienceError, experience_settings
from kapso.execution.memories.repo_memory import RepoMemoryManager
from kapso.experience.reference import parse_experience_reference
from kapso.experience.sources import ExperienceSource, resolve_source, source_exists
from kapso.gated_mcp.presets import GATES
from kapso.gated_mcp.server import run_server
from kapso.harness import load_harness
from kapso.experience.store import ExperienceStore
from kapso.learning.serving_launch import plan_serving, stage_serving, task_coordinates

# The gates a host session gets without being asked. `idea` and `code`
# join when an index is loaded; `kg` carries the write tools and `inbox` is
# campaign-only, so neither is ever a host default; `campaign` needs
# nothing and always rides.
DEFAULT_GATES = ("research", "repo_memory", "experiment_history", "bank", "campaign")
INDEX_GATES = ("idea", "code")
EXPERIMENT_HISTORY_NAME = "experiment_history.json"
# The project's config, read when `--config` is absent: the one file the
# verbs take as `--config kapso-config.yaml`.
PROJECT_CONFIG = "kapso-config.yaml"
# The label the host session's pull events carry: the session is the
# engineer's own, not one of a campaign's.
HOST_SESSION = "host"


@dataclass
class HostGates:
    """What a host session is served: the gates, the environment the
    bundled server reads them from, and one line per gate left out."""

    enabled: List[str]
    env: Dict[str, str]
    notes: List[str] = field(default_factory=list)


def repository_root(start: Path) -> Path:
    """The git top level `start` is inside, else `start` itself."""
    probe = subprocess.run(
        ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        return start
    return Path(probe.stdout.strip()).resolve()


def campaign_directory(start: Path) -> Optional[Path]:
    """The nearest ancestor of `start` (itself included) that holds a
    campaign's experiment history, else None."""
    for candidate in (start, *start.parents):
        if (candidate / ".kapso" / EXPERIMENT_HISTORY_NAME).is_file():
            return candidate
    return None


def load_sources(text: Optional[str], settings: Dict[str, Any]) -> List[ExperienceSource]:
    """The sources `--experience` names, comma-separated, each resolved
    as `build` resolves them: a bad reference or a missing source stops
    the server with the line that would make it."""
    sources: List[ExperienceSource] = []
    for raw in [part.strip() for part in (text or "").split(",") if part.strip()]:
        reference = parse_experience_reference(raw, indexes_root=Path(settings["indexes"]))
        sources.append(resolve_source(reference, settings=settings))
    return sources


def config_sources(settings: Dict[str, Any]) -> Tuple[List[ExperienceSource], List[str]]:
    """What `experience.session_sources` lists: each source that exists,
    resolved as `build` resolves it, and one note per source that does
    not (its gate stays off, the line that makes it named), never a stop.
    A wrong type is refused; a reference outside the grammar raises as
    the parser does."""
    listed = settings["session_sources"]
    if not isinstance(listed, list) or not all(isinstance(item, str) for item in listed):
        raise ExperienceError(
            "experience.session_sources must be a list of references (mine, mine:all, leeroopedia, "
            f"a bank path or git reference, an index name), not {listed!r}"
        )
    sources: List[ExperienceSource] = []
    notes: List[str] = []
    store = ExperienceStore(settings["banks"], settings["indexes"])
    for raw in listed:
        reference = parse_experience_reference(raw, indexes_root=Path(settings["indexes"]), must_exist=False)
        # A bank you read is fetched the first time, when its remote answers.
        first_fetch = reference.kind == "git" and not store.is_fetched(reference) and store.reachable(reference)
        if first_fetch or source_exists(reference, settings=settings):
            sources.append(resolve_source(reference, settings=settings))
        else:
            notes.append(f"{raw}: not found, so its gate is off ({_makes_it(reference)}); it is listed under experience.session_sources")
    return sources, notes


def _makes_it(reference) -> str:
    if reference.kind == "hosted":
        return "LEEROOPEDIA_API_KEY in .env serves it"
    if reference.kind == "index":
        return f"`kapso experience add --repo <url> --name {reference.name}` makes it"
    if reference.kind == "git":
        return "its remote did not answer `git ls-remote`; it is fetched the first time it does"
    if reference.kind == "path":
        return f"no bank at {reference.path}: a bank is a git repository whose main branch holds index.md, insights/ and procedures/"
    return "`kapso experience add mine` makes it, as a first learn does"


def project_config(root: Path) -> Optional[str]:
    """The project's `kapso-config.yaml` at the repository root `root` is
    inside (else `root` itself), when there is one."""
    candidate = repository_root(root) / PROJECT_CONFIG
    return str(candidate) if candidate.is_file() else None


def session_brief(enabled: Sequence[str], notes: Sequence[str], intro_path: Optional[str]) -> str:
    """The server's instructions, shown on connect: the staged intro (what
    the session is served, measured before) when one exists, then the
    gates on and each one off with its reason."""
    lines: List[str] = []
    if intro_path:
        lines += [Path(intro_path).read_text(encoding="utf-8").rstrip(), ""]
    lines.append(f"## Kapso tools in this session (kapso-mcp {__version__})")
    lines.append(f"- on: {', '.join(enabled)}")
    lines += [f"- {note}" for note in notes]
    return "\n".join(lines) + "\n"


def prepare_host_gates(
    *,
    root: Path,
    requested: Sequence[str],
    config: Dict[str, Any],
    sources: Sequence[ExperienceSource],
    session_dir: Path,
    probe_budget: int,
    harness_name: Optional[str] = None,
) -> HostGates:
    """Decide which of the requested gates this session can have, and
    build the environment the bundled server resolves them from. The
    loaded banks are staged into `session_dir` here (one pinned clone
    each, the serving record, the pull log and the intro the server
    serves as its instructions) when the bank gate is among the
    requested; `harness_name` is the session's coordinate (`--harness`),
    `none` without one. Request order is kept, duplicates dropped; an
    unknown gate or a campaign-only one is a configuration error, never
    silently skipped."""
    plan = plan_serving(list(sources), task_coordinates(harness_name), session_dir, probe_budget, holder="session")
    env: Dict[str, str] = {}
    enabled: List[str] = []
    notes: List[str] = []
    seen = set()
    # Whatever is loaded is staged once: the banks' pinned clones, the
    # intro naming every source (the server's instructions, so the session
    # sees what it is served on connect, a corpus or the hosted gate
    # included), the serving record and the pull log.
    staged = stage_serving(plan, list(sources)) if plan is not None else None
    if staged is not None:
        env["KAPSO_SERVING_INTRO"] = staged["intro_path"]
    for gate in requested:
        if gate in seen:
            continue
        seen.add(gate)
        if gate not in GATES:
            raise ValueError(
                f"unknown gate {gate!r}; the gates are {sorted(GATES)}"
            )
        if gate == "inbox":
            raise ValueError(
                "the inbox gate is campaign-only: a host session is the "
                "person, it does not ask one"
            )
        if gate == "research":
            if shutil.which("codex") is None:
                notes.append("research: off (the codex CLI is not on PATH)")
                continue
        elif gate == "repo_memory":
            repo_root = repository_root(root)
            if not (repo_root / RepoMemoryManager.MEMORY_REL_PATH).is_file():
                notes.append(
                    f"repo_memory: off (no {RepoMemoryManager.MEMORY_REL_PATH} "
                    f"under {repo_root}; a campaign builds one)"
                )
                continue
            env["REPO_MEMORY_ROOT"] = str(repo_root)
        elif gate == "experiment_history":
            campaign = campaign_directory(root)
            if campaign is None:
                notes.append(
                    "experiment_history: off (not inside a campaign directory)"
                )
                continue
            env["EXPERIMENT_HISTORY_PATH"] = str(
                campaign / ".kapso" / EXPERIMENT_HISTORY_NAME
            )
            embedding = config["models"]["embedding"]
            if embedding:
                env["EXPERIMENT_EMBEDDING_MODEL"] = str(embedding)
        elif gate == "bank":
            if staged is None or not plan["banks"]:
                notes.append("bank: off (no bank loaded: `kapso experience add mine` makes yours; `session_sources: [mine:all]` under `experience` in kapso-config.yaml serves it here)")
                continue
            env.update(staged["bank_serving"])
            env["KAPSO_SESSION"] = HOST_SESSION
        elif gate in INDEX_GATES:
            if plan is None or not plan["kg_index_path"]:
                notes.append(f"{gate}: off (no index loaded: `kapso experience add --repo <url> --name <name>` makes one; `session_sources` serves it here)")
                continue
            env["KG_INDEX_PATH"] = plan["kg_index_path"]
        enabled.append(gate)
    if plan is not None and plan["hosted"]:
        notes.append(
            "leeroopedia: a hosted server, not a bundled gate — add it to the "
            "host as an HTTP MCP server with your key on the URL"
        )
    return HostGates(enabled=enabled, env=env, notes=notes)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="kapso-mcp",
        description=(
            "Serve Kapso's knowledge and experience tools to the coding-agent "
            "session this runs in (stdio MCP). Add it to a host with "
            "`claude mcp add kapso -- kapso-mcp --experience mine` or "
            "`codex mcp add kapso -- kapso-mcp --experience mine`."
        ),
    )
    parser.add_argument("--version", action="version", version=f"kapso-mcp {__version__}")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Config path (default: the project root's kapso-config.yaml, else the packaged config.yaml)",
    )
    parser.add_argument(
        "--root", type=str, default=None,
        help="The directory the session works in (default: the current directory)",
    )
    parser.add_argument(
        "--experience", type=str, default=None,
        help=(
            "Comma-separated sources to serve: mine, mine:all, a bank path or git reference, "
            "an index name (see `kapso experience list`). Nothing loaded serves "
            "no bank and no index"
        ),
    )
    parser.add_argument(
        "--harness", type=str, default=None,
        help=(
            "The harness this session works under (a name, a path or a git reference): "
            "its name is the coordinate `mine` serves cards by; none means `none`"
        ),
    )
    return parser.parse_args(argv)


def remove_session_dir(session_dir: Path) -> None:
    """The session's staging (pinned clones, record, pull log, intro) goes
    with the server; a host session has no learning to harvest it for."""
    if session_dir.exists():
        shutil.rmtree(session_dir)


def main(argv: Optional[Sequence[str]] = None) -> None:
    # The engine's own errors print their message alone (cli/errors.py):
    # a bad reference, a wrong `session_sources`, a missing `--experience`
    # source are one stderr line and exit 1.
    install_error_hook()
    args = _parse_args(argv)
    root = Path(args.root).expanduser().resolve() if args.root else Path.cwd()
    config_path = args.config or project_config(root)
    print(f"kapso-mcp {__version__} (config: {config_path or 'the packaged defaults'})", file=sys.stderr)
    config = load_config(config_path)
    settings = experience_settings(config)
    if args.experience is None:
        sources, source_notes = config_sources(settings)
    else:
        # Explicit, so strict: a missing source refuses, as a build is
        # refused; an empty string loads nothing.
        sources, source_notes = load_sources(args.experience, settings), []
    harness_name = None
    if args.harness:
        # Loaded for its name alone: the coordinate the bank serves by.
        # stderr keeps stdout for the MCP transport.
        harness = load_harness(
            args.harness, settings=config["harness"],
            say=lambda line: print(f"kapso-mcp: {line}", file=sys.stderr),
        )
        harness_name = harness.name
    requested = list(DEFAULT_GATES)
    if any(source.kind == "index" for source in sources):
        requested.extend(INDEX_GATES)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    session_dir = Path(settings["session_root"]).expanduser() / f"{stamp}-{os.getpid()}"
    # The staging goes when the server does: at a normal exit, and at the
    # SIGTERM a host sends its MCP servers on shutdown (an exit through
    # the handler runs the same registration).
    atexit.register(remove_session_dir, session_dir)
    signal.signal(signal.SIGTERM, lambda _signum, _frame: sys.exit(0))
    gates = prepare_host_gates(
        root=root,
        requested=requested,
        config=config,
        sources=sources,
        session_dir=session_dir,
        probe_budget=settings["probe_budget"],
        harness_name=harness_name,
    )
    notes = [*source_notes, *gates.notes]
    for note in notes:
        print(f"kapso-mcp: {note}", file=sys.stderr)
    print(
        f"kapso-mcp: serving {', '.join(gates.enabled)} for {root}",
        file=sys.stderr,
    )
    # The bundled server reads its configuration from the environment
    # (env is the transport across the process boundary, as for campaign
    # sessions); the campaign gate's working directory and config path
    # are its params, and the brief is the server's instructions.
    # Availability was decided above, so any miss is a bug.
    os.environ.update(gates.env)
    os.environ["MCP_ENABLED_GATES"] = ",".join(gates.enabled)
    os.environ["MCP_GATE_FAILURE_POLICY"] = "error"
    asyncio.run(run_server(
        gate_params={
            "campaign": {"config_path": config_path, "cwd": str(root)},
            **{gate: {"config_path": config_path} for gate in INDEX_GATES},
        },
        instructions=session_brief(gates.enabled, notes, gates.env.get("KAPSO_SERVING_INTRO")),
    ))


if __name__ == "__main__":
    main()
