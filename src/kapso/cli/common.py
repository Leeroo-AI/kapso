"""Shared pieces of the CLI: the agent list, detached runs and the campaign
summary block. Each verb module imports what it needs from here; nothing
here imports a verb module."""

from pathlib import Path
from typing import List, Optional, Sequence

from kapso.execution.coding_agents.factory import CodingAgentFactory
from kapso.execution.inbox import beside, launch_receipt_path
from kapso.execution.launch import detach


# Available coding agents
AVAILABLE_AGENTS = [
    name
    for name in CodingAgentFactory.list_available()
    if not CodingAgentFactory.get_agent_info(name).get("inference_only", False)
]


def list_agents() -> None:
    """List available coding agents with detailed info."""
    CodingAgentFactory.print_agents_info()


# =============================================================================
# DETACHED RUNS
# =============================================================================

def _without_detach_flags(argv: Sequence[str]) -> List[str]:
    """This command line without --detach and --log, for the child."""
    kept: List[str] = []
    skip = False
    for word in argv:
        if skip:
            skip = False
        elif word == "--detach" or word.startswith("--log="):
            continue
        elif word == "--log":
            skip = True
        else:
            kept.append(word)
    return kept


def _inside(path: Path, directory: Path) -> bool:
    """Whether `path` lies under `directory` (both resolved)."""
    return path == directory or directory in path.parents


def _default_log(campaign: Path) -> Path:
    """The detached run's log: a sibling of the campaign directory, which
    must stay empty for seeding and holds only the campaign itself."""
    return beside(campaign, ".log")




def _detach(
    label: str, argv: Sequence[str], *, log_path: Path, receipt_path: Optional[Path], handoff: Sequence[str],
) -> None:
    """Run this very command again in a session of its own, its output in
    a log file, and return at once with what the person needs to follow
    it: the pid, the log, and the `handoff` lines the verb gives (where
    the status is, the watch line). A session of its own
    (`start_new_session`) is what outlives the shell call that launched
    it: a coding agent's tool call ends a plain background child when it
    returns, and the campaign with it (Codex, verified 2026-09-07); every
    host and the skill used to reinvent the nohup/setsid dance for this
    one flag. A build writes a receipt beside its log, which `kapso
    watch` reads before the status exists; a learn or an ingest writes
    none (their status is born within seconds under the learning status
    directory, and nothing reads a receipt there). The launch itself is
    `kapso.execution.launch.detach`, which the API's `detach=True` shares."""
    pid = detach(argv, log_path=log_path, receipt_path=receipt_path)
    print(f"{label} started in a session of its own (pid {pid})")
    print(f"  log:     {log_path}")
    for line in handoff:
        print(f"  {line}")


def build_handoff(campaign: Path) -> List[str]:
    """What a person needs after a detached build: the status, the watch
    line, the inbox."""
    return [
        f"status:  {campaign / '.kapso' / 'status.json'}",
        f"follow:  kapso watch {campaign} --follow",
        f"inbox:   kapso inbox {campaign}",
        "runs:    on its own until it ends; this terminal is free",
    ]


def learning_handoff(status_dir: str) -> List[str]:
    """What a person needs after a detached learn or ingest: the status
    directory, whose newest file `kapso watch` follows."""
    return [
        f"status:  {status_dir}/ (the newest file is this run's)",
        f"follow:  kapso watch {status_dir} --follow",
    ]


def _print_build_summary(solution) -> None:
    """The block after a campaign: COMPLETED, or WAITING ON YOU when it
    paused for a reply in the inbox (design v4 §4.6), each over the
    solution's own summary. A pause is not a failure: the exit code stays 0."""
    print("\n" + "=" * 60)
    campaign = str(solution.campaign)
    if solution.stopped_reason == "waiting_for_user":
        requests = solution.requests
        target = f" {requests[0]['id']}" if len(requests) > 1 else ""
        print("WAITING ON YOU")
        print("=" * 60)
        print(solution.explain())
        print(f'Reply with:  kapso inbox reply{target} "…"   (inside {campaign})')
        print(f"Any time:    kapso inbox {campaign}")
        return
    print("COMPLETED")
    print("=" * 60)
    print(solution.explain())
