"""Implementation sessions for the generic strategy.

Owns the implementation feature: the coding-agent session over a candidate
solution (claude_code or codex, with the crash-retry on the configured
fallback model), the implementation prompt build, and the
technical-difficulties fallback reconstruction. Stateless functions only —
GenericSearch assembles arguments (including its cross-phase state
callbacks) and delegates here.
"""

import copy
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from kapso.core.prompt_loader import load_prompt, render_prompt
from kapso.execution.inbox import Request, load_requests
from kapso.execution.memories.repo_memory import RepoMemoryManager
from kapso.execution.search_strategies.generic.difficulties_generator import (
    generate_technical_difficulties,
)

logger = logging.getLogger(__name__)

# The implementation output contract's terminal tags: a result event
# carrying ALL of these means the session declared itself complete (drives
# the adapter's linger-reap and truthful end-mode classification).
IMPLEMENTATION_COMPLETION_MARKERS = ["</score>", "</technical_difficulties>"]

# The prompt's closing line and checklist note, with the inbox off (today's
# exact text) and on (the one sanctioned way to ask).
CLOSING_LINE_WITHOUT_INBOX = (
    "Do not ask any questions. Implement everything as specified and run "
    "the evaluation."
)
CLOSING_LINE_WITH_INBOX = (
    "Do not ask questions in your output — text outside the final tags is "
    "never read. The one way to ask for something is the `request_from_user` "
    "tool, and only for what a person must do."
)
CHECKLIST_NOTE_WITH_INBOX = (
    " — unless you called `request_from_user`, in which case return nothing"
)
INBOX_TOOL_LINE = (
    "\n\n### Asking the person (MCP tool)\n"
    "- **request_from_user**: ask the person running this campaign for "
    "something only a person can provide; calling it STOPS your session — "
    "see \"When you are blocked on something only a person can provide\" below"
)


@dataclass(frozen=True)
class SuspendedSession:
    """A session that asked the person and was stopped (design v4 §4.2):
    the requests it filed and the handle the CLI needs to resume it."""

    request_ids: List[int]
    cli_session_id: str


@dataclass(frozen=True)
class Continuation:
    """Resume a stored CLI session with the person's reply (design v4 §4.4)."""

    cli_session_id: str
    follow_up: str


class ContinuationFailed(RuntimeError):
    """The CLI could not resume the stored session; the node stays
    suspended and the campaign pauses again."""


def run_implementation(
    *,
    solution: str,
    problem: str,
    branch_name: str,
    parent_branch_name: str,
    ideation_repo_memory_sections_consulted: Optional[List[str]],
    lane_index: int,
    workspace,
    llm,
    registered_evaluation_manifest: Optional[Dict[str, str]],
    sync_registered_evaluation: Callable[[str], None],
    implementation_gates: List[str],
    gate_failure_policy: str,
    bank_serving: Optional[Dict[str, str]] = None,
    kg_index_path: Optional[str] = None,
    implementation_cli: str,
    implementation_model: str,
    implementation_fallback_model: Optional[str],
    implementation_web: bool,
    claude_auth_settings: Dict[str, Any],
    env_strip: List[str],
    env_defaults: Dict[str, str],
    lane_env: Optional[Dict[str, str]],
    session_effort: Optional[str],
    clamped_timeout: Callable[[float], float],
    implementation_timeout: float,
    session_stream_path: Callable[[str], str],
    build_prompt: Callable[..., str],
    previous_errors_text: str,
    lane_brief: str,
    note_session_started: Callable[[], None],
    note_session_end_facts: Callable[[str], None],
    await_registered_evaluation: Callable[[str], Optional[str]],
    node_id: int = 0,
    inbox_settings: Optional[Dict[str, Any]] = None,
    continuation: Optional[Continuation] = None,
) -> Tuple[str, Dict[str, float], Optional[str], Optional[SuspendedSession]]:
    """
    Implementation using Claude Code with MCP gates (code, research).
    
    Runs the coding-agent session (with the crash-retry on the configured
    fallback model), schedules the RepoMemory update, guards the
    registered evaluation's teardown, and finalizes the session.
    
    Args:
        solution: Solution description to implement
        problem: Problem description
        branch_name: Git branch for this experiment
        parent_branch_name: Parent branch to inherit code from
        ideation_repo_memory_sections_consulted: RepoMemory sections used during ideation
        node_id: The experiment node, named to the inbox gate
        inbox_settings: The campaign's inbox block ({enabled, path,
            stop_grace_seconds}); None or disabled means no session asks
        continuation: Resume the node's stored CLI session with the
            person's reply instead of starting a fresh one

    Returns:
        Tuple of (agent output string, phase telemetry with cost/duration,
        recovered manifest line, the suspended-session record when the
        session asked the person and was stopped)
    """
    from kapso.execution.coding_agents.base import CodingAgentConfig
    from kapso.execution.coding_agents.adapters.claude_code_agent import ClaudeCodeCodingAgent
    from kapso.gated_mcp import get_mcp_config
    from kapso.execution.memories.repo_memory.observation import extract_repo_memory_sections_consulted
    
    # Create experiment session (handles git branching). A continuation
    # takes the branch as the stopped session left it.
    if continuation is not None:
        session = workspace.create_experiment_session(
            branch_name, parent_branch_name, llm=llm, continue_branch=True
        )
    else:
        session = workspace.create_experiment_session(branch_name, parent_branch_name, llm=llm)

    # The campaign inbox: the gate files requests under an id kapso mints
    # here; the adapter tails the file for that id and stops the session.
    inbox_on = bool(inbox_settings and inbox_settings.get("enabled"))
    inbox_context: Dict[str, str] = {}
    inbox_agent_settings: Dict[str, Any] = {}
    if inbox_on:
        inbox_session_id = str(uuid.uuid4())
        inbox_context = {
            "KAPSO_INBOX_PATH": str(inbox_settings["path"]),
            "KAPSO_SESSION_ID": inbox_session_id,
            "KAPSO_NODE_ID": str(node_id),
        }
        inbox_agent_settings = {
            "session_id": inbox_session_id,
            "inbox_path": str(inbox_settings["path"]),
            "inbox_stop_grace_seconds": float(inbox_settings["stop_grace_seconds"]),
        }

    # A maintainer-registered evaluation is versioned on the workspace
    # root, but sessions inherit their parent branch's tree — which may
    # predate a re-registration. Frame-sync the registered tree in so
    # every candidate runs (and is integrity-checked against) the head.
    if registered_evaluation_manifest:
        sync_registered_evaluation(session.session_folder)
    
    # 1. Load RepoMemory
    repo_memory_doc = RepoMemoryManager.ensure_exists_in_worktree(session.session_folder)
    repo_memory_brief = RepoMemoryManager.render_summary_and_toc(repo_memory_doc, max_chars=2500)
    
    # 2. Get MCP config for code + research + repo_memory gates (not idea)
    mcp_servers, mcp_tools = get_mcp_config(
        gates=implementation_gates,
        repo_root=session.session_folder,
        include_base_tools=False,
        gate_failure_policy=gate_failure_policy,
        bank_serving=bank_serving,
        kg_index_path=kg_index_path,
        **({"inbox": inbox_context} if inbox_on else {}),
    )
    
    # 3. Build full tool set for implementation (includes Write, Edit)
    # Bash is kept for running evaluation scripts, not for repo_memory access.
    # implementation_web gates the session's live-web access on BOTH CLIs:
    # here the claude whitelist's WebSearch/WebFetch, below the codex
    # --search flag — one knob, no web side-door on either path.
    implementation_allowed_tools = [
        "Read", "Write", "Edit", "Bash",
        *(["WebSearch", "WebFetch"] if implementation_web else []),
        *[t for t in mcp_tools if t.startswith("mcp__")],
    ]
    
    logger.info(f"[GenericSearch] Implementation tools: {implementation_allowed_tools}")
    
    # 4. Configure Claude Code for implementation
    if implementation_cli == "codex":
        config = CodingAgentConfig(
            agent_type="codex",
            model=implementation_model,
            debug_model=implementation_model,
            agent_specific={
                **({"env_overrides": lane_env} if lane_env else {}),
                "env_strip": env_strip,
                "env_defaults": env_defaults,
                "mcp_servers": mcp_servers,
                # Same implementation_web knob as the claude whitelist:
                # gates the codex CLI's native --search tool.
                "web_search": implementation_web,
                "timeout": clamped_timeout(implementation_timeout),
                # Lane 0 tees the live transcript to the console, same
                # policy as the claude path.
                "streaming": lane_index == 0,
                "effort": session_effort,
                # Transcript stream persisted for the difficulties
                # fallback's forensics, same as the claude path.
                "stream_artifact_path": session_stream_path(branch_name),
                # The inbox needs the thread id (the resume handle) and
                # the tail-and-stop; both ride the same settings.
                "capture_thread_id": inbox_on,
                **inbox_agent_settings,
            },
        )
    else:
        config = CodingAgentConfig(
            agent_type="claude_code",
            model=implementation_model,
            debug_model=implementation_model,
            agent_specific={
                **claude_auth_settings,
                **({"env_overrides": lane_env} if lane_env else {}),
                "env_strip": env_strip,
                "env_defaults": env_defaults,
                "mcp_servers": mcp_servers,
                "allowed_tools": implementation_allowed_tools,
                "timeout": clamped_timeout(implementation_timeout),
                # Under node expansion only lane 0 streams to the console;
                # other lanes stay buffered (their raw streams still land in
                # per-branch stream_artifact_path files).
                "streaming": lane_index == 0,
                "effort": session_effort,
                # Per-session process record: raw stream-json events land
                # here as they arrive, so a killed session still leaves its
                # forensics behind (feeds the difficulties fallback).
                "stream_artifact_path": session_stream_path(branch_name),
                # Declared-completion contract: lets the adapter reap a CLI
                # that delivered its full final report but lingers alive.
                "completion_markers": IMPLEMENTATION_COMPLETION_MARKERS,
                **inbox_agent_settings,
            }
        )

    # 5. Build implementation prompt
    repo_memory_detail_access_instructions = (
        "For detailed section content (architecture, gotchas, invariants, etc.),\n"
        "use the MCP tool: `get_repo_memory_section(section_id=\"core.architecture\")`\n"
        "Available sections: core.architecture, core.entrypoints, core.where_to_edit, core.invariants, core.testing, core.gotchas, core.dependencies\n"
        "Fallback: open `.kapso/repo_memory.json` and read `book.sections[section_id]`."
    )
    
    if continuation is not None:
        prompt = continuation.follow_up
    else:
        prompt = build_prompt(
            solution=solution,
            problem=problem,
            branch_name=branch_name,
            repo_memory_brief=repo_memory_brief,
            repo_memory_detail_access_instructions=repo_memory_detail_access_instructions,
            previous_errors=previous_errors_text,
            lane_brief=lane_brief,
            **(
                {"inbox_section": render_inbox_section(
                    load_requests(inbox_settings["path"])
                )}
                if inbox_on else {}
            ),
        )
    
    # 6. Run the implementation session
    print(f"[GenericSearch] Running {implementation_cli} implementation...")
    if implementation_cli == "codex":
        from kapso.execution.coding_agents.factory import CodingAgentFactory

        agent = CodingAgentFactory.create(config)
    else:
        agent = ClaudeCodeCodingAgent(config)
    agent.initialize(session.session_folder)

    phase_started = time.monotonic()
    phase_cost = 0.0
    suspended: Optional[SuspendedSession] = None
    try:
        note_session_started()
        if continuation is not None:
            result = agent.resume(continuation.cli_session_id, prompt)
        else:
            result = agent.generate_code(prompt)
        phase_cost = agent.get_cumulative_cost()
        agent_output = result.output if result.output else ""

        # Ground truth about HOW the session ended, for the feedback
        # judge (run #8: a self-inflicted SIGTERM was misdiagnosed as
        # the time limit, so the footgun was never named).
        meta = result.metadata or {}
        if meta.get("stopped_for_inbox"):
            request_ids = [int(i) for i in meta["inbox_request_ids"]]
            cli_session_id = str(meta.get("cli_session_id") or "")
            if not cli_session_id:
                raise RuntimeError(
                    "the session asked the person but the CLI session handle "
                    "is unknown, so it could never be resumed — a wiring bug"
                )
            suspended = SuspendedSession(request_ids, cli_session_id)
            end_facts = (
                "implementation session STOPPED to ask the person (inbox "
                f"requests {', '.join(f'#{i}' for i in request_ids)}); "
                "it will be resumed with the reply"
            )
        elif continuation is not None and not result.success:
            raise ContinuationFailed(
                f"could not resume CLI session {continuation.cli_session_id!r} "
                f"for {branch_name}: {result.error}"
            )
        elif meta.get("completed_reaped"):
            end_facts = (
                "implementation session COMPLETED its final report; the "
                "CLI process lingered and was reaped after a short grace "
                "— this was a successful session, not a failure"
            )
        elif meta.get("deadline_exceeded") and meta.get(
            "completed_before_kill"
        ):
            end_facts = (
                "implementation session COMPLETED its final report "
                "before the deadline kill — the kill reflects a lingering "
                "process, not unfinished work"
            )
        elif result.success:
            end_facts = "implementation session ended naturally"
        elif meta.get("deadline_exceeded"):
            end_facts = (
                "implementation session was KILLED BY ITS OWN DEADLINE "
                f"after {meta.get('elapsed_seconds', 0):.0f}s"
            )
        else:
            end_facts = (
                f"implementation session died prematurely ({result.error}); "
                "the deadline was NOT reached — suspect an external or "
                "self-inflicted kill"
            )
        if meta.get("last_tool"):
            end_facts += f"; last tool call before end: {meta['last_tool']}"
        note_session_end_facts(end_facts)

        if not result.success and continuation is None:
            logger.warning(f"[GenericSearch] Implementation failed: {result.error}")
            agent_output = f"Implementation failed: {result.error}\n\n{agent_output}"
            # A crash is not the deadline: the lane still owns its time.
            # Retry once on the fallback model (different model family or
            # version answers a provider-side content kill; the deadline
            # clamp keeps the retry inside the campaign budget).
            deadline_kill = bool((result.metadata or {}).get(
                "deadline_exceeded"
            ))
            if implementation_fallback_model and not deadline_kill:
                logger.warning(
                    "[GenericSearch] Retrying the implementation on "
                    f"fallback model {implementation_fallback_model}"
                )
                fallback_config = copy.deepcopy(config)
                fallback_config.model = implementation_fallback_model
                fallback_config.debug_model = (
                    implementation_fallback_model
                )
                fallback_config.agent_specific["timeout"] = (
                    clamped_timeout(implementation_timeout)
                )
                agent.cleanup()
                if implementation_cli == "codex":
                    from kapso.execution.coding_agents.factory import (
                        CodingAgentFactory,
                    )
                    agent = CodingAgentFactory.create(fallback_config)
                else:
                    agent = ClaudeCodeCodingAgent(fallback_config)
                agent.initialize(session.session_folder)
                note_session_started()
                fallback_result = agent.generate_code(
                    prompt
                    + "\n\nNOTE: a previous session on this lane ended "
                    f"prematurely ({result.error}). Its partial work is in "
                    "the branch; continue from there and finish the "
                    "implementation and evaluation."
                )
                phase_cost += agent.get_cumulative_cost()
                if fallback_result.output:
                    agent_output = fallback_result.output
                    result = fallback_result
                    note_session_end_facts(
                        "implementation session crashed and was retried on "
                        f"fallback model {implementation_fallback_model}"
                    )
    finally:
        agent.cleanup()
    telemetry = {
        "cost_usd": phase_cost,
        "duration_seconds": time.monotonic() - phase_started,
    }
    
    # 7. Update RepoMemory for this experiment branch — not for a session
    # that asked and stopped: its work is unfinished, the continuation's
    # close records it.
    run_result_payload = {
        "score": 0,
        "run_had_error": False,
        "error_message": "",
        "error_details": "",
        "feedbacks": "",
        "ideation_repo_memory_sections_consulted": ideation_repo_memory_sections_consulted or [],
    }
    
    # Extract sections consulted from changes.log
    sections_consulted = []
    try:
        changes_log_path = os.path.join(session.session_folder, "changes.log")
        if os.path.exists(changes_log_path):
            with open(changes_log_path, "r", encoding="utf-8", errors="replace") as f:
                sections_consulted = extract_repo_memory_sections_consulted(f.read())
    except Exception:
        sections_consulted = []
    run_result_payload["repo_memory_sections_consulted"] = sections_consulted
    
    # Schedule RepoMemory update for session close
    if suspended is None:
        session.schedule_repo_memory_update(
            solution_spec=solution,
            run_result=run_result_payload,
        )
    
    # 8. Registered-evaluation teardown guard: wait for a live grader
    # and stash any durable-archive recovery BEFORE finalize's rmtree.
    recovered_manifest_line = await_registered_evaluation(
        agent_output
    )

    # 9. Finalize session (commits changes; push serialized by the
    # workspace repo_lock — lane-safe under node expansion)
    workspace.finalize_session(session)

    return agent_output, telemetry, recovered_manifest_line, suspended


def build_implementation_prompt(
    *,
    solution: str,
    problem: str,
    branch_name: str,
    repo_memory_brief: str,
    repo_memory_detail_access_instructions: str,
    previous_errors: str,
    budget_status: str,
    evaluation_instructions: str,
    shared_artifacts_brief: str,
    lane_brief: str = "",
    inbox_section: str = "",
) -> str:
    """Build the implementation prompt for Claude Code. With the inbox
    off (empty ``inbox_section``) the render is byte-identical to the
    prompt before the inbox existed."""
    template = load_prompt("execution/search_strategies/generic/prompts/implementation_claude_code.md")
    inbox_on = bool(inbox_section)
    return render_prompt(
        template,
        {
            "solution": solution or "(No solution provided)",
            "problem": problem or "(No problem description provided)",
            "branch_name": branch_name,
            "repo_memory_brief": repo_memory_brief or "(No repo memory available)",
            "repo_memory_detail_access_instructions": repo_memory_detail_access_instructions,
            "previous_errors": previous_errors or "(No previous errors)",
            "budget_status": budget_status,
            "evaluation_instructions": evaluation_instructions,
            "shared_artifacts_brief": shared_artifacts_brief,
            "lane_brief": lane_brief,
            "inbox_section": ("\n\n" + inbox_section) if inbox_on else "",
            "inbox_tool_line": INBOX_TOOL_LINE if inbox_on else "",
            "inbox_checklist_note": CHECKLIST_NOTE_WITH_INBOX if inbox_on else "",
            "closing_line": (
                CLOSING_LINE_WITH_INBOX if inbox_on else CLOSING_LINE_WITHOUT_INBOX
            ),
        },
    )


# =============================================================================
# THE INBOX IN THE PROMPTS (design v4, Appendix A)
# =============================================================================

def render_inbox_state(requests: Dict[int, Request]) -> str:
    """The campaign's requests so far, for the build prompt. Empty on a
    fresh campaign (the heading is omitted)."""
    if not requests:
        return ""
    lines = ["### Requests already in this campaign's inbox"]
    for request in sorted(requests.values(), key=lambda r: r.id):
        if request.open:
            lines.append(
                f"- #{request.id} {request.key} — open, no reply yet: treat as "
                "ABSENT and do not request it again"
            )
        else:
            reply = request.reply if request.reply else "(done)"
            lines.append(
                f"- #{request.id} {request.key} — answered (node {request.node}): "
                f"{reply!r}"
            )
    return "\n".join(lines) + "\n\n"


def render_inbox_section(requests: Dict[int, Request]) -> str:
    """The implementation prompt's inbox section (Appendix A.1)."""
    template = load_prompt(
        "execution/search_strategies/generic/prompts/inbox_section.md"
    )
    return render_prompt(template, {"inbox_state": render_inbox_state(requests)})


def render_follow_up(requests: List[Request]) -> str:
    """The one user message a resumed session reads first (Appendix A.3):
    every request of the node with the person's reply, then the next
    steps the session recorded."""
    if not requests:
        raise ValueError("a continuation needs at least one answered request")
    blocks = []
    next_steps: List[str] = []
    for request in requests:
        if request.open:
            raise ValueError(f"request #{request.id} has no reply yet")
        reply = f'"{request.reply}"' if request.reply else "(done)"
        blocks.append(
            f"Request #{request.id} — {request.key}\n"
            f"  you saw: {request.hit}\n"
            f"  you tried: {request.tried}\n"
            f"  you asked them to: {request.fix}\n"
            f"  their reply: {reply}"
        )
        if request.next_steps not in next_steps:
            next_steps.append(request.next_steps)
    template = load_prompt(
        "execution/search_strategies/generic/prompts/inbox_follow_up.md"
    )
    return render_prompt(
        template,
        {
            "requests": "\n\n".join(blocks),
            "next_steps": "\n".join(f"  {step}" for step in next_steps),
        },
    )


def render_inbox_answered(requests: Dict[int, Request]) -> str:
    """The ideation prompt's block about answered requests (empty when
    none): a reply holds for the whole campaign."""
    answered = [r for r in sorted(requests.values(), key=lambda r: r.id) if not r.open]
    if not answered:
        return ""
    template = load_prompt(
        "execution/search_strategies/generic/prompts/inbox_answered.md"
    )
    lines = "\n".join(
        f"- {r.key}: {r.reply!r}" if r.reply else f"- {r.key}: provided"
        for r in answered
    )
    return "\n\n" + render_prompt(template, {"answered_lines": lines})


def ensure_technical_difficulties(
    node,
    *,
    implementation_model: str,
    claude_auth_settings: Dict[str, Any],
    env_strip: List[str],
    session_effort: Optional[str],
    clamped_timeout: Callable[[float], float],
    ideation_timeout: float,
    workspace_dir: str,
    session_stream_path: Callable[[str], str],
) -> None:
    """Run the fallback reconstruction when the implementor's
    technical_difficulties tag is missing (crashed or deadline-killed
    session, or simply omitted)."""
    if (node.technical_difficulties or "").strip():
        return
    print(
        "[GenericSearch] technical_difficulties missing — "
        "running fallback reconstruction"
    )
    node.technical_difficulties = generate_technical_difficulties(
        model=implementation_model,
        claude_auth_settings=claude_auth_settings,
        env_strip=env_strip,
        effort=session_effort,
        timeout_seconds=clamped_timeout(ideation_timeout),
        workspace_dir=node.workspace_dir or workspace_dir,
        solution=node.solution,
        stream_artifact_path=session_stream_path(node.branch_name),
    )
