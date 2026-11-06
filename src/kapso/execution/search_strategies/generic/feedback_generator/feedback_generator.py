# Feedback Generator
#
# LLM-based feedback generator that validates evaluation and decides stop/continue.
#
# The feedback generator is a coding agent that can be any of the coding agents
# in src/execution/coding_agents/ (aider, gemini, claude_code, openhands).
# Default is claude_code (adapter-resolved auth).
#
# It is responsible for:
# 1. Validating that the evaluation is fair and correct
# 2. Checking if the goal has been achieved
# 3. Extracting the evaluation score (if any)
# 4. Generating actionable feedback for the next iteration

import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

from kapso.execution.coding_agents.factory import CodingAgentFactory
from kapso.execution.coding_agents.base import CodingAgentConfig
from kapso.core.prompt_loader import load_prompt, optional_block, render_prompt


@dataclass
class FeedbackResult:
    """Result from feedback generator."""
    stop: bool                      # Whether to stop iteration
    evaluation_valid: bool          # Whether evaluation is fair/correct
    feedback: str                   # Actionable feedback for next iteration
    score: Optional[float] = None   # Extracted evaluation score (if any)
    # Served knowledge-bank cards whose guidance was load-bearing for this
    # iteration's result (the citation contract's attribution substrate —
    # learn-from-trajectories §5.1); empty when none or when serving is off.
    cards_load_bearing: List[str] = field(default_factory=list)
    # Budget telemetry for the feedback agent call itself.
    cost_usd: float = 0.0
    duration_seconds: Optional[float] = None
    # Why no verdict was reached (the judge's session ended at its
    # deadline, no tags after a retry, no room for one): the attempt is
    # unjudged, by reason, never given a fabricated verdict.
    unjudged: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "stop": self.stop,
            "evaluation_valid": self.evaluation_valid,
            "feedback": self.feedback,
            "score": self.score,
            "cards_load_bearing": self.cards_load_bearing,
            "unjudged": self.unjudged,
            "cost_usd": self.cost_usd,
            "duration_seconds": self.duration_seconds,
        }


class FeedbackGenerator:
    """
    LLM-based feedback generator.
    
    Validates evaluation and decides stop/continue.
    
    NOTE 1: The feedback generator is a coding agent that can be any of the
    coding agents in src/execution/coding_agents/ (aider, gemini, claude_code,
    openhands). Default is claude_code (adapter-resolved auth).
    
    NOTE 2: The feedback generator is also responsible for extracting the exact
    value of the evaluation score (if any) from the evaluation output and
    returning it in the FeedbackResult.
    """
    
    # Default prompt path
    PROMPT_PATH = "execution/search_strategies/generic/feedback_generator/prompts/feedback_generator.md"
    # The axes section and the frontier duty, rendered only while the
    # campaign has a lens plan (the planner names the axes).
    AXIS_FRONTIER_PROMPT_PATH = "execution/search_strategies/generic/feedback_generator/prompts/axis_frontier.md"

    # The retry resumes the judge's own session with the nudge; its run
    # entry says why (`retry_reason`), and the outcomes without a verdict
    # are the attempt's `unjudged` reasons.
    RETRY_REASON = "tags missing"
    DEADLINE_REASON = "the judge's session ended at its deadline"
    NO_TAGS_AFTER_RETRY = "no tags after a retry"
    RETRY_PROMPT = (
        "Your previous response did not contain the required tags. Respond "
        "now with ONLY the five tags — <stop>, <evaluation_valid>, <score>, "
        "<feedback>, <cards_load_bearing> — filled in for the iteration you "
        "just analyzed. No other text."
    )
    
    def __init__(self, coding_agent_config: CodingAgentConfig):
        """
        Initialize feedback generator.

        Args:
            coding_agent_config: The judge's session: the `judge` role over
                the config's hosts table, built by the orchestrator.
        """
        # Absent timeout means no judge deadline: pin the explicit None so
        # the adapter runs unbounded instead of applying its own default.
        coding_agent_config.agent_specific.setdefault("timeout", None)

        self.coding_agent_config = coding_agent_config

        # Create the coding agent
        self.agent = CodingAgentFactory.create(coding_agent_config)
        # The judge's own settings, kept apart so each attempt's staging is
        # merged onto them and not onto the previous attempt's merge.
        self._own_agent_specific = dict(coding_agent_config.agent_specific)

    @property
    def agent_type(self) -> str:
        return self.coding_agent_config.agent_type

    def attach_harness(self, staged) -> None:
        """This judgment's staged session: the harness's feedback module
        (delegates, hooks, skills, servers) when there is one, and in every
        case this attempt's own session record and stream, join the judge's
        own settings, and the agent is made again, so each judgment's runs
        land in its own record (the proof run's churn build, with no
        harness, had judged two attempts and recorded no run)."""
        self.coding_agent_config.agent_specific = staged.merged_agent_specific(self._own_agent_specific)
        self.agent = CodingAgentFactory.create(self.coding_agent_config)

    @property
    def configured_timeout_seconds(self) -> Optional[float]:
        """The block-configured call timeout (None = no deadline)."""
        configured = self.coding_agent_config.agent_specific.get("timeout")
        return None if configured is None else float(configured)
    
    def generate(
        self,
        goal: str,
        idea: str,
        code_changes_summary: str,
        base_branch: str,
        head_branch: str,
        evaluation_script_path: str,
        evaluation_result: str,
        workspace_dir: str,
        axes: str = "",
        session_end_facts: str = "",
        timeout_seconds: Optional[float] = None,
        domain_guidance: str = "",
        verification: str = "",
        retry_deadline: Optional[Callable[[], Tuple[Optional[float], Optional[str]]]] = None,
        campaign_notes: str = "",
    ) -> FeedbackResult:
        """
        Generate feedback for the iteration.
        
        Args:
            goal: The original goal/objective
            idea: The solution approach for this iteration
            code_changes_summary: Summary of code changes from the coding agent
            base_branch: Git branch/commit to diff from (parent branch)
            head_branch: Git branch/commit to diff to (current branch)
            evaluation_script_path: Path to evaluation script in workspace (e.g., kapso_evaluation/evaluate.py)
            evaluation_result: Output from running the evaluation
            workspace_dir: Path to workspace directory (agent can access full code)
            campaign_notes: The rendered section of the person's notes ("" without notes)
            
        Returns:
            FeedbackResult with:
              - stop: bool (whether to stop iteration)
              - evaluation_valid: bool (whether evaluation is fair/correct)
              - feedback: str (actionable feedback for next iteration)
              - score: Optional[float] (extracted evaluation score, if any)
        """
        # Initialize agent with workspace so it can access full code
        self.agent.initialize(workspace_dir)
        
        # Get commit message for head branch
        commit_message = self._get_commit_message(workspace_dir, head_branch)
        
        # Build the prompt
        prompt = self._build_prompt(
            goal=goal,
            idea=idea,
            code_changes_summary=code_changes_summary,
            base_branch=base_branch,
            head_branch=head_branch,
            commit_message=commit_message,
            evaluation_script_path=evaluation_script_path,
            evaluation_result=evaluation_result,
            workspace_dir=workspace_dir,
            axes=axes,
            session_end_facts=session_end_facts,
            domain_guidance=domain_guidance,
            verification=verification,
            campaign_notes=campaign_notes,
        )

        # Run the judge. The agent is persistent across iterations, so
        # this call's spend is the cumulative delta around it (the retry
        # included). The invocation's outcome is inspected before its
        # output: a first run that ended at its deadline is not retried
        # (a retry would get less still); a complete answer with no tags
        # resumes the judge's own session with the nudge, under a room
        # the caller re-computes (`retry_deadline`); a retry cut at its
        # deadline, or still without tags, leaves the attempt unjudged,
        # by reason — never a fabricated verdict.
        call_started = time.monotonic()
        cost_before = self.agent.get_cumulative_cost()
        result = self.agent.generate_code(
            prompt, timeout_seconds=timeout_seconds
        )
        unjudged = self._ended_at_deadline(result)
        if unjudged is None and not result.success:
            # The invocation itself failed (a login revoked, a quota, a
            # crashed CLI): nothing to parse and nothing to resume.
            unjudged = f"the judge's session failed: {result.error or 'no output'}"
        feedback_result = None if unjudged else self._parse_response(result.output)
        if unjudged is None and feedback_result is None:
            cli_session_id = result.metadata.get("cli_session_id")
            if not cli_session_id:
                raise RuntimeError(
                    "the judge's session cannot be resumed for its retry: the adapter "
                    "recorded no cli_session_id (a wiring bug, not a tagless answer)"
                )
            room, not_launched = retry_deadline() if retry_deadline is not None else (timeout_seconds, None)
            if not_launched is not None:
                unjudged = f"no tags, and no room for a retry ({not_launched})"
            else:
                result = self.agent.resume(
                    cli_session_id, self.RETRY_PROMPT, timeout_seconds=room, reason=self.RETRY_REASON,
                )
                unjudged = self._ended_at_deadline(result)
                feedback_result = None if unjudged else self._parse_response(result.output)
                if unjudged is None and feedback_result is None:
                    unjudged = self.NO_TAGS_AFTER_RETRY
        if feedback_result is None:
            feedback_result = FeedbackResult(
                stop=False, evaluation_valid=True, feedback="", score=None, unjudged=unjudged,
            )
        feedback_result.cost_usd = (
            self.agent.get_cumulative_cost() - cost_before
        )
        feedback_result.duration_seconds = time.monotonic() - call_started
        return feedback_result
    
    def _ended_at_deadline(self, result: Any) -> Optional[str]:
        """The deadline reason when the invocation was killed at its
        deadline without completing its turn, None otherwise."""
        metadata = result.metadata
        if metadata.get("deadline_exceeded") and not metadata.get("completed_before_kill"):
            return self.DEADLINE_REASON
        return None

    def _build_prompt(
        self,
        goal: str,
        idea: str,
        code_changes_summary: str,
        base_branch: str,
        head_branch: str,
        commit_message: str,
        evaluation_script_path: str,
        evaluation_result: str,
        workspace_dir: str,
        axes: str = "",
        session_end_facts: str = "",
        domain_guidance: str = "",
        verification: str = "",
        campaign_notes: str = "",
    ) -> str:
        """Build the prompt for the feedback generator."""
        template = load_prompt(self.PROMPT_PATH)
        return render_prompt(
            template,
            {
                "goal": goal,
                "idea": idea,
                "code_changes_summary": code_changes_summary,
                "base_branch": base_branch,
                "head_branch": head_branch,
                "commit_message": commit_message,
                "evaluation_script_path": evaluation_script_path,
                "evaluation_result": evaluation_result,
                "workspace_dir": workspace_dir,
                "axis_frontier": optional_block(
                    render_prompt(load_prompt(self.AXIS_FRONTIER_PROMPT_PATH), {"axes": axes})
                    if axes else ""
                ),
                "session_end_facts": session_end_facts
                or "(not recorded)",
                "domain_guidance": optional_block(domain_guidance),
                "verification": optional_block(verification),
                # Last, so nothing is substituted into the person's own words.
                "campaign_notes": optional_block(campaign_notes),
            }
        )
    
    def _get_commit_message(self, workspace_dir: str, branch: str) -> str:
        """
        Get the commit message for a branch.
        
        Args:
            workspace_dir: Path to the git repository
            branch: Branch name or commit ref
            
        Returns:
            Commit message string, or empty string if not found
        """
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%B", branch],
                cwd=workspace_dir,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            print(f"[FeedbackGenerator] Warning: Could not get commit message: {e}")
        return ""
    
    def _parse_response(self, response: str) -> Optional[FeedbackResult]:
        """
        Parse the agent's response into a FeedbackResult.

        Expects XML tags format from the agent:
        <stop>true/false</stop>
        <evaluation_valid>true/false</evaluation_valid>
        <score>numeric or null</score>
        <feedback>...</feedback>

        Returns None when no tag is present — the caller owns the retry
        and the explicit failure result.
        """
        import re

        # Extract values from XML tags
        def extract_tag(tag: str, text: str) -> Optional[str]:
            pattern = rf'<{tag}>\s*(.*?)\s*</{tag}>'
            match = re.search(pattern, text, re.DOTALL)
            return match.group(1).strip() if match else None
        
        stop_str = extract_tag("stop", response)
        eval_valid_str = extract_tag("evaluation_valid", response)
        score_str = extract_tag("score", response)
        feedback_str = extract_tag("feedback", response)
        cards_str = extract_tag("cards_load_bearing", response)

        # If we found at least some tags, use them
        if any([stop_str, eval_valid_str, score_str, feedback_str]):
            # Parse boolean values
            stop = stop_str.lower() == "true" if stop_str else False
            evaluation_valid = eval_valid_str.lower() != "false" if eval_valid_str else True

            # Parse score
            score = None
            if score_str and score_str.lower() != "null":
                try:
                    score = float(score_str)
                except ValueError:
                    pass

            feedback = feedback_str or ""

            # Citation contract: comma-separated card names; "none" (the
            # prompt's word), "false", "null", an empty tag or no tag mean
            # no served card was load-bearing (also the case whenever
            # serving is off and the tag never appears).
            cards_load_bearing = []
            if cards_str and cards_str.lower() not in ("none", "false", "null"):
                cards_load_bearing = [
                    name.strip().strip("[]").removeprefix("card:")
                    for name in cards_str.split(",")
                    if name.strip()
                ]

            return FeedbackResult(
                stop=stop,
                evaluation_valid=evaluation_valid,
                feedback=feedback,
                score=score,
                cards_load_bearing=cards_load_bearing,
            )

        return None
