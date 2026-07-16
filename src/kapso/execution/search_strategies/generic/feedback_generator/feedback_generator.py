# Feedback Generator
#
# LLM-based feedback generator that validates evaluation and decides stop/continue.
#
# The feedback generator is a coding agent that can be any of the coding agents
# in src/execution/coding_agents/ (aider, gemini, claude_code, openhands).
# Default is claude_code with Bedrock.
#
# It is responsible for:
# 1. Validating that the evaluation is fair and correct
# 2. Checking if the goal has been achieved
# 3. Extracting the evaluation score (if any)
# 4. Generating actionable feedback for the next iteration

import os
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

from kapso.execution.coding_agents.factory import CodingAgentFactory
from kapso.execution.coding_agents.base import CodingAgentConfig
from kapso.core.prompt_loader import load_prompt, render_prompt


@dataclass
class FeedbackResult:
    """Result from feedback generator."""
    stop: bool                      # Whether to stop iteration
    evaluation_valid: bool          # Whether evaluation is fair/correct
    feedback: str                   # Actionable feedback for next iteration
    score: Optional[float] = None   # Extracted evaluation score (if any)
    # Budget telemetry for the feedback agent call itself.
    cost_usd: float = 0.0
    duration_seconds: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "stop": self.stop,
            "evaluation_valid": self.evaluation_valid,
            "feedback": self.feedback,
            "score": self.score,
            "cost_usd": self.cost_usd,
            "duration_seconds": self.duration_seconds,
        }


class FeedbackGenerator:
    """
    LLM-based feedback generator.
    
    Validates evaluation and decides stop/continue.
    
    NOTE 1: The feedback generator is a coding agent that can be any of the
    coding agents in src/execution/coding_agents/ (aider, gemini, claude_code,
    openhands). Default is claude_code with Bedrock.
    
    NOTE 2: The feedback generator is also responsible for extracting the exact
    value of the evaluation score (if any) from the evaluation output and
    returning it in the FeedbackResult.
    """
    
    # Default prompt path
    PROMPT_PATH = "execution/search_strategies/generic/feedback_generator/prompts/feedback_generator.md"

    # Single source for the module's default call timeout; live runs get
    # theirs from the mode config's feedback_generator block.
    DEFAULT_TIMEOUT_SECONDS = 120.0

    RETRY_PROMPT = (
        "Your previous response did not contain the required tags. Respond "
        "now with ONLY the four tags — <stop>, <evaluation_valid>, <score>, "
        "<feedback> — filled in for the iteration you just analyzed. No "
        "other text."
    )
    
    def __init__(
        self,
        coding_agent_config: Optional[CodingAgentConfig] = None,
    ):
        """
        Initialize feedback generator.
        
        Args:
            coding_agent_config: Config for the coding agent. If None, defaults to claude_code with Bedrock.
        """
        # Default to claude_code with Bedrock if no config provided
        if coding_agent_config is None:
            coding_agent_config = CodingAgentConfig(
                agent_type="claude_code",
                model="us.anthropic.claude-opus-4-5-20251101-v1:0",
                debug_model="us.anthropic.claude-opus-4-5-20251101-v1:0",
                agent_specific={
                    "auth_mode": "bedrock",
                    "aws_region": os.environ.get("AWS_REGION", "us-east-1"),
                    "streaming": True,
                    "timeout": self.DEFAULT_TIMEOUT_SECONDS,
                }
            )

        self.coding_agent_config = coding_agent_config

        # Create the coding agent
        self.agent = CodingAgentFactory.create(coding_agent_config)

    @property
    def configured_timeout_seconds(self) -> float:
        """The block-configured call timeout, for budget clamping."""
        return float(
            self.coding_agent_config.agent_specific.get(
                "timeout", self.DEFAULT_TIMEOUT_SECONDS
            )
        )
    
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
        session_end_facts: str = "",
        timeout_seconds: Optional[float] = None,
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
            session_end_facts=session_end_facts,
        )
        
        # Run the coding agent to analyze and generate feedback. The agent is
        # persistent across iterations, so this call's spend is the cumulative
        # delta around it (retry included).
        call_started = time.monotonic()
        cost_before = self.agent.get_cumulative_cost()
        result = self.agent.generate_code(
            prompt, timeout_seconds=timeout_seconds
        )
        response = result.output

        feedback_result = self._parse_response(response)
        if feedback_result is None:
            # One in-session nudge: the judge failing to emit tags loses
            # the whole iteration's verdict, so a cheap retry earns its
            # cost. Still tagless after that -> an explicit failure result,
            # never a fabricated verdict.
            result = self.agent.generate_code(
                self.RETRY_PROMPT, timeout_seconds=timeout_seconds
            )
            response = result.output
            feedback_result = self._parse_response(response)
        if feedback_result is None:
            feedback_result = FeedbackResult(
                stop=False,
                evaluation_valid=True,
                feedback=(
                    "Feedback generation failed: the required tags were "
                    "missing after a retry.\n\nRaw output:\n" + response
                ),
                score=None,
            )
        feedback_result.cost_usd = (
            self.agent.get_cumulative_cost() - cost_before
        )
        feedback_result.duration_seconds = time.monotonic() - call_started
        return feedback_result
    
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
        session_end_facts: str = "",
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
                "session_end_facts": session_end_facts
                or "(not recorded)",
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
            
            return FeedbackResult(
                stop=stop,
                evaluation_valid=evaluation_valid,
                feedback=feedback,
                score=score,
            )
        
        return None
