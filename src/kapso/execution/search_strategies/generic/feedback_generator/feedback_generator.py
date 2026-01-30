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

import json
import os
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
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "stop": self.stop,
            "evaluation_valid": self.evaluation_valid,
            "feedback": self.feedback,
            "score": self.score,
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
                    "use_bedrock": True,
                    "aws_region": os.environ.get("AWS_REGION", "us-east-1"),
                    "streaming": True,
                    "timeout": 120,
                }
            )
        
        self.coding_agent_config = coding_agent_config
        
        # Create the coding agent
        self.agent = CodingAgentFactory.create(coding_agent_config)
    
    def generate(
        self,
        goal: str,
        idea: str,
        code_diff: str,
        evaluation_script_path: str,
        evaluation_result: str,
        workspace_dir: str,
    ) -> FeedbackResult:
        """
        Generate feedback for the iteration.
        
        Args:
            goal: The original goal/objective
            idea: The solution approach for this iteration
            code_diff: Git diff of implementation changes
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
        
        # Build the prompt
        prompt = self._build_prompt(
            goal=goal,
            idea=idea,
            code_diff=code_diff,
            evaluation_script_path=evaluation_script_path,
            evaluation_result=evaluation_result,
            workspace_dir=workspace_dir,
        )
        
        # Run the coding agent to analyze and generate feedback
        result = self.agent.generate_code(prompt)
        response = result.output
        
        # Parse the response - expect strict JSON
        return self._parse_response(response)
    
    def _build_prompt(
        self,
        goal: str,
        idea: str,
        code_diff: str,
        evaluation_script_path: str,
        evaluation_result: str,
        workspace_dir: str,
    ) -> str:
        """Build the prompt for the feedback generator."""
        template = load_prompt(self.PROMPT_PATH)
        return render_prompt(
            template,
            {
                "goal": goal,
                "idea": idea,
                "code_diff": code_diff,
                "evaluation_script_path": evaluation_script_path,
                "evaluation_result": evaluation_result,
                "workspace_dir": workspace_dir,
            }
        )
    
    def _parse_response(self, response: str) -> FeedbackResult:
        """
        Parse the agent's response into a FeedbackResult.
        
        Expects XML tags format from the agent:
        <stop>true/false</stop>
        <evaluation_valid>true/false</evaluation_valid>
        <score>numeric or null</score>
        <feedback>...</feedback>
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
            
            print(f"[FeedbackGenerator] Extracted from XML tags: stop={stop}, score={score}")
            return FeedbackResult(
                stop=stop,
                evaluation_valid=evaluation_valid,
                feedback=feedback,
                score=score,
            )
        
        # Fallback: try JSON parsing for backward compatibility
        return self._parse_response_json_fallback(response)
    
    def _parse_response_json_fallback(self, response: str) -> FeedbackResult:
        """
        Fallback JSON parsing for backward compatibility.
        """
        try:
            # Look for JSON block in markdown code fence
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            else:
                json_str = response.strip()
            
            data = json.loads(json_str)
            
            print(f"[FeedbackGenerator] Extracted from JSON (fallback)")
            return FeedbackResult(
                stop=data.get("stop", False),
                evaluation_valid=data.get("evaluation_valid", True),
                feedback=data.get("feedback", ""),
                score=data.get("score"),
            )
        except (json.JSONDecodeError, KeyError) as e:
            # If parsing fails, return a default result with the raw response as feedback
            print(f"[FeedbackGenerator] Warning: Failed to parse response: {e}")
            return FeedbackResult(
                stop=False,
                evaluation_valid=True,
                feedback=f"Failed to parse feedback response. Raw output: {response[:500]}",
                score=None,
            )
