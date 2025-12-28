# =============================================================================
# Decision Maker - LLM-based decisions for workflow navigation
# =============================================================================
#
# ALL decisions are made by the LLM.
# The LLM receives THE SAME context that goes to the coding agent.
#
# OUTPUT PARSING (handles any LLM, with or without JSON mode):
# 1. Try JSON mode if model supports it
# 2. Extract JSON with regex (handles text around JSON)
# 3. Retry with correction prompt if parsing fails
# 4. Final fallback: infer from keywords
#
# =============================================================================

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from src.memory.context import CognitiveContext
    from src.core.llm import LLMBackend

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"

# Regex to extract JSON object from text
JSON_PATTERN = re.compile(r'\{[^{}]*\}', re.DOTALL)

# Models known to support JSON mode
JSON_MODE_MODELS: Set[str] = {
    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo",
    "claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229",
    # Add more as needed
}


class WorkflowAction(str, Enum):
    """
    Actions the agent can take after an experiment.
    
    SIMPLIFIED: No step-level actions (ADVANCE, SKIP) since agent
    implements full solution in one go. Actions are iteration-level:
    - RETRY: Try again with same workflow (agent sees error feedback)
    - PIVOT: Get different workflow from KG  
    - COMPLETE: Goal achieved, done
    """
    RETRY = "RETRY"      # Try again with current workflow
    PIVOT = "PIVOT"      # Switch to different workflow
    COMPLETE = "COMPLETE"  # Goal achieved


@dataclass
class ActionDecision:
    """Result of an action decision."""
    action: WorkflowAction
    reasoning: str
    confidence: float
    raw_response: str = ""


class DecisionMaker:
    """
    LLM-based decision maker for workflow navigation.
    
    Works with ANY LLM - uses JSON mode when available, falls back to
    parsing strategies when not.
    """
    
    DEFAULT_MODEL = "gpt-4o-mini"
    FALLBACK_MODELS = ["gpt-4o", "claude-3-5-sonnet-20241022"]
    
    def __init__(
        self,
        llm: Optional["LLMBackend"] = None,
        model: Optional[str] = None,
    ):
        self._llm = llm
        self.model = model or self.DEFAULT_MODEL
        self._action_prompt = self._load_prompt("decide_action.md")
    
    def _get_llm(self) -> "LLMBackend":
        if self._llm is None:
            from src.core.llm import LLMBackend
            self._llm = LLMBackend()
        return self._llm
    
    def _load_prompt(self, filename: str) -> str:
        path = PROMPTS_DIR / filename
        if path.exists():
            return path.read_text()
        logger.warning(f"Prompt file not found: {path}")
        return ""
    
    def _supports_json_mode(self, model: str) -> bool:
        """Check if model supports JSON mode."""
        # Check exact match first
        if model in JSON_MODE_MODELS:
            return True
        # Check prefix match (e.g., "gpt-4o-2024-05-13" matches "gpt-4o")
        for known in JSON_MODE_MODELS:
            if model.startswith(known):
                return True
        return False
    
    # =========================================================================
    # Main Decision
    # =========================================================================
    
    def decide_action(self, context: "CognitiveContext") -> ActionDecision:
        """Decide what action to take based on current context."""
        logger.info("=== LLM DECISION MAKING ===")
        context_str = context.render()
        prompt = self._action_prompt.replace("{context}", context_str)
        
        # Log key context info
        logger.info(f"Context for decision:")
        logger.info(f"  Iteration: {context.iteration}")
        if context.last_experiment:
            exp = context.last_experiment
            logger.info(f"  Last experiment: {'SUCCESS' if exp.success else 'FAILED'}, score={exp.score}")
            if exp.feedback:
                logger.info(f"  Feedback: {exp.feedback[:100]}...")
        if context.workflow and context.workflow.current_step:
            step = context.workflow.current_step
            logger.info(f"  Current step: {step.number}. {step.title} (attempts: {step.attempts})")
        
        logger.debug(f"Decision prompt length: {len(prompt)} chars")
        
        # Call LLM (with JSON mode if supported)
        response = self._call_llm_smart(prompt)
        
        # Parse response
        decision = self._parse_action_response(response, prompt)
        
        logger.info(f"Decision: {decision.action.value} (confidence: {decision.confidence:.2f})")
        logger.info(f"Reasoning: {decision.reasoning}")
        
        return decision
    
    # =========================================================================
    # Smart LLM Call - uses JSON mode when available
    # =========================================================================
    
    def _call_llm_smart(self, prompt: str) -> str:
        """
        Call LLM with JSON mode if supported, otherwise regular call.
        
        This method is model-agnostic and gracefully handles:
        - Models with JSON mode support
        - Models without JSON mode support
        - JSON mode failures (falls back to regular call)
        """
        llm = self._get_llm()
        
        models_to_try = [self.model] + self.FALLBACK_MODELS
        last_error = None
        
        for model in models_to_try:
            try:
                # Try JSON mode if model supports it
                if self._supports_json_mode(model):
                    try:
                        response = llm.llm_completion(
                            model=model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.2,
                            response_format={"type": "json_object"},
                        )
                        return response
                    except Exception as e:
                        # JSON mode failed, try regular call
                        logger.debug(f"JSON mode failed for {model}: {e}")
                
                # Regular call (no JSON mode)
                response = llm.llm_completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                return response
                
            except Exception as e:
                logger.warning(f"LLM call failed with {model}: {e}")
                last_error = e
                continue
        
        raise RuntimeError(f"All LLM models failed. Last error: {last_error}")
    
    # =========================================================================
    # Robust Parsing (works for any LLM output)
    # =========================================================================
    
    def _parse_action_response(self, response: str, original_prompt: str) -> ActionDecision:
        """
        Parse LLM response with multiple fallback strategies.
        
        Works for:
        - Clean JSON (from JSON mode)
        - JSON in markdown code blocks
        - JSON mixed with text
        - Plain text (keyword inference)
        """
        raw = response
        
        # Strategy 1: Direct JSON parse
        parsed = self._try_parse_json(response)
        if parsed:
            return self._json_to_decision(parsed, raw)
        
        # Strategy 2: Extract from markdown code blocks
        cleaned = self._extract_from_code_block(response)
        if cleaned:
            parsed = self._try_parse_json(cleaned)
            if parsed:
                return self._json_to_decision(parsed, raw)
        
        # Strategy 3: Extract JSON with regex
        match = JSON_PATTERN.search(response)
        if match:
            parsed = self._try_parse_json(match.group())
            if parsed:
                return self._json_to_decision(parsed, raw)
        
        # Strategy 4: Retry with correction prompt
        logger.warning("Failed to parse response, retrying with correction")
        corrected = self._retry_with_correction(response)
        if corrected:
            parsed = self._try_parse_json(corrected)
            if parsed:
                return self._json_to_decision(parsed, raw + " -> " + corrected)
        
        # Strategy 5: Infer from keywords
        logger.warning("All parsing failed, inferring from text")
        return self._infer_action_from_text(response)
    
    def _try_parse_json(self, text: str) -> Optional[dict]:
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return None
    
    def _extract_from_code_block(self, response: str) -> Optional[str]:
        pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
        return None
    
    def _json_to_decision(self, data: dict, raw: str) -> ActionDecision:
        try:
            action_str = data.get("action", "RETRY").upper()
            action = WorkflowAction(action_str)
            return ActionDecision(
                action=action,
                reasoning=data.get("reasoning", "No reasoning provided"),
                confidence=float(data.get("confidence", 0.5)),
                raw_response=raw,
            )
        except (ValueError, KeyError) as e:
            logger.warning(f"Invalid action value: {e}")
            return ActionDecision(
                action=WorkflowAction.RETRY,
                reasoning=f"Parse error, defaulting to RETRY: {data}",
                confidence=0.3,
                raw_response=raw,
            )
    
    def _retry_with_correction(self, bad_response: str) -> Optional[str]:
        """Ask LLM to fix malformed response."""
        correction_prompt = f"""Your previous response was not valid JSON:

{bad_response[:500]}

Respond ONLY with valid JSON:
{{"action": "RETRY|PIVOT|COMPLETE", "reasoning": "why", "confidence": 0.8}}"""
        
        try:
            # Use regular call (no JSON mode) to avoid recursion issues
            llm = self._get_llm()
            return llm.llm_completion(
                model=self.model,
                messages=[{"role": "user", "content": correction_prompt}],
                temperature=0.1,
            )
        except Exception as e:
            logger.warning(f"Correction retry failed: {e}")
            return None
    
    def _infer_action_from_text(self, response: str) -> ActionDecision:
        """Last resort: infer action from keywords."""
        r = response.lower()
        
        if any(w in r for w in ["complete", "done", "finished", "goal achieved", "success"]):
            action = WorkflowAction.COMPLETE
        elif any(w in r for w in ["pivot", "different approach", "change workflow", "abandon", "try different"]):
            action = WorkflowAction.PIVOT
        else:
            # Default: retry with current workflow
            action = WorkflowAction.RETRY
        
        return ActionDecision(
            action=action,
            reasoning=f"Inferred from text: {response[:100]}...",
            confidence=0.2,
            raw_response=response,
        )
