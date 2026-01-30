# =============================================================================
# Insight Extractor - LLM-based generalization of errors into lessons
# =============================================================================
#
# Instead of storing raw error messages, we use an LLM to:
# 1. Understand what went wrong
# 2. Generalize into an actionable lesson
# 3. Extract specific conditions when this lesson applies
#
# This produces REUSABLE knowledge, not just error logs.
#
# Prompts:
# - Loaded from external files in src/memory/prompts/
# - extract_error_insight.md - for error generalization
# - extract_success_insight.md - for best practice extraction
#
# =============================================================================

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from kapso.core.llm import LLMBackend

logger = logging.getLogger(__name__)

# Path to prompt templates
PROMPTS_DIR = Path(__file__).parent / "prompts"


@dataclass
class ExtractedInsight:
    """
    An insight extracted by the LLM from an error or success.
    
    Attributes:
        lesson: The generalized, actionable lesson
        trigger_conditions: When this lesson applies
        suggested_fix: What to do when this occurs
        confidence: How confident the LLM is (0-1)
        original_error: The original error/feedback
        tags: Extracted keywords for retrieval
    """
    lesson: str
    trigger_conditions: str
    suggested_fix: str
    confidence: float
    original_error: str
    tags: List[str]


class InsightExtractor:
    """
    Extracts generalized, reusable insights from errors and successes.
    
    Instead of storing "ModuleNotFoundError: No module named 'peft'",
    we extract:
    - Lesson: "The 'peft' library must be installed for LoRA operations"
    - Trigger: "When using LoraConfig, get_peft_model, or PEFT classes"
    - Fix: "Run 'pip install peft' before running the script"
    
    This makes episodic memory USEFUL for future problems.
    """
    
    DEFAULT_MODEL = "gpt-4o-mini"
    
    def __init__(
        self,
        llm: Optional["LLMBackend"] = None,
        model: Optional[str] = None,
    ):
        self._llm = llm
        self.model = model or self.DEFAULT_MODEL
    
    def _get_llm(self) -> "LLMBackend":
        if self._llm is None:
            from kapso.core.llm import LLMBackend
            self._llm = LLMBackend()
        return self._llm
    
    def _load_prompt(self, filename: str) -> Optional[str]:
        """Load prompt template from external file."""
        path = PROMPTS_DIR / filename
        if path.exists():
            return path.read_text()
        return None
    
    def extract_from_error(
        self,
        error_message: str,
        goal: str,
        current_step: Optional[str] = None,
        code_snippet: Optional[str] = None,
    ) -> ExtractedInsight:
        """
        Extract a generalized insight from an error.
        
        Args:
            error_message: The raw error message
            goal: What the agent was trying to achieve
            current_step: The workflow step being attempted
            code_snippet: Relevant code that caused the error
            
        Returns:
            ExtractedInsight with generalized lesson
        """
        prompt = self._build_error_prompt(error_message, goal, current_step, code_snippet)
        
        try:
            response = self._call_llm(prompt)
            return self._parse_insight_response(response, error_message, is_error=True)
        except Exception as e:
            logger.warning(f"Insight extraction failed: {e}")
            # Fallback: create basic insight from error
            return self._fallback_error_insight(error_message)
    
    def extract_from_success(
        self,
        feedback: str,
        goal: str,
        score: float,
        current_step: Optional[str] = None,
        solution_summary: Optional[str] = None,
    ) -> ExtractedInsight:
        """
        Extract a best practice insight from a successful experiment.
        
        Args:
            feedback: Evaluator feedback
            goal: What was achieved
            score: How well it was achieved (0-1)
            current_step: The workflow step completed
            solution_summary: Summary of what worked
            
        Returns:
            ExtractedInsight with best practice
        """
        prompt = self._build_success_prompt(feedback, goal, score, current_step, solution_summary)
        
        try:
            response = self._call_llm(prompt)
            return self._parse_insight_response(response, feedback, is_error=False)
        except Exception as e:
            logger.warning(f"Success insight extraction failed: {e}")
            return self._fallback_success_insight(feedback, score)
    
    def _build_error_prompt(
        self,
        error_message: str,
        goal: str,
        current_step: Optional[str],
        code_snippet: Optional[str],
    ) -> str:
        """Build prompt for error insight extraction."""
        context = f"Goal: {goal}"
        if current_step:
            context += f"\nCurrent step: {current_step}"
        if code_snippet:
            context += f"\nCode:\n```\n{code_snippet}\n```"
        
        # Try external template first
        template = self._load_prompt("extract_error_insight.md")
        if template:
            return template.format(
                context=context,
                error_message=error_message,
            )
        
        # Fallback: inline prompt
        return f"""You are extracting reusable lessons from coding errors.

## Context
{context}

## Error
{error_message}

## Task
Extract a GENERALIZED, REUSABLE lesson from this error.
Don't just repeat the error - explain what went wrong and how to prevent it.

Respond in JSON:
{{
  "lesson": "A general principle that applies beyond this specific case",
  "trigger_conditions": "When/where this issue typically occurs",
  "suggested_fix": "Actionable steps to fix or prevent this",
  "confidence": 0.0-1.0,
  "tags": ["keyword1", "keyword2", "keyword3"]
}}

Make the lesson USEFUL for future similar problems.
Respond ONLY with JSON."""
    
    def _build_success_prompt(
        self,
        feedback: str,
        goal: str,
        score: float,
        current_step: Optional[str],
        solution_summary: Optional[str],
    ) -> str:
        """Build prompt for success insight extraction."""
        context = f"Goal: {goal}\nScore: {score:.2f}"
        if current_step:
            context += f"\nCompleted step: {current_step}"
        if solution_summary:
            context += f"\nSolution approach: {solution_summary}"
        
        # Try external template first
        template = self._load_prompt("extract_success_insight.md")
        if template:
            return template.format(
                context=context,
                feedback=feedback,
            )
        
        # Fallback: inline prompt
        return f"""You are extracting best practices from successful code solutions.

## Context
{context}

## Evaluator Feedback
{feedback}

## Task
Extract a REUSABLE best practice from this success.
What made this solution work well? What pattern should be repeated?

Respond in JSON:
{{
  "lesson": "A best practice or pattern that worked well",
  "trigger_conditions": "When to apply this pattern",
  "suggested_fix": "How to implement this pattern",
  "confidence": 0.0-1.0,
  "tags": ["keyword1", "keyword2", "keyword3"]
}}

Focus on PATTERNS that transfer to other problems.
Respond ONLY with JSON."""
    
    def _call_llm(self, prompt: str) -> str:
        llm = self._get_llm()
        return llm.llm_completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
    
    def _parse_insight_response(
        self,
        response: str,
        original: str,
        is_error: bool,
    ) -> ExtractedInsight:
        import json
        import re
        
        # Try to parse JSON
        try:
            # Handle markdown code blocks
            if "```" in response:
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
                if match:
                    response = match.group(1)
            
            data = json.loads(response.strip())
            
            return ExtractedInsight(
                lesson=data.get("lesson", "Unknown lesson"),
                trigger_conditions=data.get("trigger_conditions", "Unknown"),
                suggested_fix=data.get("suggested_fix", "Unknown"),
                confidence=float(data.get("confidence", 0.5)),
                original_error=original,
                tags=data.get("tags", []),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse insight response: {e}")
            if is_error:
                return self._fallback_error_insight(original)
            else:
                return self._fallback_success_insight(original, 0.5)
    
    def _fallback_error_insight(self, error_message: str) -> ExtractedInsight:
        """Create basic insight when LLM extraction fails."""
        # Extract error type
        error_type = "unknown_error"
        if "ModuleNotFoundError" in error_message:
            error_type = "missing_module"
        elif "ImportError" in error_message:
            error_type = "import_error"
        elif "SyntaxError" in error_message:
            error_type = "syntax_error"
        elif "TypeError" in error_message:
            error_type = "type_error"
        elif "AttributeError" in error_message:
            error_type = "attribute_error"
        
        return ExtractedInsight(
            lesson=f"Error occurred: {error_message}",
            trigger_conditions=f"When {error_type} happens",
            suggested_fix="Review the error and fix the underlying issue",
            confidence=0.3,
            original_error=error_message,
            tags=[error_type, "error", "fallback"],
        )
    
    def _fallback_success_insight(self, feedback: str, score: float) -> ExtractedInsight:
        """Create basic insight when LLM extraction fails."""
        return ExtractedInsight(
            lesson=f"Success: {feedback}",
            trigger_conditions="Similar goals",
            suggested_fix="Follow the same approach",
            confidence=score * 0.5,
            original_error=feedback,
            tags=["success", "fallback"],
        )

