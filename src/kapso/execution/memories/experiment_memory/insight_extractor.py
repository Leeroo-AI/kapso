# Insight Extractor
#
# LLM-based generalization of errors and successes into reusable lessons.
#
# Instead of storing raw error messages, we use an LLM to:
# 1. Understand what went wrong
# 2. Generalize into an actionable lesson
# 3. Extract specific conditions when this lesson applies
#
# This produces REUSABLE knowledge, not just error logs.

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from kapso.core.llm import LLMBackend

logger = logging.getLogger(__name__)

# Path to prompt templates
PROMPTS_DIR = Path(__file__).parent / "prompts"


class InsightType(str, Enum):
    """Categories for learned insights."""
    CRITICAL_ERROR = "critical_error"      # Mistakes to avoid
    BEST_PRACTICE = "best_practice"        # Patterns that work


@dataclass
class ExtractedInsight:
    """
    An insight extracted by the LLM from an error or success.
    
    Attributes:
        lesson: The generalized, actionable lesson
        trigger_conditions: When this lesson applies
        suggested_fix: What to do when this occurs
        confidence: How confident the LLM is (0-1)
        insight_type: Category of insight
        original_text: The original error/feedback
        tags: Extracted keywords for retrieval
    """
    lesson: str
    trigger_conditions: str
    suggested_fix: str
    confidence: float
    insight_type: InsightType
    original_text: str
    tags: List[str] = field(default_factory=list)
    
    def to_formatted_string(self) -> str:
        """Format insight for display/storage."""
        return (
            f"{self.lesson}\n"
            f"→ When: {self.trigger_conditions}\n"
            f"→ Fix: {self.suggested_fix}"
        )


class InsightExtractor:
    """
    Extracts generalized, reusable insights from errors and successes.
    
    Instead of storing "ModuleNotFoundError: No module named 'peft'",
    we extract:
    - Lesson: "The 'peft' library must be installed for LoRA operations"
    - Trigger: "When using LoraConfig, get_peft_model, or PEFT classes"
    - Fix: "Run 'pip install peft' before running the script"
    
    This makes experiment history USEFUL for future problems.
    
    Usage:
        extractor = InsightExtractor()
        insight = extractor.extract_from_error(
            error_message="ModuleNotFoundError: No module named 'peft'",
            goal="Fine-tune LLaMA with LoRA",
        )
        print(insight.lesson)
    """
    
    DEFAULT_MODEL = "gpt-4o-mini"
    
    def __init__(
        self,
        llm: Optional["LLMBackend"] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize insight extractor.
        
        Args:
            llm: LLM backend (lazy-loaded if not provided)
            model: Model to use for extraction (default: gpt-4o-mini)
        """
        self._llm = llm
        self.model = model or self.DEFAULT_MODEL
    
    def _get_llm(self) -> "LLMBackend":
        """Lazy-load LLM backend."""
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
        solution: Optional[str] = None,
    ) -> ExtractedInsight:
        """
        Extract a generalized insight from an error.
        
        Args:
            error_message: The raw error message
            goal: What the agent was trying to achieve
            solution: The solution that was attempted (optional)
            
        Returns:
            ExtractedInsight with generalized lesson
        """
        prompt = self._build_error_prompt(error_message, goal, solution)
        
        try:
            response = self._call_llm(prompt)
            return self._parse_insight_response(
                response, 
                error_message, 
                InsightType.CRITICAL_ERROR
            )
        except Exception as e:
            logger.warning(f"Insight extraction failed: {e}")
            return self._fallback_error_insight(error_message)
    
    def extract_from_success(
        self,
        feedback: str,
        score: float,
        goal: str,
        solution: Optional[str] = None,
    ) -> ExtractedInsight:
        """
        Extract a best practice insight from a successful experiment.
        
        Args:
            feedback: Evaluator feedback
            score: How well it was achieved (0-1)
            goal: What was achieved
            solution: The solution that worked (optional)
            
        Returns:
            ExtractedInsight with best practice
        """
        prompt = self._build_success_prompt(feedback, goal, score, solution)
        
        try:
            response = self._call_llm(prompt)
            return self._parse_insight_response(
                response, 
                feedback, 
                InsightType.BEST_PRACTICE
            )
        except Exception as e:
            logger.warning(f"Success insight extraction failed: {e}")
            return self._fallback_success_insight(feedback, score)
    
    def _build_error_prompt(
        self,
        error_message: str,
        goal: str,
        solution: Optional[str],
    ) -> str:
        """Build prompt for error insight extraction."""
        context = f"Goal: {goal}"
        if solution:
            # Truncate solution to avoid huge prompts
            solution_preview = solution[:1000] + "..." if len(solution) > 1000 else solution
            context += f"\nSolution attempted:\n```\n{solution_preview}\n```"
        
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
        solution: Optional[str],
    ) -> str:
        """Build prompt for success insight extraction."""
        context = f"Goal: {goal}\nScore: {score:.2f}"
        if solution:
            solution_preview = solution[:1000] + "..." if len(solution) > 1000 else solution
            context += f"\nSolution:\n```\n{solution_preview}\n```"
        
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
        """Call LLM with JSON mode."""
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
        insight_type: InsightType,
    ) -> ExtractedInsight:
        """Parse LLM response into ExtractedInsight."""
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
                insight_type=insight_type,
                original_text=original,
                tags=data.get("tags", []),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse insight response: {e}")
            if insight_type == InsightType.CRITICAL_ERROR:
                return self._fallback_error_insight(original)
            else:
                return self._fallback_success_insight(original, 0.5)
    
    def _fallback_error_insight(self, error_message: str) -> ExtractedInsight:
        """Create basic insight when LLM extraction fails."""
        # Extract error type from message
        # Check specific patterns first, then general error types
        error_type = "unknown_error"
        error_lower = error_message.lower()
        
        # Check specific patterns first (before general error types)
        if "OOM" in error_message or "out of memory" in error_lower:
            error_type = "memory_error"
        elif "CUDA" in error_message or "cuda" in error_lower:
            error_type = "cuda_error"
        # Then check general error types
        elif "ModuleNotFoundError" in error_message:
            error_type = "missing_module"
        elif "ImportError" in error_message:
            error_type = "import_error"
        elif "SyntaxError" in error_message:
            error_type = "syntax_error"
        elif "TypeError" in error_message:
            error_type = "type_error"
        elif "AttributeError" in error_message:
            error_type = "attribute_error"
        elif "ValueError" in error_message:
            error_type = "value_error"
        elif "KeyError" in error_message:
            error_type = "key_error"
        elif "RuntimeError" in error_message:
            error_type = "runtime_error"
        
        # Truncate long error messages
        lesson = error_message[:500] if len(error_message) > 500 else error_message
        
        return ExtractedInsight(
            lesson=f"Error: {lesson}",
            trigger_conditions=f"When {error_type} occurs",
            suggested_fix="Review the error and fix the underlying issue",
            confidence=0.3,
            insight_type=InsightType.CRITICAL_ERROR,
            original_text=error_message,
            tags=[error_type, "error", "fallback"],
        )
    
    def _fallback_success_insight(self, feedback: str, score: float) -> ExtractedInsight:
        """Create basic insight when LLM extraction fails."""
        # Truncate long feedback
        lesson = feedback[:500] if len(feedback) > 500 else feedback
        
        return ExtractedInsight(
            lesson=f"Success: {lesson}",
            trigger_conditions="Similar goals",
            suggested_fix="Follow the same approach",
            confidence=score * 0.5,
            insight_type=InsightType.BEST_PRACTICE,
            original_text=feedback,
            tags=["success", "fallback"],
        )
