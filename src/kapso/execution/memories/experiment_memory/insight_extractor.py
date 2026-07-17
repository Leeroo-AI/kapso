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
    
    DEFAULT_MODEL = "utility"
    
    def __init__(
        self,
        llm: Optional["LLMBackend"] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize insight extractor.
        
        Args:
            llm: LLM backend (lazy-loaded if not provided)
            model: Explicit model or role (default: utility)
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
    
    def extract(
        self,
        technical_difficulties: str,
        feedback: str,
        score: Optional[float],
        goal: str,
        solution: Optional[str] = None,
    ) -> ExtractedInsight:
        """Extract one generalized insight from a finished experiment.

        Runs unconditionally for every node — no score threshold and no
        success/error branching. The primary source is the implementor's
        own technical_difficulties report (the process record); the judge's
        feedback and the outcome provide the result context. The model
        classifies the insight_type (critical_error vs best_practice) in
        its JSON output.
        """
        prompt = self._build_prompt(
            technical_difficulties, feedback, score, goal, solution
        )

        try:
            response = self._call_llm(prompt)
            return self._parse_insight_response(
                response,
                technical_difficulties or feedback,
            )
        except Exception as e:
            logger.warning(f"Insight extraction failed: {e}")
            return self._fallback_insight(technical_difficulties or feedback)

    def _build_prompt(
        self,
        technical_difficulties: str,
        feedback: str,
        score: Optional[float],
        goal: str,
        solution: Optional[str],
    ) -> str:
        """Build the single extraction prompt."""
        context = f"Goal: {goal}"
        if score is not None:
            context += f"\nFinal score: {score}"
        if solution:
            # Truncate solution to avoid huge prompts
            solution_preview = (
                solution[:1000] + "..." if len(solution) > 1000 else solution
            )
            context += f"\nSolution attempted:\n```\n{solution_preview}\n```"

        template = self._load_prompt("extract_insight.md")
        if template is None:
            raise FileNotFoundError(
                "extract_insight.md prompt template is missing"
            )
        return template.format(
            context=context,
            technical_difficulties=technical_difficulties or "(none reported)",
            feedback=feedback or "(no feedback)",
        )

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
    ) -> ExtractedInsight:
        """Parse LLM response into ExtractedInsight (type comes from the JSON)."""
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
                insight_type=InsightType(data["insight_type"]),
                original_text=original,
                tags=data.get("tags", []),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse insight response: {e}")
            return self._fallback_insight(original)

    def _fallback_insight(self, source_text: str) -> ExtractedInsight:
        """Degraded insight when LLM extraction or parsing fails: keep the
        head of the source material so the lesson is retrievable at all."""
        lesson = source_text[:500] if len(source_text) > 500 else source_text
        return ExtractedInsight(
            lesson=lesson or "No difficulty or feedback material available",
            trigger_conditions="Similar goals",
            suggested_fix="Review the original experiment record",
            confidence=0.3,
            insight_type=InsightType.BEST_PRACTICE,
            original_text=source_text,
            tags=["fallback"],
        )
