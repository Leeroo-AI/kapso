# Researcher
#
# A wrapper around OpenAI's `web_search` tool for deep public web research.
#
# Design goals:
# - Support three modes: idea, implementation, study
# - Accept single mode or list of modes
# - Return appropriate type based on input
# - Keep prompt templates in markdown files for easy iteration

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Union, overload

from openai import OpenAI

from kapso.knowledge_base.types import Source, ResearchFindings
from kapso.researcher.research_findings import (
    ResearchMode,
    ResearchModeInput,
    parse_idea_results,
    parse_implementation_results,
    parse_study_result,
)

logger = logging.getLogger(__name__)

# Type definitions
ResearchDepth = Literal["light", "deep"]

# Return type for single mode
ResearchResultSingle = Union[List[Source.Idea], List[Source.Implementation], Source.ResearchReport]

# Prompts directory
_PROMPTS_DIR = Path(__file__).parent / "prompts"


@dataclass
class Researcher:
    """
    Deep public web research using OpenAI Responses API + `web_search`.
    
    Supports three research modes:
    - idea: Conceptual understanding, returns List[Source.Idea]
    - implementation: Working code snippets, returns List[Source.Implementation]
    - study: Comprehensive research report, returns Source.ResearchReport
    
    Usage:
        researcher = Researcher()
        
        # Single mode - returns List[Source.Idea]
        ideas = researcher.research("How to fine-tune LLMs?", mode="idea", top_k=5)
        
        # Multiple modes - returns ResearchFindings
        findings = researcher.research("LLM fine-tuning", mode=["idea", "implementation"], top_k=5)
        for idea in findings.ideas:
            print(idea.to_string())
        for impl in findings.implementations:
            print(impl.to_string())
    """

    # Model choice (internal, not exposed in public API)
    model: str = "gpt-5.2"

    def __post_init__(self) -> None:
        # Create client once per instance
        self._client = OpenAI()

    def research(
        self,
        query: str,
        *,
        mode: ResearchModeInput,
        top_k: int = 5,
        depth: ResearchDepth = "deep",
    ) -> Union[List[Source.Idea], List[Source.Implementation], Source.ResearchReport, ResearchFindings]:
        """
        Run deep web research.
        
        Args:
            query: What we want to learn from public sources.
            mode: Research mode (required). Can be:
                - Single mode: "idea", "implementation", or "study"
                - List of modes: ["idea", "implementation"]
            top_k: Maximum number of results (for idea/implementation modes).
            depth: Research depth ("light" or "deep").
        
        Returns:
            - List[Source.Idea] if mode="idea"
            - List[Source.Implementation] if mode="implementation"
            - Source.ResearchReport if mode="study"
            - ResearchFindings if mode is a list
        """
        # Validate query
        query = (query or "").strip()
        if not query:
            raise ValueError("query must be a non-empty string")
        
        # Normalize mode to list
        modes = self._normalize_modes(mode)
        
        # Map depth to reasoning effort
        reasoning_effort = self._get_reasoning_effort(depth)
        
        # Single mode - return direct type
        if len(modes) == 1:
            logger.info(f"Running research in '{modes[0]}' mode for: {query[:50]}...")
            return self._run_single_mode(
                query=query,
                mode=modes[0],
                top_k=top_k,
                reasoning_effort=reasoning_effort,
            )
        
        # Multiple modes - return ResearchFindings
        findings = ResearchFindings(query=query)
        
        for m in modes:
            logger.info(f"Running research in '{m}' mode for: {query[:50]}...")
            result = self._run_single_mode(
                query=query,
                mode=m,
                top_k=top_k,
                reasoning_effort=reasoning_effort,
            )
            
            if m == "idea":
                findings.ideas = result
            elif m == "implementation":
                findings.implementations = result
            else:  # study
                findings.report = result
        
        return findings

    def _normalize_modes(self, mode: ResearchModeInput) -> List[ResearchMode]:
        """Normalize mode input to a list of modes."""
        if isinstance(mode, str):
            if mode not in ("idea", "implementation", "study"):
                raise ValueError(f"Invalid mode: {mode}. Must be 'idea', 'implementation', or 'study'")
            return [mode]
        elif isinstance(mode, list):
            for m in mode:
                if m not in ("idea", "implementation", "study"):
                    raise ValueError(f"Invalid mode: {m}. Must be 'idea', 'implementation', or 'study'")
            return mode
        else:
            raise ValueError(f"mode must be a string or list (got {type(mode)})")

    def _get_reasoning_effort(self, depth: ResearchDepth) -> str:
        """Map depth to OpenAI reasoning effort."""
        if depth == "light":
            return "medium"
        elif depth == "deep":
            return "high"
        else:
            raise ValueError(f"depth must be 'light' or 'deep' (got {depth!r})")

    def _run_single_mode(
        self,
        query: str,
        mode: ResearchMode,
        top_k: int,
        reasoning_effort: str,
    ) -> ResearchResultSingle:
        """
        Run research for a single mode.
        
        Args:
            query: The research query
            mode: The mode to run
            top_k: Max results (for idea/implementation modes)
            reasoning_effort: OpenAI reasoning effort level
            
        Returns:
            List[Source.Idea], List[Source.Implementation], or Source.ResearchReport based on mode
        """
        # Build prompt
        prompt = self._build_research_prompt(query=query, mode=mode, top_k=top_k)
        
        try:
            # Build request params
            # Note: reasoning.effort is only supported by certain models (e.g., o1, o3)
            request_params = {
                "model": self.model,
                "tools": [{"type": "web_search"}],
                "input": prompt,
                "max_output_tokens": 32000,
            }
            
            # Only add reasoning for models that support it
            if self.model.startswith("o1") or self.model.startswith("o3"):
                request_params["reasoning"] = {"effort": reasoning_effort}
            
            response = self._client.responses.create(**request_params)
            raw_text = response.output_text or ""
        except Exception as e:
            logger.exception(f"Research failed for mode '{mode}': {e}")
            # Return empty results on error
            if mode == "idea":
                return []
            elif mode == "implementation":
                return []
            else:  # study
                return ResearchReport(query=query, content="")
        
        # Parse based on mode
        if mode == "idea":
            return parse_idea_results(raw_text, query)
        elif mode == "implementation":
            return parse_implementation_results(raw_text, query)
        else:  # study
            return parse_study_result(raw_text, query)

    def _build_research_prompt(
        self,
        *,
        query: str,
        mode: ResearchMode,
        top_k: int,
    ) -> str:
        """
        Build the full prompt for a research request.
        
        Combines the envelope template with mode-specific instructions.
        """
        # Load mode-specific instructions
        mode_instructions = self._load_mode_instructions(mode)
        
        # Load envelope template
        envelope_path = _PROMPTS_DIR / "research_envelope.md"
        if not envelope_path.exists():
            raise FileNotFoundError(f"Missing research envelope prompt file: {envelope_path}")
        envelope_template = envelope_path.read_text(encoding="utf-8")
        
        # Format the prompt
        return envelope_template.format(
            query=query,
            mode=mode,
            top_k=top_k,
            mode_instructions=mode_instructions,
        )

    def _load_mode_instructions(self, mode: ResearchMode) -> str:
        """
        Load mode-specific instruction block from markdown file.
        """
        path = _PROMPTS_DIR / f"{mode}.md"
        if not path.exists():
            raise FileNotFoundError(f"Missing prompt file for mode '{mode}': {path}")
        return path.read_text(encoding="utf-8").strip()
