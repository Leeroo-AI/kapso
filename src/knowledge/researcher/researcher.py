# Researcher
#
# A wrapper around OpenAI's `web_search` tool for deep public web research.
#
# Design goals:
# - Support three modes: idea, implementation, study
# - Accept mode as string or list (default: all three modes)
# - Return structured ResearchFindings with parsed results
# - Keep prompt templates in markdown files for easy iteration

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Union

from openai import OpenAI

from src.knowledge.researcher.research_findings import (
    ResearchFindings,
    ResearchReport,
    IdeaResult,
    ImplementationResult,
    parse_research_result,
    merge_findings,
)

logger = logging.getLogger(__name__)

# Type definitions
ResearchMode = Literal["idea", "implementation", "study"]
ResearchModeInput = Union[ResearchMode, List[ResearchMode]]
ResearchDepth = Literal["light", "deep"]

# Default modes when none specified
DEFAULT_MODES: List[ResearchMode] = ["idea", "implementation", "study"]

# Prompts directory
_PROMPTS_DIR = Path(__file__).parent / "prompts"


@dataclass
class Researcher:
    """
    Deep public web research using OpenAI Responses API + `web_search`.
    
    Supports three research modes:
    - idea: Conceptual understanding, returns List[IdeaResult]
    - implementation: Working code snippets, returns List[ImplementationResult]
    - study: Comprehensive research report, returns ResearchReport
    
    Usage:
        researcher = Researcher()
        
        # Default: all three modes
        findings = researcher.research("How to fine-tune LLMs?", top_k=5)
        
        # Single mode
        findings = researcher.research("RAG", mode="idea", top_k=3)
        
        # Multiple modes
        findings = researcher.research("RAG", mode=["idea", "implementation"])
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
        mode: ResearchModeInput = None,
        top_k: int = 5,
        depth: ResearchDepth = "deep",
    ) -> ResearchFindings:
        """
        Run deep web research and return a ResearchFindings object.
        
        Args:
            query: What we want to learn from public sources.
            mode: Research mode(s). Can be:
                - Single mode: "idea", "implementation", or "study"
                - List of modes: ["idea", "implementation"]
                - None (default): runs all three modes
            top_k: Maximum number of results per mode (default: 5).
                   Only applies to idea and implementation modes.
            depth: Research depth. Maps to OpenAI reasoning effort:
                - "light" -> "medium"
                - "deep" -> "high"
        
        Returns:
            ResearchFindings with:
            - .ideas: List[IdeaResult] (if idea mode was run)
            - .implementations: List[ImplementationResult] (if implementation mode was run)
            - .report: ResearchReport (if study mode was run)
        """
        # Validate query
        query = (query or "").strip()
        if not query:
            raise ValueError("query must be a non-empty string")
        
        # Normalize mode to list
        modes = self._normalize_modes(mode)
        
        # Map depth to reasoning effort
        reasoning_effort = self._get_reasoning_effort(depth)
        
        # Run each mode and collect results
        findings_list = []
        for m in modes:
            logger.info(f"Running research in '{m}' mode for: {query[:50]}...")
            findings = self._run_single_mode(
                query=query,
                mode=m,
                top_k=top_k,
                reasoning_effort=reasoning_effort,
            )
            findings_list.append(findings)
        
        # Merge results if multiple modes
        if len(findings_list) == 1:
            return findings_list[0]
        else:
            return merge_findings(findings_list, query=query, top_k=top_k)

    def _normalize_modes(self, mode: ResearchModeInput) -> List[ResearchMode]:
        """
        Normalize mode input to a list of modes.
        
        Args:
            mode: Single mode, list of modes, or None
            
        Returns:
            List of modes to run
        """
        if mode is None:
            return DEFAULT_MODES.copy()
        elif isinstance(mode, str):
            if mode not in ("idea", "implementation", "study"):
                raise ValueError(f"Invalid mode: {mode}. Must be 'idea', 'implementation', or 'study'")
            return [mode]
        elif isinstance(mode, list):
            for m in mode:
                if m not in ("idea", "implementation", "study"):
                    raise ValueError(f"Invalid mode: {m}. Must be 'idea', 'implementation', or 'study'")
            return mode
        else:
            raise ValueError(f"mode must be a string, list, or None (got {type(mode)})")

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
    ) -> ResearchFindings:
        """
        Run research for a single mode.
        
        Args:
            query: The research query
            mode: The mode to run
            top_k: Max results (for idea/implementation modes)
            reasoning_effort: OpenAI reasoning effort level
            
        Returns:
            ResearchFindings for this mode
        """
        # Build prompt
        prompt = self._build_research_prompt(query=query, mode=mode, top_k=top_k)
        
        try:
            # Build request params
            # Note: reasoning.effort is only supported by certain models (e.g., o1, o3)
            # For other models, we skip it
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
            # Return empty findings on error
            return ResearchFindings(query=query, modes=[mode], top_k=top_k)
        
        # Parse the output
        return parse_research_result(
            raw_output=raw_text,
            mode=mode,
            query=query,
            top_k=top_k,
        )

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
