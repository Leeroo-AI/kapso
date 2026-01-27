# Researcher
#
# A small wrapper around OpenAI's `web_search` tool.
#
# Design goals:
# - Return `ResearchFindings` with fluent accessors (.repos(), .ideas()).
# - Keep prompt templates in markdown files (per mode) for easy iteration.
# - Keep the prompt builder as a private method on the class (requested).

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from openai import OpenAI

from src.knowledge.learners.sources import Source
from src.knowledge.researcher.research_findings import (
    ResearchFindings,
    IdeaInfo,
    parse_repos_from_report,
    parse_ideas_from_report,
)

logger = logging.getLogger(__name__)

# Public type for call sites (Kapso, tests, etc.)
ResearchMode = Literal["idea", "implementation", "both"]
ResearchDepth = Literal["light", "deep"]


_PROMPTS_DIR = Path(__file__).parent / "prompts"


@dataclass
class Researcher:
    """
    Deep public web research using OpenAI Responses API + `web_search`.
    
    Notes:
    - MVP uses a single Responses call. The model can invoke `web_search` multiple times internally.
    - Output is a markdown report with inline URLs (citations).
    """

    # Keep model choice internal for now. The public API does not expose it.
    # Default chosen by the user for production usage.
    model: str = "gpt-5.2"

    def __post_init__(self) -> None:
        # Create client once per instance.
        self._client = OpenAI()

    def research(
        self,
        objective: str,
        *,
        mode: ResearchMode = "both",
        depth: ResearchDepth = "deep",
    ) -> ResearchFindings:
        """
        Run deep web research and return a `ResearchFindings` object.
        
        Args:
            objective: What we want to learn from public sources.
            mode: "idea" | "implementation" | "both"
            depth: "light" | "deep"
                Maps to OpenAI `reasoning.effort`:
                - light -> "medium"
                - deep  -> "high"
        
        Returns:
            ResearchFindings with fluent accessors:
            - .repos(top_k) -> List[Source.Repo] for learn()
            - .ideas(top_k) -> str for evolve() context
            - .source -> Source.Research for direct KG ingestion
        """
        objective = (objective or "").strip()
        if not objective:
            raise ValueError("objective must be a non-empty string")

        prompt = self._build_research_prompt(objective=objective, mode=mode)
        if depth == "light":
            reasoning_effort = "medium"
        elif depth == "deep":
            reasoning_effort = "high"
        else:
            raise ValueError(f"depth must be 'light' or 'deep' (got {depth!r})")

        try:
            response = self._client.responses.create(
                model=self.model,
                tools=[{"type": "web_search"}],
                input=prompt,
                reasoning={"effort": reasoning_effort},
            )
            raw_text = response.output_text or ""
            report = self._extract_research_result(raw_text)
        except Exception as e:
            # Keep the failure visible to downstream callers while still returning a valid artifact.
            #
            # Why:
            # - The caller may choose to ingest this into the KG for debugging.
            # - The caller may choose to show it in logs.
            logger.exception("Researcher failed: %s", e)
            report = (
                "## Web research failed\n\n"
                f"**Objective**: {objective}\n\n"
                f"**Error**: {e}\n\n"
                "### Troubleshooting\n"
                "- Ensure `OPENAI_API_KEY` is set (typically via `.env`).\n"
                "- Ensure your OpenAI account has access to a model that supports `web_search`.\n"
            )

        # Create Source.Research for KG ingestion
        source = Source.Research(objective=objective, mode=mode, report_markdown=report)
        
        # Parse structured data from report
        repos = parse_repos_from_report(report)
        ideas = parse_ideas_from_report(report)
        
        return ResearchFindings(_source=source, _repos=repos, _ideas=ideas)

    @staticmethod
    def _extract_research_result(text: str) -> str:
        """
        Extract the report from <research_result>...</research_result> tags.
        
        Why:
        - We want a stable, machine-parseable output boundary.
        - The prompt templates instruct the model to wrap the final report in these tags.
        
        Behavior:
        - If the tags are present, return ONLY the inner content (trimmed).
        - If tags are missing or malformed, fall back to the whole output (trimmed),
          and log a warning so we can improve prompts over time.
        """
        start_tag = "<research_result>"
        end_tag = "</research_result>"

        if not text:
            return ""

        start = text.find(start_tag)
        if start == -1:
            logger.warning("Missing <research_result> tags in web research output; returning raw output.")
            return text.strip()

        start_content = start + len(start_tag)
        end = text.find(end_tag, start_content)
        if end == -1:
            logger.warning("Missing </research_result> end tag in web research output; returning raw output.")
            return text.strip()

        extracted = text[start_content:end].strip()
        if not extracted:
            logger.warning("Empty <research_result> block in web research output; returning raw output.")
            return text.strip()

        # Optional sanity check: warn if there is non-whitespace outside the tag block.
        prefix = text[:start].strip()
        suffix = text[end + len(end_tag) :].strip()
        if prefix or suffix:
            logger.warning(
                "Found extra text outside <research_result> block; ignoring it (prefix=%s, suffix=%s).",
                bool(prefix),
                bool(suffix),
            )

        return extracted

    def _build_research_prompt(self, *, objective: str, mode: ResearchMode) -> str:
        """
        Build a single prompt that encourages multi-search behavior.
        
        This method:
        - loads the mode-specific instruction block from markdown, and
        - wraps it in a stable prompt envelope (citations + output format).
        """
        mode_instructions = self._load_mode_instructions(mode)

        # Load the envelope template from file. Can be edited without code changes.
        envelope_path = _PROMPTS_DIR / "research_envelope.md"
        if not envelope_path.exists():
            raise FileNotFoundError(f"Missing research envelope prompt file: {envelope_path}")
        envelope_template = envelope_path.read_text(encoding="utf-8")

        return envelope_template.format(
            objective=objective,
            mode=mode,
            mode_instructions=mode_instructions,
        )

    def _load_mode_instructions(self, mode: ResearchMode) -> str:
        """
        Load the mode-specific instruction block from markdown.
        
        We keep mode prompts as files so they can be iterated on quickly.
        """
        path = _PROMPTS_DIR / f"{mode}.md"
        if not path.exists():
            raise FileNotFoundError(f"Missing web research prompt file for mode '{mode}': {path}")
        return path.read_text(encoding="utf-8").strip()

