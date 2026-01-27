# Research Findings
#
# Data structures for research results.
#
# This module provides:
# - Idea: Single idea from research (query + source + content)
# - Implementation: Single implementation from research (query + source + content)
# - ResearchReport: Study mode research report (query + content)
# - ResearchFindings: Wrapper for multi-mode results
#
# Usage:
#     # Single mode
#     ideas = researcher.research("How to fine-tune LLMs", mode="idea", top_k=5)
#     for idea in ideas:
#         print(idea.to_string())
#     
#     # Multiple modes
#     findings = researcher.research("LLM fine-tuning", mode=["idea", "implementation"], top_k=5)
#     for idea in findings.ideas:
#         print(idea.to_string())
#     for impl in findings.implementations:
#         print(impl.to_string())

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Result Data Classes
# =============================================================================

@dataclass
class Idea:
    """
    A single research idea from web research.
    
    Produced by: researcher.research(query, mode="idea")
    Used in: kapso.evolve(context=[idea.to_string()])
    Learnable: pipeline.run(idea)
    """
    query: str      # Original research query
    source: str     # URL where this idea came from
    content: str    # Full content with sections
    
    def to_string(self) -> str:
        """Format idea as context string for LLM prompts."""
        return f"# Research Idea\nQuery: {self.query}\nSource: {self.source}\n\n{self.content}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {"query": self.query, "source": self.source, "content": self.content}
    
    def __str__(self) -> str:
        return self.to_string()


@dataclass
class Implementation:
    """
    A single implementation from web research.
    
    Produced by: researcher.research(query, mode="implementation")
    Used in: kapso.evolve(context=[impl.to_string()])
    Learnable: pipeline.run(impl)
    """
    query: str      # Original research query
    source: str     # URL where this implementation came from
    content: str    # Full content with code snippet
    
    def to_string(self) -> str:
        """Format implementation as context string for LLM prompts."""
        return f"# Implementation\nQuery: {self.query}\nSource: {self.source}\n\n{self.content}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {"query": self.query, "source": self.source, "content": self.content}
    
    def __str__(self) -> str:
        return self.to_string()


@dataclass
class ResearchReport:
    """
    A comprehensive research report (academic paper style).
    
    Produced by: researcher.research(query, mode="study")
    Used in: kapso.evolve(context=[report.to_string()])
    Learnable: pipeline.run(report)
    """
    query: str      # Original research query
    content: str    # Full markdown report
    
    def to_string(self) -> str:
        """Format report as context string for LLM prompts."""
        return f"# Research Report\nQuery: {self.query}\n\n{self.content}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {"query": self.query, "content": self.content}
    
    def __str__(self) -> str:
        return self.to_string()


@dataclass
class ResearchFindings:
    """
    Wrapper for multi-mode research results.
    
    Produced by: researcher.research(query, mode=["idea", "implementation"])
    Contains results from multiple modes in a single object.
    """
    query: str
    ideas: List[Idea] = field(default_factory=list)
    implementations: List[Implementation] = field(default_factory=list)
    report: Optional[ResearchReport] = None
    
    def to_string(self) -> str:
        """Format all findings as context string for LLM prompts."""
        parts = [f"# Research Findings\nQuery: {self.query}\n"]
        
        if self.ideas:
            parts.append("\n## Ideas\n")
            for idea in self.ideas:
                parts.append(f"### {idea.source}\n{idea.content}\n")
        
        if self.implementations:
            parts.append("\n## Implementations\n")
            for impl in self.implementations:
                parts.append(f"### {impl.source}\n{impl.content}\n")
        
        if self.report:
            parts.append("\n## Report\n")
            parts.append(self.report.content)
        
        return "\n".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "ideas": [i.to_dict() for i in self.ideas],
            "implementations": [i.to_dict() for i in self.implementations],
            "report": self.report.to_dict() if self.report else None,
        }
    
    def __str__(self) -> str:
        return self.to_string()


# =============================================================================
# Type Definitions
# =============================================================================

ResearchMode = Literal["idea", "implementation", "study"]
ResearchModeInput = Union[ResearchMode, List[ResearchMode]]


# =============================================================================
# Parsing Functions
# =============================================================================

def _extract_tag(text: str, tag: str) -> Optional[str]:
    """
    Extract content from a single XML tag.
    
    Args:
        text: The text to search in
        tag: The tag name (without angle brackets)
        
    Returns:
        The content inside the tag, or None if not found
    """
    # Use non-greedy match to handle nested content
    match = re.search(rf'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
    return match.group(1).strip() if match else None


def _extract_research_content(raw_output: str) -> Optional[str]:
    """
    Extract content from <research_result> tags.
    
    Handles truncated output by extracting everything after opening tag
    if closing tag is missing.
    """
    # First try with closing tag
    match = re.search(r'<research_result>(.*?)</research_result>', raw_output, re.DOTALL)
    
    # If no closing tag found, try to extract everything after opening tag
    if not match:
        match = re.search(r'<research_result>(.*)', raw_output, re.DOTALL)
        if match:
            logger.warning("Missing </research_result> closing tag; output may have been truncated")
    
    return match.group(1).strip() if match else None


def parse_idea_results(raw_output: str, query: str) -> List[Idea]:
    """
    Parse LLM output into List[Idea].
    
    Args:
        raw_output: The raw LLM output text
        query: The original research query
        
    Returns:
        List of Idea objects
    """
    content = _extract_research_content(raw_output)
    if not content:
        logger.warning("Missing <research_result> tags in output; returning empty list")
        return []
    
    # Parse <research_item> tags
    items = re.findall(r'<research_item>(.*?)</research_item>', content, re.DOTALL)
    
    results = []
    for item in items:
        source = _extract_tag(item, "source")
        content_text = _extract_tag(item, "content")
        
        if source and content_text:
            results.append(Idea(query=query, source=source, content=content_text))
        else:
            logger.warning("Skipping research_item with missing source or content")
    
    return results


def parse_implementation_results(raw_output: str, query: str) -> List[Implementation]:
    """
    Parse LLM output into List[Implementation].
    
    Args:
        raw_output: The raw LLM output text
        query: The original research query
        
    Returns:
        List of Implementation objects
    """
    content = _extract_research_content(raw_output)
    if not content:
        logger.warning("Missing <research_result> tags in output; returning empty list")
        return []
    
    # Parse <research_item> tags
    items = re.findall(r'<research_item>(.*?)</research_item>', content, re.DOTALL)
    
    results = []
    for item in items:
        source = _extract_tag(item, "source")
        content_text = _extract_tag(item, "content")
        
        if source and content_text:
            results.append(Implementation(query=query, source=source, content=content_text))
        else:
            logger.warning("Skipping research_item with missing source or content")
    
    return results


def parse_study_result(raw_output: str, query: str) -> ResearchReport:
    """
    Parse LLM output into ResearchReport.
    
    Args:
        raw_output: The raw LLM output text
        query: The original research query
        
    Returns:
        ResearchReport object (may have empty content if parsing fails)
    """
    content = _extract_research_content(raw_output)
    if not content:
        logger.warning("Missing <research_result> tags in output; returning empty report")
        return ResearchReport(query=query, content="")
    
    return ResearchReport(query=query, content=content)
