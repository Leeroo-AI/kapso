# Research Findings
#
# Data structures and parsing for research results.
#
# This module provides:
# - IdeaResult: Single idea from research (source + content)
# - ImplementationResult: Single implementation from research (source + content)
# - ResearchReport: Study mode research report
# - ResearchFindings: Wrapper for all research outputs
#
# Usage:
#     findings = researcher.research("How to fine-tune LLMs", top_k=5)
#     
#     # Access ideas (from idea mode)
#     for idea in findings.ideas:
#         print(idea.content, idea.source)
#     
#     # Access implementations (from implementation mode)
#     for impl in findings.implementations:
#         print(impl.content)
#     
#     # Access report (from study mode)
#     print(findings.report.content)

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
class IdeaResult:
    """
    Single idea result from research.
    
    Used in 'idea' mode. Each result represents a conceptual idea or approach.
    """
    source: str  # URL where this was found
    content: str  # Description of the idea/approach
    
    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "content": self.content}
    
    def to_context_string(self) -> str:
        return f"- {self.content} ({self.source})"
    
    def __str__(self) -> str:
        return self.to_context_string()


@dataclass
class ImplementationResult:
    """
    Single implementation result from research.
    
    Used in 'implementation' mode. Each result includes working code.
    Content is freeform and includes description, code snippet, and dependencies.
    """
    source: str  # URL where this was found
    content: str  # Freeform: description + code snippet + dependencies
    
    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "content": self.content}
    
    def to_context_string(self) -> str:
        return f"**Source:** {self.source}\n\n{self.content}"
    
    def __str__(self) -> str:
        return self.to_context_string()


@dataclass
class ResearchReport:
    """
    Research report (academic paper style).
    
    Used in 'study' mode. Contains a full markdown report with sections:
    Key Takeaways, Abstract, Introduction, Background, Literature Review,
    Methodology Comparison, Implementation Guide, Evaluation, Limitations,
    Conclusion, References.
    """
    content: str  # Full markdown with all sections
    
    def to_dict(self) -> Dict[str, Any]:
        return {"content": self.content}
    
    def to_context_string(self) -> str:
        return self.content
    
    def __str__(self) -> str:
        return self.content


# =============================================================================
# Research Findings Wrapper
# =============================================================================

# Type alias for research modes
ResearchMode = Literal["idea", "implementation", "study"]


@dataclass
class ResearchFindings:
    """
    Wrapper for all research outputs.
    
    Supports multiple modes in a single call. When multiple modes are requested,
    the researcher runs each mode sequentially and merges results.
    
    Attributes:
        query: The original research query
        modes: List of modes that were run
        top_k: Maximum number of results requested per mode
        ideas: List of IdeaResult (populated if 'idea' mode was run)
        implementations: List of ImplementationResult (populated if 'implementation' mode was run)
        report: ResearchReport (populated if 'study' mode was run)
    """
    query: str
    modes: List[ResearchMode] = field(default_factory=list)
    top_k: int = 5
    
    # Populated based on mode(s):
    ideas: List[IdeaResult] = field(default_factory=list)
    implementations: List[ImplementationResult] = field(default_factory=list)
    report: Optional[ResearchReport] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "modes": self.modes,
            "top_k": self.top_k,
            "ideas": [i.to_dict() for i in self.ideas],
            "implementations": [i.to_dict() for i in self.implementations],
            "report": self.report.to_dict() if self.report else None,
        }
    
    def to_context_string(self) -> str:
        """Convert to a string suitable for LLM context."""
        parts = [f"# Research: {self.query}\n"]
        
        if self.ideas:
            parts.append("## Ideas\n")
            for idea in self.ideas:
                parts.append(idea.to_context_string())
            parts.append("")
        
        if self.implementations:
            parts.append("## Implementations\n")
            for impl in self.implementations:
                parts.append(impl.to_context_string())
                parts.append("---")
            parts.append("")
        
        if self.report:
            parts.append("## Research Report\n")
            parts.append(self.report.content)
        
        return "\n".join(parts)
    
    def __str__(self) -> str:
        return self.to_context_string()


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


def parse_research_result(
    raw_output: str,
    mode: ResearchMode,
    query: str,
    top_k: int,
) -> ResearchFindings:
    """
    Parse LLM output into ResearchFindings for a single mode.
    
    Args:
        raw_output: The raw LLM output text
        mode: The research mode that was used
        query: The original query
        top_k: The requested number of results
        
    Returns:
        ResearchFindings with parsed results
    """
    # Extract content between <research_result>...</research_result>
    # First try with closing tag
    match = re.search(r'<research_result>(.*?)</research_result>', raw_output, re.DOTALL)
    
    # If no closing tag found, try to extract everything after opening tag
    # (handles case where output was truncated)
    if not match:
        match = re.search(r'<research_result>(.*)', raw_output, re.DOTALL)
        if match:
            logger.warning("Missing </research_result> closing tag; output may have been truncated")
    
    if not match:
        logger.warning("Missing <research_result> tags in output; returning empty findings")
        return ResearchFindings(query=query, modes=[mode], top_k=top_k)
    
    content = match.group(1).strip()
    
    # Freeform/study mode: content is the full report
    if mode == "study":
        return ResearchFindings(
            query=query,
            modes=[mode],
            top_k=top_k,
            report=ResearchReport(content=content),
        )
    
    # Idea/Implementation modes: parse <research_item> tags
    items = re.findall(r'<research_item>(.*?)</research_item>', content, re.DOTALL)
    
    results = []
    for item in items:
        source = _extract_tag(item, "source")
        content_text = _extract_tag(item, "content")
        
        if source and content_text:
            if mode == "idea":
                results.append(IdeaResult(source=source, content=content_text))
            else:  # implementation
                results.append(ImplementationResult(source=source, content=content_text))
        else:
            logger.warning(f"Skipping research_item with missing source or content")
    
    if mode == "idea":
        return ResearchFindings(query=query, modes=[mode], top_k=top_k, ideas=results)
    else:
        return ResearchFindings(query=query, modes=[mode], top_k=top_k, implementations=results)


def merge_findings(findings_list: List[ResearchFindings], query: str, top_k: int) -> ResearchFindings:
    """
    Merge multiple ResearchFindings into one.
    
    Used when running multiple modes sequentially.
    
    Args:
        findings_list: List of ResearchFindings from different modes
        query: The original query
        top_k: The requested number of results
        
    Returns:
        Single merged ResearchFindings
    """
    merged = ResearchFindings(query=query, modes=[], top_k=top_k)
    
    for findings in findings_list:
        merged.modes.extend(findings.modes)
        merged.ideas.extend(findings.ideas)
        merged.implementations.extend(findings.implementations)
        if findings.report and not merged.report:
            merged.report = findings.report
    
    return merged
