# Research Findings
#
# Parsing functions for research results.
# Types are defined in src.knowledge_base.types.
#
# Usage:
#     from kapso.knowledge_base.types import Source
#     from kapso.researcher.research_findings import parse_idea_results
#     
#     ideas = parse_idea_results(raw_output, query)

from __future__ import annotations

import logging
import re
from typing import List, Literal, Optional, Union

from kapso.knowledge_base.types import Source

logger = logging.getLogger(__name__)


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


def parse_idea_results(raw_output: str, query: str) -> List[Source.Idea]:
    """
    Parse LLM output into List[Source.Idea].
    
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
            results.append(Source.Idea(query=query, source=source, content=content_text))
        else:
            logger.warning("Skipping research_item with missing source or content")
    
    return results


def parse_implementation_results(raw_output: str, query: str) -> List[Source.Implementation]:
    """
    Parse LLM output into List[Source.Implementation].
    
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
            results.append(Source.Implementation(query=query, source=source, content=content_text))
        else:
            logger.warning("Skipping research_item with missing source or content")
    
    return results


def parse_study_result(raw_output: str, query: str) -> Source.ResearchReport:
    """
    Parse LLM output into Source.ResearchReport.
    
    Args:
        raw_output: The raw LLM output text
        query: The original research query
        
    Returns:
        ResearchReport object (may have empty content if parsing fails)
    """
    content = _extract_research_content(raw_output)
    if not content:
        logger.warning("Missing <research_result> tags in output; returning empty report")
        return Source.ResearchReport(query=query, content="")
    
    return Source.ResearchReport(query=query, content=content)
