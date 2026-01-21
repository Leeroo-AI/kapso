# Research Findings
#
# Rich wrapper around research results with fluent accessors.
#
# This class provides a clean API for accessing research results:
# - .repos(top_k) -> List[Source.Repo] for learn()
# - .ideas(top_k) -> IdeaList (List[Source.Idea] with to_context_string())
# - .source -> Source.Research for direct KG ingestion
#
# Usage:
#     findings = kapso.research("How to fine-tune LLMs", depth="deep")
#     
#     # Get repos for learning
#     kapso.learn(sources=[*findings.repos(top_k=5)])
#     
#     # Get ideas for evolve context (IdeaList has to_context_string())
#     solution = kapso.evolve(goal="...", context=[findings.ideas(top_k=20)])

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from src.knowledge.learners.sources import Source, IdeaList


@dataclass
class RepoInfo:
    """Parsed repository information from research report."""
    url: str
    name: str = ""
    stars: Optional[int] = None
    description: str = ""


@dataclass
class IdeaInfo:
    """Parsed idea/insight from research report."""
    content: str
    source_url: str = ""


@dataclass
class ResearchFindings:
    """
    Rich wrapper around research results.
    
    Provides fluent accessors for different use cases:
    - .repos(top_k) -> List[Source.Repo] for learn()
    - .ideas(top_k) -> IdeaList (List[Source.Idea] with to_context_string())
    - .source -> Source.Research for direct KG ingestion
    """
    _source: Source.Research
    _repos: List[RepoInfo] = field(default_factory=list)
    _ideas: List[IdeaInfo] = field(default_factory=list)
    
    def repos(self, top_k: int = 10) -> List[Source.Repo]:
        """
        Return top-k repos as Source.Repo objects.
        
        These can be passed directly to kapso.learn().
        
        Args:
            top_k: Maximum number of repos to return (default: 10)
            
        Returns:
            List of Source.Repo objects ready for learn()
        """
        return [Source.Repo(url=r.url) for r in self._repos[:top_k]]
    
    def ideas(self, top_k: int = 20) -> IdeaList:
        """
        Return top-k ideas as IdeaList (List[Source.Idea] with to_context_string()).
        
        The returned IdeaList can be:
        - Iterated as a list of Source.Idea objects
        - Converted to string via to_context_string() or str()
        - Passed to kapso.evolve(context=[...])
        
        Args:
            top_k: Maximum number of ideas to return (default: 20)
            
        Returns:
            IdeaList with Source.Idea objects
        """
        if not self._ideas:
            # Fallback: create a single idea from the full report
            fallback_idea = Source.Idea(
                content=self._source.to_context_string(),
                source_url=""
            )
            return IdeaList([fallback_idea], objective=self._source.objective)
        
        # Convert IdeaInfo to Source.Idea
        selected = self._ideas[:top_k]
        idea_objects = [
            Source.Idea(content=info.content, source_url=info.source_url)
            for info in selected
        ]
        return IdeaList(idea_objects, objective=self._source.objective)
    
    @property
    def source(self) -> Source.Research:
        """Access raw Source.Research for direct KG ingestion."""
        return self._source
    
    def to_context_string(self) -> str:
        """Delegate to underlying source for full context."""
        return self._source.to_context_string()
    
    def __str__(self) -> str:
        """String representation uses to_context_string()."""
        return self.to_context_string()


def parse_repos_from_report(report: str) -> List[RepoInfo]:
    """
    Extract GitHub repos from the research report.
    
    Looks for URLs matching github.com pattern.
    
    Args:
        report: The markdown research report
        
    Returns:
        List of RepoInfo objects (deduplicated by URL)
    """
    repos = []
    
    # Pattern: github.com/owner/repo (captures owner and repo name)
    github_pattern = r'https?://github\.com/([^/\s\)\]]+)/([^/\s\)\]\#]+)'
    
    for match in re.finditer(github_pattern, report):
        url = f"https://github.com/{match.group(1)}/{match.group(2)}"
        # Clean up trailing punctuation
        url = url.rstrip('.,;:')
        name = f"{match.group(1)}/{match.group(2)}"
        repos.append(RepoInfo(url=url, name=name))
    
    # Deduplicate by URL
    seen = set()
    unique = []
    for r in repos:
        if r.url not in seen:
            seen.add(r.url)
            unique.append(r)
    
    return unique


def parse_ideas_from_report(report: str) -> List[IdeaInfo]:
    """
    Extract key ideas/takeaways from the research report.
    
    Looks for bullet points in Summary section and other key sections.
    
    Args:
        report: The markdown research report
        
    Returns:
        List of IdeaInfo objects
    """
    ideas = []
    
    # Extract from ## Summary section (primary source of ideas)
    summary_match = re.search(
        r'##\s*Summary\s*\n(.*?)(?=\n##|\Z)', 
        report, 
        re.DOTALL | re.IGNORECASE
    )
    if summary_match:
        summary_text = summary_match.group(1)
        # Extract bullet points (- or *)
        bullets = re.findall(r'^\s*[-*]\s*(.+)$', summary_text, re.MULTILINE)
        for bullet in bullets:
            ideas.append(IdeaInfo(content=bullet.strip()))
    
    # Also extract from ## Core concepts if Summary is sparse
    if len(ideas) < 5:
        concepts_match = re.search(
            r'##\s*Core\s+concepts?\s*\n(.*?)(?=\n##|\Z)', 
            report, 
            re.DOTALL | re.IGNORECASE
        )
        if concepts_match:
            concepts_text = concepts_match.group(1)
            bullets = re.findall(r'^\s*[-*]\s*(.+)$', concepts_text, re.MULTILINE)
            for bullet in bullets:
                ideas.append(IdeaInfo(content=bullet.strip()))
    
    # Extract from ## Trade-offs section
    tradeoffs_match = re.search(
        r'##\s*Trade-?offs?\s*\n(.*?)(?=\n##|\Z)', 
        report, 
        re.DOTALL | re.IGNORECASE
    )
    if tradeoffs_match:
        tradeoffs_text = tradeoffs_match.group(1)
        bullets = re.findall(r'^\s*[-*]\s*(.+)$', tradeoffs_text, re.MULTILINE)
        for bullet in bullets:
            ideas.append(IdeaInfo(content=bullet.strip()))
    
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for idea in ideas:
        if idea.content and idea.content not in seen:
            seen.add(idea.content)
            unique.append(idea)
    
    return unique
