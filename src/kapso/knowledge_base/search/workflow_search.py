# Workflow Repository Search (C3)
#
# Wiki-based retrieval tool that wraps KGGraphSearch with Workflow page type filter.
# Searches for starter workflow repositories and extracts GitHub URLs.
#
# Usage:
#   from kapso.knowledge_base.search.workflow_search import WorkflowRepoSearch
#   
#   search = WorkflowRepoSearch()
#   results = search.search("fine-tune LLM with LoRA")
#   
#   # Get top result with GitHub URL
#   if not results.is_empty:
#       top = results.items[0]
#       print(f"Repo: {top.title}, URL: {top.github_url}")

import re
from dataclasses import dataclass, field
from typing import List, Optional

from kapso.knowledge_base.search.base import KGSearchFilters, PageType


# =============================================================================
# Result Data Structures
# =============================================================================

@dataclass
class WorkflowRepoItem:
    """
    Single workflow repository result.
    
    Attributes:
        title: Repository/workflow name
        content: Full page content
        overview: Brief description
        github_url: Extracted GitHub repository URL
        score: Relevance score (0.0 to 1.0)
        domains: Category tags
    """
    title: str
    content: str
    overview: str
    github_url: str
    score: float = 0.0
    domains: List[str] = field(default_factory=list)
    
    def to_string(self) -> str:
        """Format item for LLM context."""
        return (
            f"### {self.title}\n"
            f"**GitHub:** {self.github_url}\n"
            f"**Overview:** {self.overview}\n\n"
            f"{self.content[:500]}..."
        )


@dataclass 
class WorkflowRepoResult:
    """
    Result from workflow repository search.
    
    Attributes:
        query: Original search query
        items: List of matching workflow repos
        source: Tool name
    """
    query: str
    items: List[WorkflowRepoItem] = field(default_factory=list)
    source: str = "workflow_repo"
    
    @property
    def is_empty(self) -> bool:
        return len(self.items) == 0
    
    @property
    def top_result(self) -> Optional[WorkflowRepoItem]:
        return self.items[0] if self.items else None
    
    def to_context_string(self, max_items: int = 5) -> str:
        """Format results as context for LLM prompts."""
        if self.is_empty:
            return f"No workflow repositories found for: {self.query}"
        parts = [item.to_string() for item in self.items[:max_items]]
        return "## Workflow Repository Search Results\n\n" + "\n\n---\n\n".join(parts)


# =============================================================================
# GitHub URL Extraction
# =============================================================================

def extract_github_url(content: str) -> str:
    """
    Extract GitHub URL from wiki page content.
    
    Looks for patterns:
    1. == Github URL == section
    2. [[source::Repo|name|URL]] - MediaWiki syntax
    3. Raw GitHub URLs
    
    Args:
        content: Wiki page content
        
    Returns:
        GitHub URL or empty string if not found
    """
    # Pattern 1: == Github URL == section
    section_pattern = r'==\s*Github URL\s*==\s*\n\s*(https://github\.com/\S+)'
    match = re.search(section_pattern, content, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Pattern 2: [[source::Repo|name|URL]]
    repo_pattern = r'\[\[source::Repo\|[^|]+\|(https://github\.com/[^\]]+)\]\]'
    match = re.search(repo_pattern, content)
    if match:
        return match.group(1)
    
    # Pattern 3: Any GitHub URL
    url_pattern = r'https://github\.com/[\w-]+/[\w.-]+'
    match = re.search(url_pattern, content)
    if match:
        return match.group(0)
    
    return ""


# =============================================================================
# Workflow Repository Search (C3)
# =============================================================================

class WorkflowRepoSearch:
    """
    Search for starter workflow repositories (C3).
    
    Wraps KGGraphSearch with page_type filter for Workflow pages.
    Extracts GitHub URLs from page content.
    
    Use when no starting repo is provided to find a relevant starter template.
    """
    
    # Filter to Workflow pages only
    WORKFLOW_TYPES = [PageType.WORKFLOW.value]
    
    def __init__(
        self,
        kg_search: Optional["KGGraphSearch"] = None,
        use_llm_reranker: bool = True,
    ):
        """
        Initialize WorkflowRepoSearch.
        
        Args:
            kg_search: Existing KGGraphSearch instance (creates via factory if None)
            use_llm_reranker: Whether to use LLM for reranking results
        """
        # Use factory to get params from config
        if kg_search is None:
            from kapso.knowledge_base.search import KnowledgeSearchFactory
            kg_search = KnowledgeSearchFactory.create("kg_graph_search")
        self._kg_search = kg_search
        self._use_llm_reranker = use_llm_reranker
    
    @property
    def name(self) -> str:
        return "workflow_repo"
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
        domains: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> WorkflowRepoResult:
        """
        Search for relevant workflow repositories.
        
        Args:
            query: Problem description or task to find a starter repo for
            top_k: Maximum number of results
            min_score: Minimum relevance score threshold
            domains: Filter by domain tags (e.g., ["LLMs", "computer_vision"])
            context: Additional context for search
            
        Returns:
            WorkflowRepoResult with matching repos and GitHub URLs
        """
        # Build filters for Workflow pages
        filters = KGSearchFilters(
            top_k=top_k,
            page_types=self.WORKFLOW_TYPES,
            min_score=min_score,
            domains=domains,
            include_content=True,
        )
        
        # Delegate to KGGraphSearch
        kg_output = self._kg_search.search(
            query=query,
            filters=filters,
            context=context,
            use_llm_reranker=self._use_llm_reranker,
        )
        
        # Convert to WorkflowRepoResult with GitHub URLs
        items = []
        for r in kg_output.results:
            # Extract GitHub URL from content
            github_url = extract_github_url(r.content or "")
            
            items.append(WorkflowRepoItem(
                title=r.page_title,
                content=r.content or r.overview,
                overview=r.overview,
                github_url=github_url,
                score=r.score,
                domains=r.domains,
            ))
        
        return WorkflowRepoResult(query=query, items=items)

