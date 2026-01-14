# Web Search Tools
#
# Web-based retrieval tools using OpenAI Responses API with web_search tool.
# - WebIdeaSearch: Searches web for concepts, principles, best practices
# - WebCodeSearch: Searches web for code implementations and documentation
#
# Usage:
#   from src.execution.web_search import WebIdeaSearch, WebCodeSearch
#   
#   idea_search = WebIdeaSearch()
#   results = idea_search.search("best practices for gradient checkpointing")
#   
#   code_search = WebCodeSearch()
#   results = code_search.search("PyTorch LoRA implementation example")

import os
from dataclasses import dataclass, field
from typing import List, Optional

from openai import OpenAI


# =============================================================================
# Result Data Structures
# =============================================================================

@dataclass
class RetrievalItem:
    """
    Single item from retrieval.
    
    Attributes:
        title: Item title
        content: Main content text  
        score: Relevance score (0.0 to 1.0)
        item_type: Type of content (WebIdea, WebCode)
    """
    title: str
    content: str
    score: float = 0.0
    item_type: str = ""
    
    def to_string(self) -> str:
        """Format item for LLM context."""
        type_str = f" ({self.item_type})" if self.item_type else ""
        return f"### {self.title}{type_str}\n{self.content}"


@dataclass
class RetrievalResult:
    """
    Result from a retrieval tool search.
    
    Attributes:
        query: Original search query
        items: List of retrieved items
        source: Tool name that produced this result
    """
    query: str
    items: List[RetrievalItem] = field(default_factory=list)
    source: str = ""
    
    @property
    def is_empty(self) -> bool:
        return len(self.items) == 0
    
    def to_context_string(self, max_items: int = 5) -> str:
        """Format results as context for LLM prompts."""
        if self.is_empty:
            return f"No results found for: {self.query}"
        parts = [item.to_string() for item in self.items[:max_items]]
        return f"## {self.source} Results\n\n" + "\n\n---\n\n".join(parts)


# =============================================================================
# Web Search Tools
# =============================================================================

class WebIdeaSearch:
    """
    Web search for ideas: concepts, principles, best practices.
    
    Uses OpenAI Responses API with web_search tool.
    Use in expand/select phase when wiki doesn't have relevant principles.
    """
    
    DEFAULT_MODEL = "gpt-4.1"
    
    def __init__(self, model: Optional[str] = None):
        """
        Initialize WebIdeaSearch.
        
        Args:
            model: Model to use for web search (default: gpt-4.1)
        """
        self._client = OpenAI()
        self._model = model or self.DEFAULT_MODEL
    
    @property
    def name(self) -> str:
        return "web_idea"
    
    def search(self, query: str, top_k: int = 5) -> RetrievalResult:
        """
        Search the web for conceptual knowledge.
        
        Args:
            query: Search query (concept, principle, best practice)
            top_k: Ignored for web search (model controls result count)
            
        Returns:
            RetrievalResult with web search findings
        """
        # Craft prompt for idea-focused search
        prompt = f"""Search the web for expert knowledge about: {query}

Provide a structured summary with:
1. **Core Principles**: The fundamental concepts and why they matter
2. **Best Practices**: Recommended approaches from authoritative sources
3. **Trade-offs**: Key decisions and their consequences
4. **When to Use**: Scenarios where this approach excels
5. **Common Pitfalls**: Mistakes to avoid

Focus on theoretical understanding and practical guidance, not code.
Cite sources where possible."""
        
        try:
            response = self._client.responses.create(
                model=self._model,
                tools=[{"type": "web_search"}],
                input=prompt,
            )
            
            # Extract output text from response
            output_text = response.output_text
            
            items = [
                RetrievalItem(
                    title="Web Search Results",
                    content=output_text,
                    score=1.0,
                    item_type="WebIdea",
                )
            ]
            
        except Exception as e:
            print(f"WebIdeaSearch error: {e}")
            items = []
        
        return RetrievalResult(query=query, items=items, source=self.name)


class WebCodeSearch:
    """
    Web search for code: implementations, examples, documentation.
    
    Uses OpenAI Responses API with web_search tool.
    Use in implement/debug phase when wiki doesn't have relevant code.
    """
    
    DEFAULT_MODEL = "gpt-4.1"
    
    def __init__(self, model: Optional[str] = None):
        """
        Initialize WebCodeSearch.
        
        Args:
            model: Model to use for web search (default: gpt-4.1)
        """
        self._client = OpenAI()
        self._model = model or self.DEFAULT_MODEL
    
    @property
    def name(self) -> str:
        return "web_code"
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        language: str = "",
        framework: str = "",
    ) -> RetrievalResult:
        """
        Search the web for code implementations.
        
        Args:
            query: Search query (code task, library usage, error fix)
            top_k: Ignored for web search (model controls result count)
            language: Programming language hint (e.g., "Python")
            framework: Framework hint (e.g., "PyTorch")
            
        Returns:
            RetrievalResult with code examples from web
        """
        # Build search hints
        lang_hint = f" in {language}" if language else ""
        framework_hint = f" using {framework}" if framework else ""
        
        # Craft prompt for code-focused search
        prompt = f"""Search the web for production-quality code examples for: {query}{lang_hint}{framework_hint}

Provide:
1. **Complete Code**: Working examples with all necessary imports
2. **Dependencies**: Required packages and version compatibility
3. **Usage Example**: How to call/use the code in practice
4. **Configuration Options**: Key parameters and their effects
5. **Error Handling**: Common errors and how to handle them

Prioritize code from official documentation, GitHub repos, and trusted sources.
Include any important caveats or known issues."""
        
        try:
            response = self._client.responses.create(
                model=self._model,
                tools=[{"type": "web_search"}],
                input=prompt,
            )
            
            # Extract output text from response
            output_text = response.output_text
            
            items = [
                RetrievalItem(
                    title="Web Code Search Results",
                    content=output_text,
                    score=1.0,
                    item_type="WebCode",
                )
            ]
            
        except Exception as e:
            print(f"WebCodeSearch error: {e}")
            items = []
        
        return RetrievalResult(query=query, items=items, source=self.name)
