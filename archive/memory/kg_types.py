# =============================================================================
# KG Types - Data structures matching wiki graph structure
# =============================================================================
#
# Wiki Structure (from developer_docs/wiki_page_structure.md):
#
#   Workflow
#     ├── uses_heuristic → Heuristic (workflow-level)
#     └── step → Principle
#                  ├── implemented_by → Implementation (mandatory)
#                  │                      ├── requires_env → Environment
#                  │                      └── uses_heuristic → Heuristic
#                  └── uses_heuristic → Heuristic (principle-level)
#
# These types capture the FULL content from each page, not truncated summaries.
# =============================================================================

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class KGTier(Enum):
    """Which retrieval tier produced this knowledge."""
    TIER1_EXACT = "tier1_exact"           # Found exact workflow in KG
    TIER2_RELEVANT = "tier2_relevant"     # No workflow, found relevant pages
    TIER3_ERROR = "tier3_error"           # Error-targeted retrieval


# =============================================================================
# Page Types - Matching wiki structure (leaf nodes first)
# =============================================================================

@dataclass
class Heuristic:
    """
    Wisdom/tips - leaf node in the graph.
    
    Can be linked from Workflow, Principle, or Implementation.
    Contains tips, tricks, optimizations, debugging advice.
    """
    id: str
    title: str
    overview: str = ""
    content: str = ""      # FULL content, not truncated
    code_snippets: List[str] = field(default_factory=list)
    
    def render(self) -> str:
        """Render heuristic for agent context."""
        lines = [f"**{self.title}**: {self.content or self.overview}"]
        if self.code_snippets:
            # Show ALL code snippets - pages are already size-limited
            for snippet in self.code_snippets:
                lines.append(f"```python\n{snippet.strip()}\n```")
        return "\n".join(lines)


@dataclass
class Environment:
    """
    Hardware/software requirements - leaf node.
    
    Linked from Implementation via requires_env.
    """
    id: str
    title: str
    overview: str = ""
    content: str = ""
    requirements: str = ""  # Specific requirements (CUDA version, RAM, etc.)
    
    def render(self) -> str:
        """Render environment for agent context."""
        return f"**{self.title}**: {self.requirements or self.overview}"


@dataclass
class Implementation:
    """
    Concrete code/API - linked from Principle via implemented_by.
    
    Contains the actual code examples and API usage.
    """
    id: str
    title: str
    overview: str = ""
    content: str = ""      # FULL implementation guide
    code_snippets: List[str] = field(default_factory=list)  # ALL code examples
    environment: Optional[Environment] = None
    heuristics: List[Heuristic] = field(default_factory=list)
    
    def render(self) -> str:
        """Render implementation for agent context."""
        lines = [f"**Implementation: {self.title}**"]
        
        if self.overview:
            lines.append(self.overview)
        
        # All code snippets - these are the most valuable
        for snippet in self.code_snippets:
            lines.append(f"```python\n{snippet.strip()}\n```")
        
        # Implementation-level heuristics
        for h in self.heuristics:
            lines.append(f"- 💡 {h.content or h.overview}")
        
        # Environment requirements
        if self.environment:
            lines.append(f"**Requires:** {self.environment.requirements or self.environment.overview}")
        
        return "\n".join(lines)


@dataclass
class Principle:
    """
    Theory/concept - linked from Workflow step.
    
    Library-agnostic explanation of what and why.
    Must have at least one Implementation (executable constraint).
    """
    id: str
    title: str
    overview: str = ""
    content: str = ""      # FULL principle explanation
    implementations: List[Implementation] = field(default_factory=list)
    heuristics: List[Heuristic] = field(default_factory=list)
    
    def render(self) -> str:
        """Render principle for agent context."""
        lines = [f"### {self.title}"]
        
        # Full principle content
        if self.content:
            lines.append(self.content)
        elif self.overview:
            lines.append(self.overview)
        lines.append("")
        
        # Implementations with their code
        for impl in self.implementations:
            lines.append(impl.render())
            lines.append("")
        
        # Principle-level heuristics
        if self.heuristics:
            lines.append("**Guidelines:**")
            for h in self.heuristics:
                lines.append(f"- {h.content or h.overview}")
        
        return "\n".join(lines)


@dataclass
class WorkflowStep:
    """
    A step in a workflow - links to a Principle.
    
    Each step represents one Principle with its full linked knowledge.
    """
    number: int
    principle: Principle
    status: str = "pending"     # For tracking (though not used in current design)
    attempts: int = 0
    last_error: Optional[str] = None


@dataclass 
class Workflow:
    """
    Complete workflow from KG with full graph structure.
    
    Entry point - the "recipe" for achieving a goal.
    Contains ordered steps, each linking to Principles.
    """
    id: str
    title: str
    overview: str = ""
    content: str = ""
    source: str = "kg_exact"
    confidence: float = 0.0
    steps: List[WorkflowStep] = field(default_factory=list)
    heuristics: List[Heuristic] = field(default_factory=list)  # Workflow-level
    
    def render(self) -> str:
        """Render full workflow for agent context."""
        lines = []
        lines.append(f"## Implementation Guide: {self.title}")
        lines.append(f"*Source: {self.source} (confidence: {self.confidence:.0%})*")
        lines.append("")
        
        # Workflow-level heuristics (general guidelines)
        if self.heuristics:
            lines.append("### General Guidelines")
            for h in self.heuristics:
                lines.append(f"- {h.content or h.overview}")
            lines.append("")
        
        # Each step with full principle content
        for step in self.steps:
            lines.append(f"### Step {step.number}: {step.principle.title}")
            
            # Principle content
            if step.principle.content:
                lines.append(step.principle.content)
            elif step.principle.overview:
                lines.append(step.principle.overview)
            lines.append("")
            
            # All implementations for this principle
            for impl in step.principle.implementations:
                lines.append(impl.render())
                lines.append("")
            
            # Principle-level heuristics
            for h in step.principle.heuristics:
                lines.append(f"- 💡 {h.content or h.overview}")
            
            lines.append("")
        
        return "\n".join(lines)


# =============================================================================
# KGKnowledge - Unified structure for all tiers
# =============================================================================

@dataclass
class KGKnowledge:
    """
    Unified knowledge structure for all retrieval tiers.
    
    TIER 1 (EXACT): workflow is populated with full graph structure
    TIER 2 (RELEVANT): workflow is None, principles list populated
    TIER 3 (ERROR): adds error_heuristics and alternatives to existing
    
    Usage:
        knowledge = retriever.retrieve_knowledge(goal)
        context_text = knowledge.render()  # Ready for agent
    """
    # Source metadata
    tier: KGTier = KGTier.TIER2_RELEVANT
    confidence: float = 0.0
    query_used: str = ""
    source_pages: List[str] = field(default_factory=list)
    
    # TIER 1: Full workflow with graph-traversed content
    workflow: Optional[Workflow] = None
    
    # TIER 2: Relevant Principles (each has nested implementations/heuristics)
    principles: List[Principle] = field(default_factory=list)
    
    # TIER 3: Error-specific additions (appended to existing knowledge)
    error_heuristics: List[Heuristic] = field(default_factory=list)
    alternative_implementations: List[Implementation] = field(default_factory=list)
    
    def render(self) -> str:
        """
        Render knowledge to text for agent context.
        
        Handles all tiers:
        - TIER 1: Structured workflow with steps
        - TIER 2: Principles list (each has nested implementations/heuristics)
        - TIER 3: Appends error-specific content
        """
        lines = []
        
        if self.workflow:
            # =========================================================
            # TIER 1: Structured workflow
            # =========================================================
            lines.append(self.workflow.render())
        
        elif self.principles:
            # =========================================================
            # TIER 2: Relevant Principles (each has nested impls/heuristics)
            # =========================================================
            lines.append("## Relevant Knowledge")
            lines.append("*No specific workflow found. Here is relevant knowledge from KG:*")
            lines.append("")
            
            for p in self.principles:
                lines.append(p.render())
                lines.append("")
        
        else:
            lines.append("## Implementation Guide")
            lines.append("*No specific guidance found in knowledge graph.*")
            lines.append("")
        
        # =========================================================
        # TIER 3: Error-specific additions (appended to any tier)
        # =========================================================
        if self.error_heuristics:
            lines.append("---")
            lines.append("## Error Recovery Tips")
            for h in self.error_heuristics:
                lines.append(f"### ⚠️ {h.title}")
                lines.append(h.content or h.overview)
                if h.code_snippets:
                    # Show ALL code snippets - pages are already size-limited
                    for snippet in h.code_snippets:
                        lines.append(f"```python\n{snippet.strip()}\n```")
            lines.append("")
        
        if self.alternative_implementations:
            lines.append("## Alternative Approaches")
            for impl in self.alternative_implementations:
                lines.append(impl.render())
                lines.append("")
        
        return "\n".join(lines)
    
    def add_error_knowledge(
        self, 
        error_heuristics: List[Heuristic] = None,
        alternative_implementations: List[Implementation] = None,
    ) -> "KGKnowledge":
        """
        Add TIER 3 error-specific knowledge.
        
        Modifies in place and returns self for chaining.
        """
        if error_heuristics:
            self.error_heuristics.extend(error_heuristics)
        if alternative_implementations:
            self.alternative_implementations.extend(alternative_implementations)
        self.tier = KGTier.TIER3_ERROR
        return self
    
    @property
    def has_workflow(self) -> bool:
        """Check if this knowledge includes a workflow."""
        return self.workflow is not None
    
    @property
    def step_count(self) -> int:
        """Get number of steps if workflow exists."""
        if self.workflow:
            return len(self.workflow.steps)
        return 0


