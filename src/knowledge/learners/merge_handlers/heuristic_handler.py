# Heuristic Merge Handler
#
# Handles merge logic for Heuristic pages.
# Heuristics are tribal knowledge, tips, and optimizations.

from src.knowledge.learners.merge_handlers.base import MergeHandler
from src.knowledge.search.base import WikiPage


class HeuristicMergeHandler(MergeHandler):
    """
    Merge handler for Heuristic pages.
    
    Heuristics capture tribal knowledge and best practices.
    Merge if addressing same problem/optimization.
    """
    
    @property
    def page_type(self) -> str:
        return "Heuristic"
    
    @property
    def merge_instructions(self) -> str:
        return """## Heuristic Merge Instructions

### Search Strategy
Search for heuristics addressing same problem/optimization. Look for:
- Same problem domain (e.g., OOM errors, slow training)
- Same optimization target (e.g., memory, speed)
- Related techniques

### Merge Decision Criteria

**MERGE if** existing heuristic addresses same problem:
- Same optimization goal
- Same problem domain
- Complementary advice
Actions:
- Combine reasoning and evidence
- Merge benchmarks/examples
- Add new sources

**CREATE NEW if** this is different advice:
- Different problem domain
- Complementary but distinct advice
- Different optimization target

**IMPORTANT**: If proposed heuristic CONTRADICTS existing:
- Do NOT merge contradicting advice
- CREATE NEW and note the conflict in the page
- Let users evaluate both approaches

### Merge Actions

If merging with existing heuristic:
1. Combine reasoning sections
2. Add more evidence/benchmarks
3. Merge source references
4. Expand trade-off documentation
5. Call `kg_edit` with merged content

If creating new:
1. Ensure clear Rule of Thumb
2. Ensure Reasoning section explains why
3. Call `kg_index` with the page data

### Quality Check
- Must have clear Rule of Thumb (The Insight)
- Must have Reasoning section explaining why
- Should include trade-offs
- Sources should be cited (GitHub issues, blogs, papers)
"""
    
    def build_search_query(self, page: WikiPage) -> str:
        """Build query from heuristic title and overview."""
        title_words = page.page_title.replace("_", " ")
        overview_snippet = page.overview[:100] if page.overview else ""
        return f"Heuristic optimization tip: {title_words}. {overview_snippet}"

