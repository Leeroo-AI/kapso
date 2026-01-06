# Principle Merge Handler
#
# Handles merge logic for Principle pages.
# Principles are theoretical concepts (library-agnostic).

from src.knowledge.learners.merge_handlers.base import MergeHandler
from src.knowledge.search.base import WikiPage


class PrincipleMergeHandler(MergeHandler):
    """
    Merge handler for Principle pages.
    
    Principles are library-agnostic theoretical concepts.
    Merge if covering the same algorithm/technique.
    """
    
    @property
    def page_type(self) -> str:
        return "Principle"
    
    @property
    def merge_instructions(self) -> str:
        return """## Principle Merge Instructions

### Search Strategy
Search for principles about the same theoretical concept. Look for:
- Same algorithm/technique (even if different name)
- Same mathematical foundation
- Alternative names/terminology for the same idea

### Merge Decision Criteria

**MERGE if** existing principle covers the same theoretical concept:
- Same algorithm or technique
- Same mathematical foundation
- Complementary explanations of same idea
Actions:
- Keep the most rigorous theoretical explanation
- Merge mathematical formulas if complementary
- Add new sources (papers, docs)
- Update `implemented_by` links

**CREATE NEW if** this is a genuinely different concept:
- Different theoretical basis
- Different use case even if similar name

### Merge Actions

If merging with existing principle:
1. Compare theoretical explanations - keep more rigorous
2. Merge complementary math/formulas
3. Combine sources (especially academic papers)
4. Update implementation links
5. Call `kg_edit` with merged content

If creating new:
1. Ensure at least one `[[implemented_by::Implementation:X]]` link
2. Verify no actual code (only pseudo-code)
3. Call `kg_index` with the page data

### Quality Check
- Must link to at least one Implementation via `[[implemented_by::Implementation:X]]`
- Theoretical basis must be library-agnostic
- No actual implementation code (only pseudo-code allowed)
- Sources should include academic papers where applicable
"""
    
    def build_search_query(self, page: WikiPage) -> str:
        """Build query from principle title and overview."""
        title_words = page.page_title.replace("_", " ")
        overview_snippet = page.overview[:100] if page.overview else ""
        return f"Principle theory concept: {title_words}. {overview_snippet}"

