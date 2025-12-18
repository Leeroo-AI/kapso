# Implementation Merge Handler
#
# Handles merge logic for Implementation pages.
# Implementations are concrete code references (API docs).

from src.knowledge.learners.merge_handlers.base import MergeHandler
from src.knowledge.search.base import WikiPage


class ImplementationMergeHandler(MergeHandler):
    """
    Merge handler for Implementation pages.
    
    Implementations document concrete APIs/functions/classes.
    Merge if documenting the same API.
    """
    
    @property
    def page_type(self) -> str:
        return "Implementation"
    
    @property
    def merge_instructions(self) -> str:
        return """## Implementation Merge Instructions

### Search Strategy
Search for implementations of the same API/function/class. Look for:
- Same function/class from same library
- Same code signature
- Same repository source

### Merge Decision Criteria

**MERGE if** existing implementation documents the same API:
- Same function/class from same library
- Same code signature
Actions:
- Update code signature if proposed is newer
- Add new usage examples
- Expand I/O Contract details
- Add new Environment requirements

**CREATE NEW if** this is a different API or different angle:
- Different function/class
- Same API but for different Principle (1:1 mapping rule)
  - Each Principle gets its own Implementation page
  - Same API can have multiple Implementation pages for different use cases

### Merge Actions

If merging with existing implementation:
1. Compare code signatures - use newer/more complete
2. Merge usage examples (avoid duplicates)
3. Combine I/O Contract details
4. Add new environment requirements
5. Call `kg_edit` with merged content

If creating new:
1. Ensure Code Reference section has complete signature
2. Ensure Import statement is included
3. Verify I/O Contract is complete
4. Call `kg_index` with the page data

### Quality Check
- Must have Code Reference section with signature
- Must have Import statement
- I/O Contract must list inputs and outputs with types
- Usage examples should be complete and runnable
"""
    
    def build_search_query(self, page: WikiPage) -> str:
        """Build query from implementation title (usually API name)."""
        title_words = page.page_title.replace("_", " ")
        return f"Implementation API documentation: {title_words}"

