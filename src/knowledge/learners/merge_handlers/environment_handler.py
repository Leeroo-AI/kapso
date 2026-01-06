# Environment Merge Handler
#
# Handles merge logic for Environment pages.
# Environments define hardware, OS, and dependency requirements.

from src.knowledge.learners.merge_handlers.base import MergeHandler
from src.knowledge.search.base import WikiPage


class EnvironmentMergeHandler(MergeHandler):
    """
    Merge handler for Environment pages.
    
    Environments document tech stack requirements.
    Merge if same core dependencies.
    """
    
    @property
    def page_type(self) -> str:
        return "Environment"
    
    @property
    def merge_instructions(self) -> str:
        return """## Environment Merge Instructions

### Search Strategy
Search for environments with same tech stack. Look for:
- Same Python version requirements
- Same CUDA/GPU requirements
- Same core library dependencies

### Merge Decision Criteria

**MERGE if** existing environment has same core dependencies:
- Same Python version range
- Same CUDA version (if GPU required)
- Same key libraries (torch, transformers, etc.)
Actions:
- Merge dependency lists
- Combine error documentation
- Add new compatibility notes

**CREATE NEW if** this is a different stack:
- Different Python version requirements
- Different CUDA version (incompatible)
- Different core library requirements

### Merge Actions

If merging with existing environment:
1. Combine dependency lists (use stricter version constraints)
2. Merge common errors documentation
3. Add new compatibility notes
4. Update Quick Install command
5. Call `kg_edit` with merged content

If creating new:
1. Ensure System Requirements section is complete
2. Ensure Dependencies section lists all packages
3. Include Quick Install command
4. Call `kg_index` with the page data

### Quality Check
- Must have System Requirements (OS, hardware)
- Must have Dependencies section with version constraints
- Should have Quick Install command
- Common Errors section is highly valuable
"""
    
    def build_search_query(self, page: WikiPage) -> str:
        """Build query from environment title and overview."""
        title_words = page.page_title.replace("_", " ")
        overview_snippet = page.overview[:100] if page.overview else ""
        return f"Environment setup requirements: {title_words}. {overview_snippet}"

