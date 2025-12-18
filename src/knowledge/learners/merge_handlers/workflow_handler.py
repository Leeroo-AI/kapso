# Workflow Merge Handler
#
# Handles merge logic for Workflow pages.
# Workflows are recipes - ordered sequences of steps.

from src.knowledge.learners.merge_handlers.base import MergeHandler
from src.knowledge.search.base import WikiPage


class WorkflowMergeHandler(MergeHandler):
    """
    Merge handler for Workflow pages.
    
    Workflows are recipes that define ordered sequences of steps.
    Merge if covering the same end-to-end process.
    """
    
    @property
    def page_type(self) -> str:
        return "Workflow"
    
    @property
    def merge_instructions(self) -> str:
        return """## Workflow Merge Instructions

### Search Strategy
Search for workflows with similar goals/outcomes. Look for workflows that:
- Cover the same end-to-end process
- Target the same domain/task
- Use similar tech stack

### Merge Decision Criteria

**MERGE if** existing workflow covers the same end-to-end process:
- Combine unique steps from both workflows
- Keep more detailed step descriptions
- Merge source references
- Update Related Pages links with new connections

**CREATE NEW if** this is a genuinely different workflow:
- Different end goal (e.g., training vs inference)
- Different tech stack even if similar domain
- Significantly different approach to same problem

### Merge Actions

If merging with existing workflow:
1. Identify steps that are unique to proposed workflow
2. Add missing steps in correct order
3. Enhance existing step descriptions with better details
4. Add new sources/references
5. Call `kg_edit` with combined content

If creating new:
1. Validate all steps link to Principles
2. Call `kg_index` with the page data

### Quality Check
- Workflows must have clear step ordering
- Each step must link to a Principle via `[[step::Principle:X]]`
- No duplicate steps
- Overview must describe the end goal
"""
    
    def build_search_query(self, page: WikiPage) -> str:
        """Build query from workflow overview and title."""
        # Use overview for semantic match, truncated for efficiency
        overview_snippet = page.overview[:150] if page.overview else ""
        title_words = page.page_title.replace("_", " ")
        return f"Workflow for {title_words}. {overview_snippet}"

