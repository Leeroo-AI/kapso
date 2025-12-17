# Knowledge Merger
#
# Merges proposed wiki pages from repository extraction into the existing KG.
# Uses Claude Code agent to intelligently decide merge actions.
#
# Merge actions:
# - create_new: Add new page to KG
# - update_existing: Update existing page with better content
# - add_links: Add new connections to existing page
# - skip: Don't add (duplicate or low quality)

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.knowledge.search.base import (
    WikiPage,
    KGEditInput,
    DEFAULT_WIKI_DIR,
    DEFAULT_PERSIST_PATH,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to merge instructions prompt
MERGE_PROMPT_PATH = Path(__file__).parent / "prompts" / "merge_instructions.md"

# Mapping from page type to subdirectory
TYPE_TO_SUBDIR = {
    "Workflow": "workflows",
    "Principle": "principles",
    "Implementation": "implementations",
    "Environment": "environments",
    "Heuristic": "heuristics",
}


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class MergeAction:
    """
    A single merge action to be executed.
    
    Attributes:
        action: Type of action (create_new, update_existing, add_links, skip)
        proposed_index: Index in the proposed pages list
        reason: Explanation for why this action was chosen
        page: WikiPage for create_new actions
        existing_page_id: Page ID for update/add_links actions
        updates: Dict of fields to update for update_existing
        new_links: List of links to add for add_links
    """
    action: str
    proposed_index: int
    reason: str
    page: Optional[WikiPage] = None
    existing_page_id: Optional[str] = None
    updates: Optional[Dict[str, Any]] = None
    new_links: Optional[List[Dict[str, str]]] = None


@dataclass
class MergeResult:
    """
    Result of a knowledge merge operation.
    
    Attributes:
        total_proposed: Number of pages proposed
        created: Number of new pages created
        updated: Number of existing pages updated
        linked: Number of pages with new links added
        skipped: Number of pages skipped
        actions: List of all merge actions taken
        errors: List of any errors encountered
    """
    total_proposed: int = 0
    created: int = 0
    updated: int = 0
    linked: int = 0
    skipped: int = 0
    actions: List[MergeAction] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_proposed": self.total_proposed,
            "created": self.created,
            "updated": self.updated,
            "linked": self.linked,
            "skipped": self.skipped,
            "actions": [
                {
                    "action": a.action,
                    "proposed_index": a.proposed_index,
                    "reason": a.reason,
                    "page_id": a.page.id if a.page else a.existing_page_id,
                }
                for a in self.actions
            ],
            "errors": self.errors,
        }


# =============================================================================
# Helper Functions
# =============================================================================

def load_merge_prompt() -> str:
    """
    Load the merge instructions prompt from file.
    
    Returns:
        The merge prompt text
        
    Raises:
        FileNotFoundError: If prompt file doesn't exist
    """
    if not MERGE_PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Merge prompt not found: {MERGE_PROMPT_PATH}"
        )
    
    return MERGE_PROMPT_PATH.read_text(encoding="utf-8")


def get_kg_summary(wiki_dir: Path) -> List[Dict[str, str]]:
    """
    Get a summary of all existing pages in the KG.
    
    Returns a list of dicts with page_id, page_type, page_title, overview.
    This is passed to the agent for comparison with proposed pages.
    
    Args:
        wiki_dir: Path to wiki directory
        
    Returns:
        List of page summary dicts
    """
    summaries = []
    
    for type_name, subdir in TYPE_TO_SUBDIR.items():
        subdir_path = wiki_dir / subdir
        if not subdir_path.exists():
            continue
        
        for wiki_file in subdir_path.glob("*.md"):
            content = wiki_file.read_text(encoding="utf-8")
            page_title = wiki_file.stem
            page_id = f"{type_name}/{page_title}"
            
            # Extract overview from content
            overview = ""
            overview_match = re.search(
                r'== Overview ==\s*\n+(.+?)(?=\n==|\n\{\{|\Z)',
                content, re.DOTALL
            )
            if overview_match:
                overview = overview_match.group(1).strip()
                # Clean up and truncate
                overview = re.sub(r'\n+', ' ', overview)[:300]
            
            summaries.append({
                "page_id": page_id,
                "page_type": type_name,
                "page_title": page_title,
                "overview": overview,
            })
    
    logger.info(f"Found {len(summaries)} existing pages in KG")
    return summaries


def parse_merge_output(output: str) -> List[MergeAction]:
    """
    Parse Claude Code agent output into MergeAction objects.
    
    Args:
        output: Raw output from Claude Code agent
        
    Returns:
        List of MergeAction objects
    """
    # Try to find JSON in the output
    json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', output)
    
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find raw JSON object
        json_match = re.search(r'(\{[\s\S]*"actions"[\s\S]*\})', output)
        if json_match:
            json_str = json_match.group(1)
        else:
            logger.warning("Could not find JSON in agent output")
            return []
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from agent output: {e}")
        return []
    
    # Convert to MergeAction objects
    actions = []
    
    for action_data in data.get("actions", []):
        try:
            action_type = action_data.get("action", "")
            
            action = MergeAction(
                action=action_type,
                proposed_index=action_data.get("proposed_index", -1),
                reason=action_data.get("reason", ""),
            )
            
            if action_type == "create_new":
                page_data = action_data.get("page", {})
                page_type = page_data.get("page_type", "")
                page_title = page_data.get("page_title", "")
                
                action.page = WikiPage(
                    id=f"{page_type}/{page_title}",
                    page_title=page_title,
                    page_type=page_type,
                    overview=page_data.get("overview", ""),
                    content=page_data.get("content", ""),
                    domains=page_data.get("domains", []),
                    sources=page_data.get("sources", []),
                    outgoing_links=page_data.get("outgoing_links", []),
                )
            
            elif action_type == "update_existing":
                action.existing_page_id = action_data.get("existing_page_id", "")
                action.updates = action_data.get("updates", {})
            
            elif action_type == "add_links":
                action.existing_page_id = action_data.get("existing_page_id", "")
                action.new_links = action_data.get("new_links", [])
            
            actions.append(action)
            
        except Exception as e:
            logger.warning(f"Failed to parse action: {e}")
            continue
    
    logger.info(f"Parsed {len(actions)} merge actions from agent output")
    return actions


def write_new_page(page: WikiPage, wiki_dir: Path) -> bool:
    """
    Write a new WikiPage to the appropriate subdirectory.
    
    Args:
        page: WikiPage to write
        wiki_dir: Root wiki directory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        subdir = TYPE_TO_SUBDIR.get(page.page_type)
        if not subdir:
            logger.error(f"Unknown page type: {page.page_type}")
            return False
        
        # Ensure subdirectory exists
        subdir_path = wiki_dir / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)
        
        # Write the file
        file_path = subdir_path / f"{page.page_title}.md"
        
        # Add timestamp if not present in content
        content = page.content
        if "[[last_updated::" not in content:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M GMT")
            # Try to inject into metadata table
            if "|-" in content and "! Last Updated" not in content:
                # Add before closing |}
                content = content.replace("|}", f"|-\n! Last Updated\n|| [[last_updated::{timestamp}]]\n|}}")
        
        file_path.write_text(content, encoding="utf-8")
        logger.info(f"Created new page: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to write page {page.page_title}: {e}")
        return False


def update_existing_page(
    page_id: str,
    updates: Dict[str, Any],
    wiki_dir: Path,
) -> bool:
    """
    Update an existing page with new information.
    
    Args:
        page_id: Page ID (e.g., "Principle/Low_Rank_Adaptation")
        updates: Dict of fields to update
        wiki_dir: Root wiki directory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Parse page_id to get type and title
        parts = page_id.split("/", 1)
        if len(parts) != 2:
            logger.error(f"Invalid page_id format: {page_id}")
            return False
        
        page_type, page_title = parts
        subdir = TYPE_TO_SUBDIR.get(page_type)
        if not subdir:
            logger.error(f"Unknown page type: {page_type}")
            return False
        
        file_path = wiki_dir / subdir / f"{page_title}.md"
        if not file_path.exists():
            logger.error(f"Page file not found: {file_path}")
            return False
        
        content = file_path.read_text(encoding="utf-8")
        
        # Apply updates
        if "overview" in updates:
            new_overview = updates["overview"]
            # Replace content between "== Overview ==" and next section
            pattern = r'(== Overview ==\n)(.+?)(\n===|\n==|\n\{\{|\Z)'
            content = re.sub(
                pattern,
                f'\\1{new_overview}\n\\3',
                content,
                flags=re.DOTALL
            )
        
        if "sources" in updates:
            new_sources = updates["sources"]
            source_lines = []
            for src in new_sources:
                src_type = src.get("type", "Doc")
                src_title = src.get("title", "")
                src_url = src.get("url", "")
                source_lines.append(f"* [[source::{src_type}|{src_title}|{src_url}]]")
            
            sources_text = "\n".join(source_lines)
            # Replace sources section
            pattern = r'(! Knowledge Sources\n\|\|\n)(\* \[\[source::.*?\]\]\n?)+'
            content = re.sub(
                pattern,
                f'\\1{sources_text}\n',
                content,
                flags=re.DOTALL
            )
        
        # Update timestamp
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M GMT")
        content = re.sub(
            r'\[\[last_updated::[^\]]+\]\]',
            f'[[last_updated::{timestamp}]]',
            content
        )
        
        file_path.write_text(content, encoding="utf-8")
        logger.info(f"Updated page: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update page {page_id}: {e}")
        return False


def add_links_to_page(
    page_id: str,
    new_links: List[Dict[str, str]],
    wiki_dir: Path,
) -> bool:
    """
    Add new outgoing links to an existing page.
    
    Args:
        page_id: Page ID (e.g., "Workflow/QLoRA_Finetuning")
        new_links: List of link dicts with edge_type, target_type, target_id
        wiki_dir: Root wiki directory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Parse page_id
        parts = page_id.split("/", 1)
        if len(parts) != 2:
            logger.error(f"Invalid page_id format: {page_id}")
            return False
        
        page_type, page_title = parts
        subdir = TYPE_TO_SUBDIR.get(page_type)
        if not subdir:
            logger.error(f"Unknown page type: {page_type}")
            return False
        
        file_path = wiki_dir / subdir / f"{page_title}.md"
        if not file_path.exists():
            logger.error(f"Page file not found: {file_path}")
            return False
        
        content = file_path.read_text(encoding="utf-8")
        
        # Build new link lines
        link_lines = []
        for link in new_links:
            edge_type = link.get("edge_type", "related")
            target_type = link.get("target_type", "")
            target_id = link.get("target_id", "")
            link_lines.append(f"* [[{edge_type}::{target_type}:{target_id}]]")
        
        new_links_text = "\n".join(link_lines)
        
        # Find Related Pages section and append, or create section
        if "== Related Pages ==" in content:
            # Append before the last line of content or next section
            content = content.rstrip()
            content += f"\n{new_links_text}\n"
        else:
            # Add a new Related Pages section at the end
            content = content.rstrip()
            content += f"\n\n== Related Pages ==\n{new_links_text}\n"
        
        # Update timestamp
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M GMT")
        content = re.sub(
            r'\[\[last_updated::[^\]]+\]\]',
            f'[[last_updated::{timestamp}]]',
            content
        )
        
        file_path.write_text(content, encoding="utf-8")
        logger.info(f"Added {len(new_links)} links to page: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to add links to page {page_id}: {e}")
        return False


# =============================================================================
# Main Merger Class
# =============================================================================

class KnowledgeMerger:
    """
    Merge proposed wiki pages into an existing Knowledge Graph.
    
    Uses Claude Code agent to intelligently decide:
    - Which pages are genuinely new vs duplicates
    - How to update existing pages with new info
    - What new links to create between pages
    
    Example:
        from src.knowledge.learners.repo_learner import RepoLearner
        from src.knowledge.learners.knowledge_merger import KnowledgeMerger
        
        # Extract pages from repo
        learner = RepoLearner()
        pages = learner.learn_pages({"url": "https://github.com/user/repo"})
        
        # Merge into existing KG
        merger = KnowledgeMerger()
        result = merger.merge(pages, repo_url="https://github.com/user/repo")
        
        print(f"Created: {result.created}, Updated: {result.updated}")
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize KnowledgeMerger.
        
        Args:
            params: Optional parameters:
                - timeout: Claude Code timeout in seconds (default: 1800)
                - dry_run: If True, don't actually modify files (default: False)
        """
        self.params = params or {}
        self._timeout = self.params.get("timeout", 1800)
        self._dry_run = self.params.get("dry_run", False)
        self._agent = None
    
    def _initialize_agent(self, workspace: str) -> None:
        """
        Initialize Claude Code agent for merge analysis.
        
        Args:
            workspace: Path to working directory
        """
        try:
            from src.execution.coding_agents.factory import CodingAgentFactory
            
            config = CodingAgentFactory.build_config(
                agent_type="claude_code",
                agent_specific={
                    "allowed_tools": ["Read"],  # Read-only for analysis
                    "timeout": self._timeout,
                    "planning_mode": True,
                }
            )
            
            self._agent = CodingAgentFactory.create(config)
            self._agent.initialize(workspace)
            logger.info(f"Initialized Claude Code agent for merge analysis")
            
        except ImportError as e:
            logger.error(f"Could not import CodingAgentFactory: {e}")
            raise RuntimeError(
                "Claude Code agent not available. "
                "Ensure coding_agents module is properly installed."
            )
        except Exception as e:
            logger.error(f"Failed to initialize Claude Code agent: {e}")
            raise
    
    def merge(
        self,
        proposed_pages: List[WikiPage],
        repo_url: str,
        wiki_dir: Union[str, Path] = DEFAULT_WIKI_DIR,
    ) -> MergeResult:
        """
        Merge proposed pages into the existing Knowledge Graph.
        
        Args:
            proposed_pages: List of WikiPage objects to merge
            repo_url: Source repository URL (for context)
            wiki_dir: Path to wiki directory (default: data/wikis)
            
        Returns:
            MergeResult with statistics and actions taken
        """
        wiki_dir = Path(wiki_dir).expanduser().resolve()
        # Ensure the directory exists so agent initialization and KG scanning
        # do not fail on missing paths (e.g., in tests with a fresh wiki_dir).
        wiki_dir.mkdir(parents=True, exist_ok=True)
        result = MergeResult(total_proposed=len(proposed_pages))
        
        if not proposed_pages:
            logger.warning("No proposed pages to merge")
            return result
        
        try:
            # Step 1: Get summary of existing KG
            kg_summary = get_kg_summary(wiki_dir)
            
            # Step 2: Initialize agent
            self._initialize_agent(str(wiki_dir))
            
            # Step 3: Build merge prompt
            merge_prompt = self._build_merge_prompt(
                proposed_pages,
                kg_summary,
                repo_url,
            )
            
            # Step 4: Run agent to get merge decisions
            logger.info("Running Claude Code agent for merge analysis...")
            agent_result = self._agent.generate_code(merge_prompt)
            
            if not agent_result.success:
                result.errors.append(f"Agent analysis failed: {agent_result.error}")
                return result
            
            # Step 5: Parse merge actions
            actions = parse_merge_output(agent_result.output)
            
            # Step 6: Execute merge actions
            for action in actions:
                result.actions.append(action)
                
                if self._dry_run:
                    logger.info(f"[DRY RUN] Would {action.action}: {action.reason}")
                    continue
                
                success = self._execute_action(action, wiki_dir)
                
                if success:
                    if action.action == "create_new":
                        result.created += 1
                    elif action.action == "update_existing":
                        result.updated += 1
                    elif action.action == "add_links":
                        result.linked += 1
                    elif action.action == "skip":
                        result.skipped += 1
                else:
                    result.errors.append(
                        f"Failed to execute {action.action} for index {action.proposed_index}"
                    )
            
            logger.info(
                f"Merge complete: {result.created} created, {result.updated} updated, "
                f"{result.linked} linked, {result.skipped} skipped"
            )
            return result
            
        except Exception as e:
            logger.error(f"Merge failed: {e}")
            result.errors.append(str(e))
            return result
    
    def _build_merge_prompt(
        self,
        proposed_pages: List[WikiPage],
        kg_summary: List[Dict[str, str]],
        repo_url: str,
    ) -> str:
        """
        Build the prompt for the merge analysis agent.
        
        Args:
            proposed_pages: List of proposed WikiPage objects
            kg_summary: Summary of existing KG pages
            repo_url: Source repository URL
            
        Returns:
            Complete prompt string
        """
        # Load base instructions
        base_instructions = load_merge_prompt()
        
        # Format proposed pages for context
        proposed_json = json.dumps(
            [
                {
                    "index": i,
                    "page_type": p.page_type,
                    "page_title": p.page_title,
                    "overview": p.overview,
                    "domains": p.domains,
                    "outgoing_links": p.outgoing_links,
                }
                for i, p in enumerate(proposed_pages)
            ],
            indent=2
        )
        
        # Format KG summary
        kg_json = json.dumps(kg_summary, indent=2)
        
        # Build complete prompt
        prompt = f"""{base_instructions}

## Repository Source

URL: {repo_url}

## Proposed Pages (from extraction)

```json
{proposed_json}
```

## Existing Knowledge Graph Summary

```json
{kg_json}
```

Now analyze the proposed pages against the existing KG and output your merge decisions in JSON format.
"""
        
        return prompt
    
    def _execute_action(self, action: MergeAction, wiki_dir: Path) -> bool:
        """
        Execute a single merge action.
        
        Args:
            action: MergeAction to execute
            wiki_dir: Path to wiki directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if action.action == "create_new":
                if action.page:
                    return write_new_page(action.page, wiki_dir)
                return False
            
            elif action.action == "update_existing":
                if action.existing_page_id and action.updates:
                    return update_existing_page(
                        action.existing_page_id,
                        action.updates,
                        wiki_dir,
                    )
                return False
            
            elif action.action == "add_links":
                if action.existing_page_id and action.new_links:
                    return add_links_to_page(
                        action.existing_page_id,
                        action.new_links,
                        wiki_dir,
                    )
                return False
            
            elif action.action == "skip":
                # Skip actions are always "successful"
                logger.info(f"Skipped: {action.reason}")
                return True
            
            else:
                logger.warning(f"Unknown action type: {action.action}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to execute action: {e}")
            return False


# =============================================================================
# CLI Test
# =============================================================================

if __name__ == "__main__":
    """Test the KnowledgeMerger with sample data."""
    import sys
    
    print("=" * 60)
    print("KnowledgeMerger Test")
    print("=" * 60)
    
    # Create test proposed pages
    test_pages = [
        WikiPage(
            id="Principle/Test_Principle",
            page_title="Test_Principle",
            page_type="Principle",
            overview="A test principle for the merger",
            content="== Overview ==\nTest content",
            domains=["Test"],
            outgoing_links=[],
        ),
    ]
    
    # Test in dry-run mode
    merger = KnowledgeMerger(params={"dry_run": True})
    
    print(f"\nTest with {len(test_pages)} proposed pages (dry run)")
    print("-" * 60)
    
    # Get KG summary
    wiki_dir = DEFAULT_WIKI_DIR
    if wiki_dir.exists():
        summary = get_kg_summary(wiki_dir)
        print(f"Existing KG has {len(summary)} pages")
        
        # Show a few examples
        for page in summary[:3]:
            print(f"  - {page['page_id']}: {page['overview'][:50]}...")
    else:
        print(f"Wiki directory not found: {wiki_dir}")
    
    print("\n" + "=" * 60)
    print("Test complete!")

