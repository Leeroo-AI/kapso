# Phased Repository Ingestor
#
# Extracts knowledge from Git repositories using a multi-branch pipeline.
# The agent directly writes wiki pages following wiki_structure definitions.
# Workflows link to GitHub repositories containing executable implementations.
#
# Phase 0: Repository Understanding (pre-phase)
# - Parse repo structure, generate _RepoMap.md with AST info
# - Agent fills in natural language Understanding for each file
# - Subsequent phases read this file instead of re-exploring
#
# Branch 1: Workflow-Based Extraction
# 1a. Anchoring - Find workflows from README/examples, write Workflow pages + rough WorkflowIndex
# 1b. Anchoring Context - Enrich WorkflowIndex with detailed implementation context
# 2. Excavation+Synthesis - Trace imports, write Implementation-Principle PAIRS together
#    (merged to keep concepts tightly connected to implementations)
# 3. Enrichment - Mine constraints/tips, write Environment/Heuristic pages
# 4. Audit - Validate graph integrity, fix broken links
# 4b. Repo Builder - Create GitHub repositories for workflows (NEW)
#
# Branch 2: Orphan Mining (multi-step pipeline, runs after Branch 1)
# 5a. Triage (code) - Deterministic filtering into AUTO_KEEP/AUTO_DISCARD/MANUAL_REVIEW
# 5b. Review (agent) - Agent evaluates MANUAL_REVIEW files
# 5c. Create (agent) - Agent creates wiki pages for approved files
# 5d. Verify (code) - Verify all approved files have pages
# 6. Orphan Audit - Validate orphan nodes
#
# The final graph is the union of both branches.
# Workflows connect to GitHub repositories instead of Principle pages.
#
# Usage:
#     from kapso.knowledge_base.learners.ingestors import RepoIngestor
#     
#     ingestor = RepoIngestor()
#     pages = ingestor.ingest(Source.Repo("https://github.com/user/repo"))

import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from kapso.execution.coding_agents.factory import CodingAgentFactory
from kapso.knowledge_base.learners.ingestors.base import Ingestor
from kapso.knowledge_base.learners.ingestors.factory import register_ingestor
from kapso.knowledge_base.search.base import WikiPage, DEFAULT_WIKI_DIR
from kapso.knowledge_base.search.kg_graph_search import parse_wiki_directory

from kapso.knowledge_base.learners.ingestors.repo_ingestor.utils import (
    clone_repo,
    cleanup_repo,
    load_wiki_structure,
    get_repo_namespace_from_url,
)
from kapso.knowledge_base.learners.ingestors.repo_ingestor.wiki_validator import validate_wiki_directory
from kapso.knowledge_base.learners.ingestors.repo_ingestor.context_builder import (
    generate_repo_scaffold,
    get_repo_map_path,
    check_exploration_progress,
    generate_orphan_candidates,
    get_orphan_candidates_path,
    verify_orphan_completion,
)
from kapso.knowledge_base.learners.ingestors.repo_ingestor.repo_builder import (
    sanitize_repo_name,
)

logger = logging.getLogger(__name__)

# Path to prompt templates
PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_file = PROMPTS_DIR / f"{name}.md"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_file}")
    return prompt_file.read_text(encoding="utf-8")


@register_ingestor("repo")
class RepoIngestor(Ingestor):
    """
    Phased repository knowledge extraction with two-branch pipeline.
    
    The agent directly writes wiki pages following wiki_structure definitions.
    No intermediate dataclasses - clean and simple.
    
    The agent is given:
    - Read access to the cloned repository
    - Wiki structure definitions (page_definition.md, sections_definition.md)
    - Write access to wiki output directory
    
    Branch 1: Workflow-Based Extraction
    1a. Anchoring: Find workflows → write Workflow pages + rough WorkflowIndex
    1b. Anchoring Context: Enrich WorkflowIndex with detailed implementation context
    2. Excavation+Synthesis: Trace imports → write Implementation-Principle PAIRS together
       (merged phase keeps concepts tightly connected to implementations)
    3. Enrichment: Mine constraints → write Environment/Heuristic pages
    4. Audit: Validate pages → fix broken links
    4b. Repo Builder: Create GitHub repositories for workflows → update workflow pages with URLs
    
    Branch 2: Orphan Mining (Multi-Step Pipeline)
    5a. Triage (code): Deterministic filtering → AUTO_KEEP/AUTO_DISCARD/MANUAL_REVIEW
    5b. Review (agent): Evaluate MANUAL_REVIEW files → approve/reject each
    5c. Create (agent): Create wiki pages for approved files → checkpoint progress
    5d. Verify (code): Check all approved files have pages → report errors
    6. Orphan Audit: Validate orphans → check for hidden workflows, dead code
    
    The final graph is the union of both branches.
    
    Example:
        ingestor = RepoIngestor()
        pages = ingestor.ingest(Source.Repo("https://github.com/unslothai/unsloth"))
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the phased repo ingestor.
        
        Args:
            params: Optional parameters:
                - timeout: Claude Code timeout in seconds (default: 1800)
                - cleanup: Whether to cleanup cloned repos (default: True)
                - wiki_dir: Output directory for wiki pages (default: data/wikis)
                - staging_subdir: Where to stage phase outputs inside wiki_dir (default: "_staging")
                - cleanup_staging: Whether to remove the staging directory after ingest (default: False)
                - fail_on_validation_errors: If True, raise if deterministic validation fails (default: True)
                - github_repo_visibility: "private" or "public" for workflow repos (default: "private")
        """
        super().__init__(params)
        self._timeout = self.params.get("timeout", 1800)  # 30 minutes default
        self._cleanup = self.params.get("cleanup", True)
        self._wiki_dir = Path(self.params.get("wiki_dir", DEFAULT_WIKI_DIR))
        self._staging_subdir = self.params.get("staging_subdir", "_staging")
        self._cleanup_staging = self.params.get("cleanup_staging", False)
        self._fail_on_validation_errors = self.params.get("fail_on_validation_errors", True)
        self._github_repo_visibility = self.params.get("github_repo_visibility", "private")
        self._agent = None
        self._last_repo_path: Optional[Path] = None
        self._last_staging_dir: Optional[Path] = None
    
    @property
    def source_type(self) -> str:
        """Return the source type this ingestor handles."""
        return "repo"
    
    def _initialize_agent(self, workspace: str) -> None:
        """
        Initialize Claude Code agent with read + write tools.
        
        Args:
            workspace: Path to the cloned repository
        
        Supports passing through agent_specific settings from params:
            - use_bedrock: True to use AWS Bedrock instead of direct Anthropic API
            - aws_region: AWS region for Bedrock (default: "us-east-1")
            - model: Model name (required for Bedrock, e.g. "us.anthropic.claude-opus-4-5-20251101-v1:0")
        """
        # Base agent_specific config
        agent_specific = {
            # Read for repo exploration, Write for wiki pages, Bash for file ops
            # Edit included but Write is preferred for index files (Edit can fail on tables)
            "allowed_tools": ["Read", "Write", "Edit", "Bash"],
            "timeout": self._timeout,
            "planning_mode": True,
        }
        
        # Model override - important for Bedrock which requires specific model IDs
        # Bedrock model IDs look like: us.anthropic.claude-opus-4-5-20251101-v1:0
        # Direct Anthropic API uses: claude-opus-4-5
        model = self.params.get("model")
        
        # Pass through Bedrock settings from ingestor params if provided
        # This allows callers to configure AWS Bedrock mode for Claude Code
        if self.params.get("use_bedrock"):
            agent_specific["use_bedrock"] = True
            # Allow aws_region override, default is handled by ClaudeCodeCodingAgent
            if self.params.get("aws_region"):
                agent_specific["aws_region"] = self.params["aws_region"]
            # Default Bedrock model if not specified (Claude Opus 4.5 on Bedrock)
            if not model:
                model = "us.anthropic.claude-opus-4-5-20251101-v1:0"
        
        # Build config for Claude Code with read + write tools
        config = CodingAgentFactory.build_config(
            agent_type="claude_code",
            model=model,
            debug_model=model,  # Use same model for debug
            agent_specific=agent_specific,
        )
        
        self._agent = CodingAgentFactory.create(config)
        self._agent.initialize(workspace)
        logger.info(f"Initialized Claude Code agent for {workspace} (bedrock={agent_specific.get('use_bedrock', False)})")
    
    def _normalize_source(self, source: Any) -> Dict[str, Any]:
        """
        Normalize source input to a dict.
        
        Args:
            source: Source.Repo object or dict
            
        Returns:
            Dict with url and branch keys
        """
        if hasattr(source, 'to_dict'):
            return source.to_dict()
        elif isinstance(source, dict):
            return source
        else:
            raise ValueError(f"Invalid source type: {type(source)}")
    
    def _ensure_wiki_directories(self) -> None:
        """Ensure wiki subdirectories exist."""
        for subdir in ["workflows", "principles", "implementations", "environments", "heuristics"]:
            (self._wiki_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    def _build_phase_prompt(
        self,
        phase: str,
        repo_name: str,
        repo_path: str,
        repo_url: str = "",
        branch: str = "main",
    ) -> str:
        """
        Build prompt for a specific phase.
        
        Includes:
        - Phase-specific instructions from prompt template
        - Relevant wiki structure definitions
        - Path to _RepoMap file (for context from Phase 0)
        - Paths for input/output
        
        Args:
            phase: Phase name (repo_understanding, anchoring, excavation, etc.)
            repo_name: Name of the repository
            repo_path: Path to the cloned repository
            repo_url: Repository URL (for Phase 0)
            branch: Git branch (for Phase 0)
            
        Returns:
            Complete prompt string
        """
        # Load base prompt template
        base_prompt = _load_prompt(phase)
        
        # Determine which wiki structures to include
        structures_needed = {
            # Phase 0: Repository understanding (no wiki structures needed)
            "repo_understanding": [],
            # Branch 1: Workflow-based extraction
            # Phase 1a: anchoring creates workflows and rough WorkflowIndex
            "anchoring": ["workflow"],
            # Phase 1b: anchoring_context enriches WorkflowIndex with implementation details
            "anchoring_context": [],  # No wiki structures needed, just reads existing files
            # Merged excavation+synthesis writes Implementation-Principle pairs together
            "excavation_synthesis": ["implementation", "principle"],
            "enrichment": ["environment", "heuristic"],
            "audit": ["workflow", "principle", "implementation", "environment", "heuristic"],
            # Branch 2: Orphan mining (multi-step)
            "orphan_review": [],  # No wiki structures needed for review
            "orphan_create": ["implementation", "principle"],  # Needs both structures
            "orphan_audit": ["workflow", "implementation", "principle", "heuristic"],
        }
        
        # Load relevant wiki structure definitions
        structure_content = {}
        for page_type in structures_needed.get(phase, []):
            try:
                structure_content[f"{page_type}_structure"] = load_wiki_structure(page_type)
            except FileNotFoundError:
                structure_content[f"{page_type}_structure"] = f"(No structure defined for {page_type})"
        
        # Get path to the _RepoMap file (used by all phases after Phase 0)
        repo_map_path = get_repo_map_path(self._wiki_dir, repo_name)
        
        # Get path to orphan candidates file (used by orphan phases)
        candidates_path = get_orphan_candidates_path(self._wiki_dir)
        
        # Format the prompt with all variables
        format_vars = {
            "repo_name": repo_name,
            "repo_path": repo_path,
            "repo_url": repo_url,
            "branch": branch,
            "wiki_dir": str(self._wiki_dir),
            "repo_map_path": str(repo_map_path),
            "candidates_path": str(candidates_path),  # For orphan phases
            **structure_content,
        }
        
        try:
            return base_prompt.format(**format_vars)
        except KeyError as e:
            # If template has variables we don't have, just return with partial formatting
            logger.warning(f"Missing format variable in {phase} prompt: {e}")
            return base_prompt
    
    def _run_phase(self, phase: str, repo_name: str, repo_path: str, repo_url: str = "", branch: str = "main") -> bool:
        """
        Run a single phase of the extraction pipeline.
        
        Args:
            phase: Phase name
            repo_name: Repository name
            repo_path: Path to cloned repository
            repo_url: Repository URL (optional, for Phase 0)
            branch: Git branch (optional, for Phase 0)
            
        Returns:
            True if phase succeeded, False otherwise
        """
        start = time.time()
        logger.info(f"Running {phase} phase...")
        
        prompt = self._build_phase_prompt(phase, repo_name, repo_path, repo_url, branch)
        result = self._agent.generate_code(prompt)
        
        elapsed = time.time() - start
        
        if not result.success:
            logger.error(f"{phase} phase failed after {elapsed:.1f}s: {result.error}")
            return False
        
        # Log timing from agent metadata if available, otherwise use our own timer
        agent_elapsed = result.metadata.get("elapsed_seconds") if result.metadata else None
        elapsed_str = f"{agent_elapsed:.1f}s" if agent_elapsed else f"{elapsed:.1f}s"
        logger.info(f"{phase} phase complete ({elapsed_str})")
        return True
    
    def _run_phase_zero(self, repo_name: str, repo_path: Path, repo_url: str, branch: str) -> bool:
        """
        Run Phase 0: Repository Understanding with verification loop.
        
        This phase:
        1. Generates a scaffold _RepoMap.md with AST-extracted info
        2. Agent fills in natural language Understanding for each file
        3. Verifies ALL files are explored; if not, re-runs with feedback
        4. Subsequent phases read this file instead of re-exploring
        
        Args:
            repo_name: Repository name
            repo_path: Path to cloned repository
            repo_url: Repository URL
            branch: Git branch
            
        Returns:
            True if phase succeeded (ALL files explored), False otherwise
        """
        start = time.time()
        logger.info("Running Phase 0: Repository Understanding...")
        
        # Step 1: Generate the scaffold with AST info
        # This creates both the compact index AND per-file detail pages
        logger.info("Generating repository scaffold...")
        scaffold = generate_repo_scaffold(
            repo_path=repo_path,
            repo_name=repo_name,
            repo_url=repo_url,
            branch=branch,
            wiki_dir=self._wiki_dir,  # Pass wiki_dir to create _files/ directory
        )
        
        # Step 2: Write scaffold to _RepoMap file
        repo_map_path = get_repo_map_path(self._wiki_dir, repo_name)
        repo_map_path.write_text(scaffold, encoding="utf-8")
        logger.info(f"Wrote scaffold to {repo_map_path}")
        
        # Step 3: Run the repo_understanding prompt with verification loop
        max_attempts = 3  # Maximum number of Phase 0 attempts
        
        for attempt in range(1, max_attempts + 1):
            logger.info(f"Phase 0 attempt {attempt}/{max_attempts}...")
            
            # Build the base prompt
            prompt = self._build_phase_prompt(
                phase="repo_understanding",
                repo_name=repo_name,
                repo_path=str(repo_path),
                repo_url=repo_url,
                branch=branch,
            )
            
            # If this is a retry, add feedback about missing files
            if attempt > 1:
                explored, total, unexplored = check_exploration_progress(repo_map_path)
                feedback = self._build_phase_zero_feedback(explored, total, unexplored)
                prompt = feedback + "\n\n---\n\n" + prompt
                logger.info(f"Added feedback for {len(unexplored)} unexplored files")
            
            result = self._agent.generate_code(prompt)
            
            if not result.success:
                logger.error(f"Phase 0 attempt {attempt} failed: {result.error}")
                continue  # Try again
            
            # Step 4: Verify all files are explored
            explored, total, unexplored = check_exploration_progress(repo_map_path)
            
            if total == 0:
                logger.warning("No files found in index, cannot verify exploration")
                break
            
            if explored >= total:
                elapsed = time.time() - start
                logger.info(f"Phase 0 complete: {explored}/{total} files explored ({elapsed:.1f}s)")
                return True
            
            # Not all files explored
            pct = (explored / total * 100) if total > 0 else 0
            logger.warning(
                f"Phase 0 attempt {attempt}: Only {explored}/{total} files explored ({pct:.0f}%). "
                f"{len(unexplored)} files remaining."
            )
            
            if attempt < max_attempts:
                logger.info("Re-running Phase 0 with feedback on missing files...")
        
        # Exhausted all attempts
        elapsed = time.time() - start
        explored, total, _ = check_exploration_progress(repo_map_path)
        logger.warning(
            f"Phase 0 finished with {explored}/{total} files explored after {max_attempts} attempts ({elapsed:.1f}s). "
            "Continuing with partial coverage."
        )
        return explored > 0  # Return True if we made some progress

    def _build_phase_zero_feedback(self, explored: int, total: int, unexplored: List[str]) -> str:
        """
        Build feedback message for Phase 0 retry.
        
        This augments the prompt with specific guidance on what files are missing.
        """
        lines = []
        lines.append("# ⚠️ INCOMPLETE: Resume Previous Work")
        lines.append("")
        lines.append(f"**Current progress:** {explored}/{total} files explored ({explored/total*100:.0f}%)")
        lines.append("")
        lines.append("**You MUST complete the remaining files.** Do NOT start over.")
        lines.append("")
        lines.append("## Remaining Files to Explore")
        lines.append("")
        
        # Group unexplored by directory for easier navigation
        by_dir: Dict[str, List[str]] = {}
        for path in unexplored:
            parts = path.split("/")
            dir_path = "/".join(parts[:-1]) if len(parts) > 1 else "(root)"
            if dir_path not in by_dir:
                by_dir[dir_path] = []
            by_dir[dir_path].append(parts[-1])
        
        # List by directory (max 50 files shown)
        shown = 0
        for dir_path, files in sorted(by_dir.items()):
            if shown >= 50:
                remaining = sum(len(f) for f in by_dir.values()) - shown
                lines.append(f"\n... and {remaining} more files")
                break
            lines.append(f"**{dir_path}/**")
            for f in files[:10]:
                lines.append(f"- `{f}`")
                shown += 1
            if len(files) > 10:
                lines.append(f"- ... +{len(files)-10} more in this directory")
            lines.append("")
        
        lines.append("---")
        lines.append("")
        lines.append("**Instructions:** Pick up where you left off. For each file above:")
        lines.append("1. Read the source file and its detail page")
        lines.append("2. Fill in the Understanding section")
        lines.append("3. Update the index row (⬜ → ✅, add Purpose)")
        lines.append("")
        lines.append("**Sync the index after every 5-10 files.**")
        lines.append("")
        
        return "\n".join(lines)
    
    def _run_repo_builder_phase(self, repo_name: str, repo_url: str) -> None:
        """
        Run the Repository Builder phase to create GitHub repos for workflows.
        
        Fully agentic approach: the agent reads WorkflowIndex and source code,
        then designs and creates an appropriate repository structure based on
        the workflow's domain.
        
        The agent handles:
        - Reading workflow context from WorkflowIndex and source code
        - Designing domain-appropriate repository structure
        - Implementing clean, runnable code
        - Creating user-friendly documentation
        - GitHub repository creation and pushing
        
        Args:
            repo_name: Repository namespace
            repo_url: Source repository URL (for reference)
        """
        import re
        
        logger.info("Running Repository Builder phase (agentic)...")
        
        # Get path to WorkflowIndex
        workflow_index_path = self._wiki_dir / "_WorkflowIndex.md"
        
        if not workflow_index_path.exists():
            logger.warning("WorkflowIndex not found, skipping repo builder phase")
            return
        
        # Read WorkflowIndex to find all workflows
        content = workflow_index_path.read_text(encoding="utf-8")
        
        # Find workflow sections: ## Workflow: {repo_name}_WorkflowName
        workflow_pattern = rf"## Workflow:\s*({re.escape(repo_name)}_\w+)"
        workflows = re.findall(workflow_pattern, content)
        
        if not workflows:
            logger.warning("No workflows found in WorkflowIndex")
            return
        
        logger.info(f"Found {len(workflows)} workflows to build repositories for")
        
        # Load the agentic repo_builder prompt template
        base_prompt = _load_prompt("repo_builder")
        
        # Process each workflow
        repos_created = 0
        workflow_results = []
        
        for workflow_name in workflows:
            try:
                logger.info(f"Building repository for {workflow_name}")
                
                # Prepare result file path - use ABSOLUTE paths for agent
                # The agent runs in a different workspace, so relative paths won't work
                wiki_dir_abs = self._wiki_dir.resolve()
                repo_path_abs = self._last_repo_path.resolve() if self._last_repo_path else Path.cwd()
                result_file = wiki_dir_abs / "_repo_builder_result.txt"
                
                if result_file.exists():
                    result_file.unlink()  # Remove any previous result
                
                # Build the agentic prompt with full context
                # The agent will read WorkflowIndex and source code itself
                suggested_repo_name = sanitize_repo_name(workflow_name)
                prompt = base_prompt.format(
                    workflow_name=workflow_name,
                    repo_path=str(repo_path_abs),
                    wiki_dir=str(wiki_dir_abs),
                    suggested_repo_name=suggested_repo_name,
                    visibility=self._github_repo_visibility,
                    result_file=str(result_file),
                )
                
                # Run the agent to create the repository
                start = time.time()
                result = self._agent.generate_code(prompt)
                elapsed = time.time() - start
                
                logger.info(f"Agent completed for {workflow_name} in {elapsed:.1f}s (success={result.success})")
                
                # Check result file for GitHub URL
                # Check REGARDLESS of result.success - agent may have completed successfully
                if result_file.exists():
                    github_url = result_file.read_text(encoding="utf-8").strip()
                    if github_url and github_url.startswith("https://github.com/"):
                        # Update the Workflow page with the GitHub URL
                        self._update_workflow_github_url(workflow_name, github_url)
                        repos_created += 1
                        logger.info(f"Created repository: {github_url}")
                        workflow_results.append((workflow_name, "SUCCESS", github_url))
                    else:
                        logger.warning(f"Invalid GitHub URL in result file: {github_url}")
                        if not result.success:
                            workflow_results.append((workflow_name, "FAILED", f"Agent error: {result.error}, Invalid URL: {github_url}"))
                        else:
                            workflow_results.append((workflow_name, "FAILED", f"Invalid URL: {github_url}"))
                else:
                    # No result file - this is a real failure
                    if not result.success:
                        logger.error(f"Agent failed for {workflow_name}: {result.error}")
                        workflow_results.append((workflow_name, "FAILED", f"Agent error: {result.error}"))
                    else:
                        logger.warning(f"Result file not found for {workflow_name}")
                        workflow_results.append((workflow_name, "FAILED", "No result file"))
                    
            except Exception as e:
                logger.error(f"Error processing workflow {workflow_name}: {e}")
                workflow_results.append((workflow_name, "ERROR", str(e)))
                continue
        
        # Write detailed report
        report_path = self._wiki_dir / "_reports" / "phase_repo_builder.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        report_lines = [
            "# Repository Builder Report (Agentic)",
            "",
            "## Summary",
            f"- Workflows processed: {len(workflows)}",
            f"- Repositories created: {repos_created}",
            f"- Visibility: {self._github_repo_visibility}",
            f"- Mode: Fully agentic (agent designed repository structure)",
            "",
            "## Workflow Details",
            "",
            "| Workflow | Status | Details |",
            "|----------|--------|---------|",
        ]
        
        for wf_name, status, details in workflow_results:
            status_emoji = "✅" if status == "SUCCESS" else "⚠️" if status == "SKIPPED" else "❌"
            report_lines.append(f"| {wf_name} | {status_emoji} {status} | {details} |")
        
        report_lines.extend([
            "",
            "## Status",
            "✅ All repositories created successfully" if repos_created == len(workflows) 
            else f"⚠️ {len(workflows) - repos_created} workflows could not be processed",
        ])
        
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        logger.info(f"Repository Builder phase complete: {repos_created}/{len(workflows)} repos created")
    
    def _update_workflow_github_url(self, workflow_name: str, github_url: str) -> None:
        """
        Update a Workflow page with its GitHub repository URL.
        
        Args:
            workflow_name: Workflow page name
            github_url: GitHub repository URL
        """
        import re
        
        workflow_path = self._wiki_dir / "workflows" / f"{workflow_name}.md"
        
        if not workflow_path.exists():
            logger.warning(f"Workflow page not found: {workflow_path}")
            return
        
        content = workflow_path.read_text(encoding="utf-8")
        
        # Replace PENDING placeholder with actual URL
        # Pattern: [[github_url::PENDING_REPO_BUILD]] or [[github_url::PENDING]]
        updated = re.sub(
            r'\[\[github_url::(PENDING_REPO_BUILD|PENDING)\]\]',
            f'[[github_url::{github_url}]]',
            content
        )
        
        # If no placeholder found, try to update the GitHub URL section
        if updated == content:
            # Look for == GitHub URL == section and update it
            github_section_pattern = r'(== GitHub URL ==.*?)(\[\[github_url::)([^\]]+)(\]\])'
            match = re.search(github_section_pattern, content, re.DOTALL)
            if match:
                updated = re.sub(
                    github_section_pattern,
                    f'\\1\\2{github_url}\\4',
                    content,
                    flags=re.DOTALL
                )
        
        if updated != content:
            workflow_path.write_text(updated, encoding="utf-8")
            logger.info(f"Updated {workflow_name} with GitHub URL: {github_url}")
        else:
            # Append GitHub URL section if not found (minimal format)
            github_section = f"""

== GitHub URL ==

[[github_url::{github_url}]]
"""
            workflow_path.write_text(content + github_section, encoding="utf-8")
            logger.info(f"Added GitHub URL section to {workflow_name}")
    
    def _collect_written_pages(self, repo_name: str) -> List[WikiPage]:
        """
        Collect WikiPage objects from files written by agent.
        
        Scans wiki_dir for .md files and parses them.
        
        Args:
            repo_name: Repository name to filter pages
            
        Returns:
            List of WikiPage objects
        """
        try:
            pages = parse_wiki_directory(self._wiki_dir)
            
            # IMPORTANT:
            # RepoIngestor runs phases in an isolated staging wiki_dir, so parsing that
            # directory returns only pages for this ingestion run.
            logger.info(f"Collected {len(pages)} pages from {self._wiki_dir}")
            return pages
            
        except Exception as e:
            logger.error(f"Failed to collect pages: {e}")
            return []
    
    def ingest(self, source: Any) -> List[WikiPage]:
        """
        Run the two-branch extraction pipeline (7 phases total).
        
        Branch 1 (Workflow-based): anchoring → excavation → synthesis → enrichment → audit
        Branch 2 (Orphan mining): orphan_mining → orphan_audit
        
        Args:
            source: Source.Repo object or dict with:
                - url: GitHub repository URL (required)
                - branch: Branch to clone (default: "main")
            
        Returns:
            List of WikiPage objects extracted from the repository
            (union of both branches)
            
        Raises:
            ValueError: If url is not provided
            RuntimeError: If cloning or extraction fails
        """
        # Normalize source to dict
        source_data = self._normalize_source(source)
        
        url = source_data.get("url", "")
        branch = source_data.get("branch", "main")
        
        if not url:
            raise ValueError("Repository URL is required")
        
        repo_path = None
        staging_dir = None
        wiki_dir_original: Optional[Path] = None
        
        try:
            wiki_dir_original = self._wiki_dir
            # Step 1: Clone the repository
            repo_path = clone_repo(url, branch)
            self._last_repo_path = repo_path
            repo_name = get_repo_namespace_from_url(url)
            
            # Step 1b: Create isolated staging directory for this ingestion run.
            # This prevents cross-repo contamination when wiki_dir is shared.
            run_id = uuid.uuid4().hex[:12]
            staging_dir = (self._wiki_dir / self._staging_subdir / repo_name / run_id)
            self._last_staging_dir = staging_dir
            logger.info(f"Repo ingestion staging wiki_dir: {staging_dir}")
            
            # IMPORTANT: phases should only read/write inside the staging wiki_dir,
            # never directly in the global wiki_dir.
            self._wiki_dir = staging_dir
            
            # Step 2: Ensure wiki directories exist
            self._ensure_wiki_directories()
            
            # Step 3: Initialize Claude Code agent
            self._initialize_agent(str(repo_path))
            
            # Step 4: Run Phase 0 - Repository Understanding
            logger.info("=" * 60)
            logger.info("PHASE 0: Repository Understanding")
            logger.info("=" * 60)
            
            success = self._run_phase_zero(repo_name, repo_path, url, branch)
            if not success:
                logger.warning("Phase 0 failed, continuing with limited context...")
            
            # Step 5: Run Branch 1 - Workflow-based extraction
            logger.info("=" * 60)
            logger.info("BRANCH 1: Workflow-Based Extraction")
            logger.info("=" * 60)
            
            # Two-phase anchoring:
            # - anchoring: Creates Workflow pages + rough WorkflowIndex
            # - anchoring_context: Enriches WorkflowIndex with detailed implementation context
            # Merged excavation+synthesis keeps Implementation-Principle pairs together
            # This prevents disconnect between concepts and their implementations
            branch1_phases = ["anchoring", "anchoring_context", "excavation_synthesis", "enrichment", "audit"]
            
            for phase in branch1_phases:
                success = self._run_phase(phase, repo_name, str(repo_path), url, branch)
                if not success:
                    logger.warning(f"Phase {phase} failed, continuing to next phase...")
                    # Continue even if a phase fails - try to extract what we can
            
            # Step 5b: Run Repository Builder Phase (creates GitHub repos for workflows)
            logger.info("=" * 60)
            logger.info("PHASE 4b: Repository Builder")
            logger.info("=" * 60)
            self._run_repo_builder_phase(repo_name, url)
            
            # Step 6: Run Branch 2 - Orphan mining (multi-step pipeline)
            logger.info("=" * 60)
            logger.info("BRANCH 2: Orphan Mining")
            logger.info("=" * 60)
            
            # Step 6a: Triage (code-based, deterministic)
            # Generates _orphan_candidates.md with AUTO_KEEP, AUTO_DISCARD, MANUAL_REVIEW
            logger.info("Step 6a: Orphan Triage (deterministic)...")
            candidates_path = generate_orphan_candidates(
                repo_map_path=get_repo_map_path(self._wiki_dir, repo_name),
                wiki_dir=self._wiki_dir,
                repo_name=repo_name,
            )
            logger.info(f"Orphan candidates written to: {candidates_path}")
            
            # Step 6b: Review (agent evaluates MANUAL_REVIEW files)
            logger.info("Step 6b: Orphan Review (agent evaluation)...")
            success = self._run_phase("orphan_review", repo_name, str(repo_path), url, branch)
            if not success:
                logger.warning("Orphan review phase failed, continuing...")
            
            # Step 6c: Create (agent creates wiki pages for approved files)
            logger.info("Step 6c: Orphan Create (page generation)...")
            success = self._run_phase("orphan_create", repo_name, str(repo_path), url, branch)
            if not success:
                logger.warning("Orphan create phase failed, continuing...")
            
            # Step 6d: Verify (code-based verification)
            logger.info("Step 6d: Orphan Verification (deterministic)...")
            verify_success, verify_report = verify_orphan_completion(self._wiki_dir, repo_name)
            if not verify_success:
                logger.warning(f"Orphan verification found issues:\n{verify_report}")
                # Write verification report to _reports directory
                reports_dir = self._wiki_dir / "_reports"
                reports_dir.mkdir(parents=True, exist_ok=True)
                (reports_dir / "phase5d_orphan_verify.md").write_text(verify_report, encoding="utf-8")
            else:
                logger.info("Orphan verification passed")
            
            # Phase 7: Orphan Audit (final validation)
            logger.info("Phase 7: Orphan Audit...")
            success = self._run_phase("orphan_audit", repo_name, str(repo_path), url, branch)
            if not success:
                logger.warning("Orphan audit phase failed, continuing...")
            
            # Step 7: Collect pages written by agent (union of both branches)
            report = validate_wiki_directory(self._wiki_dir)
            if report.errors:
                msg = (
                    f"Deterministic wiki validation failed for {repo_name} "
                    f"({len(report.errors)} errors, {len(report.warnings)} warnings)."
                )
                logger.error(msg)
                # One extra targeted audit pass (best-effort) with concrete failures.
                audit_prompt = self._build_phase_prompt("audit", repo_name, str(repo_path))
                audit_prompt += "\n\n## Deterministic Validator Findings (MUST FIX)\n\n"
                audit_prompt += report.to_text()
                _ = self._agent.generate_code(audit_prompt)
                # Re-validate after the extra audit pass.
                report = validate_wiki_directory(self._wiki_dir)
                if report.errors and self._fail_on_validation_errors:
                    raise RuntimeError(
                        f"Wikis are still invalid after re-audit. "
                        f"Fix errors in staging dir: {self._wiki_dir}\n\n{report.to_text()}"
                    )
            
            pages = self._collect_written_pages(repo_name)
            
            # Step 8: Publish phase outputs into the final wiki_dir (optional but default-safe)
            # We only publish if staging is inside wiki_dir; copy is limited to this run.
            # This preserves existing behavior for extract-only runs, while avoiding
            # phases accidentally reading/editing other repos.
            if wiki_dir_original and wiki_dir_original != self._wiki_dir:
                # Ensure final output dirs exist.
                for subdir in ["workflows", "principles", "implementations", "environments", "heuristics"]:
                    (wiki_dir_original / subdir).mkdir(parents=True, exist_ok=True)
                
                for subdir in ["workflows", "principles", "implementations", "environments", "heuristics"]:
                    src_dir = self._wiki_dir / subdir
                    dst_dir = wiki_dir_original / subdir
                    if not src_dir.exists():
                        continue
                    for md_file in src_dir.glob("*.md"):
                        # Copy into final directory, overwriting same-name pages.
                        shutil.copy2(md_file, dst_dir / md_file.name)
                logger.info(f"Published staged wiki files into final wiki_dir: {wiki_dir_original}")
                
                # Restore global wiki_dir for downstream callers.
                self._wiki_dir = wiki_dir_original
            
            logger.info(f"Extracted {len(pages)} pages from {url}")
            return pages
            
        finally:
            # Restore wiki_dir if we swapped to staging.
            if wiki_dir_original is not None and staging_dir and self._wiki_dir == staging_dir:
                self._wiki_dir = wiki_dir_original
            
            # Cleanup cloned repository
            if self._cleanup and repo_path:
                cleanup_repo(repo_path)
                self._last_repo_path = None
            
            # Cleanup staging outputs if requested
            if self._cleanup_staging and staging_dir and staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
                self._last_staging_dir = None
    
    def get_last_repo_path(self) -> Optional[Path]:
        """
        Get the path to the last cloned repository.
        
        Useful for debugging or if cleanup=False was set.
        
        Returns:
            Path to the cloned repo, or None if cleaned up
        """
        return self._last_repo_path

    def get_last_staging_dir(self) -> Optional[Path]:
        """
        Get the path to the last staging wiki directory.
        
        Useful for debugging phase outputs before merge.
        """
        return self._last_staging_dir

