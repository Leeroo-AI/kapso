# Basic Linear Search Strategy
#
# Simple sequential search: generate one solution per iteration,
# implement it, and keep track of the best result.
# No tree structure - just iterate and improve.
#
# Key difference from linear_search:
# - Uses Claude Code as the ideation agent (not engine-mediated ReAct)
# - Connected to MCP gates (idea, code, research, experiment_history) for external knowledge
# - Read-only access to codebase during ideation
# - Full RepoMemory access via CLI

import logging
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple

from kapso.execution.search_strategies.base import (
    SearchStrategy,
    SearchStrategyConfig,
    SearchNode,
)
from kapso.execution.search_strategies.factory import register_strategy
from kapso.repo_memory import RepoMemoryManager
from kapso.core.prompt_loader import load_prompt, render_prompt

logger = logging.getLogger(__name__)


@register_strategy("basic_linear_search")
class BasicLinearSearch(SearchStrategy):
    """
    Basic linear search strategy with Claude Code ideation and implementation.
    
    Each iteration:
    1. Generate a solution using Claude Code + MCP gates (idea, code, research, experiment_history)
    2. Implement and evaluate using Claude Code + MCP gates (code, research)
    3. Generate feedback
    4. Store result and continue
    
    Key features:
    - Claude Code as ideation agent with read-only codebase access
    - Claude Code as implementation agent with full write access
    - MCP gates for external knowledge (wiki_idea_search, wiki_code_search, research_*, experiment_history)
    - RepoMemory access via CLI for architecture understanding
    
    Config params:
        - idea_generation_model: Model for solution generation (default: claude-opus-4-5-20251101)
        - implementation_model: Model for implementation (default: claude-opus-4-5-20251101)
        - use_bedrock: Use AWS Bedrock (default: True)
        - aws_region: AWS region (default: us-east-1)
        - ideation_timeout: Timeout for ideation in seconds (default: 300)
        - implementation_timeout: Timeout for implementation in seconds (default: 600)
        - ideation_gates: MCP gates for ideation (default: ["idea", "code", "research", "experiment_history"])
        - implementation_gates: MCP gates for implementation (default: ["code", "research"])
    """
    
    def __init__(self, config: SearchStrategyConfig, workspace_dir: Optional[str] = None, import_from_checkpoint: bool = False):
        """Initialize basic linear search strategy."""
        super().__init__(config, workspace_dir, import_from_checkpoint)
        
        # Config params for ideation
        self.idea_generation_model = self.params.get(
            "idea_generation_model", 
            "us.anthropic.claude-opus-4-5-20251101-v1:0"
        )
        self.use_bedrock = self.params.get("use_bedrock", True)
        self.aws_region = self.params.get("aws_region", "us-east-1")
        self.ideation_timeout = self.params.get("ideation_timeout", 300)
        # Include experiment_history gate by default for ideation
        self.ideation_gates = self.params.get("ideation_gates", ["idea", "code", "research", "experiment_history"])
        
        # Config params for implementation
        self.implementation_model = self.params.get(
            "implementation_model",
            "us.anthropic.claude-opus-4-5-20251101-v1:0"
        )
        self.implementation_timeout = self.params.get("implementation_timeout", 600)
        self.implementation_gates = self.params.get("implementation_gates", ["code", "research"])
        
        # Experiment history path (set by orchestrator)
        self.experiment_history_path = self.params.get(
            "experiment_history_path",
            os.path.join(self.workspace_dir, ".kapso", "experiment_history.json")
        )
        
        # State
        if not import_from_checkpoint: 
            self.node_history: List[SearchNode] = []
        self.iteration_count = 0

        print(f"[BasicLinearSearch] Initialized:")
        print(f"  - idea_generation_model: {self.idea_generation_model}")
        print(f"  - implementation_model: {self.implementation_model}")
        print(f"  - use_bedrock: {self.use_bedrock}")
        print(f"  - ideation_gates: {self.ideation_gates}")
        print(f"  - implementation_gates: {self.implementation_gates}")
        print(f"  - experiment_history_path: {self.experiment_history_path}")
        print(f"  - feedback_generator: {'configured' if self.feedback_generator else 'not configured'}")
        
        # Initialize workspace with empty main file only for empty workspaces.
        # If the workspace is seeded from an existing repo, we must not overwrite it.
        if workspace_dir is None and not self.workspace.is_seeded:
            self._initialize_workspace()
    
    def _initialize_workspace(self) -> None:
        """Create initial empty main file."""
        session = self.workspace.create_experiment_session(
            branch_name=self.workspace.get_current_branch()
        )
        session.generate_code(
            f"<problem>\n{self.problem_handler.get_problem_context()}\n</problem>\n\n"
            + "Create an empty main with a main() function placeholder. No comments."
        )
        self.workspace.finalize_session(session)
        self.workspace.repo.git.stash()

    def run(self, context: Any, budget_progress: float = 0.0) -> SearchNode:
        """
        Execute one iteration of basic linear search.
        
        Node lifecycle:
        1. Generate solution (agent queries experiment history via MCP)
        2. Implement (developer agent handles implementation + evaluation)
        3. Extract results from agent output
        4. Generate feedback
        
        Args:
            context: Either a ContextData object (legacy) or a problem string
            budget_progress: Budget progress percentage (0-100)
        
        Returns:
            SearchNode with solution, evaluation_output, feedback, should_stop
        """
        self.iteration_count += 1
        print(f"\n[BasicLinearSearch] Iteration {self.iteration_count}, budget: {budget_progress:.1f}%")
        
        # Extract problem from context (support both string and ContextData)
        if isinstance(context, str):
            problem = context
        else:
            problem = str(getattr(context, "problem", context))
        
        # Determine parent branch once at the start
        parent_branch = self._get_best_branch()
        
        # Step 1: Generate solution (agent queries experiment history via MCP)
        solution, ideation_sections = self._generate_solution(problem, parent_branch)
        print(f"[BasicLinearSearch] Generated solution ({len(solution)} chars)")
        
        # Create node
        node = SearchNode(
            node_id=len(self.node_history),
            parent_node_id=self._get_best_node_id(),
            solution=solution,
            workspace_dir=self.workspace_dir,
        )
        
        # Step 2: Implement - developer agent handles everything
        branch_name = f"basic_linear_exp_{node.node_id}"
        
        print(f"[BasicLinearSearch] Implementing on branch: {branch_name} (from {parent_branch})")
        
        agent_output = self._implement(
            solution=solution,
            problem=problem,
            branch_name=branch_name,
            parent_branch_name=parent_branch,
            ideation_repo_memory_sections_consulted=ideation_sections,
        )
        
        # Update node with implementation results
        node.branch_name = branch_name
        node.agent_output = agent_output
        node.code_diff = self._get_code_diff(branch_name, parent_branch)
        
        # Step 3: Extract results from agent output JSON
        agent_result = self._extract_agent_result(agent_output)
        
        if agent_result:
            node.code_changes_summary = agent_result.get("code_changes_summary", "")
            node.evaluation_script_path = agent_result.get("evaluation_script_path", "")
            node.evaluation_output = agent_result.get("evaluation_output", agent_output)
            # Score from agent result (may be overridden by feedback generator)
            if agent_result.get("score") is not None:
                node.score = float(agent_result.get("score", 0.0))
            print(f"[BasicLinearSearch] Extracted result from agent JSON")
        else:
            # Fallback: use raw agent output
            node.evaluation_output = agent_output
            print(f"[BasicLinearSearch] Warning: No JSON result from agent, using raw output")
        
        # Step 4: Generate feedback
        self._generate_feedback(node)
        
        # Store node
        self.node_history.append(node)
        
        print(f"[BasicLinearSearch] ✓ Node {node.node_id} completed: score={node.score}, should_stop={node.should_stop}")
        
        return node

    def _generate_solution(self, problem: str, parent_branch: str) -> Tuple[str, List[str]]:
        """
        Generate solution using Claude Code with MCP gates.
        
        Uses Claude Code as ideation agent with:
        - Read-only access to repo (Read, MCP tools for repo_memory)
        - RepoMemory via CLI
        - Idea/Code/Research/ExperimentHistory gates via MCP
        
        Args:
            problem: Problem description
            parent_branch: Git branch to base ideation on
            
        Returns:
            Tuple of (solution_text, sections_consulted)
        """
        from kapso.execution.coding_agents.base import CodingAgentConfig
        from kapso.execution.coding_agents.adapters.claude_code_agent import ClaudeCodeCodingAgent
        from kapso.knowledge.gated_mcp import get_mcp_config
        
        # 1. Load RepoMemory (read-only)
        repo_memory_doc = RepoMemoryManager.load_from_git_branch(
            self.workspace.repo, parent_branch
        ) or {}
        repo_memory_brief = RepoMemoryManager.render_summary_and_toc(
            repo_memory_doc, max_chars=2500
        )
        
        # 2. Get MCP config for idea + code + research + experiment_history gates
        mcp_servers, mcp_tools = get_mcp_config(
            gates=self.ideation_gates,
            experiment_history_path=self.experiment_history_path,
            include_base_tools=False,
        )
        
        # 3. Build restricted tool set (read-only for ideation)
        # Only allow Read plus MCP tools (repo_memory is now via MCP)
        ideation_allowed_tools = [
            "Read",
            *[t for t in mcp_tools if t.startswith("mcp__")],
        ]
        
        logger.info(f"[BasicLinearSearch] Ideation tools: {ideation_allowed_tools}")
        
        # 4. Configure Claude Code for ideation (read-only mode)
        config = CodingAgentConfig(
            agent_type="claude_code",
            model=self.idea_generation_model,
            debug_model=self.idea_generation_model,
            agent_specific={
                "use_bedrock": self.use_bedrock,
                "aws_region": self.aws_region,
                "mcp_servers": mcp_servers,
                "allowed_tools": ideation_allowed_tools,
                "timeout": self.ideation_timeout,
                "streaming": True,
                "planning_mode": False,  # Direct execution for ideation
            }
        )
        
        # 5. Build ideation prompt
        prompt = self._build_ideation_prompt(
            problem=problem,
            repo_memory_brief=repo_memory_brief,
        )
        
        # 6. Run Claude Code for ideation
        print(f"[BasicLinearSearch] Running Claude Code ideation...")
        agent = ClaudeCodeCodingAgent(config)
        agent.initialize(self.workspace_dir)
        
        try:
            result = agent.generate_code(prompt)
            
            if not result.success:
                logger.warning(f"[BasicLinearSearch] Ideation failed: {result.error}")
                # Return a fallback solution
                return self._fallback_solution(problem), []
            
            # Extract solution from output
            solution = self._extract_solution_from_output(result.output)
            sections_consulted = self._extract_sections_consulted(result.output)
            
            print(f"[BasicLinearSearch] Ideation complete, sections consulted: {sections_consulted}")
            return solution, sections_consulted
            
        finally:
            agent.cleanup()
    
    def _build_ideation_prompt(
        self,
        problem: str,
        repo_memory_brief: str,
    ) -> str:
        """Build the ideation prompt for Claude Code."""
        # Load and render the prompt template
        template = load_prompt("execution/prompts/ideation_claude_code.md")
        return render_prompt(
            template,
            {
                "problem": problem or "(No problem description provided)",
                "repo_memory_brief": repo_memory_brief or "(No repo memory available)",
            },
        )
    
    def _extract_solution_from_output(self, output: str) -> str:
        """Extract solution from Claude Code output."""
        # Look for <solution>...</solution> tags
        match = re.search(r'<solution>(.*?)</solution>', output, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Fallback: look for markdown headers that indicate a solution
        # Try to find "# Core Idea" section
        core_idea_match = re.search(r'#\s*Core Idea.*', output, re.DOTALL)
        if core_idea_match:
            return core_idea_match.group(0).strip()
        
        # Last resort: return entire output (may contain useful info)
        logger.warning("[BasicLinearSearch] Could not extract solution tags, using full output")
        return output
    
    def _extract_sections_consulted(self, output: str) -> List[str]:
        """Extract RepoMemory sections consulted from Claude Code output."""
        # Look for repo_memory MCP tool calls
        sections = re.findall(r'get_repo_memory_section.*section_id.*["\']([^"\']+)["\']', output)
        # Also look for direct section references in tool calls
        sections.extend(re.findall(r'get-section\s+["\']?(\S+?)["\']?\s', output))
        # Deduplicate while preserving order
        seen = set()
        result = []
        for s in sections:
            # Clean up section ID (remove quotes, trailing punctuation)
            s = s.strip('"\'.,;:')
            if s and s not in seen:
                seen.add(s)
                result.append(s)
        return result
    
    def _fallback_solution(self, problem: str) -> str:
        """Generate a fallback solution when Claude Code ideation fails."""
        return f"""# Core Idea
Implement a baseline solution for the given problem.

# Solution Steps
1. Analyze the problem requirements
2. Implement a straightforward solution
3. Add basic error handling
4. Create evaluation metrics

# Hyperparameters
- Use default values from the problem description

# Rationale
Fallback solution due to ideation failure. Focus on correctness over optimization.

Problem: {problem}"""

    def _implement(
        self,
        solution: str,
        problem: str,
        branch_name: str,
        parent_branch_name: str = "main",
        ideation_repo_memory_sections_consulted: Optional[List[str]] = None,
    ) -> str:
        """
        Implementation using Claude Code with MCP gates (code, research).
        
        Overrides base class to use Claude Code with Bedrock and MCP gates
        instead of the default coding agent from config.
        
        Args:
            solution: Solution description to implement
            problem: Problem description
            branch_name: Git branch for this experiment
            parent_branch_name: Parent branch to inherit code from
            ideation_repo_memory_sections_consulted: RepoMemory sections used during ideation
            
        Returns:
            The agent's output string
        """
        from kapso.execution.coding_agents.base import CodingAgentConfig
        from kapso.execution.coding_agents.adapters.claude_code_agent import ClaudeCodeCodingAgent
        from kapso.knowledge.gated_mcp import get_mcp_config
        from kapso.repo_memory.observation import extract_repo_memory_sections_consulted
        
        # Create experiment session (handles git branching)
        session = self.workspace.create_experiment_session(branch_name, parent_branch_name, llm=self.llm)
        
        # 1. Load RepoMemory
        repo_memory_doc = RepoMemoryManager.ensure_exists_in_worktree(session.session_folder)
        repo_memory_brief = RepoMemoryManager.render_summary_and_toc(repo_memory_doc, max_chars=2500)
        
        # 2. Get MCP config for code + research gates (not idea)
        mcp_servers, mcp_tools = get_mcp_config(
            gates=self.implementation_gates,
            include_base_tools=False,
        )
        
        # 3. Build full tool set for implementation (includes Write, Edit)
        implementation_allowed_tools = [
            "Read", "Write", "Edit", "Bash",
            *[t for t in mcp_tools if t.startswith("mcp__")],
        ]
        
        logger.info(f"[BasicLinearSearch] Implementation tools: {implementation_allowed_tools}")
        
        # 4. Configure Claude Code for implementation
        config = CodingAgentConfig(
            agent_type="claude_code",
            model=self.implementation_model,
            debug_model=self.implementation_model,
            agent_specific={
                "use_bedrock": self.use_bedrock,
                "aws_region": self.aws_region,
                "mcp_servers": mcp_servers,
                "allowed_tools": implementation_allowed_tools,
                "timeout": self.implementation_timeout,
                "streaming": True,
            }
        )
        
        # 5. Build implementation prompt
        repo_memory_detail_access_instructions = (
            "For detailed section content (architecture, gotchas, invariants, etc.),\n"
            "use the MCP tool: `get_repo_memory_section(section_id=\"core.architecture\")`\n"
            "Available sections: core.architecture, core.entrypoints, core.where_to_edit, core.invariants, core.testing, core.gotchas, core.dependencies\n"
            "Fallback: open `.kapso/repo_memory.json` and read `book.sections[section_id]`."
        )
        
        prompt = self._build_implementation_prompt(
            solution=solution,
            problem=problem,
            branch_name=branch_name,
            repo_memory_brief=repo_memory_brief,
            repo_memory_detail_access_instructions=repo_memory_detail_access_instructions,
            previous_errors="\n".join(str(e) for e in self.previous_errors[-self.recent_error_count:]),
        )
        
        # 6. Run Claude Code for implementation
        print(f"[BasicLinearSearch] Running Claude Code implementation...")
        agent = ClaudeCodeCodingAgent(config)
        agent.initialize(session.session_folder)
        
        try:
            result = agent.generate_code(prompt)
            agent_output = result.output if result.output else ""
            
            if not result.success:
                logger.warning(f"[BasicLinearSearch] Implementation failed: {result.error}")
                agent_output = f"Implementation failed: {result.error}\n\n{agent_output}"
        finally:
            agent.cleanup()
        
        # 7. Update RepoMemory for this experiment branch
        run_result_payload = {
            "score": 0,
            "run_had_error": False,
            "error_message": "",
            "error_details": "",
            "feedbacks": "",
            "ideation_repo_memory_sections_consulted": ideation_repo_memory_sections_consulted or [],
        }
        
        # Extract sections consulted from changes.log
        sections_consulted = []
        try:
            changes_log_path = os.path.join(session.session_folder, "changes.log")
            if os.path.exists(changes_log_path):
                with open(changes_log_path, "r", encoding="utf-8", errors="replace") as f:
                    sections_consulted = extract_repo_memory_sections_consulted(f.read())
        except Exception:
            sections_consulted = []
        run_result_payload["repo_memory_sections_consulted"] = sections_consulted
        
        # Schedule RepoMemory update for session close
        session.schedule_repo_memory_update(
            solution_spec=solution,
            run_result=run_result_payload,
        )
        
        # 8. Finalize session (commits changes)
        self.workspace.finalize_session(session)
        
        return agent_output
    
    def _build_implementation_prompt(
        self,
        solution: str,
        problem: str,
        branch_name: str,
        repo_memory_brief: str,
        repo_memory_detail_access_instructions: str,
        previous_errors: str,
    ) -> str:
        """Build the implementation prompt for Claude Code."""
        template = load_prompt("execution/prompts/implementation_claude_code.md")
        return render_prompt(
            template,
            {
                "solution": solution or "(No solution provided)",
                "problem": problem or "(No problem description provided)",
                "branch_name": branch_name,
                "repo_memory_brief": repo_memory_brief or "(No repo memory available)",
                "repo_memory_detail_access_instructions": repo_memory_detail_access_instructions,
                "previous_errors": previous_errors or "(No previous errors)",
            },
        )

    def _get_best_branch(self) -> str:
        """Get the branch of the best node, or main if none."""
        best = self.get_best_experiment()
        if best:
            return best.branch_name
        return "main"
    
    def _get_best_node_id(self) -> Optional[int]:
        """Get the node_id of the best node, or None if none."""
        best = self.get_best_experiment()
        if best:
            return best.node_id
        return None

    def get_experiment_history(self, best_last: bool = False) -> List[SearchNode]:
        """Return all nodes, optionally sorted by score."""
        if best_last:
            return sorted(
                self.node_history,
                key=lambda node: (
                    not node.had_error,
                    (node.score or 0) if self.problem_handler.maximize_scoring else -(node.score or 0)
                )
            )
        return self.node_history
    
    def get_best_experiment(self) -> Optional[SearchNode]:
        """Return the best successful node."""
        valid = [node for node in self.node_history if not node.had_error]
        if not valid:
            return None
        return max(
            valid,
            key=lambda x: (x.score or 0) if self.problem_handler.maximize_scoring else -(x.score or 0)
        )

    def checkout_to_best_experiment_branch(self) -> None:
        """Checkout to the best node's branch."""
        best = self.get_best_experiment()
        if best:
            print(f"[BasicLinearSearch] Checking out to best branch: {best.branch_name} (score={best.score})")
            self.workspace.switch_branch(best.branch_name)
        else:
            print("[BasicLinearSearch] No successful experiments to checkout")

    def export_checkpoint(self) -> None:
        with open(os.path.join(self.workspace_dir, 'checkpoint.pkl'), 'wb') as f:
            pickle.dump(self.node_history, f)

    def import_checkpoint(self) -> None:
        try:
            with open(os.path.join(self.workspace_dir, 'checkpoint.pkl'), 'rb') as f:
                self.node_history = pickle.load(f)
        except FileNotFoundError:
            print("[BasicLinearSearch] No checkpoint found")
            raise FileNotFoundError(f"[BasicLinearSearch] No checkpoint found in {self.workspace_dir}")
