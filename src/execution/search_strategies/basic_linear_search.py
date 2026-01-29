# Basic Linear Search Strategy
#
# Simple sequential search: generate one solution per iteration,
# implement it, and keep track of the best result.
# No tree structure - just iterate and improve.
#
# Key difference from linear_search:
# - Uses Claude Code as the ideation agent (not engine-mediated ReAct)
# - Connected to MCP gates (idea, code, research) for external knowledge
# - Read-only access to codebase during ideation
# - Full RepoMemory access via CLI

import logging
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple

from src.execution.context_manager.types import ContextData
from src.execution.search_strategies.base import (
    SearchStrategy,
    SearchStrategyConfig,
    SearchNode,
)
from src.execution.search_strategies.factory import register_strategy
from src.repo_memory import RepoMemoryManager
from src.core.prompt_loader import load_prompt, render_prompt

logger = logging.getLogger(__name__)


@register_strategy("basic_linear_search")
class BasicLinearSearch(SearchStrategy):
    """
    Basic linear search strategy with Claude Code ideation.
    
    Each iteration:
    1. Generate a solution using Claude Code + MCP gates (idea, code, research)
    2. Implement and evaluate the solution (developer agent)
    3. Generate feedback
    4. Store result and continue
    
    Key features:
    - Claude Code as ideation agent with read-only codebase access
    - MCP gates for external knowledge (wiki_idea_search, wiki_code_search, research_*)
    - RepoMemory access via CLI for architecture understanding
    
    Config params:
        - idea_generation_model: Model for solution generation (default: claude-opus-4-5-20251101)
        - use_bedrock: Use AWS Bedrock (default: True)
        - aws_region: AWS region (default: us-east-1)
        - ideation_timeout: Timeout for ideation in seconds (default: 300)
        - ideation_gates: MCP gates to enable (default: ["idea", "code", "research"])
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
        self.ideation_gates = self.params.get("ideation_gates", ["idea", "code", "research"])
        
        # State
        if not import_from_checkpoint: 
            self.node_history: List[SearchNode] = []
        self.iteration_count = 0

        print(f"[BasicLinearSearch] Initialized:")
        print(f"  - idea_generation_model: {self.idea_generation_model}")
        print(f"  - use_bedrock: {self.use_bedrock}")
        print(f"  - ideation_gates: {self.ideation_gates}")
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

    def run(self, context: ContextData, budget_progress: float = 0.0) -> SearchNode:
        """
        Execute one iteration of basic linear search.
        
        Node lifecycle:
        1. Generate solution
        2. Implement (developer agent handles implementation + evaluation)
        3. Extract results from agent output
        4. Generate feedback
        
        Returns:
            SearchNode with solution, evaluation_output, feedback, should_stop
        """
        self.iteration_count += 1
        print(f"\n[BasicLinearSearch] Iteration {self.iteration_count}, budget: {budget_progress:.1f}%")
        
        # Determine parent branch once at the start
        parent_branch = self._get_best_branch()
        
        # Step 1: Generate solution
        solution, ideation_sections = self._generate_solution(context, parent_branch)
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
            context=context,
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

    def _generate_solution(self, context: ContextData, parent_branch: str) -> Tuple[str, List[str]]:
        """
        Generate solution using Claude Code with MCP gates.
        
        Uses Claude Code as ideation agent with:
        - Read-only access to repo (Read, Bash for repo_memory_cli.py)
        - RepoMemory via CLI
        - Idea/Code/Research gates via MCP
        
        Args:
            context: ContextData with problem and experiment history
            parent_branch: Git branch to base ideation on
            
        Returns:
            Tuple of (solution_text, sections_consulted)
        """
        from src.execution.coding_agents.base import CodingAgentConfig
        from src.execution.coding_agents.adapters.claude_code_agent import ClaudeCodeCodingAgent
        from src.knowledge.gated_mcp import get_mcp_config
        
        # 1. Load RepoMemory (read-only)
        repo_memory_doc = RepoMemoryManager.load_from_git_branch(
            self.workspace.repo, parent_branch
        ) or {}
        repo_memory_brief = RepoMemoryManager.render_summary_and_toc(
            repo_memory_doc, max_chars=2500
        )
        
        # 2. Get MCP config for idea + code + research gates
        mcp_servers, mcp_tools = get_mcp_config(
            gates=self.ideation_gates,
            include_base_tools=False,
        )
        
        # 3. Build restricted tool set (read-only for ideation)
        # Only allow Read and Bash (for repo_memory_cli.py), plus MCP tools
        ideation_allowed_tools = [
            "Read",
            "Bash",  # For repo_memory_cli.py
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
            problem=str(getattr(context, "problem", "")),
            experiment_history=str(context.additional_info or ""),
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
                return self._fallback_solution(context), []
            
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
        experiment_history: str,
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
                "experiment_history": experiment_history or "(No previous experiments)",
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
        # Look for repo_memory_cli.py get-section calls
        sections = re.findall(r'repo_memory_cli\.py\s+get-section\s+(\S+)', output)
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
    
    def _fallback_solution(self, context: ContextData) -> str:
        """Generate a fallback solution when Claude Code ideation fails."""
        problem = str(getattr(context, "problem", ""))
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
