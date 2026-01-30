# Linear Search Strategy
#
# Simple sequential search: generate one solution per iteration,
# implement it, and keep track of the best result.
# No tree structure - just iterate and improve.

import os
import pickle
from typing import List, Optional

from src.execution.types import ContextData
from src.execution.search_strategies.base import (
    SearchStrategy,
    SearchStrategyConfig,
    SearchNode,
)
from src.execution.search_strategies.factory import register_strategy


@register_strategy("linear_search")
class LinearSearch(SearchStrategy):
    """
    Simple linear search strategy.
    
    Each iteration:
    1. Generate a solution based on problem + previous results
    2. Implement and evaluate the solution (developer agent)
    3. Generate feedback
    4. Store result and continue
    
    Config params:
        - idea_generation_model: Model for solution generation (default: gpt-4.1-mini)
    """
    
    def __init__(self, config: SearchStrategyConfig, workspace_dir: Optional[str] = None, import_from_checkpoint: bool = False):
        """Initialize linear search strategy."""
        super().__init__(config, workspace_dir, import_from_checkpoint)
        
        # Config params
        self.idea_generation_model = self.params.get("idea_generation_model", "gpt-4.1-mini")
        
        # State
        if not import_from_checkpoint: 
            self.node_history: List[SearchNode] = []
        self.iteration_count = 0

        print(f"[LinearSearch] Initialized:")
        print(f"  - idea_generation_model: {self.idea_generation_model}")
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
        Execute one iteration of linear search.
        
        Node lifecycle:
        1. Generate solution
        2. Implement (developer agent handles implementation + evaluation)
        3. Extract results from agent output
        4. Generate feedback
        
        Returns:
            SearchNode with solution, evaluation_output, feedback, should_stop
        """
        self.iteration_count += 1
        print(f"\n[LinearSearch] Iteration {self.iteration_count}, budget: {budget_progress:.1f}%")
        
        # Step 1: Generate solution
        solution, ideation_sections = self._generate_solution(context, budget_progress)
        print(f"[LinearSearch] Generated solution ({len(solution)} chars)")
        
        # Create node
        node = SearchNode(
            node_id=len(self.node_history),
            parent_node_id=self._get_best_node_id(),
            solution=solution,
            workspace_dir=self.workspace_dir,
        )
        
        # Step 2: Implement - developer agent handles everything
        branch_name = f"linear_exp_{node.node_id}"
        parent_branch = self._get_best_branch()
        
        print(f"[LinearSearch] Implementing on branch: {branch_name} (from {parent_branch})")
        
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
            print(f"[LinearSearch] Extracted result from agent JSON")
        else:
            # Fallback: use raw agent output
            node.evaluation_output = agent_output
            print(f"[LinearSearch] Warning: No JSON result from agent, using raw output")
        
        # Step 4: Generate feedback
        self._generate_feedback(node)
        
        # Store node
        self.node_history.append(node)
        
        print(f"[LinearSearch] ✓ Node {node.node_id} completed: score={node.score}, should_stop={node.should_stop}")
        
        return node

    def _generate_solution(self, context: ContextData, budget_progress: float) -> tuple[str, list[str]]:
        """Generate a solution based on problem, workflow guidance, and previous experiments."""
        # Build prompt with history
        history_summary = ""
        if self.node_history:
            best = self.get_best_experiment()
            recent = self.node_history[-3:]  # Last 3 nodes
            
            history_summary = "\n\nPrevious experiments:\n"
            for node in recent:
                status = f"score={node.score}" if not node.had_error else "FAILED"
                history_summary += f"- {status}: {node.solution[:200]}...\n"
                if node.feedback:
                    history_summary += f"  Feedback: {node.feedback[:150]}...\n"
            
            if best:
                history_summary += f"\nBest so far (score={best.score}): {best.solution[:300]}..."
        
        # Get workflow guidance from cognitive context manager (if available)
        workflow_guidance = ""
        if context.additional_info:
            workflow_guidance = f"""
WORKFLOW GUIDANCE (from knowledge base):
{context.additional_info}

Use the implementation guide above to structure your solution.
Follow the steps and tips provided.
"""
        
        output_requirements = f"""
                Requirements:
                - Provide a complete, implementable solution
- Follow the workflow guidance if provided
                - Include specific steps, methods, and hyperparameters
                - If there are previous experiments, improve upon the best one
- Budget progress: {budget_progress:.1f}%

                Format your solution with:
                # Core Idea: 
                [Brief description]

                # Solution Steps:
[Detailed implementation steps - follow workflow guidance]

                # Hyperparameters:
                [Specific values to use]
""".strip()
        
        # Build user message with all context
        user_message = f"""
Problem: {str(getattr(context, "problem", ""))}

{workflow_guidance}

{history_summary}

Additional Knowledge: {str(context.kg_results if context.kg_results else "")}

{output_requirements}
"""
        solution = self.llm.llm_completion_with_system_prompt(
            model=self.idea_generation_model,
            system_prompt="You are a world class problem solver generating solutions.",
            user_message=user_message,
        )
        # Return empty list for sections since we no longer use RepoMemory ReAct loop
        return solution, []

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
            print(f"[LinearSearch] Checking out to best branch: {best.branch_name} (score={best.score})")
            self.workspace.switch_branch(best.branch_name)
        else:
            print("[LinearSearch] No successful experiments to checkout")

    def export_checkpoint(self) -> None:
        with open(os.path.join(self.workspace_dir, 'checkpoint.pkl'), 'wb') as f:
            pickle.dump(self.node_history, f)

    def import_checkpoint(self) -> None:
        try:
            with open(os.path.join(self.workspace_dir, 'checkpoint.pkl'), 'rb') as f:
                self.node_history = pickle.load(f)
        except FileNotFoundError:
            print("[LinearSearch] No checkpoint found")
            raise FileNotFoundError(f"[LinearSearch] No checkpoint found in {self.workspace_dir}")
