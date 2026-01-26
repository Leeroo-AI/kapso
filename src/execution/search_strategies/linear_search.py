# Linear Search Strategy
#
# Simple sequential search: generate one solution per iteration,
# implement it, and keep track of the best result.
# No tree structure - just iterate and improve.

import os
import pickle
from typing import List, Optional

from src.execution.context_manager.types import ContextData
from src.execution.search_strategies.base import (
    SearchStrategy,
    SearchStrategyConfig,
    ExperimentResult,
)
from src.execution.search_strategies.factory import register_strategy
from src.execution.ideation.repo_memory_react import ideate_solution_with_repo_memory_react


@register_strategy("linear_search")
class LinearSearch(SearchStrategy):
    """
    Simple linear search strategy.
    
    Each iteration:
    1. Generate a solution based on problem + previous results
    2. Implement and debug the solution
    3. Store result and continue
    
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
            self.experiment_history: List[ExperimentResult] = []
        self.iteration_count = 0

        print(f"[LinearSearch] Initialized:")
        print(f"  - idea_generation_model: {self.idea_generation_model}")
        
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

    def run(self, context: ContextData, budget_progress: float = 0.0) -> ExperimentResult:
        """
        Execute one iteration of linear search.
        
        Simple approach:
        1. Generate a solution considering previous experiments
        2. Implement it (developer agent handles implementation + evaluation)
        3. Store result
        
        Returns:
            ExperimentResult with solution, evaluation_output, code_diff, workspace_dir
        """
        import json
        
        self.iteration_count += 1
        print(f"\n[LinearSearch] Iteration {self.iteration_count}, budget: {budget_progress:.1f}%")
        
        # Generate solution
        solution, ideation_sections = self._generate_solution(context, budget_progress)
        print(f"[LinearSearch] Generated solution ({len(solution)} chars)")
        
        # Implement - developer agent handles everything
        branch_name = f"linear_exp_{len(self.experiment_history)}"
        parent_branch = self._get_best_branch()
        
        print(f"[LinearSearch] Implementing on branch: {branch_name} (from {parent_branch})")
        
        agent_output = self._implement(
            solution=solution,
            context=context,
            branch_name=branch_name,
            parent_branch_name=parent_branch,
            ideation_repo_memory_sections_consulted=ideation_sections,
        )
        
        # Get code diff for this branch
        code_diff = self._get_code_diff(branch_name, parent_branch)
        
        # Read evaluation result from kapso_evaluation/result.json (written by developer agent)
        evaluation_output = agent_output
        evaluation_script_path = ""
        score = 0.0
        had_error = False
        error_message = ""
        
        result_json_path = os.path.join(self.workspace_dir, "kapso_evaluation", "result.json")
        if os.path.exists(result_json_path):
            try:
                with open(result_json_path, 'r') as f:
                    eval_result = json.load(f)
                evaluation_output = eval_result.get("evaluation_output", agent_output)
                evaluation_script_path = eval_result.get("evaluation_script_path", "")
                score = float(eval_result.get("score", 0.0))
                print(f"[LinearSearch] Read evaluation result from {result_json_path}")
            except Exception as e:
                print(f"[LinearSearch] Warning: Could not read result.json: {e}")
        
        # Store result
        experiment_result = ExperimentResult(
            node_id=len(self.experiment_history),
            solution=solution,
            score=score,
            branch_name=branch_name,
            had_error=had_error,
            error_message=error_message,
            output=agent_output,
            detailed_output=agent_output,
            feedbacks="",
            evaluation_output=evaluation_output,
            evaluation_script_path=evaluation_script_path,
            code_diff=code_diff,
            workspace_dir=self.workspace_dir,
        )
        self.experiment_history.append(experiment_result)
        
        print(f"[LinearSearch] ✓ Experiment completed with score: {score}")
        
        return experiment_result
    
    def _get_code_diff(self, branch_name: str, parent_branch: str) -> str:
        """Get git diff between branch and parent."""
        try:
            diff = self.workspace.repo.git.diff(parent_branch, branch_name)
            return diff
        except Exception as e:
            print(f"[LinearSearch] Warning: Could not get diff: {e}")
            return ""

    def _generate_solution(self, context: ContextData, budget_progress: float) -> tuple[str, list[str]]:
        """Generate a solution based on problem, workflow guidance, and previous experiments."""
        parent_branch = self._get_best_branch()
        
        # Build prompt with history
        history_summary = ""
        if self.experiment_history:
            best = self.get_best_experiment()
            recent = self.experiment_history[-3:]  # Last 3 experiments
            
            history_summary = "\n\nPrevious experiments:\n"
            for exp in recent:
                status = f"score={exp.score}" if not exp.had_error else "FAILED"
                history_summary += f"- {status}: {exp.solution[:200]}...\n"
            
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
        
        solution, sections = ideate_solution_with_repo_memory_react(
            llm=self.llm,
            model=self.idea_generation_model,
            repo=self.workspace.repo,
            base_branch=parent_branch,
            problem=str(getattr(context, "problem", "")),
            workflow_guidance=workflow_guidance or "",
            history_summary=history_summary or "",
            additional_knowledge=str(context.kg_results if context.kg_results else ""),
            output_requirements=output_requirements,
        )
        return solution, sections

    def _get_best_branch(self) -> str:
        """Get the branch of the best experiment, or main if none."""
        best = self.get_best_experiment()
        if best:
            return best.branch_name
        return "main"

    def get_experiment_history(self, best_last: bool = False) -> List[ExperimentResult]:
        """Return all experiments, optionally sorted by score."""
        if best_last:
            return sorted(
                self.experiment_history,
                key=lambda exp: (
                    not exp.had_error,
                    exp.score if self.problem_handler.maximize_scoring else -exp.score
                )
            )
        return self.experiment_history
    
    def get_best_experiment(self) -> Optional[ExperimentResult]:
        """Return the best successful experiment."""
        valid = [exp for exp in self.experiment_history if not exp.had_error]
        if not valid:
            return None
        return max(
            valid,
            key=lambda x: x.score if self.problem_handler.maximize_scoring else -x.score
        )

    def checkout_to_best_experiment_branch(self) -> None:
        """Checkout to the best experiment's branch."""
        best = self.get_best_experiment()
        if best:
            print(f"[LinearSearch] Checking out to best branch: {best.branch_name} (score={best.score})")
            self.workspace.switch_branch(best.branch_name)
        else:
            print("[LinearSearch] No successful experiments to checkout")

    def export_checkpoint(self) -> None:
        with open(os.path.join(self.workspace_dir, 'checkpoint.pkl'), 'wb') as f:
            pickle.dump(self.experiment_history, f)

    def import_checkpoint(self) -> None:
        try:
            with open(os.path.join(self.workspace_dir, 'checkpoint.pkl'), 'rb') as f:
                self.experiment_history = pickle.load(f)
        except FileNotFoundError:
            print("[LinearSearch] No checkpoint found")
            raise FileNotFoundError(f"[LinearSearch] No checkpoint found in {self.workspace_dir}")
