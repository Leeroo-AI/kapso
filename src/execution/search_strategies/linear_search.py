# Linear Search Strategy
#
# Simple sequential search: generate one solution per iteration,
# implement it, and keep track of the best result.
# No tree structure - just iterate and improve.

from typing import List, Optional

from src.execution.context_manager.types import ContextData
from src.execution.search_strategies.base import (
    SearchStrategy,
    SearchStrategyConfig,
    ExperimentResult,
)
from src.execution.search_strategies.factory import register_strategy


@register_strategy("linear_search")
class LinearSearch(SearchStrategy):
    """
    Simple linear search strategy.
    
    Each iteration:
    1. Generate a solution based on problem + previous results
    2. Implement and debug the solution
    3. Store result and continue
    
    Config params:
        - code_debug_tries: Max debug attempts (default: 3)
        - idea_generation_model: Model for solution generation (default: gpt-4.1-mini)
    """
    
    def __init__(self, config: SearchStrategyConfig):
        """Initialize linear search strategy."""
        super().__init__(config)
        
        # Config params
        self.code_debug_tries = self.params.get("code_debug_tries", 3)
        self.idea_generation_model = self.params.get("idea_generation_model", "gpt-4.1-mini")
        
        # State
        self.experiment_history: List[ExperimentResult] = []
        self.iteration_count = 0
        
        print(f"[LinearSearch] Initialized:")
        print(f"  - code_debug_tries: {self.code_debug_tries}")
        print(f"  - idea_generation_model: {self.idea_generation_model}")
        
        # Initialize workspace with empty main file
        self._initialize_workspace()
    
    def _initialize_workspace(self) -> None:
        """Create initial empty main file."""
        session = self.workspace.create_experiment_session(
            branch_name=self.workspace.get_current_branch()
        )
        session.generate_code(
            f"<problem>\n{self.problem_handler.get_problem_context()}\n</problem>\n\n"
            + "Create an empty main.py with a main() function placeholder. No comments."
        )
        self.workspace.finalize_session(session)
        self.workspace.repo.git.stash()

    def run(self, context: ContextData, budget_progress: float = 0.0) -> None:
        """
        Execute one iteration of linear search.
        
        Simple approach:
        1. Generate a solution considering previous experiments
        2. Implement it
        3. Store result
        """
        self.iteration_count += 1
        print(f"\n[LinearSearch] Iteration {self.iteration_count}, budget: {budget_progress:.1f}%")
        
        # Generate solution
        solution = self._generate_solution(context, budget_progress)
        print(f"[LinearSearch] Generated solution ({len(solution)} chars)")
        
        # Implement and run
        branch_name = f"linear_exp_{len(self.experiment_history)}"
        parent_branch = self._get_best_branch()
        
        print(f"[LinearSearch] Implementing on branch: {branch_name} (from {parent_branch})")
        
        result = self._implement_n_debug(
            solution=solution,
            context=context,
            code_debug_tries=self.code_debug_tries,
            branch_name=branch_name,
            parent_branch_name=parent_branch,
        )
        
        # Store result
        experiment_result = ExperimentResult(
            node_id=len(self.experiment_history),
            solution=solution,
            score=result.score,
            branch_name=branch_name,
            had_error=result.run_had_error,
            error_message=result.error_message,
            output=result.output,
            detailed_output=result.detailed_output,
            feedbacks=result.feedbacks,
        )
        self.experiment_history.append(experiment_result)
        
        if result.run_had_error:
            print(f"[LinearSearch] ❌ Experiment failed: {result.error_message[:100]}...")
        else:
            print(f"[LinearSearch] ✓ Experiment completed with score: {result.score}")

    def _generate_solution(self, context: ContextData, budget_progress: float) -> str:
        """Generate a solution based on problem and previous experiments."""
        
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
        
        prompt = f"""Generate a solution for this problem. Be specific and detailed.

PROBLEM:
{context.problem}

{history_summary}

KNOWLEDGE BASE:
{context.kg_results if context.kg_results else "No additional knowledge available."}

Requirements:
- Provide a complete, implementable solution
- Include specific steps, methods, and hyperparameters
- If there are previous experiments, improve upon the best one
- Consider the current budget progress: {budget_progress:.1f}%

Format your solution with:
# Core Idea: 
[Brief description]

# Solution Steps:
[Detailed implementation steps]

# Hyperparameters:
[Specific values to use]
"""
        
        response = self.llm.llm_completion(
            model=self.idea_generation_model,
            messages=[{"role": "user", "content": prompt}],
        )
        
        return response

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
