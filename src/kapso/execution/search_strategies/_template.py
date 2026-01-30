"""
Search Strategy Template

Copy this file and rename it to create your own search strategy.

Steps:
1. Copy this file: cp _template.py my_strategy.py
2. Rename the class and update the @register_strategy decorator
3. Implement the abstract methods
4. Add presets to strategies.yaml (optional but recommended)
5. Test: python -c "from kapso.execution.search_strategies import SearchStrategyFactory; print(SearchStrategyFactory.list_available())"

Your strategy will be auto-discovered when the module loads!
"""

from typing import Any, Dict, List, Optional

from kapso.execution.types import ContextData
from kapso.execution.search_strategies.base import (
    SearchStrategy,
    SearchStrategyConfig,
    ExperimentResult,
)
from kapso.execution.search_strategies.factory import register_strategy


# =============================================================================
# STEP 1: Choose a unique name for your strategy
# =============================================================================
# Change "my_strategy" to your strategy name (lowercase, underscores ok)
# This name is used in config.yaml: search_strategy.type: "my_strategy"

@register_strategy("my_strategy")
class MyStrategy(SearchStrategy):
    """
    Brief description of your search strategy.
    
    Explain the algorithm:
    - How does it explore solutions?
    - What makes it different from other strategies?
    
    Config params (from strategies.yaml or config.yaml):
        - param1: Description of param1
        - param2: Description of param2
    """
    
    # =========================================================================
    # STEP 2: Initialize your strategy
    # =========================================================================
    
    def __init__(self, config: SearchStrategyConfig):
        """
        Initialize your search strategy.
        
        Available from config:
        - config.problem_handler: ProblemHandler for running experiments
        - config.llm: LLM backend for generating solutions
        - config.coding_agent_config: Config for the coding agent
        - config.params: Dict of your strategy-specific parameters
        """
        # IMPORTANT: Call super().__init__() first!
        # This sets up: self.problem_handler, self.llm, self.workspace, self.params
        super().__init__(config)
        
        # Extract your parameters with defaults
        self.param1 = self.params.get("param1", 10)
        self.param2 = self.params.get("param2", "default_value")
        self.code_debug_tries = self.params.get("code_debug_tries", 3)
        
        # Initialize your strategy-specific state
        self.experiment_history: List[ExperimentResult] = []
        self.iteration_count = 0
        
        # Optional: Print initialization info
        print(f"[MyStrategy] Initialized with param1={self.param1}, param2={self.param2}")
        
        # Optional: Initialize workspace with empty main file
        # self._initialize_workspace()
    
    def _initialize_workspace(self) -> None:
        """Optional: Create initial files in workspace."""
        session = self.workspace.create_experiment_session(
            branch_name=self.workspace.get_current_branch()
        )
        session.generate_code(
            f"<problem>\n{self.problem_handler.get_problem_context()}\n</problem>\n\n"
            + "Create an empty main.py file with a main() function placeholder."
        )
        self.workspace.finalize_session(session)
        self.workspace.repo.git.stash()

    # =========================================================================
    # STEP 3: Implement the main search loop
    # =========================================================================
    
    def run(self, context: ContextData, budget_progress: float = 0.0) -> None:
        """
        Execute one iteration of your search strategy.
        
        This is called repeatedly by the OrchestratorAgent until:
        - Max iterations reached
        - Time/cost budget exhausted
        - Problem stop condition met
        
        Args:
            context: Contains problem description, experiment history, KG results
                - context.problem: Problem description string
                - context.additional_info: Previous experiment summaries
                - context.kg_results: Knowledge graph results (if enabled)
            budget_progress: 0-100 indicating how much budget is consumed
        
        Your implementation should:
        1. Generate solution ideas (use self.llm)
        2. Implement solutions (use self._implement_n_debug)
        3. Store results in self.experiment_history
        """
        self.iteration_count += 1
        print(f"[MyStrategy] Running iteration {self.iteration_count}, budget: {budget_progress:.1f}%")
        
        # ----- EXAMPLE: Generate a solution using LLM -----
        solution = self._generate_solution(context)
        
        # ----- EXAMPLE: Implement and run the solution -----
        branch_name = f"experiment_{len(self.experiment_history)}"
        
        result = self._implement_n_debug(
            solution=solution,
            context=context,
            code_debug_tries=self.code_debug_tries,
            branch_name=branch_name,
            parent_branch_name="main",  # Or use parent from your tree structure
        )
        
        # ----- EXAMPLE: Store the result -----
        experiment_result = ExperimentResult(
            node_id=len(self.experiment_history),
            solution=solution,
            score=result.score,
            branch_name=branch_name,
            had_error=result.run_had_error,
            error_message=result.error_message,
            output=result.output,
        )
        self.experiment_history.append(experiment_result)
        
        print(f"[MyStrategy] Experiment completed: score={result.score}, error={result.run_had_error}")
    
    def _generate_solution(self, context: ContextData) -> str:
        """
        Example: Generate a solution using the LLM.
        
        Customize this for your strategy's solution generation approach.
        """
        prompt = f"""
        Generate a solution for the following problem:
        
        {context.problem}
        
        Previous experiments:
        {context.additional_info}
        
        Provide a detailed solution with clear steps.
        """
        
        response = self.llm.llm_completion(
            model="gpt-4.1-mini",  # Or use a param: self.params.get("model")
            messages=[{"role": "user", "content": prompt}],
        )
        
        return response

    # =========================================================================
    # STEP 4: Implement history and best experiment retrieval
    # =========================================================================
    
    def get_experiment_history(self, best_last: bool = False) -> List[ExperimentResult]:
        """
        Return all experiment results.
        
        Args:
            best_last: If True, sort so best experiments are at the end
            
        Returns:
            List of ExperimentResult objects
        """
        if best_last:
            # Sort by score (adjust based on maximize_scoring)
            return sorted(
                self.experiment_history,
                key=lambda exp: (
                    not exp.had_error,  # Successful experiments first
                    exp.score if self.problem_handler.maximize_scoring else -exp.score
                )
            )
        return self.experiment_history
    
    def get_best_experiment(self) -> Optional[ExperimentResult]:
        """
        Return the best experiment result so far.
        
        Returns:
            Best ExperimentResult, or None if no successful experiments
        """
        valid_experiments = [exp for exp in self.experiment_history if not exp.had_error]
        
        if not valid_experiments:
            return None
        
        return max(
            valid_experiments,
            key=lambda x: x.score if self.problem_handler.maximize_scoring else -x.score
        )
    
    def checkout_to_best_experiment_branch(self) -> None:
        """
        Checkout git to the best experiment's branch.
        
        Called at the end to prepare the final solution.
        """
        best = self.get_best_experiment()
        if best:
            print(f"[MyStrategy] Checking out to best branch: {best.branch_name}")
            self.workspace.switch_branch(best.branch_name)
        else:
            print("[MyStrategy] No successful experiments to checkout")


# =============================================================================
# STEP 5 (Optional): Add presets to strategies.yaml
# =============================================================================
"""
Add this to strategies.yaml:

  my_strategy:
    description: "Brief description of your strategy"
    presets:
      FAST:
        params:
          param1: 5
          param2: "fast_mode"
          code_debug_tries: 2
      THOROUGH:
        params:
          param1: 20
          param2: "thorough_mode"
          code_debug_tries: 5
    default_preset: "FAST"
"""


# =============================================================================
# STEP 6 (Optional): Add tests
# =============================================================================
if __name__ == "__main__":
    # Quick test that the strategy is registered
    from kapso.execution.search_strategies import (
        SearchStrategyFactory
    )
    
    print("Available strategies:", SearchStrategyFactory.list_available())
    
    if SearchStrategyFactory.is_available("my_strategy"):
        print("✓ my_strategy is registered!")
    else:
        print("✗ my_strategy is NOT registered - check decorator name")

