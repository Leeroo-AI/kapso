# Benchmark Tree Search Strategy
#
# Tree search that uses handler.run() for evaluation.
# For use with MLE-Bench and ALE-Bench.
#
# This strategy inherits all tree search logic (expand, select, prune)
# but replaces agent-based evaluation with handler.run().

from typing import Optional

from src.execution.types import ContextData
from src.execution.search_strategies.base import SearchNode, SearchStrategyConfig
from src.execution.search_strategies.llm_tree_search import LlmSteeredTreeSearch, TreeSearchNode
from src.execution.search_strategies.factory import register_strategy


@register_strategy("benchmark_tree_search")
class BenchmarkTreeSearch(LlmSteeredTreeSearch):
    """
    Tree search for benchmarks (MLE-Bench, ALE-Bench).
    
    Inherits all tree search logic (expand, select, prune, solution_generation)
    but uses handler.run() for evaluation instead of agent-based evaluation
    with JSON extraction.
    
    Key differences from parent:
    - _run_for_node() calls _evaluate_with_handler() instead of _extract_agent_result()
    - Skips _generate_feedback() (handler provides feedback via ProblemRunResult)
    - Checks handler.stop_condition() for should_stop
    """
    
    def __init__(self, config: SearchStrategyConfig, workspace_dir: Optional[str] = None, import_from_checkpoint: bool = False):
        super().__init__(config, workspace_dir, import_from_checkpoint)
        print(f"[BenchmarkTreeSearch] Initialized (handler-based evaluation)")
    
    def _run_for_node(
        self, 
        node: TreeSearchNode, 
        context: ContextData, 
        branch_name: str, 
        budget_progress: float = 0.0
    ) -> None:
        """
        Run experiment for a single node with handler-based evaluation.
        
        Same as parent but:
        - Uses _evaluate_with_handler() instead of _extract_agent_result()
        - Skips _generate_feedback() (handler provides feedback)
        - Checks handler.stop_condition() for should_stop
        """
        print(
            f"Budget progress: {budget_progress}\n" + "#" * 100 + "\n" 
            + f"[Benchmark] Experiment at node {node.node_id} "
            + f"(parent: {node.parent_node.node_id if node.parent_node else None}):\n"
            + f"{node.solution[:500]}..."
            + "\n" + "#" * 100
        )

        node.node_event_history.append([self.experimentation_count, "experiment"])
        node.branch_name = branch_name
        node.workspace_dir = self.workspace_dir
        
        # Step 1: Implement (same as parent)
        agent_output = self._implement(
            node.solution,
            context,
            branch_name=branch_name,
            parent_branch_name=self._get_closest_experimented_parent(node).branch_name,
            ideation_repo_memory_sections_consulted=node.ideation_repo_memory_sections_consulted,
        )
        
        node.agent_output = agent_output
        node.code_diff = self._get_code_diff(
            branch_name, 
            self._get_closest_experimented_parent(node).branch_name
        )
        
        # Step 2: DIFFERENT - Use handler.run() for evaluation
        # (instead of _extract_agent_result + _generate_feedback)
        self._evaluate_with_handler(node, node.solution)
        
        # Step 3: Check handler's stop condition
        node.should_stop = self._check_handler_stop_condition()
        
        with self.node_history_lock:
            self.node_history.append(node)
        
        print(f"[Benchmark] Experiment at branch {branch_name} ended: score={node.score}, should_stop={node.should_stop}")
    
    def run(self, context: ContextData, budget_progress: float = 0.0) -> Optional[SearchNode]:
        """
        Execute one iteration of tree search.
        
        Same as parent - the handler stop condition is checked in _run_for_node().
        """
        return super().run(context, budget_progress)
