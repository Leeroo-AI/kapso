# LLM-Steered Tree Search Strategy
#
# Tree-based exploration of solutions with LLM guidance.
# Registered as "llm_tree_search" via the factory decorator.

import json
import os
import pickle
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.execution.context_manager.types import ContextData
from src.execution.search_strategies.base import (
    SearchStrategy,
    SearchStrategyConfig,
    ExperimentResult,
)
from src.execution.search_strategies.factory import register_strategy


class Node:
    """Node in the solution search tree."""
    
    def __init__(self, node_id: int, parent_node=None, solution: str = "", branch_name: str = None):
        self.parent_node = parent_node
        self.solution = solution
        self.node_id = node_id
        self.children: List['Node'] = []
        self.branch_name = branch_name
        self.is_terminated = False
        self.experiment_result: Optional[ExperimentResult] = None
        self.node_event_history: List = []
        self.is_root = parent_node is None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


@register_strategy("llm_tree_search")
class LlmSteeredTreeSearch(SearchStrategy):
    """
    LLM-steered tree search strategy for experiment generation.
    
    Uses a tree structure to explore solutions, with LLM guidance for:
    - Solution generation (expand nodes)
    - Solution selection (pick best nodes to experiment)
    - Solution pruning (remove unpromising nodes)
    
    Config params:
        - reasoning_effort: LLM reasoning effort level
        - code_debug_tries: Max debug attempts per solution
        - node_expansion_limit: Max nodes to expand per iteration
        - node_expansion_new_childs_count: New solutions per expansion
        - idea_generation_steps: Refinement steps for solutions
        - first_experiment_factor: Multiplier for first iteration
        - experimentation_per_run: Experiments per iteration
        - per_step_maximum_solution_count: Max solutions per step
        - exploration_budget_percent: When to switch to exploitation
        - idea_generation_model: Model for generating ideas
        - idea_generation_ensemble_models: Models for ensemble generation
    """
    
    def __init__(self, config: SearchStrategyConfig, workspace_dir: Optional[str] = None, import_from_checkpoint: bool = False):
        """Initialize LLM-steered tree search."""
        super().__init__(config, workspace_dir, import_from_checkpoint)
        
        # Extract config params with defaults
        params = config.params
        self.reasoning_effort = params.get("reasoning_effort", "medium")
        self.code_debug_tries = params.get("code_debug_tries", 5)
        self.node_expansion_limit = params.get("node_expansion_limit", 2)
        self.node_expansion_new_childs_count = params.get("node_expansion_new_childs_count", 5)
        self.idea_generation_steps = params.get("idea_generation_steps", 1)
        self.first_experiment_factor = params.get("first_experiment_factor", 1)
        self.experimentation_per_run = params.get("experimentation_per_run", 1)
        self.per_step_maximum_solution_count = params.get("per_step_maximum_solution_count", 10)
        self.exploration_budget_percent = params.get("exploration_budget_percent", 30)
        self.idea_generation_model = params.get("idea_generation_model", "gpt-4.1-mini")
        self.idea_generation_ensemble_models = params.get(
            "idea_generation_ensemble_models", 
            ["gpt-4.1-mini"]
        )

        print(f"[LlmSteeredTreeSearch] Initialized with params:")
        print(f"  - node_expansion_limit: {self.node_expansion_limit}")
        print(f"  - code_debug_tries: {self.code_debug_tries}")
        print(f"  - idea_generation_model: {self.idea_generation_model}")

        # Tree state
        self.experimentation_count = 0

        if not import_from_checkpoint:
            self.experiment_history: List[ExperimentResult] = []
            self.nodes: List[Node] = []
            # Initialize root nodes
            for i in range(self.node_expansion_limit * 4):
                self.nodes.append(Node(node_id=i, branch_name=self.workspace.get_current_branch(), solution="Root node to be expanded for new and diverse ideas."))
        
        # Thread locks
        self.experiment_history_lock = threading.Lock()
        self.nodes_lock = threading.Lock()

        # Initialize with empty main file
        if workspace_dir is None:
            self._initialize_workspace()

    def _initialize_workspace(self) -> None:
        """Create initial empty main file in workspace."""
        session = self.workspace.create_experiment_session(
            branch_name=self.workspace.get_current_branch()
        )
        session.generate_code(
            f"<problem>\n {self.problem_handler.get_problem_context()} \n </problem>\n\n"
            + "implement an empty main file for <problem> without anything extra implementation. "
            + "Do not write any comment or any other text in the code. just an empty main file with an empty main function"
        )
        self.workspace.finalize_session(session)
        self.workspace.repo.git.stash()

    # =========================================================================
    # Abstract Method Implementations
    # =========================================================================

    def run(self, context: ContextData, budget_progress: float = 0.0) -> None:
        """
        Execute one iteration of tree search.
        
        Steps:
        1. Prune unpromising solutions (if budget > 20%)
        2. Expand promising nodes with new solutions
        3. Select best nodes to experiment
        4. Run experiments in parallel
        """
        self.experimentation_count += 1
        
        # Prune after initial exploration
        if budget_progress >= 20:
            self.prune_bad_solutions(context)
        
        # Expand nodes
        self.expand(context, budget_progress)
        
        # Select and run experiments
        experiments_count = self.experimentation_per_run
        if len(self.experiment_history) == 0:
            experiments_count *= self.first_experiment_factor
        
        best_nodes = self.select(
            context, 
            top_k=experiments_count, 
            exclude_experimented_nodes=True
        )
        
        with self.experiment_history_lock:
            base_experiment_count = len(self.experiment_history)
        
        branch_names = [f'experiment_{base_experiment_count + i}' for i in range(len(best_nodes))]

        # Run experiments in parallel
        def run_node(node, branch_name):
            self._run_for_node(node, context, branch_name, budget_progress)
        
        with ThreadPoolExecutor(max_workers=len(best_nodes) + 1) as executor:
            futures = [
                executor.submit(run_node, node, branch_name) 
                for node, branch_name in zip(best_nodes, branch_names)
            ]
            self._run_futures(executor, futures)

    def get_experiment_history(self, best_last: bool = False) -> List[ExperimentResult]:
        """Get all experiment results, optionally sorted by score."""
        if best_last:
            return sorted(
                self.experiment_history,
                key=lambda exp: (
                    (not exp.had_error, exp.score) 
                    if self.problem_handler.maximize_scoring 
                    else (not exp.had_error, -exp.score)
                )
            )
        return self.experiment_history
    
    def get_best_experiment(self) -> Optional[ExperimentResult]:
        """Get the best experiment result."""
        valid_experiments = [exp for exp in self.experiment_history if not exp.had_error]
        if not valid_experiments:
            return None
        return max(
            valid_experiments, 
            key=lambda x: x.score if self.problem_handler.maximize_scoring else -x.score
        )

    def checkout_to_best_experiment_branch(self) -> None:
        """Checkout git to the best experiment's branch."""
        best_experiment = self.get_best_experiment()
        if best_experiment:
            print("#" * 100)
            print(f"Checking out to the best experiment branch: {best_experiment.branch_name}")
            print("#" * 100)
            self.workspace.switch_branch(best_experiment.branch_name)

    # =========================================================================
    # Tree Operations
    # =========================================================================

    def expand(self, context: ContextData, budget_progress: float) -> None:
        """Expand selected nodes with new solution candidates."""
        top_experiments = self.get_experiment_history(best_last=True)[-self.node_expansion_limit:]
        
        if budget_progress >= self.exploration_budget_percent:
            print("Expanding top Nodes for exploitation.")
            selected_nodes = [self.nodes[exp.node_id] for exp in top_experiments]
        elif len(self.experiment_history) == 0:
            print("Expanding first iteration")
            selected_nodes = self.nodes[:self.node_expansion_limit]
        else:
            print("Expanding by LLM selection for exploration.")
            selected_nodes = self.select(
                context,
                top_k=self.node_expansion_limit,
                selection_criteria="Expected score + potential for further improvements of score.",
                exclude_root_nodes=False,
            )
        
        with ThreadPoolExecutor(max_workers=len(selected_nodes) + 1) as executor:
            futures = [
                executor.submit(self._expand_node, context, node, budget_progress) 
                for node in selected_nodes
            ]
            self._run_futures(executor, futures)

    def _expand_node(self, context: ContextData, node: Node, budget_progress: float) -> None:
        """Generate new child solutions for a node."""
        expansion_count = self.node_expansion_new_childs_count
        if len(self.experiment_history) == 0:
            expansion_count *= self.first_experiment_factor
        
        new_solutions = self.solution_generation(
            context,
            parent_solution=node.solution,
            final_solution_count=expansion_count,
            step_count=self.idea_generation_steps,
            per_step_solution_count=min(expansion_count, self.per_step_maximum_solution_count),
        )
        
        for new_solution in new_solutions:
            with self.nodes_lock:
                new_node = Node(
                    node_id=len(self.nodes), 
                    parent_node=node, 
                    solution=new_solution
                )
                self.nodes.append(new_node)
            new_node.node_event_history.append([self.experimentation_count, "create"])
            node.children.append(new_node)
        
        if new_solutions:
            node.node_event_history.append([self.experimentation_count, "expand"])

    def select(
        self, 
        context: ContextData, 
        top_k: int = 1, 
        selection_criteria: str = "Best expected score, speed, and diversity.",
        exclude_experimented_nodes: bool = False,
        exclude_root_nodes: bool = True,
    ) -> List[Node]:
        """Select best nodes using LLM guidance."""
        leaf_nodes = [node for node in self.nodes if node.is_leaf and not node.is_terminated]
        
        if exclude_experimented_nodes:
            leaf_nodes = [node for node in leaf_nodes if node.experiment_result is None]
        if exclude_root_nodes:
            leaf_nodes = [node for node in leaf_nodes if not node.is_root]

        if top_k >= len(leaf_nodes):
            return leaf_nodes
        
        system_prompt = f"""
            you are a world class problem solver. You are given a list of solutions and you must select the top {top_k} solutions that are the best.
            requirements:
            - your output must be a list of {top_k} ids.
            - make sure to consider the previous experiments according to their score and the reliable knowledge base information in your selection.
            - selection criteria is ** {selection_criteria} **.
            - For each selection you must write a reason why you selected that solution.
            - output must always be a python list of ids between <output> and </output> tags. eg:
                Reason for solution id 2: ...
                Reason for solution id 4: ...
                <output>
                [2, 4]
                </output>
        """ 
        
        user_prompt = (
            f"Problem: {context.problem} \n\n Additional information: {context.additional_info} \n\n"
            + f"Reliable knowledge base information: {context.kg_results} \n\n"
            + "Candidate Solutions for selection:\n" 
            + "\n\n".join([f" id: {node.node_id}, solution: {node.solution}" for node in leaf_nodes])
            + f'\n\n Provide the list of top {top_k} ids between <output> and </output> tags.'
        )
        
        output = self.llm.llm_completion_with_system_prompt(
            model=self.idea_generation_model,
            system_prompt=system_prompt,
            user_message=user_prompt,
            reasoning_effort=self.reasoning_effort,
        )
        
        selected_node_ids = eval(re.findall(r'<output>(.*?)</output>', output, re.DOTALL)[0].strip())
        return [node for node in leaf_nodes if node.node_id in selected_node_ids]

    def prune_bad_solutions(self, context: ContextData) -> None:
        """Remove unpromising solutions using LLM guidance."""
        leaf_nodes = [
            node for node in self.nodes 
            if node.is_leaf and not node.is_terminated and node.experiment_result is None
        ]
        
        if len(leaf_nodes) <= 1:
            return 
        
        system_prompt = f"""
            you are a world class problem solver. You are given a problem and its history, and you have a list of candidate solutions.
            Considering the previous experiments, their score and feedbacks, you must select the solutions that are not promising and are unable to improve the score to be deleted from the candidates.
            requirements:
            - your output must be a list of ids of the bad solutions between <output> and </output> tags.
            - You must select at least {len(leaf_nodes)//20} and at most {len(leaf_nodes)//5} solutions to be deleted.
            - Your selection must be based on the previous experiments and their final score.
            - For every node, write why you think it should be deleted and has no more room for improvement.
            - Output example:
                Reason 1: ...
                Reason 5: ...
                <output>
                [1, 5]
                </output>
        """
        
        user_prompt = (
            f"Problem: {context.problem} \n\n "
            f"Additional information: {context.additional_info} \n\n "
            f"Reliable knowledge base information: {context.kg_results} \n\n "
            f"Candidate Solutions for deletion:\n"
            + "\n\n".join([f" id: {node.node_id}, solution: {node.solution}" for node in leaf_nodes])
        )
        
        output = self.llm.llm_completion_with_system_prompt(
            model=self.idea_generation_model,
            system_prompt=system_prompt,
            user_message=user_prompt,
            reasoning_effort=self.reasoning_effort,
        )
        
        selected_node_ids = eval(re.findall(r'<output>(.*?)</output>', output, re.DOTALL)[0].strip())
        
        for node in leaf_nodes:
            if node.node_id in selected_node_ids:
                node.node_event_history.append([self.experimentation_count, "terminate"])
                node.is_terminated = True

    # =========================================================================
    # Experiment Execution
    # =========================================================================

    def _run_for_node(
        self, 
        node: Node, 
        context: ContextData, 
        branch_name: str, 
        budget_progress: float = 0.0
    ) -> None:
        """Run experiment for a single node."""
        print(
            f"Budget progress: {budget_progress}\n" + "#" * 100 + "\n" 
            + f"Initiating experiment at node {node.node_id} "
            + f"(parent: {node.parent_node.node_id if node.parent_node else None}):\n"
            + f"{node.solution[:500]}..."
            + "\n" + "#" * 100
        )

        node.node_event_history.append([self.experimentation_count, "experiment"])
        node.branch_name = branch_name
        
        result = self._implement_n_debug(
            node.solution,
            context,
            code_debug_tries=self.code_debug_tries,
            branch_name=branch_name,
            parent_branch_name=self._get_closest_experimented_parent(node).branch_name,
        )

        if result.run_had_error:
            node.is_terminated = True
            node.node_event_history.append([self.experimentation_count, "terminate"])

        experiment_result = ExperimentResult(
            solution=node.solution,
            score=result.score,
            branch_name=branch_name,
            node_id=node.node_id,
            had_error=result.run_had_error,
            error_message=result.error_message,
            output=result.output,
            detailed_output=result.detailed_output,
            feedbacks=result.feedbacks
        )
        
        with self.experiment_history_lock:
            node.experiment_result = experiment_result
            self.experiment_history.append(experiment_result)
        
        print(f"Experiment at branch {branch_name} ended with score: {experiment_result.score}, error: {experiment_result.had_error}")

    def _get_closest_experimented_parent(self, node: Node) -> Node:
        """Find the closest ancestor with an experiment result."""
        while node.parent_node is not None and node.experiment_result is None:
            node = node.parent_node
        return node

    # =========================================================================
    # Solution Generation
    # =========================================================================

    def solution_generation(
        self, 
        context: ContextData,
        final_solution_count: int,
        step_count: int,
        per_step_solution_count: int = 3,
        parent_solution: str = "",
    ) -> List[str]:
        """Generate new solutions using LLM."""
        if final_solution_count > per_step_solution_count * len(self.idea_generation_ensemble_models):
            per_step_solution_count = final_solution_count // len(self.idea_generation_ensemble_models) + 1
        
        solutions = ""
        solution_generation_prompt = """
            You are a world class problem solver. Generate {per_step_solution_count} exact solutions for the given problem that are the best and significantly better than the previous experiments.
            Requirement:
            - Each solution must be exact and high level steps specific enough to be coded.
            - If parent solution exists, the newly proposed solutions must extend, improve, or tune it (at least one from each).
            - Solutions must be significantly different from each other.
            - Solutions must not reference to each other parts and parent parts. Each solution must be self-contained.
            - CRITICAL: ** Put solutions between <solution> and </solution> tags. ** e.g.:
                Solution 1:
                <solution>
                   # Core Idea: 
                    ...
                   # Body:
                    ...
                   # Runtime expectation:
                    t1 seconds
                </solution>
                
                Solution 2:
                <solution>
                   # Core Idea: 
                    ...
                   #Body:
                    ...
                   # Runtime expectation:
                    t2 seconds.
                </solution>
                ...
        """
        
        for i in range(step_count):
            system_prompt = solution_generation_prompt.format(per_step_solution_count=per_step_solution_count)
            user_prompt = f"""
                # Problem: \n {context.problem} \n\n 
                # Additional information:\n {context.additional_info} \n\n 
                # Reliable knowledge base information:\n {context.kg_results} \n\n 
                # Parent solution:\n {parent_solution} \n\n
                # Last iteration proposed solutions:\n {solutions} 
            """
            
            new_solutions = self.llm.llm_multiple_completions(
                models=(
                    self.idea_generation_ensemble_models 
                    if self.experiment_history 
                    else self.idea_generation_ensemble_models * 2
                ),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                reasoning_effort=self.reasoning_effort
            )
            solutions = "\n".join(new_solutions)
        
        solutions_list = re.findall(r'<solution>(.*?)</solution>', solutions, re.DOTALL)

        if final_solution_count >= len(solutions_list):
            return solutions_list

        # Select best solutions if we have too many
        final_solution = self.llm.llm_completion_with_system_prompt(
            model=self.idea_generation_model,
            system_prompt=f""" 
                You are a world class problem solver. Choose {final_solution_count} best solutions from the list.
                Output must be a list of solution ids between <output> and </output> tags.
                <output> [1, 2, 3] </output>
            """,
            user_message=f"""
                # Problem: \n {context.problem} \n\n 
                # Solutions list:\n {chr(10).join([f"Solution id {idx}: {solution}" for idx, solution in enumerate(solutions_list)])} 
            """,
            reasoning_effort=self.reasoning_effort,
        )
        
        final_solutions_ids = eval(re.findall(r'<output>(.*?)</output>', final_solution, re.DOTALL)[0].strip())
        return [solutions_list[int(id)] for id in final_solutions_ids if id < len(solutions_list)]

    # =========================================================================
    # Utilities
    # =========================================================================

    def _run_futures(self, executor, futures) -> None:
        """Run futures and handle keyboard interrupt."""
        all_futures = list(futures)        
        try:
            for future in as_completed(all_futures):
                future.result()
        except KeyboardInterrupt:
            print("\nKilling NOW...")
            raise

    def export_nodes_to_json(self, log_dir: str = "tmp/log/tree") -> None:
        """Export tree state to JSON for debugging."""
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.problem_handler.problem_id}_experiment_{self.experimentation_count}_{timestamp}.json"
        filepath = os.path.join(log_dir, filename)
        
        nodes_data = [{
            "node_id": node.node_id,
            "solution": node.solution,
            "is_terminated": node.is_terminated,
            "experiment_result": {
                "score": node.experiment_result.score,
                "had_error": node.experiment_result.had_error,
            } if node.experiment_result else None,
            "node_event_history": node.node_event_history,
            "children_ids": [child.node_id for child in node.children]
        } for node in self.nodes]
        
        with open(filepath, 'w') as f:
            json.dump(nodes_data, f, indent=2)

    def export_checkpoint(self) -> None:
        with open(os.path.join(self.workspace_dir, 'checkpoint.pkl'), 'wb') as f:
            pickle.dump({
                "experiment_history": self.experiment_history,
                "nodes": self.nodes,
            }, f)

    def import_checkpoint(self) -> None:
        try:
            with open(os.path.join(self.workspace_dir, 'checkpoint.pkl'), 'rb') as f:
                checkpoint = pickle.load(f)
            self.experiment_history = checkpoint["experiment_history"]
            self.nodes = checkpoint["nodes"]
            print(f"[LlmSteeredTreeSearch] Checkpoint imported successfully from {self.workspace_dir}")
            print(f"[LlmSteeredTreeSearch] Experiment history: {len(self.experiment_history)}")
            print(f"[LlmSteeredTreeSearch] Nodes: {len(self.nodes)}")
            print( f"[LlmSteeredTreeSearch] last Node: {self.nodes[-1]}")
        except FileNotFoundError:
            print("[LlmSteeredTreeSearch] No checkpoint found")
            raise FileNotFoundError(f"[LlmSteeredTreeSearch] No checkpoint found in {self.workspace_dir}")