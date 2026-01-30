# Benchmark Tree Search Strategy
#
# Tree search that uses handler.run() for evaluation.
# For use with MLE-Bench and ALE-Bench.
#
# This is a self-contained tree search strategy that includes all tree logic.
# Uses handler.run() for evaluation instead of agent-based evaluation.

import json
import os
import pickle
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from kapso.execution.types import ContextData
from kapso.execution.search_strategies.base import (
    SearchStrategy,
    SearchStrategyConfig,
    SearchNode,
)
from kapso.execution.search_strategies.factory import register_strategy
from kapso.execution.memories.repo_memory import RepoMemoryManager
from kapso.execution.memories.repo_memory.observation import extract_repo_memory_sections_consulted
from kapso.core.prompt_loader import load_prompt, render_prompt


# =============================================================================
# Tree Search Node
# =============================================================================

@dataclass
class TreeSearchNode(SearchNode):
    """
    Node in the solution search tree.
    
    Extends SearchNode with tree-specific fields for parent/child relationships
    and tree search state.
    """
    # Tree structure (not using dataclass default for mutable - set in __post_init__)
    parent_node: Optional["TreeSearchNode"] = None
    children: List["TreeSearchNode"] = field(default_factory=list)
    
    # Tree search state
    is_terminated: bool = False
    is_root: bool = False
    node_event_history: List = field(default_factory=list)
    
    # Observability: which RepoMemory sections were consulted during ideation
    ideation_repo_memory_sections_consulted: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Set is_root and parent_node_id based on parent_node."""
        self.is_root = self.parent_node is None
        # Update parent_node_id from parent_node reference
        if self.parent_node is not None:
            self.parent_node_id = self.parent_node.node_id
    
    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


# =============================================================================
# Benchmark Tree Search Strategy
# =============================================================================

@register_strategy("benchmark_tree_search")
class BenchmarkTreeSearch(SearchStrategy):
    """
    Tree search for benchmarks (MLE-Bench, ALE-Bench).
    
    Uses a tree structure to explore solutions, with LLM guidance for:
    - Solution generation (expand nodes)
    - Solution selection (pick best nodes to experiment)
    - Solution pruning (remove unpromising nodes)
    
    Key difference from generic search:
    - Uses handler.run() for evaluation instead of agent-based evaluation
    - Skips feedback generator (handler provides feedback via ProblemRunResult)
    - Checks handler.stop_condition() for should_stop
    
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
        """Initialize benchmark tree search."""
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

        print(f"[BenchmarkTreeSearch] Initialized with params:")
        print(f"  - node_expansion_limit: {self.node_expansion_limit}")
        print(f"  - idea_generation_model: {self.idea_generation_model}")
        print(f"  - handler-based evaluation: True")

        # Tree state
        self.experimentation_count = 0

        if not import_from_checkpoint:
            self.node_history: List[TreeSearchNode] = []  # All experimented nodes
            self.nodes: List[TreeSearchNode] = []  # All tree nodes (including unexperimented)
            # Initialize root nodes
            for i in range(self.node_expansion_limit * 4):
                self.nodes.append(TreeSearchNode(
                    node_id=i, 
                    branch_name=self.workspace.get_current_branch(), 
                    solution="Root node to be expanded for new and diverse ideas."
                ))
        
        # Thread locks
        self.node_history_lock = threading.Lock()
        self.nodes_lock = threading.Lock()

        # Error tracking for implementation
        self.previous_errors: List[str] = []
        self.recent_error_count = 10

        # Initialize with an empty main file ONLY when starting from an empty workspace.
        # If the workspace is seeded from an existing repo, we must not overwrite it.
        if workspace_dir is None and not self.workspace.is_seeded:
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

    def run(self, context: ContextData, budget_progress: float = 0.0) -> Optional[SearchNode]:
        """
        Execute one iteration of tree search.
        
        Steps:
        1. Prune unpromising solutions (if budget > 20%)
        2. Expand promising nodes with new solutions
        3. Select best nodes to experiment
        4. Run experiments in parallel
        
        Args:
            context: Either a ContextData object or a problem string
            budget_progress: Budget progress percentage (0-100)
        
        Returns:
            The best SearchNode from this iteration (with should_stop set)
        """
        self.experimentation_count += 1
        
        # Normalize context: support both string and ContextData
        if isinstance(context, str):
            context = ContextData(
                problem=context,
                additional_info="",
                kg_results="",
                kg_code_results="",
            )
        
        # Prune after initial exploration
        if budget_progress >= 20:
            self.prune_bad_solutions(context)
        
        # Expand nodes
        self.expand(context, budget_progress)
        
        # Select and run experiments
        experiments_count = self.experimentation_per_run
        if len(self.node_history) == 0:
            experiments_count *= self.first_experiment_factor
        
        best_nodes = self.select(
            context, 
            top_k=experiments_count, 
            exclude_experimented_nodes=True
        )
        
        with self.node_history_lock:
            base_experiment_count = len(self.node_history)
        
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
        
        # Return the best node from this iteration (for orchestrator to check should_stop)
        return self.get_best_experiment()

    def get_experiment_history(self, best_last: bool = False) -> List[SearchNode]:
        """Get all experiment results, optionally sorted by score."""
        if best_last:
            return sorted(
                self.node_history,
                key=lambda node: (
                    (not node.had_error, node.score or 0) 
                    if self.problem_handler.maximize_scoring 
                    else (not node.had_error, -(node.score or 0))
                )
            )
        return self.node_history
    
    def get_best_experiment(self) -> Optional[SearchNode]:
        """Get the best experiment result."""
        valid_nodes = [node for node in self.node_history if not node.had_error]
        if not valid_nodes:
            return None
        return max(
            valid_nodes, 
            key=lambda x: (x.score or 0) if self.problem_handler.maximize_scoring else -(x.score or 0)
        )

    def checkout_to_best_experiment_branch(self) -> None:
        """Checkout git to the best experiment's branch."""
        best_node = self.get_best_experiment()
        if best_node:
            print("#" * 100)
            print(f"Checking out to the best experiment branch: {best_node.branch_name}")
            print("#" * 100)
            self.workspace.switch_branch(best_node.branch_name)

    # =========================================================================
    # Tree Operations
    # =========================================================================

    def expand(self, context: ContextData, budget_progress: float) -> None:
        """Expand selected nodes with new solution candidates."""
        top_nodes = self.get_experiment_history(best_last=True)[-self.node_expansion_limit:]
        
        if budget_progress >= self.exploration_budget_percent:
            print("Expanding top Nodes for exploitation.")
            selected_nodes = [self.nodes[node.node_id] for node in top_nodes]
        elif len(self.node_history) == 0:
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

    def _expand_node(self, context: ContextData, node: TreeSearchNode, budget_progress: float) -> None:
        """Generate new child solutions for a node."""
        expansion_count = self.node_expansion_new_childs_count
        if len(self.node_history) == 0:
            expansion_count *= self.first_experiment_factor

        # Ground ideation in the closest experimented parent branch memory.
        # This keeps new solutions consistent with the actual inherited code state.
        base_branch_name = self._get_closest_experimented_parent(node).branch_name or self.workspace.get_current_branch()
        
        new_solutions, ideation_sections = self.solution_generation(
            context,
            parent_solution=node.solution,
            final_solution_count=expansion_count,
            step_count=self.idea_generation_steps,
            per_step_solution_count=min(expansion_count, self.per_step_maximum_solution_count),
            base_branch_name=base_branch_name,
        )
        
        for new_solution in new_solutions:
            with self.nodes_lock:
                new_node = TreeSearchNode(
                    node_id=len(self.nodes), 
                    parent_node=node, 
                    solution=new_solution
                )
                new_node.ideation_repo_memory_sections_consulted = list(ideation_sections or [])
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
    ) -> List[TreeSearchNode]:
        """Select best nodes using LLM guidance."""
        leaf_nodes = [node for node in self.nodes if node.is_leaf and not node.is_terminated]
        
        if exclude_experimented_nodes:
            leaf_nodes = [node for node in leaf_nodes if node.score is None]
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
            - For each selection you must write a 1 sentence reason why you selected that solution.
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
            if node.is_leaf and not node.is_terminated and node.score is None
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
    # Experiment Execution (Handler-based)
    # =========================================================================
    
    def _run_for_node(
        self, 
        node: TreeSearchNode, 
        context: ContextData, 
        branch_name: str, 
        budget_progress: float = 0.0
    ) -> None:
        """
        Run experiment for a single node with handler-based evaluation.
        
        Uses handler.run() for evaluation instead of agent-based evaluation.
        Skips feedback generator (handler provides feedback via ProblemRunResult).
        Checks handler.stop_condition() for should_stop.
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
        
        # Step 1: Implement (using base class method)
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
        
        # Step 2: Use handler.run() for evaluation (instead of agent-based)
        self._evaluate_with_handler(node, node.solution)
        
        # Step 3: Check handler's stop condition
        node.should_stop = self._check_handler_stop_condition()
        
        with self.node_history_lock:
            self.node_history.append(node)
        
        print(f"[Benchmark] Experiment at branch {branch_name} ended: score={node.score}, should_stop={node.should_stop}")
    
    def _get_closest_experimented_parent(self, node: TreeSearchNode) -> TreeSearchNode:
        """Find the closest ancestor that has been experimented (has score)."""
        while node.parent_node is not None and node.score is None:
            node = node.parent_node
        return node

    # =========================================================================
    # Implementation Methods (Benchmark-specific)
    # =========================================================================

    def implement_solution(
        self, 
        solution: str, 
        context: ContextData, 
        session
    ) -> str:
        """
        Have the developer agent implement a solution.
        
        The developer agent is responsible for:
        - Implementing the solution
        - Building evaluation in kapso_evaluation/
        - Running the evaluation
        - Handling any errors/retries internally
        
        Args:
            solution: The solution description to implement
            context: Problem context with KG results
            session: Experiment session with coding agent
            
        Returns:
            The agent's output (contains implementation results)
        """
        repo_memory_doc = RepoMemoryManager.ensure_exists_in_worktree(session.session_folder)
        repo_memory_brief = RepoMemoryManager.render_summary_and_toc(repo_memory_doc, max_chars=2500)

        # By default, agents can read the JSON file directly.
        agent_type = getattr(getattr(self.workspace, "coding_agent_config", None), "agent_type", "")
        if agent_type == "claude_code":
            repo_memory_detail_access_instructions = (
                "For detailed section content (architecture, gotchas, invariants, etc.),\n"
                "use the MCP tool: `get_repo_memory_section(section_id=\"core.architecture\")`\n"
                "Available sections: core.architecture, core.entrypoints, core.where_to_edit, core.invariants, core.testing, core.gotchas, core.dependencies\n"
                "Fallback: open `.kapso/repo_memory.json` and read `book.sections[section_id]`."
            )
        else:
            repo_memory_detail_access_instructions = (
                "For detailed section content (architecture, gotchas, invariants, etc.),\n"
                "read: `.kapso/repo_memory.json` and look up by section ID from the TOC."
            )

        template = load_prompt("execution/search_strategies/generic/prompts/coding_agent_implement.md")
        developer_prompt = render_prompt(
            template,
            {
                "previous_errors": "\n".join(
                    str(e) for e in self.previous_errors[-self.recent_error_count:]
                ),
                "branch_name": session.branch_name,
                "repo_memory_brief": repo_memory_brief,
                "repo_memory_detail_access_instructions": repo_memory_detail_access_instructions,
                "kg_code_results": str(getattr(context, "kg_code_results", "")),
                "problem": str(getattr(context, "problem", "")),
                "solution": str(solution or ""),
            },
        )

        # Run the developer agent
        result = session.generate_code(developer_prompt)
        return result.output if hasattr(result, 'output') else str(result)

    def _implement(
        self, 
        solution: str, 
        context: ContextData, 
        branch_name: str, 
        parent_branch_name: str = "main",
        ideation_repo_memory_sections_consulted: Optional[List[str]] = None,
    ) -> str:
        """
        Full implementation flow.
        
        Creates a session, runs the developer agent, and finalizes.
        
        Args:
            solution: Solution description to implement
            context: Problem context
            branch_name: Git branch for this experiment
            parent_branch_name: Parent branch to inherit code from
            ideation_repo_memory_sections_consulted: RepoMemory sections used during ideation
            
        Returns:
            The agent's output string
        """
        session = self.workspace.create_experiment_session(branch_name, parent_branch_name, llm=self.llm)
        agent_output = self.implement_solution(solution, context, session)

        # Update RepoMemory for this experiment branch
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
        
        self.workspace.finalize_session(session)
        return agent_output

    def _evaluate_with_handler(self, node: SearchNode, solution: str) -> SearchNode:
        """
        Evaluate using handler.run() for benchmark compatibility.
        
        Maps ProblemRunResult fields to SearchNode:
        - result.score -> node.score
        - result.output -> node.evaluation_output
        - result.run_had_error -> node.had_error
        - result.error_message -> node.error_message
        - result.feedbacks -> node.feedback
        
        Args:
            node: SearchNode to populate with evaluation results
            solution: Solution string to pass to handler
            
        Returns:
            Updated SearchNode with score, evaluation_output, etc.
        """
        from kapso.environment.handlers.base import ProblemRunResult
        
        # Check if handler has run() method
        if not hasattr(self.problem_handler, 'run') or not callable(self.problem_handler.run):
            print(f"[BenchmarkTreeSearch] Warning: handler has no run() method, skipping handler evaluation")
            return node
        
        # Prepare run directory
        run_data_dir = os.path.join(self.workspace_dir, "kapso_evaluation")
        os.makedirs(run_data_dir, exist_ok=True)
        
        # Call handler's run()
        try:
            print(f"[BenchmarkTreeSearch] Calling handler.run() for evaluation...")
            result: ProblemRunResult = self.problem_handler.run(
                file_path=self.workspace_dir,
                run_data_dir=run_data_dir,
                solution=solution,
            )
            
            # Map ProblemRunResult fields to SearchNode
            node.score = result.score
            node.evaluation_output = result.output or result.detailed_output
            node.had_error = result.run_had_error
            node.error_message = result.error_message or result.error_details
            node.feedback = result.feedbacks
            
            print(f"[BenchmarkTreeSearch] Handler evaluation: score={node.score}, had_error={node.had_error}")
            
        except Exception as e:
            print(f"[BenchmarkTreeSearch] Handler evaluation failed: {e}")
            node.had_error = True
            node.error_message = str(e)
        
        return node

    def _check_handler_stop_condition(self) -> bool:
        """
        Check handler's stop_condition() for benchmark compatibility.
        
        Returns:
            True if handler says to stop, False otherwise
        """
        if hasattr(self.problem_handler, 'stop_condition') and callable(self.problem_handler.stop_condition):
            return self.problem_handler.stop_condition()
        return False

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
        base_branch_name: str = "main",
    ) -> Tuple[List[str], List[str]]:
        """Generate new solutions using LLM completion."""
        if final_solution_count > per_step_solution_count * len(self.idea_generation_ensemble_models):
            per_step_solution_count = final_solution_count // len(self.idea_generation_ensemble_models) + 1
        
        solutions = ""
        all_sections_consulted: List[str] = []
        solution_generation_prompt = """
            You are a world class problem solver. Generate {per_step_solution_count} exact solutions for the given problem that are the best and significantly better than the previous experiments.
            Requirement:
            - Each solution must be exact and high level steps specific enough to be coded.
            - If parent solution exists, the newly proposed solutions must improve it either by 1-extend and add something to parent solution, 2-remove and change and improve parts, 3-big tune, 4-small tune it or 5-completely change the core idea (at least one from each).
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
            output_requirements = (
                solution_generation_prompt.format(per_step_solution_count=per_step_solution_count).strip()
                + "\n\nCRITICAL: Put solutions exactly between <solution> and </solution> tags (as shown above)."
            )
            history = (
                f"# Parent solution:\n{parent_solution}\n\n"
                f"# Last iteration proposed solutions:\n{solutions}"
            )

            # Build user message with all context
            user_message = f"""
Problem: {str(getattr(context, "problem", ""))}

Workflow Guidance: {str(getattr(context, "additional_info", "") or "")}

History: {history}

Additional Knowledge: {str(getattr(context, "kg_results", "") or "")}

{output_requirements}
"""
            solutions = self.llm.llm_completion_with_system_prompt(
                model=self.idea_generation_model,
                system_prompt="You are a world class problem solver generating solutions.",
                user_message=user_message,
                reasoning_effort=self.reasoning_effort,
            )
        
        solutions_list = re.findall(r'<solution>(.*?)</solution>', solutions, re.DOTALL)

        if final_solution_count >= len(solutions_list):
            return solutions_list, all_sections_consulted

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
                \n\n Provide the list of top {final_solution_count} ids between <output> and </output> tags.
            """,
            reasoning_effort=self.reasoning_effort,
        )
        
        final_solutions_ids = eval(re.findall(r'<output>(.*?)</output>', final_solution, re.DOTALL)[0].strip())
        return (
            [solutions_list[int(id)] for id in final_solutions_ids if id < len(solutions_list)],
            all_sections_consulted,
        )

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
            "score": node.score,
            "had_error": node.had_error,
            "node_event_history": node.node_event_history,
            "children_ids": [child.node_id for child in node.children]
        } for node in self.nodes]
        
        with open(filepath, 'w') as f:
            json.dump(nodes_data, f, indent=2)

    def export_checkpoint(self) -> None:
        with open(os.path.join(self.workspace_dir, 'checkpoint.pkl'), 'wb') as f:
            pickle.dump({
                "node_history": self.node_history,
                "nodes": self.nodes,
            }, f)

    def import_checkpoint(self) -> None:
        try:
            with open(os.path.join(self.workspace_dir, 'checkpoint.pkl'), 'rb') as f:
                checkpoint = pickle.load(f)
            self.node_history = checkpoint["node_history"]
            self.nodes = checkpoint["nodes"]
            print(f"[BenchmarkTreeSearch] Checkpoint imported successfully from {self.workspace_dir}")
            print(f"[BenchmarkTreeSearch] Node history: {len(self.node_history)}")
            print(f"[BenchmarkTreeSearch] Nodes: {len(self.nodes)}")
            print(f"[BenchmarkTreeSearch] last Node: {self.nodes[-1]}")
        except FileNotFoundError:
            print("[BenchmarkTreeSearch] No checkpoint found")
            raise FileNotFoundError(f"[BenchmarkTreeSearch] No checkpoint found in {self.workspace_dir}")
