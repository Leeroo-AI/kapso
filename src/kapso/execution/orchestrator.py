# Orchestrator Agent
#
# Main orchestrator that coordinates the experimentation loop.
# Uses pluggable search strategies and knowledge retrievers.
#
# In the new design:
# - Developer agent builds evaluation in kapso_evaluation/
# - Developer agent runs evaluation and reports results
# - FeedbackGenerator decides when to stop
# - ExperimentHistoryStore persists experiment results for MCP access
# - Experiment history is accessed via MCP tools (not context managers)

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from kapso.knowledge_base.search import (
    KnowledgeSearch,
    KnowledgeSearchFactory,
)
from kapso.execution.search_strategies import (
    SearchStrategy,
    SearchStrategyFactory,
)
from kapso.execution.coding_agents.factory import CodingAgentFactory
from kapso.execution.search_strategies.generic import FeedbackGenerator, FeedbackResult
from kapso.environment.handlers.base import ProblemHandler
from kapso.core.llm import LLMBackend
from kapso.core.config import load_mode_config
from kapso.execution.search_strategies.base import ExperimentResult
from kapso.execution.memories.experiment_memory import ExperimentHistoryStore


@dataclass
class SolveResult:
    """Result from orchestrator.solve()."""
    best_experiment: Optional[ExperimentResult]
    final_feedback: Optional[FeedbackResult]
    stopped_reason: str  # "goal_achieved", "max_iterations", "budget_exhausted", "legacy_stop"
    iterations_run: int
    total_cost: float


class OrchestratorAgent:
    """
    Main orchestrator agent that coordinates the experimentation loop.
    
    Uses pluggable components:
    - Search strategies (registered via @register_strategy decorator)
    - Knowledge search backends (registered via @register_knowledge_search decorator)
    - Coding agents (Aider, Gemini, Claude Code, OpenHands)
    - Feedback generator (LLM-based, decides when to stop)
    - Experiment history store (persists results for MCP access)
    
    Args:
        problem_handler: Handler for the problem being solved
        config_path: Path to benchmark-specific config.yaml file
        mode: Configuration mode to use (if None, uses default_mode from config)
        coding_agent: Coding agent to use (overrides config if specified)
        is_kg_active: Whether to use the knowledge graph
        goal: The goal/objective for the evolve process
    """
    
    def __init__(
        self, 
        problem_handler: ProblemHandler,
        config_path: Optional[str] = None,
        mode: Optional[str] = None,
        coding_agent: Optional[str] = None,
        is_kg_active: bool = False,
        knowledge_search: Optional[KnowledgeSearch] = None,
        workspace_dir: Optional[str] = None,
        start_from_checkpoint: bool = False,
        initial_repo: Optional[str] = None,
        eval_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        goal: Optional[str] = None,
    ):
        self.problem_handler = problem_handler
        self.llm = LLMBackend()
        self.config_path = config_path
        self.mode = mode
        self.goal = goal or ""
        # Optional: seed experiments from an existing local repo (copy/clone into workspace).
        self.initial_repo = initial_repo
        # Optional: directories to copy into workspace
        self.eval_dir = eval_dir
        self.data_dir = data_dir
        
        # Load config once and store for reuse
        self.mode_config = load_mode_config(config_path, mode)
        
        # Determine workspace directory for experiment history
        self._workspace_dir = workspace_dir
        
        # Create experiment history store
        # Path is determined after search strategy creates workspace
        self.experiment_store: Optional[ExperimentHistoryStore] = None
        
        # Create feedback generator FIRST (needed by search strategy)
        self.feedback_generator = self._create_feedback_generator(coding_agent)
        
        # Track feedback for next iteration
        self.current_feedback: Optional[str] = None
        self.last_feedback_result: Optional[FeedbackResult] = None
        
        # Create search strategy (uses feedback_generator)
        self.search_strategy = self._create_search_strategy(
            coding_agent=coding_agent,
            workspace_dir=workspace_dir,
            start_from_checkpoint=start_from_checkpoint,
        )
        
        # Now create experiment history store with the actual workspace path
        experiment_history_path = os.path.join(
            self.search_strategy.workspace_dir, 
            ".kapso", 
            "experiment_history.json"
        )
        self.experiment_store = ExperimentHistoryStore(
            json_path=experiment_history_path,
            weaviate_url=os.environ.get("WEAVIATE_URL"),
            goal=self.goal,
        )
        
        # Create knowledge search backend (or use provided instance).
        # This allows Kapso.evolve() to inject a concrete backend (e.g., kg_graph_search)
        # without relying on config defaults (which may point to a different backend).
        if knowledge_search is not None:
            self.knowledge_search = knowledge_search
            self._owns_knowledge_search = False
        else:
            self.knowledge_search = self._create_knowledge_search(
                is_kg_active=is_kg_active,
            )
            # We created it inside the orchestrator → we should close it.
            self._owns_knowledge_search = True
    
    def _create_feedback_generator(
        self,
        coding_agent: Optional[str] = None,
    ) -> FeedbackGenerator:
        """
        Create feedback generator.
        
        Uses the same coding agent type as the developer agent by default.
        """
        # Get coding agent config from mode config
        mode_config = self.mode_config
        # Check for dedicated feedback_generator config first, fall back to coding_agent
        feedback_config = mode_config.get('feedback_generator', {}) if mode_config else {}
        coding_config = mode_config.get('coding_agent', {}) if mode_config else {}
        
        # Use feedback_generator config if available, otherwise fall back to coding_agent config
        if feedback_config:
            agent_type = feedback_config.get('type', 'claude_code')
            agent_model = feedback_config.get('model')
            agent_debug_model = feedback_config.get('debug_model')
            agent_specific = feedback_config.get('agent_specific', {})
        else:
            # Fall back to coding_agent config
            agent_type = coding_agent or coding_config.get('type', 'claude_code')
            agent_model = coding_config.get('model')
            agent_debug_model = coding_config.get('debug_model')
            agent_specific = coding_config.get('agent_specific', {})
        
        # Build config for feedback generator
        feedback_agent_config = CodingAgentFactory.build_config(
            agent_type=agent_type,
            model=agent_model,
            debug_model=agent_debug_model,
            agent_specific=agent_specific,
        )
        
        return FeedbackGenerator(
            coding_agent_config=feedback_agent_config,
        )

    def _create_search_strategy(
        self,
        coding_agent: Optional[str],
        workspace_dir: Optional[str],
        start_from_checkpoint: bool,
    ) -> SearchStrategy:
        """
        Create search strategy from config.
        
        Args:
            coding_agent: Override coding agent type
            
        Returns:
            Configured SearchStrategy instance
        """
        mode_config = self.mode_config
        
        if not mode_config:
            # Use defaults
            strategy_type = "generic"
            strategy_params = {}
            coding_agent_type = coding_agent or "claude_code"
            coding_agent_model = None
            coding_agent_debug_model = None
        else:
            # Extract search strategy config
            search_config = mode_config.get('search_strategy', {})
            strategy_type = search_config.get('type', 'generic')
            strategy_params = search_config.get('params', {})
            
            # If no search_strategy section, use legacy format
            if not search_config:
                strategy_type = "generic"
                strategy_params = {
                    'reasoning_effort': mode_config.get('reasoning_effort', 'medium'),
                    'code_debug_tries': mode_config.get('code_debug_tries', 5),
                    'node_expansion_limit': mode_config.get('node_expansion_limit', 2),
                    'node_expansion_new_childs_count': mode_config.get('node_expansion_new_childs_count', 5),
                    'idea_generation_steps': mode_config.get('idea_generation_steps', 1),
                    'first_experiment_factor': mode_config.get('first_experiment_factor', 1),
                    'experimentation_per_run': mode_config.get('experimentation_per_run', 1),
                    'per_step_maximum_solution_count': mode_config.get('per_step_maximum_solution_count', 10),
                    'exploration_budget_percent': mode_config.get('exploration_budget_percent', 30),
                    'idea_generation_model': mode_config.get('idea_generation_model', 'gpt-4.1-mini'),
                    'idea_generation_ensemble_models': mode_config.get('idea_generation_ensemble_models', ['gpt-4.1-mini']),
                }
            
            # Extract coding agent config
            coding_config = mode_config.get('coding_agent', {})
            if coding_agent:
                coding_agent_type = coding_agent
                coding_agent_model = None
                coding_agent_debug_model = None
                coding_agent_specific = None
            elif coding_config:
                coding_agent_type = coding_config.get('type', 'aider')
                coding_agent_model = coding_config.get('model')
                coding_agent_debug_model = coding_config.get('debug_model')
                # Support agent_specific from YAML config (e.g., use_bedrock for Claude Code)
                coding_agent_specific = coding_config.get('agent_specific')
            else:
                coding_agent_type = 'aider'
                coding_agent_model = mode_config.get('developer_model')
                coding_agent_debug_model = mode_config.get('developer_debug_model')
                coding_agent_specific = None
        
        # Build coding agent config
        coding_agent_config = CodingAgentFactory.build_config(
            agent_type=coding_agent_type,
            model=coding_agent_model,
            debug_model=coding_agent_debug_model,
            agent_specific=coding_agent_specific,
        )
        
        # Create strategy via factory
        return SearchStrategyFactory.create(
            strategy_type=strategy_type,
            problem_handler=self.problem_handler,
            llm=self.llm,
            coding_agent_config=coding_agent_config,
            params=strategy_params,
            workspace_dir=workspace_dir,
            start_from_checkpoint=start_from_checkpoint,
            initial_repo=self.initial_repo,
            eval_dir=self.eval_dir,
            data_dir=self.data_dir,
            feedback_generator=self.feedback_generator,
            goal=self.goal,
        )

    def _create_knowledge_search(
        self,
        is_kg_active: bool,
    ) -> KnowledgeSearch:
        """
        Create knowledge search backend from config.
        
        Args:
            is_kg_active: Whether to enable knowledge graph
            
        Returns:
            Configured KnowledgeSearch instance
        """
        mode_config = self.mode_config
        
        # Check for knowledge_search config (new format)
        ks_config = mode_config.get('knowledge_search', {})
        
        if ks_config:
            return KnowledgeSearchFactory.create_from_config(ks_config)
        
        # Check for legacy knowledge_retriever config
        kr_config = mode_config.get('knowledge_retriever', {})
        
        if kr_config:
            # Convert legacy config to new format
            return KnowledgeSearchFactory.create_from_config({
                "type": "kg_llm_navigation",
                "enabled": kr_config.get("enabled", True),
                "params": kr_config.get("params"),
                "preset": kr_config.get("preset"),
            })
        
        # Check use_knowledge_graph flag
        if 'use_knowledge_graph' in mode_config:
            kg_enabled = mode_config.get('use_knowledge_graph', False)
            if kg_enabled or is_kg_active:
                return KnowledgeSearchFactory.create(
                    search_type="kg_llm_navigation",
                )
            else:
                return KnowledgeSearchFactory.create_null()
        
        # Fall back to is_kg_active parameter
        if is_kg_active:
            return KnowledgeSearchFactory.create(
                search_type="kg_llm_navigation",
            )
        
        # Default: disabled
        return KnowledgeSearchFactory.create_null()

    def get_cumulative_cost(self) -> float:
        """Get total cost from all components."""
        return (
            self.llm.get_cumulative_cost() 
            + self.search_strategy.workspace.get_cumulative_cost()
        )

    def solve(
        self, 
        experiment_max_iter: int = 20, 
        time_budget_minutes: Optional[int] = None, 
        cost_budget: Optional[float] = None
    ) -> SolveResult:
        """
        Run the main experimentation loop.
        
        In the new design:
        1. Developer agent implements solution and runs evaluation
        2. Feedback generator validates evaluation and decides stop/continue
        3. Loop continues until goal reached or budget exhausted
        4. Experiment history is accessed via MCP tools (not context managers)
        
        Stops when ANY of these conditions is met:
        1. Feedback generator says STOP (goal achieved)
        2. Budget exhausted (time/cost/iterations)
        3. Legacy: problem_handler.stop_condition() (for backward compatibility)
        
        Args:
            experiment_max_iter: Maximum number of experiment iterations
            time_budget_minutes: Time budget in minutes (optional, no limit if not set)
            cost_budget: Maximum cost in dollars (optional, no limit if not set)
            
        Returns:
            SolveResult with best_experiment, final_feedback, stopped_reason
        """
        start_time = time.time()
        stopped_reason = "max_iterations"  # default
        iterations_run = 0
        
        # Get problem context once (experiment history is accessed via MCP)
        problem = self.problem_handler.get_problem_context()
        
        try:
            for i in range(experiment_max_iter):
                iterations_run = i + 1
                
                # Calculate budget progress (0-100)
                # Only include time/cost if budgets are set
                progress_factors = [i / experiment_max_iter]
                if time_budget_minutes is not None:
                    progress_factors.append((time.time() - start_time) / (time_budget_minutes * 60))
                if cost_budget is not None:
                    progress_factors.append(self.get_cumulative_cost() / cost_budget)
                budget_progress = max(progress_factors) * 100
                
                # Check budget exhaustion
                if budget_progress >= 100:
                    print(f"[Orchestrator] Stopping: budget exhausted")
                    stopped_reason = "budget_exhausted"
                    break
                
                # Build context with problem and feedback
                # Experiment history is accessed via MCP tools by the agent
                context = problem
                if self.current_feedback:
                    context = f"{problem}\n\n## Feedback from Previous Iteration\n\n{self.current_feedback}\n\nPlease address the above feedback in this iteration."
                
                # Run one iteration of search strategy
                # Search strategy handles: solution generation, implementation, feedback
                # Returns SearchNode with all data including should_stop
                node = self.search_strategy.run(
                    context, 
                    budget_progress=budget_progress
                )
                
                # Skip if no result (shouldn't happen but be safe)
                if node is None:
                    print(f"[Orchestrator] Warning: No result from iteration {i+1}")
                    continue
                
                # Add experiment to history store (for MCP access)
                if self.experiment_store:
                    self.experiment_store.add_experiment(node)
                
                # Log result
                print(f"[Orchestrator] Iteration {i+1} result:")
                print(f"  - Score: {node.score}")
                print(f"  - Should stop: {node.should_stop}")
                print(f"  - Evaluation valid: {node.evaluation_valid}")
                print(f"  - Feedback: {node.feedback or ''}")
                
                # Store feedback result for return value
                if node.feedback:
                    from kapso.execution.search_strategies.generic import FeedbackResult
                    self.last_feedback_result = FeedbackResult(
                        stop=node.should_stop,
                        evaluation_valid=node.evaluation_valid,
                        feedback=node.feedback,
                        score=node.score,
                    )
                
                # Check if search strategy says stop
                if node.should_stop:
                    print(f"[Orchestrator] Stopping: goal achieved")
                    stopped_reason = "goal_achieved"
                    break
                
                # Store feedback for next iteration
                self.current_feedback = node.feedback

                print(
                    f"Experiment {i+1} completed with cumulative cost: ${self.get_cumulative_cost():.3f}", 
                    '#' * 100,
                    '\n', 
                    self.search_strategy.get_best_experiment(), 
                    '\n', 
                    '#' * 100
                )
                self.search_strategy.export_checkpoint()
        finally:
            # Best-effort cleanup: prevents leaked sockets from KG/Episodic clients.
            
            # Close experiment history store
            if self.experiment_store:
                try:
                    self.experiment_store.close()
                except Exception:
                    pass
            
            # Close knowledge search only if the orchestrator created it.
            if getattr(self, "_owns_knowledge_search", False) and hasattr(self.knowledge_search, "close"):
                try:
                    self.knowledge_search.close()
                except Exception:
                    pass

        return SolveResult(
            best_experiment=self.search_strategy.get_best_experiment(),
            final_feedback=self.last_feedback_result,
            stopped_reason=stopped_reason,
            iterations_run=iterations_run,
            total_cost=self.get_cumulative_cost(),
        )
