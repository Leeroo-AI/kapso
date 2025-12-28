# Orchestrator Agent
#
# Main orchestrator that coordinates the experimentation loop.
# Uses pluggable search strategies, context managers, and knowledge retrievers.

import time
from typing import Any, Dict, Optional

from src.execution.context_manager import (
    ContextManager,
    ContextManagerFactory,
)
from src.knowledge.search import (
    KnowledgeSearch,
    KnowledgeSearchFactory,
)
from src.execution.search_strategies import (
    SearchStrategy,
    SearchStrategyFactory,
)
from src.execution.coding_agents.factory import CodingAgentFactory
from src.environment.handlers.base import ProblemHandler
from src.core.llm import LLMBackend
from src.core.config import load_mode_config


class OrchestratorAgent:
    """
    Main orchestrator agent that coordinates the experimentation loop.
    
    Uses pluggable components:
    - Search strategies (registered via @register_strategy decorator)
    - Context managers (registered via @register_context_manager decorator)
    - Knowledge search backends (registered via @register_knowledge_search decorator)
    - Coding agents (Aider, Gemini, Claude Code, OpenHands)
    
    Args:
        problem_handler: Handler for the problem being solved
        config_path: Path to benchmark-specific config.yaml file
        mode: Configuration mode to use (if None, uses default_mode from config)
        coding_agent: Coding agent to use (overrides config if specified)
        is_kg_active: Whether to use the knowledge graph
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
    ):
        self.problem_handler = problem_handler
        self.llm = LLMBackend()
        self.config_path = config_path
        self.mode = mode
        
        # Load config once and store for reuse
        self.mode_config = load_mode_config(config_path, mode)
        
        # Create search strategy 
        self.search_strategy = self._create_search_strategy(
            coding_agent=coding_agent,
            workspace_dir=workspace_dir,
            start_from_checkpoint=start_from_checkpoint,
        )
        
        # Create knowledge search backend (or use provided instance).
        # This allows Expert.build() to inject a concrete backend (e.g., kg_graph_search)
        # without relying on config defaults (which may point to a different backend).
        if knowledge_search is not None:
            self.knowledge_search = knowledge_search
        else:
            self.knowledge_search = self._create_knowledge_search(
                is_kg_active=is_kg_active,
            )
        
        # Create context manager with injected search backend
        self.context_manager = self._create_context_manager()

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
            strategy_type = "llm_tree_search"
            strategy_params = {}
            coding_agent_type = coding_agent or "aider"
            coding_agent_model = None
            coding_agent_debug_model = None
        else:
            # Extract search strategy config
            search_config = mode_config.get('search_strategy', {})
            strategy_type = search_config.get('type', 'llm_tree_search')
            strategy_params = search_config.get('params', {})
            
            # If no search_strategy section, use legacy format
            if not search_config:
                strategy_type = "llm_tree_search"
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
            elif coding_config:
                coding_agent_type = coding_config.get('type', 'aider')
                coding_agent_model = coding_config.get('model')
                coding_agent_debug_model = coding_config.get('debug_model')
            else:
                coding_agent_type = 'aider'
                coding_agent_model = mode_config.get('developer_model')
                coding_agent_debug_model = mode_config.get('developer_debug_model')
        
        # Build coding agent config
        coding_agent_config = CodingAgentFactory.build_config(
            agent_type=coding_agent_type,
            model=coding_agent_model,
            debug_model=coding_agent_debug_model,
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

    def _create_context_manager(self) -> ContextManager:
        """
        Create context manager with injected knowledge search backend.
        
        Returns:
            Configured ContextManager instance
        """
        mode_config = self.mode_config
        
        # Check for context_manager config
        cm_config = mode_config.get('context_manager', {})
        
        # If context_manager is explicitly configured, use it
        if cm_config and cm_config.get('type'):
            return ContextManagerFactory.create_from_config(
                config=cm_config,
                problem_handler=self.problem_handler,
                search_strategy=self.search_strategy,
                knowledge_search=self.knowledge_search,
            )
        
        # If KG is active and no context_manager specified, use cognitive
        # for full workflow support
        if self.knowledge_search and self.knowledge_search.is_enabled():
            return ContextManagerFactory.create(
                context_manager_type="cognitive",
                problem_handler=self.problem_handler,
                search_strategy=self.search_strategy,
                knowledge_search=self.knowledge_search,
            )
        
        # Default: use kg_enriched context manager
        return ContextManagerFactory.create(
            context_manager_type="kg_enriched",
            problem_handler=self.problem_handler,
            search_strategy=self.search_strategy,
            knowledge_search=self.knowledge_search,
        )

    def get_cumulative_cost(self) -> float:
        """Get total cost from all components."""
        return (
            self.llm.get_cumulative_cost() 
            + self.search_strategy.workspace.get_cumulative_cost() 
            + self.problem_handler.llm.get_cumulative_cost()
        )

    def solve(
        self, 
        experiment_max_iter: int = 20, 
        time_budget_minutes: int = 24*60, 
        cost_budget: float = 300
    ):
        """
        Run the main experimentation loop.
        
        Stops when ANY of these conditions is met:
        1. LLM decision says COMPLETE (context_manager.should_stop())
        2. Score threshold reached (problem_handler.stop_condition())
        3. Budget exhausted (time/cost/iterations)
        
        Args:
            experiment_max_iter: Maximum number of experiment iterations
            time_budget_minutes: Time budget in minutes
            cost_budget: Maximum cost in dollars
            
        Returns:
            Best experiment result
        """
        start_time = time.time()
        
        for i in range(experiment_max_iter):
            # Calculate budget progress (0-100)
            budget_progress = max(
                (time.time() - start_time) / (time_budget_minutes * 60),
                i / experiment_max_iter,
                self.get_cumulative_cost() / cost_budget
            ) * 100
            
            # Check stopping conditions (score threshold or budget)
            if self.problem_handler.stop_condition() or budget_progress >= 100:
                print(f"[Orchestrator] Stopping: score threshold or budget reached")
                break
            
            # Get context (decision happens inside for cognitive context manager)
            experiment_context = self.context_manager.get_context(budget_progress=budget_progress)
            
            # Check if LLM decided COMPLETE
            if self.context_manager.should_stop():
                print(f"[Orchestrator] Stopping: LLM decided COMPLETE")
                break
            
            # Run one iteration of search strategy
            self.search_strategy.run(experiment_context, budget_progress=budget_progress)

            print(
                f"Experiment {i+1} completed with cumulative cost: ${self.get_cumulative_cost():.3f}", 
                '#' * 100,
                '\n', 
                self.search_strategy.get_best_experiment(), 
                '\n', 
                '#' * 100
            )
            self.search_strategy.export_checkpoint()

        return self.search_strategy.get_best_experiment()
