# Tinkerer Agent - Main Entry Point
#
# The primary user-facing API for the Tinkerer Agent system.
# Provides a clean interface for the "Brain to Binary" workflow:
#   Tinkerer.learn() -> Tinkerer.evolve() -> Tinkerer.deploy() -> Software.run()
#
# Usage:
#     from src.tinkerer import Tinkerer, Source, DeployStrategy
#     
#     tinkerer = Tinkerer()
#     tinkerer.learn(Source.Repo("https://github.com/..."), target_kg="https://skills.leeroo.com")
#     solution = tinkerer.evolve(goal="Create a triage agent")
#     software = tinkerer.deploy(solution, strategy=DeployStrategy.LOCAL)
#     result = software.run({"input": "data"})

import os
from typing import Any, Dict, List, Optional, Union

from src.execution.orchestrator import OrchestratorAgent
from src.execution.solution import SolutionResult
from src.environment.handlers.generic import GenericProblemHandler
from src.knowledge.search import KnowledgeSearchFactory
from src.knowledge.learners import Source, KnowledgePipeline

# Placeholder types for unimplemented learning
class KnowledgeChunk:
    pass

LearnerFactory = None  # Learning not implemented yet
from src.deployment import (
    Software,
    DeployConfig,
    DeployStrategy,
    DeploymentFactory,
)


# =============================================================================
# EXPERT
# =============================================================================

# Path to default configuration
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


class Tinkerer:
    """
    The main Tinkerer Agent class.
    
    A Tinkerer is an intelligent agent that can:
    1. Learn from various sources (repos, papers, files, past solutions)
    2. Evolve software to solve goals using experimentation
    3. Improve over time through the feedback loop
    
    Usage:
        # Simple usage
        tinkerer = Tinkerer(kg_location="https://skills.leeroo.com")
        tinkerer.learn(Source.Repo("https://github.com/alpaca/alpaca-py"), target_kg="https://skills.leeroo.com")
        solution = tinkerer.evolve(goal="Create a momentum trading bot")
        software = tinkerer.deploy(solution)
        result = software.run({"ticker": "AAPL"})
        
        # Advanced usage with evaluator and stop condition
        solution = tinkerer.evolve(
            goal="Build a classifier with 95% accuracy",
            evaluator="regex_pattern",
            evaluator_params={"pattern": r"Accuracy: ([\\d.]+)"},
            stop_condition="threshold",
            stop_condition_params={"threshold": 0.95},
        )
    """
    
    # Mapping from Source type to Learner type
    _SOURCE_TO_LEARNER = {
        Source.Repo: "repo",
        Source.Paper: "paper",
        Source.File: "file",
        Source.Doc: "file",  # Use file learner for now
        Source.Solution: "experiment",
    }
    
    def __init__(
        self, 
        kg_location: str = "default",
        config_path: Optional[str] = None,
    ):
        """
        Initialize an Tinkerer agent.
        
        Args:
            kg_location: Path to local KG or URL to remote KG
            config_path: Path to configuration file (uses default if not provided)
        """
        self.kg_location = kg_location
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        
        # Initialize knowledge search (for querying during evolve)
        # Default to disabled for now - can be enabled via config
        self.knowledge_search = KnowledgeSearchFactory.create_null()
        
        # Track learned knowledge chunks (in-memory for MVP)
        self._learned_chunks: List[KnowledgeChunk] = []
        
        print(f"Initialized Tinkerer (KG: {kg_location})")
    
    def learn(
        self, 
        *sources: Union[Source.Repo, Source.Paper, Source.File, Source.Doc, Source.Solution],
        target_kg: str = "https://skills.leeroo.com",
    ):
        """
        Learn from one or more knowledge sources.
        
        This ingests knowledge into the Tinkerer's brain (KG).
        
        Args:
            *sources: One or more Source objects
            target_kg: URL or path to the target knowledge graph/wiki to update.
                       Defaults to "https://skills.leeroo.com".
            
        Example:
            tinkerer.learn(
                Source.Repo("https://github.com/user/repo"),
                Source.Paper("./paper.pdf"),
            )
            
            # With custom KG target
            tinkerer.learn(
                Source.File("./notes.md"),
                target_kg="https://custom-wiki.example.com",
            )
        """
        print(f"Learning to KG: {target_kg}")
        for source in sources:
            source_type = type(source)
            learner_type = self._SOURCE_TO_LEARNER.get(source_type)
            
            if learner_type is None:
                print(f"Warning: Unknown source type {source_type}, skipping")
                continue
            
            # Create the appropriate learner
            print(f"  ⚠️ Learning from {source_type.__name__} not yet implemented"); continue  # learner = LearnerFactory.create(learner_type)
            
            # Get data from source
            if source_type == Source.Solution:
                source_data = source.obj  # Pass Solution object directly
            else:
                source_data = source.to_dict()
            
            # Learn and collect chunks
            chunks = learner.learn(source_data)
            self._learned_chunks.extend(chunks)
            
            # Index into KG if enabled
            if self.knowledge_search.is_enabled():
                for chunk in chunks:
                    self.knowledge_search.index(chunk.to_dict())
    
    def evolve(
        self,
        goal: str,
        context: Optional[List[Any]] = None,
        constraints: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        starting_repo_path: Optional[str] = None,
        max_iterations: int = 10,
        # --- Configuration options ---
        mode: Optional[str] = None,
        coding_agent: Optional[str] = None,
        # --- Execution options ---
        language: str = "python",
        main_file: str = "main.py",
        timeout: int = 300,
        data_dir: Optional[str] = None,
        # --- Evaluation options ---
        evaluator: str = "no_score",
        evaluator_params: Optional[Dict[str, Any]] = None,
        stop_condition: str = "never",
        stop_condition_params: Optional[Dict[str, Any]] = None,
    ) -> SolutionResult:
        """
        Evolve a solution for the given goal.
        
        Uses the Tinkerer's knowledge (KG) and online experimentation to
        generate robust software.
        
        Args:
            goal: The high-level objective (problem description)
            context: Optional list of Source objects to learn before evolving
            constraints: List of constraints (e.g., ["latency < 50ms"])
            output_path: Where to save the generated code
            starting_repo_path: Optional local path to an existing repository to improve.
                If provided, Praxium will clone/copy it into the experiment workspace and
                run the experiment loop on top of that baseline.
            max_iterations: Maximum experiment iterations (default: 10)
            
            mode: Configuration mode (GENERIC, MINIMAL, TREE_SEARCH, etc.)
            coding_agent: Coding agent to use (aider, gemini, claude_code, openhands)
            
            language: Programming language (default: python)
            main_file: Entry point file (default: main.py)
            timeout: Execution timeout in seconds (default: 300)
            data_dir: Path to data files needed for the evolution
            
            evaluator: Evaluator type (no_score, regex_pattern, llm_judge, etc.)
            evaluator_params: Parameters for the evaluator
            stop_condition: Stop condition (never, threshold, plateau, etc.)
            stop_condition_params: Parameters for stop condition
            
        Returns:
            SolutionResult with code_path, experiment_logs, and metadata
        """
        print(f"\n{'='*60}")
        print(f"EVOLVING: {goal}")
        print(f"{'='*60}")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Language: {language}")
        print(f"  Main file: {main_file}")
        if data_dir:
            print(f"  Data dir: {data_dir}")
        print(f"  Evaluator: {evaluator}")
        print(f"  Stop condition: {stop_condition}")
        print(f"  Coding agent: {coding_agent or 'from config'}")
        if starting_repo_path:
            print(f"  Starting repo: {starting_repo_path}")
        print()
        
        # Learn from context if provided
        if context:
            print(f"Learning from {len(context)} context sources...")
            self.learn(*context)
        
        # Build problem description
        problem = self._build_problem_description(goal, constraints)
        
        # Create problem handler with all options
        handler = GenericProblemHandler(
            problem_description=problem,
            main_file=main_file,
            language=language,
            timeout=timeout,
            data_dir=data_dir,
            evaluator=evaluator,
            evaluator_params=evaluator_params or {},
            stop_condition=stop_condition,
            stop_condition_params=stop_condition_params or {},
            additional_context=self._get_kg_context(),
        )
        
        # Create orchestrator
        orchestrator = OrchestratorAgent(
            handler,
            config_path=self.config_path,
            mode=mode,
            coding_agent=coding_agent,
            is_kg_active=self.knowledge_search.is_enabled(),
            knowledge_search=self.knowledge_search if self.knowledge_search.is_enabled() else None,
            # IMPORTANT:
            # - Many callers (CLI + E2E tests) pass `output_path` expecting the final repo to live there.
            # - The orchestration layer owns the experiment workspace (a git repo with branches).
            # - Therefore, when `output_path` is provided, we must use it as the workspace directory
            #   so `solution.code_path` points at a real git repo (with `.praxium/repo_memory.json`).
            workspace_dir=output_path,
            starting_repo_path=starting_repo_path,
        )
        
        # Run experimentation
        print("Running experiments...")
        orchestrator.solve(experiment_max_iter=max_iterations)
        
        # Collect results
        experiment_logs = self._extract_experiment_logs(orchestrator)
        workspace_path = orchestrator.search_strategy.workspace.workspace_dir
        
        # Checkout to best solution
        orchestrator.search_strategy.checkout_to_best_experiment_branch()
        
        # Final evaluation
        final_result = handler.final_evaluate(workspace_path)
        
        # Use custom output path if provided
        code_path = output_path or workspace_path
        cost = orchestrator.get_cumulative_cost()
        
        # Create solution result
        solution = SolutionResult(
            goal=goal,
            code_path=code_path,
            experiment_logs=experiment_logs,
            metadata={
                "constraints": constraints or [],
                "iterations": max_iterations,
                "cost": f"${cost:.3f}",
                "language": language,
                "evaluator": evaluator,
                "final_evaluation": final_result,
            }
        )
        
        print(f"\n{'='*60}")
        print("Evolution Complete")
        print(f"{'='*60}")
        print(f"Solution at: {code_path}")
        print(f"Experiments run: {len(experiment_logs)}")
        print(f"Total cost: ${cost:.3f}")
        
        return solution
    
    def deploy(
        self,
        solution: SolutionResult,
        strategy: DeployStrategy = DeployStrategy.AUTO,
        env_vars: Optional[Dict[str, str]] = None,
        coding_agent: str = "claude_code",
    ) -> Software:
        """
        Deploy a solution to create running software.
        
        Uses the deployment pipeline:
        1. Selector: Analyzes solution and selects strategy (if AUTO)
        2. Adapter: Adapts and deploys via coding agent
        3. Runner: Creates execution backend
        
        Args:
            solution: The SolutionResult from evolve()
            strategy: Where to deploy (AUTO, LOCAL, DOCKER, MODAL, BENTOML)
                - AUTO: System analyzes code and chooses best strategy
                - LOCAL: Run as local Python process (fastest)
                - DOCKER: Run in Docker container (isolated)
                - MODAL: Deploy to Modal.com (serverless, GPU)
                - BENTOML: Deploy with BentoML (production ML)
            env_vars: Environment variables to pass to the software
            coding_agent: Which coding agent for adaptation
            
        Returns:
            Software instance with unified interface:
            - .run(inputs) -> {"status": "success", "output": ...}
            - .stop() -> cleanup resources
            - .logs() -> execution logs
            - .is_healthy() -> health check
            
        Example:
            solution = tinkerer.evolve(goal="Create a trading bot")
            software = tinkerer.deploy(solution, strategy=DeployStrategy.LOCAL)
            result = software.run({"ticker": "AAPL"})
            software.stop()
        """
        print(f"\n{'='*60}")
        print(f"DEPLOYING: {solution.goal}")
        print(f"{'='*60}")
        print(f"  Strategy: {strategy}")
        print(f"  Code path: {solution.code_path}")
        print()
        
        config = DeployConfig(
            solution=solution,
            env_vars=env_vars,
            coding_agent=coding_agent,
        )
        
        return DeploymentFactory.create(strategy, config)
    
    def _build_problem_description(
        self, 
        goal: str, 
        constraints: Optional[List[str]]
    ) -> str:
        """Build the full problem description for the orchestrator."""
        parts = [f"# Goal\n{goal}"]
        
        if constraints:
            parts.append("\n# Constraints")
            for c in constraints:
                parts.append(f"- {c}")
        
        return "\n".join(parts)
    
    def _get_kg_context(self) -> str:
        """Get relevant knowledge from learned chunks."""
        if not self._learned_chunks:
            return ""
        
        # For MVP, just concatenate recent chunks
        recent = self._learned_chunks[-5:]  # Last 5 chunks
        context_parts = ["# Knowledge from Tinkerer's Brain"]
        for chunk in recent:
            context_parts.append(f"- [{chunk.chunk_type}] {chunk.content[:200]}...")
        
        return "\n".join(context_parts)
    
    def _extract_experiment_logs(self, orchestrator: OrchestratorAgent) -> List[str]:
        """Extract experiment history as string logs."""
        logs = []
        history = orchestrator.search_strategy.get_experiment_history()
        
        for exp in history:
            if hasattr(exp, 'had_error') and exp.had_error:
                logs.append(f"Failed: {exp.solution[:100]}... (Error: {exp.error_message})")
            else:
                score = getattr(exp, 'score', 'N/A')
                logs.append(f"Success: {exp.solution[:100]}... (Score: {score})")
        
        return logs


# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

__all__ = [
    "Tinkerer",
    "Source",
    "SolutionResult",
    "Software",
    "DeployStrategy",
    "DeployConfig",
    "DeploymentFactory",
]
