# Deployment Factory
#
# Factory for creating deployed Software instances.
# Handles the full deployment pipeline:
# 1. Select strategy (if AUTO)
# 2. Adapt and deploy (coding agent does both)
# 3. Create appropriate runner
# 4. Return unified Software instance
#
# The coding agent is responsible for:
# - Creating deployment files
# - Running the deploy command
# - Reporting the endpoint URL
#
# Usage:
#     software = expert.deploy(solution, strategy=DeployStrategy.LOCAL)
#     result = software.run({"input": "data"})

from typing import Optional, List

from src.deployment.base import (
    Software,
    DeployConfig,
    DeployStrategy,
    DeploymentSetting,
    DeploymentInfo,
)
from src.deployment.software import DeployedSoftware
from src.deployment.strategies.base import Runner
from src.deployment.strategies import StrategyRegistry


class DeploymentFactory:
    """
    Factory for creating Software instances.
    
    Handles the full deployment pipeline:
    1. Select strategy (if AUTO)
    2. Adapt and deploy (coding agent handles both)
    3. Create appropriate runner
    4. Return unified Software instance
    
    The coding agent creates deployment files, runs the deploy command,
    and reports the endpoint. No separate deployment step is needed.
    """
    
    @classmethod
    def create(
        cls,
        strategy: DeployStrategy,
        config: DeployConfig,
        strategies: Optional[List[str]] = None,
    ) -> Software:
        """
        Create a deployed Software instance.
        
        Args:
            strategy: Deployment strategy (AUTO, LOCAL, DOCKER, etc.)
            config: Deployment configuration
            strategies: Optional list of allowed strategies (for AUTO selection)
            
        Returns:
            Software instance with unified interface
        """
        print(f"[Deployment] Creating deployment...")
        
        # Phase 1: Selection
        if strategy == DeployStrategy.AUTO:
            print(f"[Deployment] Phase 1: Selecting optimal strategy...")
            setting = cls._select_strategy(config, strategies)
        else:
            # Validate explicit strategy is allowed
            if strategies and strategy.value not in strategies:
                raise ValueError(f"Strategy '{strategy.value}' not in allowed: {strategies}")
            print(f"[Deployment] Phase 1: Using specified strategy: {strategy.value}")
            setting = cls._create_setting(strategy)
        
        print(f"[Deployment] Selected: {setting.strategy} ({setting.reasoning})")
        
        # Phase 2: Adaptation (coding agent adapts AND deploys)
        print(f"[Deployment] Phase 2: Adapting and deploying...")
        adaptation = cls._adapt_repo(config, setting, strategies)
        
        if not adaptation.success:
            raise RuntimeError(f"Adaptation failed: {adaptation.error}")
        
        # Endpoint comes from the coding agent's output (it runs deployment)
        endpoint = adaptation.run_interface.get("endpoint")
        print(f"[Deployment] Adaptation complete. Files changed: {len(adaptation.files_changed)}")
        if endpoint:
            print(f"[Deployment] Endpoint: {endpoint}")
        
        # Phase 3: Create Runner
        print(f"[Deployment] Phase 3: Creating runner...")
        runner = cls._create_runner(config, setting, adaptation)
        
        # Phase 4: Create unified Software
        info = DeploymentInfo(
            strategy=setting.strategy,
            provider=setting.provider,
            endpoint=endpoint,
            adapted_path=adaptation.adapted_path,
            adapted_files=adaptation.files_changed,
            resources=setting.resources,
        )
        
        print(f"[Deployment] Ready.")
        
        return DeployedSoftware(config=config, runner=runner, info=info)
    
    @classmethod
    def _select_strategy(
        cls, 
        config: DeployConfig,
        strategies: Optional[List[str]] = None,
    ) -> DeploymentSetting:
        """
        Use LLM-based selector to choose best strategy.
        
        Args:
            config: Deployment configuration
            strategies: Optional list of allowed strategies
        """
        from src.deployment.selector.agent import SelectorAgent
        
        selector = SelectorAgent()
        return selector.select(config.solution, strategies=strategies)
    
    @classmethod
    def _create_setting(
        cls, 
        strategy: DeployStrategy,
    ) -> DeploymentSetting:
        """Create setting from explicit strategy (user-specified)."""
        from src.deployment.selector.agent import SelectorAgent
        
        selector = SelectorAgent()
        return selector._create_setting_for_strategy(strategy.value)
    
    @classmethod
    def _adapt_repo(
        cls,
        config: DeployConfig,
        setting: DeploymentSetting,
        strategies: Optional[List[str]] = None,
    ):
        """
        Adapt the repo for deployment.
        
        Uses coding agent if available, falls back to minimal adaptation.
        
        Args:
            config: Deployment configuration
            setting: Selected deployment setting
            strategies: Optional list of allowed strategies
        """
        from src.deployment.adapter.agent import AdapterAgent
        
        adapter = AdapterAgent(
            coding_agent_type=config.coding_agent,
            max_retries=2,
        )
        
        return adapter.adapt(
            solution=config.solution,
            setting=setting,
            allowed_strategies=strategies,
        )
    
    @classmethod
    def _create_runner(
        cls,
        config: DeployConfig,
        setting: DeploymentSetting,
        adaptation,
    ) -> Runner:
        """
        Create the appropriate runner for the strategy.
        
        Uses strategy packages from strategies/ directory.
        """
        interface = adaptation.run_interface
        interface_type = interface.get("type", "function")
        strategy = setting.strategy
        adapted_path = adaptation.adapted_path
        
        # Import runners from strategy packages
        if strategy == "local" or interface_type == "function":
            from src.deployment.strategies.local.runner import LocalRunner
            return LocalRunner(
                code_path=adapted_path,
                module=interface.get("module", "main"),
                callable=interface.get("callable", "predict"),
            )
        
        elif strategy == "docker" or interface_type == "http":
            from src.deployment.strategies.docker.runner import DockerRunner
            return DockerRunner(
                endpoint=interface.get("endpoint", "http://localhost:8000"),
                predict_path=interface.get("path", "/predict"),
                timeout=config.timeout,
                code_path=adapted_path,
            )
        
        elif strategy == "modal" or interface_type == "modal":
            from src.deployment.strategies.modal.runner import ModalRunner
            app_name = interface.get("app_name", adapted_path.replace("/", "-").replace(".", "-"))
            return ModalRunner(
                app_name=app_name,
                function_name=interface.get("callable", "predict"),
                code_path=adapted_path,
            )
        
        elif strategy == "bentoml" or interface_type == "bentocloud":
            from src.deployment.strategies.bentoml.runner import BentoMLRunner
            deployment_name = interface.get("deployment_name", adapted_path.split("/")[-1])
            return BentoMLRunner(
                deployment_name=deployment_name,
                endpoint=interface.get("endpoint"),
                predict_path=interface.get("path", "/predict"),
                code_path=adapted_path,
            )
        
        elif strategy == "langgraph" or interface_type == "langgraph":
            from src.deployment.strategies.langgraph.runner import LangGraphRunner
            return LangGraphRunner(
                deployment_url=interface.get("deployment_url"),
                assistant_id=interface.get("assistant_id", "agent"),
                code_path=adapted_path,
            )
        
        else:
            # Default to local runner
            from src.deployment.strategies.local.runner import LocalRunner
            return LocalRunner(
                code_path=adapted_path,
                module="main",
                callable="predict",
            )
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        """List all available deployment strategies from registry."""
        registry = StrategyRegistry.get()
        return registry.list_strategies()
    
    @classmethod
    def explain_strategy(cls, strategy: str) -> str:
        """Get explanation of a deployment strategy from its selector instruction."""
        registry = StrategyRegistry.get()
        
        if not registry.strategy_exists(strategy):
            return f"Unknown strategy: {strategy}"
        
        # Extract summary from selector instruction
        instruction = registry.get_selector_instruction(strategy)
        
        # Look for Summary section
        import re
        match = re.search(r'##\s*Summary\s*\n+([^\n#]+)', instruction, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Fallback to first line
        lines = instruction.strip().split('\n')
        for line in lines:
            if line.strip() and not line.startswith('#'):
                return line.strip()
        
        return f"Deployment strategy: {strategy}"
    
    @classmethod
    def print_strategies_info(cls) -> None:
        """Print information about all deployment strategies."""
        print("\nAvailable Deployment Strategies:")
        print("=" * 50)
        for strategy in cls.list_strategies():
            desc = cls.explain_strategy(strategy)
            print(f"  {strategy}: {desc}")
        print()
