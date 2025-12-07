# Deployment Factory
#
# Factory for creating deployed Software instances.
# Handles the full deployment pipeline:
# 1. Select strategy (if AUTO)
# 2. Adapt the repo for the strategy
# 3. Create appropriate runner
# 4. Return unified Software instance
#
# Usage:
#     from src.deployment import DeploymentFactory, DeployStrategy, DeployConfig
#     
#     config = DeployConfig(code_path="./solution", goal="My Goal")
#     software = DeploymentFactory.create(DeployStrategy.AUTO, config)
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
    2. Adapt the repo for the strategy
    3. Create appropriate runner
    4. Return unified Software instance
    
    Usage:
        software = DeploymentFactory.create(
            strategy=DeployStrategy.AUTO,
            config=DeployConfig(code_path="./solution", goal="My Goal"),
        )
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
            setting = cls._create_setting(strategy, config.code_path)
        
        print(f"[Deployment] Selected: {setting.strategy} ({setting.reasoning})")
        
        # Phase 2: Adaptation
        print(f"[Deployment] Phase 2: Adapting repository...")
        adaptation = cls._adapt_repo(config, setting, strategies)
        
        if not adaptation.success:
            raise RuntimeError(f"Adaptation failed: {adaptation.error}")
        
        print(f"[Deployment] Adaptation complete. Files changed: {len(adaptation.files_changed)}")
        
        # Phase 3: Deploy to cloud (if applicable)
        endpoint = None
        if setting.strategy in ["modal", "bentoml", "langgraph"]:
            print(f"[Deployment] Phase 3: Deploying to {setting.strategy}...")
            deploy_result = cls._deploy(config.code_path, setting.strategy, adaptation.deploy_script)
            if deploy_result.get("success"):
                endpoint = deploy_result.get("endpoint")
                print(f"[Deployment] Deployed! Endpoint: {endpoint or 'N/A'}")
            else:
                print(f"[Deployment] Deploy warning: {deploy_result.get('error', 'Unknown')}")
        
        # Phase 4: Create Runner
        print(f"[Deployment] Phase 4: Creating runner...")
        # Update interface with endpoint if we got one from deployment
        if endpoint:
            adaptation.run_interface["endpoint"] = endpoint
            adaptation.run_interface["deployment_url"] = endpoint
        runner = cls._create_runner(config, setting, adaptation)
        
        # Phase 5: Create unified Software
        info = DeploymentInfo(
            strategy=setting.strategy,
            provider=setting.provider,
            deploy_command=adaptation.deploy_script,
            endpoint=endpoint or adaptation.run_interface.get("endpoint"),
            adapted_files=adaptation.files_changed,
            resources=setting.resources,
        )
        
        print(f"[Deployment] Ready. Deploy command: {adaptation.deploy_script}")
        
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
        return selector.select(config.code_path, config.goal, strategies=strategies)
    
    @classmethod
    def _create_setting(
        cls, 
        strategy: DeployStrategy,
        code_path: str,
    ) -> DeploymentSetting:
        """Create setting from explicit strategy (user-specified)."""
        from src.deployment.selector.agent import SelectorAgent
        
        selector = SelectorAgent()
        return selector._create_setting_for_strategy(strategy.value, code_path)
    
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
            solution_path=config.code_path,
            goal=config.goal,
            setting=setting,
            strategies=strategies,
            validate=config.validate,
        )
    
    @classmethod
    def _deploy(
        cls,
        code_path: str,
        strategy: str,
        deploy_script: str,
    ) -> dict:
        """
        Execute deployment to cloud provider.
        
        Args:
            code_path: Path to the adapted code
            strategy: Deployment strategy (modal, bentoml, langgraph)
            deploy_script: Command to run for deployment
            
        Returns:
            Dict with success, endpoint, and error keys
        """
        import subprocess
        import os
        
        result = {"success": False, "endpoint": None, "error": None}
        
        try:
            # Run the deploy command
            proc = subprocess.run(
                deploy_script,
                shell=True,
                cwd=code_path,
                capture_output=True,
                text=True,
                timeout=300,
                env={**os.environ},
            )
            
            if proc.returncode == 0:
                result["success"] = True
                # Try to extract endpoint from output
                output = proc.stdout + proc.stderr
                for line in output.split("\n"):
                    if "https://" in line:
                        # Extract URL from line
                        import re
                        urls = re.findall(r'https://[^\s<>"]+', line)
                        if urls:
                            result["endpoint"] = urls[0]
                            break
            else:
                result["error"] = proc.stderr[:200] if proc.stderr else "Deploy failed"
                
        except subprocess.TimeoutExpired:
            result["error"] = "Deploy timeout (300s)"
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
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
        
        # Import runners from strategy packages
        if strategy == "local" or interface_type == "function":
            from src.deployment.strategies.local.runner import LocalRunner
            return LocalRunner(
                code_path=config.code_path,
                module=interface.get("module", "main"),
                callable=interface.get("callable", "predict"),
            )
        
        elif strategy == "docker" or interface_type == "http":
            from src.deployment.strategies.docker.runner import DockerRunner
            return DockerRunner(
                endpoint=interface.get("endpoint", f"http://localhost:{config.port}"),
                predict_path=interface.get("path", "/predict"),
                timeout=config.timeout,
                code_path=config.code_path,
            )
        
        elif strategy == "modal" or interface_type == "modal":
            from src.deployment.strategies.modal.runner import ModalRunner
            app_name = interface.get("app_name", config.code_path.replace("/", "-").replace(".", "-"))
            return ModalRunner(
                app_name=app_name,
                function_name=interface.get("callable", "predict"),
                code_path=config.code_path,
            )
        
        elif strategy == "bentoml" or interface_type == "bentocloud":
            from src.deployment.strategies.bentoml.runner import BentoMLRunner
            deployment_name = interface.get("deployment_name", config.code_path.split("/")[-1])
            return BentoMLRunner(
                deployment_name=deployment_name,
                endpoint=interface.get("endpoint"),
                predict_path=interface.get("path", "/predict"),
                code_path=config.code_path,
            )
        
        elif strategy == "langgraph" or interface_type == "langgraph":
            from src.deployment.strategies.langgraph.runner import LangGraphRunner
            return LangGraphRunner(
                deployment_url=interface.get("deployment_url"),
                assistant_id=interface.get("assistant_id", "agent"),
                code_path=config.code_path,
            )
        
        else:
            # Default to local runner
            from src.deployment.strategies.local.runner import LocalRunner
            return LocalRunner(
                code_path=config.code_path,
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
