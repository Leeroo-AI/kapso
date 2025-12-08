# Adapter Agent
#
# Uses coding agents to adapt a solution for a specific deployment target.
# Loads instructions from the strategies/ registry.
#
# Usage:
#     adapter = AdapterAgent(coding_agent_type="claude_code")
#     result = adapter.adapt(solution, setting)

import re
import shutil
from pathlib import Path
from typing import List, Optional

from src.deployment.base import DeploymentSetting, AdaptationResult
from src.deployment.strategies import StrategyRegistry
from src.execution.coding_agents.factory import CodingAgentFactory
from src.execution.solution import SolutionResult


# Path to the adaptation prompt template
ADAPTATION_PROMPT_PATH = Path(__file__).parent / "adaptation_prompt.md"


class AdapterAgent:
    """
    Uses coding agents to adapt a solution for deployment.
    
    Flow:
    1. Load instructions from strategies registry
    2. Create coding agent (Claude Code, Aider, etc.)
    3. Generate adaptation prompt
    4. Execute coding agent to transform the code and deploy
    5. Validate the adaptation
    6. Return result with deploy script and run interface
    """
    
    def __init__(
        self,
        coding_agent_type: str = "claude_code",
        model: str = "claude-opus-4-5",
        fallback_agent_type: str = "gemini",
        max_retries: int = 2,
    ):
        """
        Initialize the adapter agent.
        
        Args:
            coding_agent_type: Primary coding agent (claude_code, aider, gemini)
            model: LLM model for the primary coding agent
            fallback_agent_type: Secondary coding agent if primary fails
            max_retries: Maximum retry attempts if adaptation fails
        """
        self.coding_agent_type = coding_agent_type
        self.model = model
        self.fallback_agent_type = fallback_agent_type
        self.max_retries = max_retries
        self.registry = StrategyRegistry.get()
    
    def _create_adapted_workspace(self, original_path: str, strategy: str) -> str:
        """
        Create a copy of the solution for adaptation.
        
        The original solution is never modified. All adaptation happens
        in the new workspace at {original_path}_adapted_{strategy}.
        
        Args:
            original_path: Path to the original solution
            strategy: Deployment strategy name (used in directory name)
            
        Returns:
            Path to the adapted workspace
        """
        original = Path(original_path)
        adapted = Path(f"{original_path}_adapted_{strategy}")
        
        # Remove existing adapted workspace if it exists
        if adapted.exists():
            shutil.rmtree(adapted)
        
        # Copy the original solution to the adapted workspace
        shutil.copytree(original, adapted)
        
        return str(adapted)
    
    def adapt(
        self,
        solution: SolutionResult,
        setting: DeploymentSetting,
        allowed_strategies: Optional[List[str]] = None,
    ) -> AdaptationResult:
        """
        Adapt a solution for the specified deployment setting.
        
        Creates a copy of the solution at {code_path}_adapted_{strategy} and
        performs adaptation there. The original solution is never modified.
        
        Args:
            solution: The SolutionResult from Expert.build()
            setting: Selected deployment configuration
            allowed_strategies: Optional list of allowed strategies
            
        Returns:
            AdaptationResult with deploy script and run interface
        """
        logs: List[str] = []
        logs.append(f"Adapting for {setting.strategy} deployment")
        
        # Extract from solution
        original_path = solution.code_path
        goal = solution.goal
        
        # Validate strategy is available
        available = self.registry.list_strategies(allowed=allowed_strategies)
        if setting.strategy not in available:
            return AdaptationResult(
                success=False,
                adapted_path=original_path,
                deploy_script="",
                run_interface={},
                error=f"Strategy '{setting.strategy}' not available. Options: {available}",
                logs=logs,
            )
        
        # 1. Create adapted workspace (copy original, don't modify it)
        adapted_path = self._create_adapted_workspace(original_path, setting.strategy)
        logs.append(f"Created adapted workspace: {adapted_path}")
        
        # Track endpoint extracted from agent output
        endpoint: Optional[str] = None
        agent_output: str = ""
        
        # 2. Load target-specific instructions from registry
        target_instructions = self.registry.get_adapter_instruction(setting.strategy)
        logs.append(f"Loaded instructions for {setting.strategy}")
        
        # 3. Create and run coding agent on the adapted workspace
        try:
            config = CodingAgentFactory.build_config(
                agent_type=self.coding_agent_type,
                model=self.model,
                workspace=adapted_path,
            )
            agent = CodingAgentFactory.create(config)
            agent.initialize(adapted_path)
            
            logs.append(f"Initialized {self.coding_agent_type} agent")
            
            # Build prompt
            prompt = self._build_adaptation_prompt(
                goal=goal,
                setting=setting,
                target_instructions=target_instructions,
            )
            
            # Execute coding agent (agent also runs deployment via Bash tool)
            result = agent.generate_code(prompt)
            logs.append(f"Coding agent completed: success={result.success}")
            
            if not result.success:
                return AdaptationResult(
                    success=False,
                    adapted_path=adapted_path,
                    deploy_script="",
                    run_interface={},
                    error=result.error or "Coding agent failed",
                    logs=logs,
                )
            
            files_changed = result.files_changed
            agent_output = result.output
            logs.append(f"Files changed: {files_changed}")
            
            # Extract endpoint URL from agent output
            endpoint = self._extract_endpoint_from_output(agent_output)
            if endpoint:
                logs.append(f"Deployment endpoint extracted: {endpoint}")
            
            agent.cleanup()
            
        except (ImportError, ValueError) as e:
            logs.append(f"Primary agent ({self.coding_agent_type}) not available: {e}")
            files_changed, agent_output = self._run_fallback_agent(
                adapted_path, goal, setting, target_instructions, logs
            )
            if files_changed is None:
                return AdaptationResult(
                    success=False,
                    adapted_path=adapted_path,
                    deploy_script="",
                    run_interface={},
                    error="Both primary and fallback agents failed",
                    logs=logs,
                )
            endpoint = self._extract_endpoint_from_output(agent_output)
            if endpoint:
                logs.append(f"Deployment endpoint extracted: {endpoint}")
                
        except Exception as e:
            logs.append(f"Primary agent ({self.coding_agent_type}) error: {e}")
            files_changed, agent_output = self._run_fallback_agent(
                adapted_path, goal, setting, target_instructions, logs
            )
            if files_changed is None:
                return AdaptationResult(
                    success=False,
                    adapted_path=adapted_path,
                    deploy_script="",
                    run_interface={},
                    error=str(e),
                    logs=logs,
                )
            endpoint = self._extract_endpoint_from_output(agent_output)
            if endpoint:
                logs.append(f"Deployment endpoint extracted: {endpoint}")
        
        # 4. Build run interface (how to call the deployed software)
        run_interface = self._build_run_interface(setting.strategy, endpoint)
        
        logs.append(f"Adaptation complete at: {adapted_path}")
        
        return AdaptationResult(
            success=True,
            adapted_path=adapted_path,
            deploy_script="",  # Agent already ran deployment
            run_interface=run_interface,
            files_changed=files_changed if isinstance(files_changed, list) else [],
            logs=logs,
        )
    
    def _run_fallback_agent(
        self,
        adapted_path: str,
        goal: str,
        setting: DeploymentSetting,
        target_instructions: str,
        logs: List[str],
    ) -> tuple:
        """
        Run the fallback coding agent when primary agent fails.
        
        Args:
            adapted_path: Path to the adapted workspace (copy of original)
            goal: The original goal/objective
            setting: Selected deployment configuration
            target_instructions: Strategy-specific instructions
            logs: Log list to append to
        
        Returns:
            Tuple of (files_changed, agent_output), or (None, "") if fails
        """
        logs.append(f"Attempting fallback agent: {self.fallback_agent_type}")
        
        try:
            fallback_config = CodingAgentFactory.build_config(
                agent_type=self.fallback_agent_type,
                workspace=adapted_path,
            )
            fallback_agent = CodingAgentFactory.create(fallback_config)
            fallback_agent.initialize(adapted_path)
            
            logs.append(f"Initialized fallback agent: {self.fallback_agent_type}")
            
            prompt = self._build_adaptation_prompt(
                goal=goal,
                setting=setting,
                target_instructions=target_instructions,
            )
            
            result = fallback_agent.generate_code(prompt)
            logs.append(f"Fallback agent completed: success={result.success}")
            
            if not result.success:
                logs.append(f"Fallback agent failed: {result.error}")
                return None, ""
            
            files_changed = result.files_changed
            agent_output = result.output
            logs.append(f"Fallback agent files changed: {files_changed}")
            
            fallback_agent.cleanup()
            return (files_changed if isinstance(files_changed, list) else [], agent_output)
            
        except Exception as e:
            logs.append(f"Fallback agent ({self.fallback_agent_type}) also failed: {e}")
            return None, ""
    
    def _build_adaptation_prompt(
        self,
        goal: str,
        setting: DeploymentSetting,
        target_instructions: str,
    ) -> str:
        """
        Build the prompt for the coding agent.
        
        Loads template from adaptation_prompt.md and fills in placeholders.
        """
        template = ADAPTATION_PROMPT_PATH.read_text()
        
        return template.format(
            goal=goal,
            strategy=setting.strategy,
            provider=setting.provider or "N/A",
            interface=setting.interface,
            resources=setting.resources,
            target_instructions=target_instructions,
        )
    
    def _build_run_interface(self, strategy: str, endpoint: Optional[str]) -> dict:
        """
        Build the run interface for the deployed software.
        
        The coding agent follows standard conventions:
        - main.py with predict() function
        - Endpoint reported via <endpoint_url> tag
        
        Args:
            strategy: Deployment strategy name
            endpoint: Endpoint URL extracted from agent output (if any)
            
        Returns:
            Interface dict for the Runner
        """
        # Map strategy to interface type
        type_map = {
            "local": "function",
            "docker": "http",
            "modal": "modal",
            "bentoml": "bentocloud",
            "langgraph": "langgraph",
        }
        
        interface = {
            "type": type_map.get(strategy, "function"),
            "module": "main",
            "callable": "predict",
        }
        
        # Add endpoint if available (from agent's deployment output)
        if endpoint:
            interface["endpoint"] = endpoint
            interface["deployment_url"] = endpoint
        
        return interface
    
    def _extract_endpoint_from_output(self, output: str) -> Optional[str]:
        """
        Extract deployment endpoint URL from coding agent output.
        
        The agent is instructed to output the endpoint in XML-style tags:
        <endpoint_url>https://...</endpoint_url>
        """
        if not output:
            return None
        
        match = re.search(r'<endpoint_url>\s*(https?://[^\s<>]+)\s*</endpoint_url>', output)
        if match:
            return match.group(1).strip()
        
        return None
