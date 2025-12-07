# Adapter Agent
#
# Uses coding agents to adapt a solution for a specific deployment target.
# Loads instructions from the strategies/ registry.
#
# Usage:
#     adapter = AdapterAgent(coding_agent_type="claude_code")
#     result = adapter.adapt(solution_path, goal, setting)

import re
from pathlib import Path
from typing import List, Optional

from src.deployment.base import DeploymentSetting, AdaptationResult
from src.deployment.adapter.validator import AdaptationValidator
from src.deployment.strategies import StrategyRegistry


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
        model: str = "claude-sonnet-4-20250514",
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
        self._validator = AdaptationValidator()
        self.registry = StrategyRegistry.get()
    
    def adapt(
        self,
        solution_path: str,
        goal: str,
        setting: DeploymentSetting,
        strategies: Optional[List[str]] = None,
        validate: bool = True,
    ) -> AdaptationResult:
        """
        Adapt a solution for the specified deployment setting.
        
        Args:
            solution_path: Path to the solution repository
            goal: Original goal/objective
            setting: Selected deployment configuration
            strategies: Optional allowed strategies (for validation)
            validate: Whether to run validation after adaptation
            
        Returns:
            AdaptationResult with deploy script and run interface
        """
        logs: List[str] = []
        logs.append(f"Adapting for {setting.strategy} deployment")
        
        # Track endpoint extracted from agent output
        endpoint: Optional[str] = None
        agent_output: str = ""
        
        # Validate strategy is available
        available = self.registry.list_strategies(allowed=strategies)
        if setting.strategy not in available:
            return AdaptationResult(
                success=False,
                adapted_path=solution_path,
                deploy_script="",
                run_interface={},
                error=f"Strategy '{setting.strategy}' not available. Options: {available}",
                logs=logs,
            )
        
        # 1. Load target-specific instructions from registry
        target_instructions = self.registry.get_adapter_instruction(setting.strategy)
        logs.append(f"Loaded instructions for {setting.strategy}")
        
        # 2. Create and run coding agent
        try:
            from src.execution.coding_agents.factory import CodingAgentFactory
            
            config = CodingAgentFactory.build_config(
                agent_type=self.coding_agent_type,
                model=self.model,
                workspace=solution_path,
            )
            agent = CodingAgentFactory.create(config)
            agent.initialize(solution_path)
            
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
                    adapted_path=solution_path,
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
                solution_path, goal, setting, target_instructions, logs
            )
            if files_changed is None:
                return AdaptationResult(
                    success=False,
                    adapted_path=solution_path,
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
                solution_path, goal, setting, target_instructions, logs
            )
            if files_changed is None:
                return AdaptationResult(
                    success=False,
                    adapted_path=solution_path,
                    deploy_script="",
                    run_interface={},
                    error=str(e),
                    logs=logs,
                )
            endpoint = self._extract_endpoint_from_output(agent_output)
            if endpoint:
                logs.append(f"Deployment endpoint extracted: {endpoint}")
        
        # 3. Validate adaptation
        if validate:
            validation = self._validator.validate(solution_path, setting)
            logs.extend(validation.logs)
            
            if not validation.success:
                logs.append(f"Validation failed: {validation.error}")
                return AdaptationResult(
                    success=False,
                    adapted_path=solution_path,
                    deploy_script="",
                    run_interface={},
                    error=validation.error,
                    logs=logs,
                )
        
        # 4. Generate deploy script and run interface
        deploy_script = self._get_deploy_script(setting.strategy, target_instructions)
        run_interface = self._get_run_interface(solution_path, setting.strategy, target_instructions)
        
        # Add endpoint to run_interface if extracted
        if endpoint:
            run_interface["endpoint"] = endpoint
            run_interface["deployment_url"] = endpoint
            logs.append(f"Endpoint added to run_interface: {endpoint}")
        
        logs.append("Adaptation complete")
        
        return AdaptationResult(
            success=True,
            adapted_path=solution_path,
            deploy_script=deploy_script,
            run_interface=run_interface,
            files_changed=files_changed if isinstance(files_changed, list) else [],
            logs=logs,
        )
    
    def _run_fallback_agent(
        self,
        solution_path: str,
        goal: str,
        setting: DeploymentSetting,
        target_instructions: str,
        logs: List[str],
    ) -> tuple:
        """
        Run the fallback coding agent when primary agent fails.
        
        Returns:
            Tuple of (files_changed, agent_output), or (None, "") if fails
        """
        logs.append(f"Attempting fallback agent: {self.fallback_agent_type}")
        
        try:
            from src.execution.coding_agents.factory import CodingAgentFactory
            
            fallback_config = CodingAgentFactory.build_config(
                agent_type=self.fallback_agent_type,
                workspace=solution_path,
            )
            fallback_agent = CodingAgentFactory.create(fallback_config)
            fallback_agent.initialize(solution_path)
            
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
    
    def _get_deploy_script(self, strategy: str, instructions: str) -> str:
        """
        Extract deploy command from adapter instructions.
        
        Looks for "## DEPLOY COMMAND" section with bash code block.
        """
        match = re.search(
            r'##\s*DEPLOY\s*COMMAND\s*\n+```bash\s*\n(.+?)\n```',
            instructions,
            re.IGNORECASE | re.DOTALL
        )
        if match:
            return match.group(1).strip()
        
        # Fallback defaults
        defaults = {
            "local": "python main.py",
            "docker": "docker build -t solution . && docker run -p 8000:8000 solution",
            "modal": "modal deploy modal_app.py",
            "bentoml": "python deploy.py",
            "langgraph": "langgraph deploy",
        }
        return defaults.get(strategy, "python main.py")
    
    def _get_run_interface(
        self,
        solution_path: str,
        strategy: str,
        instructions: str,
    ) -> dict:
        """
        Extract run interface from adapter instructions.
        
        Looks for "## RUN INTERFACE" section.
        """
        interface = {}
        
        # Try to extract from instructions
        match = re.search(
            r'##\s*RUN\s*INTERFACE\s*\n((?:[-*]\s*\w+:.+\n?)+)',
            instructions,
            re.IGNORECASE
        )
        if match:
            for line in match.group(1).strip().split('\n'):
                if ':' in line:
                    key, value = line.lstrip('-* ').split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Handle special placeholders
                    if value == "derived from path":
                        path = Path(solution_path)
                        value = path.name.replace("_", "-").replace(" ", "-")
                    
                    interface[key] = value
        
        # Ensure we have at least type
        if "type" not in interface:
            type_map = {
                "local": "function",
                "docker": "http",
                "modal": "modal",
                "bentoml": "bentocloud",
                "langgraph": "langgraph",
            }
            interface["type"] = type_map.get(strategy, "function")
        
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
