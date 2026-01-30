# Adapter Agent
#
# Uses coding agents to adapt a solution for a specific deployment target.
# Loads instructions from the strategies/ registry.
#
# Usage:
#     adapter = AdapterAgent(coding_agent_type="claude_code")
#     result = adapter.adapt(solution, setting)

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from kapso.deployment.base import DeploymentSetting, AdaptationResult
from kapso.deployment.strategies import StrategyRegistry
from kapso.execution.coding_agents.factory import CodingAgentFactory
from kapso.execution.solution import SolutionResult


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
        
        # Path to the adaptation prompt template
        self.adaptation_prompt_path = Path(__file__).parent / "adaptation_prompt.txt"
    
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
            solution: The SolutionResult from Kapso.evolve()
            setting: Selected deployment configuration
            allowed_strategies: Optional list of allowed strategies
            
        Returns:
            AdaptationResult with run interface and adapted path
        """
        print(f"[Adapter] Adapting for {setting.strategy} deployment")
        
        # Extract from solution
        original_path = solution.code_path
        goal = solution.goal
        
        # Validate strategy is available
        available = self.registry.list_strategies(allowed=allowed_strategies)
        if setting.strategy not in available:
            return AdaptationResult(
                success=False,
                adapted_path=original_path,
                run_interface={},
                error=f"Strategy '{setting.strategy}' not available. Options: {available}",
            )
        
        # 1. Create adapted workspace (copy original, don't modify it)
        adapted_path = self._create_adapted_workspace(original_path, setting.strategy)
        print(f"[Adapter] Created adapted workspace: {adapted_path}")
        
        # Track values extracted from agent output
        endpoint: Optional[str] = None
        run_interface_from_agent: Optional[Dict[str, Any]] = None
        agent_output: str = ""
        
        # 2. Load target-specific instructions from registry
        target_instructions = self.registry.get_adapter_instruction(setting.strategy)
        
        # 3. Create and run coding agent on the adapted workspace
        try:
            config = CodingAgentFactory.build_config(
                agent_type=self.coding_agent_type,
                model=self.model,
                workspace=adapted_path,
            )
            agent = CodingAgentFactory.create(config)
            agent.initialize(adapted_path)
            
            print(f"[Adapter] Running {self.coding_agent_type} agent...")
            
            # Build prompt
            prompt = self._build_adaptation_prompt(
                goal=goal,
                setting=setting,
                target_instructions=target_instructions,
            )
            
            # Execute coding agent (agent also runs deployment via Bash tool)
            result = agent.generate_code(prompt)
            
            if not result.success:
                return AdaptationResult(
                    success=False,
                    adapted_path=adapted_path,
                    run_interface={},
                    error=result.error or "Coding agent failed",
                )
            
            files_changed = result.files_changed
            agent_output = result.output
            print(f"[Adapter] Files changed: {len(files_changed) if files_changed else 0}")
            
            # Extract run_interface and endpoint from agent output
            run_interface_from_agent = self._extract_run_interface_from_output(agent_output)
            endpoint = self._extract_endpoint_from_output(agent_output)
            if run_interface_from_agent:
                print(f"[Adapter] Run interface from agent: {run_interface_from_agent}")
            if endpoint:
                print(f"[Adapter] Endpoint: {endpoint}")
            
            agent.cleanup()
            
        except (ImportError, ValueError) as e:
            print(f"[Adapter] Primary agent not available: {e}")
            files_changed, agent_output = self._run_fallback_agent(
                adapted_path, goal, setting, target_instructions
            )
            if files_changed is None:
                return AdaptationResult(
                    success=False,
                    adapted_path=adapted_path,
                    run_interface={},
                    error="Both primary and fallback agents failed",
                )
            run_interface_from_agent = self._extract_run_interface_from_output(agent_output)
            endpoint = self._extract_endpoint_from_output(agent_output)
            if run_interface_from_agent:
                print(f"[Adapter] Run interface from agent: {run_interface_from_agent}")
            if endpoint:
                print(f"[Adapter] Endpoint: {endpoint}")
                
        except Exception as e:
            print(f"[Adapter] Primary agent error: {e}")
            files_changed, agent_output = self._run_fallback_agent(
                adapted_path, goal, setting, target_instructions
            )
            if files_changed is None:
                return AdaptationResult(
                    success=False,
                    adapted_path=adapted_path,
                    run_interface={},
                    error=str(e),
                )
            run_interface_from_agent = self._extract_run_interface_from_output(agent_output)
            endpoint = self._extract_endpoint_from_output(agent_output)
            if run_interface_from_agent:
                print(f"[Adapter] Run interface from agent: {run_interface_from_agent}")
            if endpoint:
                print(f"[Adapter] Endpoint: {endpoint}")
        
        # 4. Build run interface (how to call the deployed software)
        # Use agent output if available, otherwise fall back to strategy defaults
        run_interface = self._build_run_interface(
            strategy=setting.strategy,
            endpoint=endpoint,
            agent_run_interface=run_interface_from_agent,
        )
        
        print(f"[Adapter] Complete: {adapted_path}")
        
        return AdaptationResult(
            success=True,
            adapted_path=adapted_path,
            run_interface=run_interface,
            files_changed=files_changed if isinstance(files_changed, list) else [],
        )
    
    def _run_fallback_agent(
        self,
        adapted_path: str,
        goal: str,
        setting: DeploymentSetting,
        target_instructions: str,
    ) -> tuple:
        """
        Run the fallback coding agent when primary agent fails.
        
        Args:
            adapted_path: Path to the adapted workspace (copy of original)
            goal: The original goal/objective
            setting: Selected deployment configuration
            target_instructions: Strategy-specific instructions
        
        Returns:
            Tuple of (files_changed, agent_output), or (None, "") if fails
        """
        print(f"[Adapter] Trying fallback agent: {self.fallback_agent_type}")
        
        try:
            fallback_config = CodingAgentFactory.build_config(
                agent_type=self.fallback_agent_type,
                workspace=adapted_path,
            )
            fallback_agent = CodingAgentFactory.create(fallback_config)
            fallback_agent.initialize(adapted_path)
            
            prompt = self._build_adaptation_prompt(
                goal=goal,
                setting=setting,
                target_instructions=target_instructions,
            )
            
            result = fallback_agent.generate_code(prompt)
            
            if not result.success:
                print(f"[Adapter] Fallback agent failed: {result.error}")
                return None, ""
            
            files_changed = result.files_changed
            agent_output = result.output
            
            fallback_agent.cleanup()
            return (files_changed if isinstance(files_changed, list) else [], agent_output)
            
        except Exception as e:
            print(f"[Adapter] Fallback agent also failed: {e}")
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
        
        Note: Uses .replace() instead of .format() because the template and
        target_instructions contain Python code examples with dictionary literals
        like {"status": "success"}, which .format() would incorrectly interpret
        as format placeholders.
        """
        template = self.adaptation_prompt_path.read_text()
        
        # Use .replace() instead of .format() to avoid interpreting
        # dictionary literals in code examples as format placeholders
        result = template
        result = result.replace("{goal}", goal)
        result = result.replace("{strategy}", setting.strategy)
        result = result.replace("{provider}", setting.provider or "N/A")
        result = result.replace("{interface}", setting.interface)
        result = result.replace("{resources}", str(setting.resources))
        result = result.replace("{target_instructions}", target_instructions)
        
        return result
    
    def _build_run_interface(
        self,
        strategy: str,
        endpoint: Optional[str],
        agent_run_interface: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """
        Build the run interface for the deployed software.
        
        Priority:
        1. Use run_interface from agent output (if provided)
        2. Fall back to default from strategy's adapter_instruction.md
        
        Args:
            strategy: Deployment strategy name
            endpoint: Endpoint URL extracted from agent output (if any)
            agent_run_interface: Run interface JSON from agent output (if any)
            
        Returns:
            Interface dict for the Runner
        """
        # Start with agent-provided interface or strategy defaults from registry
        if agent_run_interface:
            interface = agent_run_interface.copy()
        else:
            # Get defaults from strategy's adapter_instruction.md (no hardcoding!)
            interface = self.registry.get_default_run_interface(strategy)
        
        # Ensure we have at least a type (safe fallback)
        if "type" not in interface:
            interface["type"] = "function"
        
        # Add endpoint if available (from agent's deployment output)
        if endpoint:
            interface["endpoint"] = endpoint
            interface["deployment_url"] = endpoint
        
        return interface
    
    def _extract_run_interface_from_output(self, output: str) -> Optional[Dict[str, Any]]:
        """
        Extract run_interface JSON from coding agent output.
        
        The agent is instructed to output the run interface in XML-style tags:
        <run_interface>{"type": "function", "module": "main", ...}</run_interface>
        
        Args:
            output: Full output from the coding agent
            
        Returns:
            Parsed run_interface dict, or None if not found/invalid
        """
        if not output:
            return None
        
        # Extract JSON from <run_interface>...</run_interface> tags
        match = re.search(
            r'<run_interface>\s*(\{[^<]+\})\s*</run_interface>',
            output,
            re.DOTALL
        )
        
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError as e:
                print(f"[Adapter] Warning: Invalid run_interface JSON: {e}")
                return None
        
        return None
    
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
