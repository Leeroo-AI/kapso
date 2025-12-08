"""
Deployment Selector Agent

Uses coding agent to analyze repositories and select optimal deployment strategies.
Loads strategy information from the strategies/ registry.

Example:
    selector = SelectorAgent()
    setting = selector.select(code_path, goal)
    setting = selector.select(code_path, goal, strategies=["local", "modal"])
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.deployment.base import DeploymentSetting
from src.deployment.strategies import StrategyRegistry
from src.execution.coding_agents.factory import CodingAgentFactory
from src.execution.solution import SolutionResult

# Path to the selection prompt template
SELECTION_PROMPT_PATH = Path(__file__).parent / "selection_prompt.md"


class SelectorAgent:
    """
    Coding agent for deployment strategy selection.
    
    Analyzes repositories and determines:
    - Deployment strategy (local, docker, modal, bentoml, langgraph)
    - Resource requirements (CPU, memory, GPU)
    
    Uses StrategyRegistry to discover available strategies and their
    selector_instruction.md files.
    """
    
    def __init__(
        self,
        coding_agent_type: str = "claude_code",
        model: str = "claude-opus-4-5",
    ):
        """
        Initialize selector agent.
        
        Args:
            coding_agent_type: Coding agent to use (claude_code, aider, gemini)
            model: LLM model for the coding agent
        """
        self.coding_agent_type = coding_agent_type
        self.model = model
        self.registry = StrategyRegistry.get()
    
    def select(
        self,
        solution: SolutionResult,
        strategies: Optional[List[str]] = None,
        resources: Optional[Dict[str, Any]] = None,
    ) -> DeploymentSetting:
        """
        Select deployment configuration for a solution.
        
        Args:
            solution: The SolutionResult from Expert.build()
            strategies: Optional list of strategies to consider (default: all)
            resources: Optional user-specified resources
            
        Returns:
            Complete deployment setting
        """
        # Extract from solution
        code_path = solution.code_path
        goal = solution.goal
        
        # Get available strategies (filtered if specified)
        available = self.registry.list_strategies(allowed=strategies)
        
        if not available:
            raise ValueError(f"No valid strategies found. Requested: {strategies}")
        
        # If only one option, skip agent
        if len(available) == 1:
            strategy = available[0]
            return self._create_setting(strategy, resources, "Single option selected")
        
        # Use coding agent to select best strategy
        result = self._query_agent(code_path, goal, available)
        
        if result:
            strategy = result.get("strategy", "local")
            # Validate strategy is in allowed list
            if strategy not in available:
                strategy = available[0]
            
            # Use LLM-suggested resources if not provided
            if resources is None:
                resources = result.get("resources")
            
            reasoning = result.get("reasoning", f"LLM selected {strategy}")
        else:
            # Fallback to local
            strategy = "local" if "local" in available else available[0]
            reasoning = "Fallback to default"
        
        return self._create_setting(strategy, resources, reasoning)
    
    def _create_setting(
        self,
        strategy: str,
        resources: Optional[Dict[str, Any]],
        reasoning: str,
    ) -> DeploymentSetting:
        """Create DeploymentSetting from strategy name."""
        # Parse selector instruction to extract metadata
        selector_md = self.registry.get_selector_instruction(strategy)
        
        # Extract interface and provider from markdown
        interface = self._extract_field(selector_md, "Interface", "function")
        provider = self._extract_field(selector_md, "Provider", None)
        if provider == "None":
            provider = None
        
        # Get default resources if needed and not provided
        if resources is None:
            default_resources = self._extract_default_resources(selector_md)
            if default_resources:
                resources = default_resources
        
        return DeploymentSetting(
            strategy=strategy,
            provider=provider,
            resources=resources or {},
            interface=interface,
            reasoning=reasoning,
        )
    
    def _extract_field(self, md_content: str, field_name: str, default: str) -> str:
        """Extract a field value from markdown content."""
        # Look for patterns like "## Interface\nfunction" or "- Interface: function"
        patterns = [
            rf'##\s*{field_name}\s*\n+([^\n#(]+)',  # Stop at ( for descriptions
            rf'{field_name}:\s*([^\n(]+)',           # Stop at ( for descriptions
            rf'\*\*{field_name}\*\*:\s*([^\n(]+)',   # Stop at ( for descriptions
        ]
        
        for pattern in patterns:
            match = re.search(pattern, md_content, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                # Clean up common suffixes
                value = value.split('(')[0].strip()
                return value
        
        return default
    
    def _extract_default_resources(self, md_content: str) -> Optional[Dict[str, Any]]:
        """Extract default resources from selector instruction."""
        # Look for "Default: gpu=T4, memory=16Gi" pattern
        match = re.search(r'Default:\s*([^\n]+)', md_content)
        if match:
            resources = {}
            parts = match.group(1).split(',')
            for part in parts:
                if '=' in part:
                    key, value = part.strip().split('=', 1)
                    resources[key.strip()] = value.strip()
            if resources:
                return resources
        return None
    
    def _query_agent(
        self,
        code_path: str,
        goal: str,
        strategies: List[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Use coding agent to analyze code and select best strategy.
        
        Args:
            code_path: Repository path
            goal: Project goal
            strategies: Available strategies to choose from
            
        Returns:
            Parsed response or None on failure
        """
        prompt = self._build_prompt(goal, strategies)
        
        try:
            config = CodingAgentFactory.build_config(
                agent_type=self.coding_agent_type,
                model=self.model,
                workspace=code_path,
            )
            agent = CodingAgentFactory.create(config)
            agent.initialize(code_path)
            
            print(f"[Selector] Running {self.coding_agent_type} agent...")
            result = agent.generate_code(prompt)
            
            agent.cleanup()
            
            if result.success and result.output:
                return self._parse_response(result.output)
            return None
            
        except Exception as e:
            print(f"[Selector] Agent error: {e}")
            return None
    
    def _build_prompt(self, goal: str, strategies: List[str]) -> str:
        """Build selector prompt with strategy descriptions."""
        
        # Gather selector instructions for each strategy
        strategy_descriptions = []
        for name in strategies:
            instruction = self.registry.get_selector_instruction(name)
            strategy_descriptions.append(instruction)
        
        template = SELECTION_PROMPT_PATH.read_text()
        
        return template.format(
            goal=goal,
            strategy_descriptions="\n".join(strategy_descriptions),
            allowed_strategies=", ".join(strategies),
        )
    
    def _parse_response(self, output: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM output."""
        if not output:
            return None
        
        # Try code block extraction
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', output)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
        
        # Try direct JSON extraction
        try:
            start = output.find('{')
            end = output.rfind('}')
            if start != -1 and end != -1:
                return json.loads(output[start:end+1])
        except json.JSONDecodeError:
            pass
        
        return None
    
    def _create_setting_for_strategy(
        self,
        strategy: str,
    ) -> DeploymentSetting:
        """Create setting for explicit strategy (for factory use)."""
        return self._create_setting(strategy, None, f"User specified {strategy}")
    
    def explain(self, solution: SolutionResult) -> str:
        """Get human-readable explanation of selection."""
        setting = self.select(solution)
        
        return "\n".join([
            "Deployment Selection Analysis",
            "=" * 40,
            "",
            f"Strategy:  {setting.strategy}",
            f"Interface: {setting.interface}",
            f"Provider:  {setting.provider or 'N/A'}",
            f"Resources: {setting.resources}",
            "",
            f"Reasoning: {setting.reasoning}",
        ])
