"""
Deployment Selector Agent

Uses LLM to analyze repositories and select optimal deployment strategies.
Loads strategy information from the strategies/ registry.

Example:
    selector = SelectorAgent()
    setting = selector.select(code_path, goal)
    setting = selector.select(code_path, goal, strategies=["local", "modal"])
"""

from __future__ import annotations

import json
import re
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.llm import LLMBackend
from src.deployment.base import DeploymentSetting
from src.deployment.strategies import StrategyRegistry


class SelectorAgent:
    """
    LLM-based agent for deployment strategy selection.
    
    Analyzes repositories and determines:
    - Deployment strategy (local, docker, modal, bentoml, langgraph)
    - Resource requirements (CPU, memory, GPU)
    
    Uses StrategyRegistry to discover available strategies and their
    selector_instruction.md files.
    
    Attributes:
        model: LLM model for selection
        registry: Strategy registry for discovering available options
    """
    
    _CORRECTION_PROMPT_PATH = Path(__file__).parent / "correction_prompt.md"
    
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize selector agent.
        
        Args:
            model: LLM model for selection
        """
        self.model = model
        self.registry = StrategyRegistry.get()
    
    @cached_property
    def _correction_prompt_template(self) -> str:
        """Load correction prompt template (cached)."""
        if self._CORRECTION_PROMPT_PATH.exists():
            return self._CORRECTION_PROMPT_PATH.read_text()
        return "Extract valid JSON from this text:\n{text}"
    
    def select(
        self,
        code_path: str,
        goal: str,
        strategies: Optional[List[str]] = None,
        resources: Optional[Dict[str, Any]] = None,
    ) -> DeploymentSetting:
        """
        Select deployment configuration for a repository.
        
        Args:
            code_path: Path to the solution directory
            goal: What the code is supposed to do
            strategies: Optional list of strategies to consider (default: all)
            resources: Optional user-specified resources
            
        Returns:
            Complete deployment setting
        """
        # Get available strategies (filtered if specified)
        available = self.registry.list_strategies(allowed=strategies)
        
        if not available:
            raise ValueError(f"No valid strategies found. Requested: {strategies}")
        
        # If only one option, skip LLM
        if len(available) == 1:
            strategy = available[0]
            return self._create_setting(strategy, resources, "Single option selected")
        
        # Query LLM to select best strategy
        result = self._query_llm(code_path, goal, available)
        
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
    
    def _query_llm(
        self,
        code_path: str,
        goal: str,
        strategies: List[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Query LLM to select best strategy.
        
        Args:
            code_path: Repository path
            goal: Project goal
            strategies: Available strategies to choose from
            
        Returns:
            Parsed LLM response or None on failure
        """
        prompt = self._build_prompt(goal, strategies)
        
        try:
            llm = LLMBackend()
            response = llm.llm_completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            
            if response:
                return self._parse_response(response)
            return None
            
        except Exception as e:
            print(f"[Selector] LLM error: {e}")
            return None
    
    def _build_prompt(self, goal: str, strategies: List[str]) -> str:
        """Build selector prompt with strategy descriptions."""
        
        # Gather selector instructions for each strategy
        strategy_descriptions = []
        for name in strategies:
            instruction = self.registry.get_selector_instruction(name)
            strategy_descriptions.append(instruction)
        
        return f"""Analyze this project and select the best deployment strategy.

## Goal
{goal}

## Available Strategies

{chr(10).join(strategy_descriptions)}

## Your Task

Based on the goal, select the most appropriate deployment strategy.
Consider:
- Complexity of the solution
- Resource requirements (GPU, memory)
- Whether it's an agent/LangGraph application
- Production vs development needs

## Output Format

Return ONLY a JSON object:
```json
{{
    "strategy": "<strategy_name>",
    "resources": {{"gpu": "...", "memory": "..."}},
    "reasoning": "<brief explanation>"
}}
```

The strategy MUST be one of: {', '.join(strategies)}
"""
    
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
        code_path: str,
    ) -> DeploymentSetting:
        """Create setting for explicit strategy (for factory use)."""
        return self._create_setting(strategy, None, f"User specified {strategy}")
    
    def explain(self, code_path: str, goal: str) -> str:
        """Get human-readable explanation of selection."""
        setting = self.select(code_path, goal)
        
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
