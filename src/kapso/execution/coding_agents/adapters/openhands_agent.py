# OpenHands Coding Agent Adapter
#
# Uses OpenHands LLM interface for code generation.
# Falls back to direct LLM call if full SDK not available.
#
# Key features:
# - Model-agnostic (can use any LLM via litellm)
# - Simple code generation via LLM
# - File parsing and writing
#
# Requires:
# - openhands package: pip install openhands-ai

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from kapso.execution.coding_agents.base import (
    CodingAgentInterface, 
    CodingAgentConfig, 
    CodingResult
)


class OpenHandsCodingAgent(CodingAgentInterface):
    """
    OpenHands-based coding agent.
    
    Uses OpenHands LLM interface for code generation with
    file parsing and writing capabilities.
    
    Features:
    - Model-agnostic (works with any LLM via litellm)
    - Code extraction and file writing
    - Cost tracking
    
    Configuration (agent_specific):
    - max_tokens: 4096 (default)
    - temperature: 0.1 (default)
    
    Environment:
    - Requires API key for chosen model (OPENAI_API_KEY, etc.)
    """
    
    def __init__(self, config: CodingAgentConfig):
        """Initialize OpenHands coding agent."""
        super().__init__(config)
        self.workspace: Optional[str] = None
        self.llm = None
        
        # Get OpenHands-specific settings
        self._max_tokens = config.agent_specific.get("max_tokens", 4096)
        self._temperature = config.agent_specific.get("temperature", 0.1)
        
        # Initialize the LLM
        self._init_llm()
    
    def _init_llm(self):
        """Initialize the OpenHands LLM interface."""
        try:
            from openhands.llm import LLM
            
            # Create LLM instance with model configuration
            self.llm = LLM(
                model=self.config.model,
                temperature=self._temperature,
                max_output_tokens=self._max_tokens,
            )
        except ImportError:
            # Fall back to litellm directly if openhands.llm not available
            try:
                import litellm
                self._use_litellm_directly = True
            except ImportError:
                raise ImportError(
                    "Neither openhands.llm nor litellm available. "
                    "Run: pip install openhands-ai or pip install litellm"
                )
        except Exception as e:
            # If LLM init fails, we'll use litellm directly
            self._use_litellm_directly = True
    
    def initialize(self, workspace: str) -> None:
        """
        Initialize OpenHands agent for the workspace.
        
        Args:
            workspace: Path to the working directory
        """
        self.workspace = workspace
    
    def generate_code(self, prompt: str, debug_mode: bool = False) -> CodingResult:
        """
        Generate code using OpenHands LLM.
        
        Args:
            prompt: The implementation or debugging instructions
            debug_mode: If True, use debug model
            
        Returns:
            CodingResult with OpenHands' response
        """
        if self.workspace is None:
            return CodingResult(
                success=False,
                output="",
                error="Agent not initialized. Call initialize() first."
            )
        
        model = self.config.debug_model if debug_mode else self.config.model
        
        try:
            # Build the system prompt for code generation
            system_prompt = self._build_system_prompt()
            full_prompt = system_prompt + "\n\n" + prompt
            
            # Generate response
            if hasattr(self, '_use_litellm_directly') and self._use_litellm_directly:
                output_text, cost = self._generate_with_litellm(model, full_prompt)
            else:
                output_text, cost = self._generate_with_openhands(model, full_prompt)
            
            # Parse code blocks and write to files
            files_changed = self._parse_and_write_files(output_text)
            
            self._cumulative_cost += cost
            
            return CodingResult(
                success=True,
                output=output_text,
                files_changed=files_changed,
                cost=cost,
                metadata={
                    "model": model,
                    "temperature": self._temperature,
                }
            )
        except Exception as e:
            return CodingResult(
                success=False,
                output="",
                error=str(e)
            )
    
    def _generate_with_openhands(self, model: str, prompt: str) -> tuple:
        """Generate using OpenHands LLM."""
        from openhands.llm import LLM
        
        llm = LLM(
            model=model,
            temperature=self._temperature,
            max_output_tokens=self._max_tokens,
        )
        
        response = llm.completion(
            messages=[{"role": "user", "content": prompt}]
        )
        
        output_text = response.choices[0].message.content
        
        # Estimate cost
        cost = self._estimate_cost(len(prompt), len(output_text))
        
        return output_text, cost
    
    def _generate_with_litellm(self, model: str, prompt: str) -> tuple:
        """Fallback: Generate using litellm directly."""
        import litellm
        
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        
        output_text = response.choices[0].message.content
        
        # Get cost from response if available
        cost = 0.0
        if hasattr(response, 'usage'):
            # Rough estimate based on token counts
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = (input_tokens * 0.001 + output_tokens * 0.002) / 1000
        
        return output_text, cost
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for code generation."""
        return """You are an expert programmer implementing code changes.

When generating code, use this format for each file:

```filename.py
<complete file content>
```

Rules:
- Output complete file contents, not diffs or patches
- Include all imports and dependencies
- Write clean, functional code
- Create/modify files in the current working directory
- Do not include any conversational text outside code blocks
- If modifying an existing file, output the complete new content
"""
    
    def _parse_and_write_files(self, output: str) -> List[str]:
        """
        Parse code blocks from output and write to files.
        
        Looks for patterns like:
        ```filename.py
        code content
        ```
        
        Args:
            output: LLM's response text
            
        Returns:
            List of files that were written
        """
        files_changed = []
        
        # Pattern to match ```filename\n...\n```
        pattern = r'```(?:(?P<lang>\w+):)?(?P<filename>[\w./\\-]+\.\w+)\n(?P<content>.*?)```'
        
        matches = re.finditer(pattern, output, re.DOTALL)
        
        for match in matches:
            filename = match.group('filename')
            content = match.group('content')
            
            # Write the file
            filepath = Path(self.workspace) / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                f.write(content)
            
            files_changed.append(str(filepath))
        
        # Also try simpler pattern for unmarked code blocks
        if not files_changed:
            simple_pattern = r'```(\w+)?\n(.*?)```'
            matches = re.finditer(simple_pattern, output, re.DOTALL)
            
            for i, match in enumerate(matches):
                lang = match.group(1) or 'py'
                content = match.group(2)
                
                # Determine filename from content or use default
                filename = self._infer_filename(content, lang, i)
                
                filepath = Path(self.workspace) / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)
                
                with open(filepath, 'w') as f:
                    f.write(content)
                
                files_changed.append(str(filepath))
        
        return files_changed
    
    def _infer_filename(self, content: str, lang: str, index: int) -> str:
        """Infer filename from content or language."""
        ext_map = {
            'python': 'py', 'py': 'py',
            'javascript': 'js', 'js': 'js',
            'typescript': 'ts', 'ts': 'ts',
            'cpp': 'cpp', 'c++': 'cpp', 'c': 'c',
            'java': 'java',
            'rust': 'rs',
            'go': 'go',
        }
        
        ext = ext_map.get(lang.lower(), lang) if lang else 'py'
        
        # Check for main function to name as main.ext
        if 'def main' in content or 'int main' in content or 'func main' in content:
            return f'main.{ext}'
        
        return f'generated_{index}.{ext}' if index > 0 else f'main.{ext}'
    
    def _estimate_cost(self, input_len: int, output_len: int) -> float:
        """
        Estimate cost based on character counts.
        
        Rough estimate: 4 chars per token, GPT-4 pricing
        """
        input_tokens = input_len / 4
        output_tokens = output_len / 4
        
        # GPT-4o-mini pricing: $0.15/1M input, $0.6/1M output
        cost = (input_tokens * 0.15 + output_tokens * 0.6) / 1_000_000
        return cost
    
    def cleanup(self) -> None:
        """Clean up OpenHands resources."""
        self.workspace = None
        self.llm = None
    
    def supports_native_git(self) -> bool:
        """OpenHands doesn't handle git commits natively."""
        return False
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Return OpenHands' capabilities."""
        return {
            "native_git": False,
            "sandbox": False,  # Simplified version without Docker
            "planning_mode": True,
            "cost_tracking": True,
            "streaming": False,
        }
