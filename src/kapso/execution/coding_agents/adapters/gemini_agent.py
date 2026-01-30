# Gemini Coding Agent Adapter
#
# Uses Google's Gemini API/SDK for code generation.
# Supports both Python SDK and CLI modes.
#
# Key features:
# - ReAct loop with built-in tools
# - Model Context Protocol (MCP) support
# - Code execution capabilities
#
# Requires: GOOGLE_API_KEY in environment

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from kapso.execution.coding_agents.base import (
    CodingAgentInterface, 
    CodingAgentConfig, 
    CodingResult
)


class GeminiCodingAgent(CodingAgentInterface):
    """
    Gemini-based coding agent.
    
    Uses Google's Gemini models for code generation via the
    google-genai Python SDK.
    
    Features:
    - Powerful reasoning with Gemini models
    - Code execution support
    - Multi-modal capabilities
    
    Configuration (agent_specific):
    - temperature: 0.1 (default)
    - max_output_tokens: 8192 (default)
    
    Environment:
    - GOOGLE_API_KEY: Required for authentication
    """
    
    def __init__(self, config: CodingAgentConfig):
        """Initialize Gemini coding agent."""
        super().__init__(config)
        self.workspace: Optional[str] = None
        self.client = None
        
        # Get Gemini-specific settings
        self._temperature = config.agent_specific.get("temperature", 0.1)
        self._max_tokens = config.agent_specific.get("max_output_tokens", 8192)
        
        # Initialize the Gemini client
        self._init_client()
    
    def _init_client(self):
        """Initialize the Gemini client."""
        try:
            from google import genai
            
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not set in environment")
            
            self.client = genai.Client(api_key=api_key)
        except ImportError:
            raise ImportError(
                "google-genai package not installed. "
                "Run: pip install google-genai"
            )
    
    def initialize(self, workspace: str) -> None:
        """
        Initialize Gemini agent for the workspace.
        
        Args:
            workspace: Path to the working directory
        """
        self.workspace = workspace
    
    def generate_code(self, prompt: str, debug_mode: bool = False) -> CodingResult:
        """
        Generate code using Gemini.
        
        Args:
            prompt: The implementation or debugging instructions
            debug_mode: If True, use debug model
            
        Returns:
            CodingResult with Gemini's response
        """
        if self.client is None:
            return CodingResult(
                success=False,
                output="",
                error="Gemini client not initialized"
            )
        
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
            
            # Call Gemini API
            response = self.client.models.generate_content(
                model=model,
                contents=[
                    {"role": "user", "parts": [{"text": system_prompt + "\n\n" + prompt}]}
                ],
                config={
                    "temperature": self._temperature,
                    "max_output_tokens": self._max_tokens,
                }
            )
            
            output_text = response.text
            
            # Parse code blocks and write to files
            files_changed = self._parse_and_write_files(output_text)
            
            # Calculate cost (approximate)
            cost = self._estimate_cost(response)
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
        Parse code blocks from Gemini output and write to files.
        
        Looks for patterns like:
        ```filename.py
        code content
        ```
        
        Args:
            output: Gemini's response text
            
        Returns:
            List of files that were written
        """
        files_changed = []
        
        # Pattern to match ```filename\n...\n```
        # Matches: ```main.py or ```python:main.py or ```filename.ext
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
            # Look for common file patterns in the output
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
    
    def _estimate_cost(self, response) -> float:
        """
        Estimate cost from Gemini API response.
        
        Gemini pricing is approximate here - actual pricing varies by model.
        """
        # Rough estimate: $0.00025 per 1K input tokens, $0.0005 per 1K output tokens
        # This is a placeholder - actual costs should be calculated based on usage metadata
        try:
            if hasattr(response, 'usage_metadata'):
                input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
                output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
                return (input_tokens * 0.00025 + output_tokens * 0.0005) / 1000
        except:
            pass
        return 0.01  # Default estimate
    
    def cleanup(self) -> None:
        """Clean up Gemini resources."""
        self.workspace = None
        # Client can be reused, no cleanup needed
    
    def supports_native_git(self) -> bool:
        """Gemini doesn't handle git commits."""
        return False
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Return Gemini's capabilities."""
        return {
            "native_git": False,
            "sandbox": False,
            "planning_mode": True,  # ReAct loop
            "cost_tracking": True,
            "streaming": False,
        }

