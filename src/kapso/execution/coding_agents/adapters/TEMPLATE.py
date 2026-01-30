# ============================================================================
# TEMPLATE: New Coding Agent Adapter
# ============================================================================
#
# This is a template for creating new coding agent adapters.
#
# To add a new agent:
#
# 1. Copy this file:
#    cp TEMPLATE.py your_agent_agent.py
#
# 2. Rename the class:
#    class YourAgentCodingAgent(CodingAgentInterface):
#
# 3. Implement the abstract methods:
#    - initialize(workspace)
#    - generate_code(prompt, debug_mode)
#    - cleanup()
#
# 4. Add entry to ../agents.yaml:
#    your_agent:
#      description: "Your agent description"
#      adapter_class: "YourAgentCodingAgent"
#      adapter_module: "kapso.execution.coding_agents.adapters.your_agent_agent"
#      supports_native_git: false
#      default_model: "your-model"
#      default_debug_model: "your-debug-model"
#      env_vars:
#        - "YOUR_API_KEY"
#      install_command: "pip install your-package"
#      agent_specific:
#        option1: "default_value"
#
# 5. Your agent will be auto-registered on startup!
#
# ============================================================================

import os
import re
from typing import Dict, List, Optional

from kapso.execution.coding_agents.base import (
    CodingAgentInterface,
    CodingAgentConfig,
    CodingResult,
)


class TemplateAgentCodingAgent(CodingAgentInterface):
    """
    Template coding agent adapter.
    
    Replace this docstring with a description of your agent.
    
    Configuration options (in agent_specific):
        - option1: Description of option1
        - option2: Description of option2
    
    Environment variables:
        - YOUR_API_KEY: API key for your service
    """
    
    def __init__(self, config: CodingAgentConfig):
        """
        Initialize the agent with configuration.
        
        Args:
            config: CodingAgentConfig from factory
        """
        super().__init__(config)
        
        # Workspace will be set in initialize()
        self.workspace: Optional[str] = None
        
        # Client/connection (initialized in initialize())
        self.client = None
        
        # Extract agent-specific config with defaults
        self._option1 = config.agent_specific.get("option1", "default_value")
        self._option2 = config.agent_specific.get("option2", 100)
        
        # Track if initialized
        self._initialized = False
    
    def initialize(self, workspace: str) -> None:
        """
        Initialize the agent for a specific workspace.
        
        Called once when ExperimentSession is created.
        Set up any client connections, load models, etc.
        
        IMPORTANT: Do NOT perform git operations here.
        Git is managed by ExperimentSession.
        
        Args:
            workspace: Absolute path to the working directory
        """
        self.workspace = os.path.abspath(workspace)
        
        # TODO: Initialize your client/connection
        # Example:
        #
        # api_key = os.getenv("YOUR_API_KEY")
        # if not api_key:
        #     raise ValueError("YOUR_API_KEY environment variable not set")
        #
        # self.client = YourClient(api_key=api_key)
        
        self._initialized = True
    
    def generate_code(self, prompt: str, debug_mode: bool = False) -> CodingResult:
        """
        Generate or modify code based on the prompt.
        
        This is the main method. The agent should:
        1. Process the prompt
        2. Call the underlying API/service
        3. Parse the response
        4. Write files to the workspace
        5. Return a CodingResult
        
        Args:
            prompt: Implementation or debugging instructions
            debug_mode: If True, use debug model
            
        Returns:
            CodingResult with:
            - success: Whether generation succeeded
            - output: Agent's response text
            - files_changed: List of modified file paths
            - error: Error message if failed
            - cost: API cost in dollars
            - commit_message: Optional suggested commit message
            
        IMPORTANT: Do NOT commit changes. ExperimentSession handles commits.
        """
        if not self._initialized:
            return CodingResult(
                success=False,
                output="",
                error="Agent not initialized. Call initialize() first.",
            )
        
        # Select model based on mode
        model = self.config.debug_model if debug_mode else self.config.model
        
        try:
            # TODO: Replace this with your actual implementation
            #
            # Example flow:
            # 1. Call your API
            #    response = self.client.generate(
            #        prompt=prompt,
            #        model=model,
            #        temperature=self._option1,
            #    )
            #
            # 2. Parse response for code blocks
            #    code_blocks = self._parse_code_blocks(response.text)
            #
            # 3. Write files
            #    files_changed = []
            #    for filename, content in code_blocks:
            #        file_path = os.path.join(self.workspace, filename)
            #        os.makedirs(os.path.dirname(file_path), exist_ok=True)
            #        with open(file_path, 'w') as f:
            #            f.write(content)
            #        files_changed.append(file_path)
            #
            # 4. Calculate cost
            #    cost = response.usage.total_tokens * COST_PER_TOKEN
            #    self._cumulative_cost += cost
            #
            # 5. Return result
            #    return CodingResult(
            #        success=True,
            #        output=response.text,
            #        files_changed=files_changed,
            #        cost=cost,
            #        metadata={"model": model},
            #    )
            
            # Placeholder implementation
            return CodingResult(
                success=False,
                output="",
                error="Template agent not implemented. Replace with your implementation.",
            )
            
        except Exception as e:
            return CodingResult(
                success=False,
                output="",
                error=f"Generation failed: {str(e)}",
            )
    
    def cleanup(self) -> None:
        """
        Clean up resources.
        
        Called when ExperimentSession is closed.
        Close connections, remove temp files, etc.
        """
        self.client = None
        self.workspace = None
        self._initialized = False
    
    def supports_native_git(self) -> bool:
        """
        Return True if agent handles its own git commits.
        
        Most agents return False - ExperimentSession handles commits.
        Only Aider returns True because it auto-commits with meaningful messages.
        
        If you return True:
        - Your agent MUST commit changes in generate_code()
        - ExperimentSession will NOT commit after your calls
        
        If you return False (recommended):
        - Just write files in generate_code()
        - ExperimentSession will commit with LLM-generated message
        
        Returns:
            True if agent auto-commits, False otherwise
        """
        return False
    
    def get_capabilities(self) -> Dict[str, bool]:
        """
        Return agent capabilities for feature detection.
        
        Used by the orchestrator to understand what the agent can do.
        
        Returns:
            Dictionary of capability flags
        """
        return {
            # Does agent handle its own git commits?
            "native_git": self.supports_native_git(),
            
            # Does agent run in a sandboxed environment?
            "sandbox": False,
            
            # Does agent support multi-step planning?
            "planning_mode": False,
            
            # Does agent track API costs?
            "cost_tracking": True,
            
            # Does agent support streaming output?
            "streaming": False,
        }
    
    # =========================================================================
    # Helper Methods (add your own as needed)
    # =========================================================================
    
    def _parse_code_blocks(self, text: str) -> List[tuple]:
        """
        Parse code blocks from LLM response.
        
        Example helper for extracting code from markdown-style responses.
        
        Args:
            text: LLM response text
            
        Returns:
            List of (filename, content) tuples
        """
        # Pattern: ```filename\ncontent\n```
        pattern = r'```(\S+)\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        code_blocks = []
        for filename, content in matches:
            # Skip language-only markers like ```python
            if '.' in filename or '/' in filename:
                code_blocks.append((filename, content.strip()))
        
        return code_blocks
    
    def _write_file(self, filename: str, content: str) -> str:
        """
        Write content to a file in the workspace.
        
        Args:
            filename: Relative path within workspace
            content: File content
            
        Returns:
            Absolute path to the written file
        """
        file_path = os.path.join(self.workspace, filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        return file_path

