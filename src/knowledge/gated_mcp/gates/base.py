"""
Base class for tool gates.

A gate is a group of related tools with shared backend and configuration.
Each gate defines its tools and handles tool calls.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# MCP types (imported conditionally)
try:
    from mcp.types import Tool, TextContent
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    Tool = None
    TextContent = None

logger = logging.getLogger(__name__)


@dataclass
class GateConfig:
    """Configuration for a single gate instance."""
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


class ToolGate(ABC):
    """
    Abstract base class for tool gates.
    
    A gate groups related tools and manages their shared backend.
    Subclasses must implement:
    - name: Gate identifier (e.g., "kg", "idea")
    - description: Human-readable description
    - get_tools(): Return list of MCP Tool definitions
    - handle_call(): Handle tool invocations
    """
    
    # Subclasses must define these
    name: str = ""
    description: str = ""
    
    def __init__(self, config: Optional[GateConfig] = None):
        """
        Initialize the gate with optional configuration.
        
        Args:
            config: Gate configuration with params. If None, uses defaults.
        """
        self._config = config or GateConfig()
    
    def get_param(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration parameter.
        
        Args:
            key: Parameter name
            default: Default value if not set
            
        Returns:
            Parameter value or default
        """
        return self._config.params.get(key, default)
    
    @abstractmethod
    def get_tools(self) -> List["Tool"]:
        """
        Return the list of MCP tools provided by this gate.
        
        Returns:
            List of Tool definitions
        """
        pass
    
    @abstractmethod
    async def handle_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Optional[List["TextContent"]]:
        """
        Handle a tool call.
        
        Args:
            tool_name: Name of the tool being called
            arguments: Tool arguments from the MCP client
            
        Returns:
            List of TextContent responses, or None if tool not handled
        """
        pass
    
    def get_tool_names(self) -> List[str]:
        """
        Get list of tool names provided by this gate.
        
        Returns:
            List of tool name strings
        """
        return [tool.name for tool in self.get_tools()]
    
    async def _run_sync(self, func: Callable, *args, **kwargs) -> Any:
        """
        Run a synchronous function in an executor.
        
        MCP handlers are async, but search backends are sync.
        This helper bridges the gap without blocking the event loop.
        
        Args:
            func: Synchronous function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
        """
        loop = asyncio.get_event_loop()
        # Use default executor (thread pool)
        return await loop.run_in_executor(
            None,
            lambda: func(*args, **kwargs)
        )
