"""
RepoMemory Gate
===============

Provides MCP tools for accessing repo memory (.kapso/repo_memory.json).

Tools:
- get_repo_memory_section: Get a specific section by ID
- list_repo_memory_sections: List available section IDs (TOC)
- get_repo_memory_summary: Get summary + TOC

This replaces the CLI-based access pattern, solving path issues when
running from different directories.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from mcp.types import Tool, TextContent
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    Tool = None
    TextContent = None

from kapso.gated_mcp.gates.base import ToolGate, GateConfig

logger = logging.getLogger(__name__)


class RepoMemoryGate(ToolGate):
    """
    Gate for repo memory access tools.
    
    Provides access to .kapso/repo_memory.json for coding agents
    to understand repository architecture, gotchas, and key patterns.
    
    Tools:
    - get_repo_memory_section: Get detailed section content
    - list_repo_memory_sections: List available sections (TOC)
    - get_repo_memory_summary: Get summary + TOC overview
    """
    
    name = "repo_memory"
    description = "Tools for accessing repository memory"
    
    def __init__(self, config: Optional[GateConfig] = None):
        """Initialize repo memory gate."""
        super().__init__(config)
        self._manager = None  # Lazy loaded
    
    def _get_repo_root(self) -> str:
        """
        Get repo root path.
        
        Resolution order:
        1. Gate params (repo_root)
        2. Environment variable (REPO_MEMORY_ROOT)
        3. Current working directory
        """
        # From params
        repo_root = self.get_param("repo_root")
        if repo_root:
            return str(Path(repo_root).resolve())
        
        # From environment
        env_root = os.environ.get("REPO_MEMORY_ROOT")
        if env_root:
            return str(Path(env_root).resolve())
        
        # Default to CWD
        return str(Path.cwd())
    
    def _get_manager(self):
        """Lazy load the RepoMemoryManager."""
        if self._manager is None:
            from kapso.execution.memories.repo_memory import RepoMemoryManager
            self._manager = RepoMemoryManager
        return self._manager
    
    def _load_doc(self, repo_root: str) -> Optional[Dict[str, Any]]:
        """Load repo memory document from worktree."""
        manager = self._get_manager()
        return manager.load_from_worktree(repo_root)
    
    def get_tools(self) -> List["Tool"]:
        """Return repo memory tools."""
        if not HAS_MCP:
            return []
        
        return [
            Tool(
                name="get_repo_memory_section",
                description=(
                    "Get a specific section from repo memory by ID. "
                    "Returns detailed content including claims with evidence, entrypoints, or key files. "
                    "Use this to understand specific aspects of the repository architecture."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "section_id": {
                            "type": "string",
                            "description": (
                                "Section ID to retrieve. Common sections: "
                                "core.architecture, core.entrypoints, core.where_to_edit, "
                                "core.invariants, core.testing, core.gotchas, core.dependencies"
                            ),
                        },
                        "repo_root": {
                            "type": "string",
                            "description": "Optional repo root path. Defaults to current working directory.",
                        },
                    },
                    "required": ["section_id"],
                },
            ),
            Tool(
                name="list_repo_memory_sections",
                description=(
                    "List all available section IDs in repo memory (Table of Contents). "
                    "Returns section IDs with titles. Use this to discover what sections are available."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "repo_root": {
                            "type": "string",
                            "description": "Optional repo root path. Defaults to current working directory.",
                        },
                    },
                },
            ),
            Tool(
                name="get_repo_memory_summary",
                description=(
                    "Get repo memory summary and table of contents. "
                    "Returns a high-level overview of the repository including summary and all section IDs. "
                    "Use this first to understand the repository structure before diving into specific sections."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "repo_root": {
                            "type": "string",
                            "description": "Optional repo root path. Defaults to current working directory.",
                        },
                    },
                },
            ),
        ]
    
    async def handle_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Optional[List["TextContent"]]:
        """Handle repo memory tool calls."""
        if not HAS_MCP:
            return None
        
        if tool_name == "get_repo_memory_section":
            return await self._handle_get_section(arguments)
        elif tool_name == "list_repo_memory_sections":
            return await self._handle_list_sections(arguments)
        elif tool_name == "get_repo_memory_summary":
            return await self._handle_get_summary(arguments)
        
        return None
    
    async def _handle_get_section(self, arguments: Dict[str, Any]) -> List["TextContent"]:
        """Handle get_repo_memory_section tool call."""
        section_id = arguments.get("section_id", "")
        repo_root = arguments.get("repo_root") or self._get_repo_root()
        
        if not section_id:
            return [TextContent(type="text", text="Error: section_id is required")]
        
        try:
            doc = await self._run_sync(self._load_doc, repo_root)
            if not doc:
                manager = self._get_manager()
                memory_path = os.path.join(repo_root, manager.MEMORY_REL_PATH)
                return [TextContent(
                    type="text",
                    text=f"RepoMemory not found at {memory_path}. Run kapso.evolve() first to generate it."
                )]
            
            manager = self._get_manager()
            # No truncation - return full section content
            text = manager.get_section(doc, section_id, max_chars=999999999)
            return [TextContent(type="text", text=text)]
        except Exception as e:
            logger.error(f"get_repo_memory_section failed: {e}")
            return [TextContent(type="text", text=f"Error getting section: {e}")]
    
    async def _handle_list_sections(self, arguments: Dict[str, Any]) -> List["TextContent"]:
        """Handle list_repo_memory_sections tool call."""
        repo_root = arguments.get("repo_root") or self._get_repo_root()
        
        try:
            doc = await self._run_sync(self._load_doc, repo_root)
            if not doc:
                manager = self._get_manager()
                memory_path = os.path.join(repo_root, manager.MEMORY_REL_PATH)
                return [TextContent(
                    type="text",
                    text=f"RepoMemory not found at {memory_path}. Run kapso.evolve() first to generate it."
                )]
            
            manager = self._get_manager()
            toc = manager.list_sections(doc)
            
            lines = ["# Available Repo Memory Sections\n"]
            for item in toc:
                sid = (item or {}).get("id", "")
                title = (item or {}).get("title", "")
                one_liner = (item or {}).get("one_liner", "")
                if sid:
                    lines.append(f"- **{sid}**: {title}")
                    if one_liner:
                        lines.append(f"  - {one_liner}")
            
            return [TextContent(type="text", text="\n".join(lines))]
        except Exception as e:
            logger.error(f"list_repo_memory_sections failed: {e}")
            return [TextContent(type="text", text=f"Error listing sections: {e}")]
    
    async def _handle_get_summary(self, arguments: Dict[str, Any]) -> List["TextContent"]:
        """Handle get_repo_memory_summary tool call."""
        repo_root = arguments.get("repo_root") or self._get_repo_root()
        
        try:
            doc = await self._run_sync(self._load_doc, repo_root)
            if not doc:
                manager = self._get_manager()
                memory_path = os.path.join(repo_root, manager.MEMORY_REL_PATH)
                return [TextContent(
                    type="text",
                    text=f"RepoMemory not found at {memory_path}. Run kapso.evolve() first to generate it."
                )]
            
            manager = self._get_manager()
            # No truncation - return full summary
            text = manager.render_summary_and_toc(doc, max_chars=999999999)
            return [TextContent(type="text", text=text)]
        except Exception as e:
            logger.error(f"get_repo_memory_summary failed: {e}")
            return [TextContent(type="text", text=f"Error getting summary: {e}")]
