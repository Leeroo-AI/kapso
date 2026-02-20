# Agentic KG Search
#
# Uses a Claude Code agent (via Bedrock) with read-only KG MCP tools
# to iteratively search, read pages, and synthesize structured responses.
#
# Architecture:
# - Agent is initialized once with KG MCP server (search_knowledge,
#   get_wiki_page, get_page_structure) and Read tool only.
# - Each of the 7 public methods builds a task-specific prompt and
#   calls agent.generate_code() to get a synthesized answer.
# - Follows the same CodingAgentFactory pattern as KnowledgeMerger.
# - All agent/MCP defaults live in knowledge_search.yaml under
#   searches.agentic_kg_search; runtime agent_config overrides them.
#
# Usage:
#     from kapso.knowledge_base.search.agentic_kg_search import AgenticKGSearch
#
#     search = AgenticKGSearch()
#     answer = search.search_knowledge("What is LoRA?")

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from kapso.execution.coding_agents.factory import CodingAgentFactory

logger = logging.getLogger(__name__)

# Path to the shared search configuration YAML
_SEARCH_YAML_PATH = Path(__file__).parent / "knowledge_search.yaml"


def _load_yaml_config() -> Dict[str, Any]:
    """Load the agentic_kg_search section from knowledge_search.yaml."""
    try:
        with open(_SEARCH_YAML_PATH, "r") as f:
            data = yaml.safe_load(f) or {}
        return data.get("searches", {}).get("agentic_kg_search", {})
    except Exception as e:
        logger.warning(f"Failed to load {_SEARCH_YAML_PATH}: {e}")
        return {}


class AgenticKGSearch:
    """
    Agentic search layer over the KG knowledge base.

    Each tool invocation runs a Claude Code agent that iteratively
    searches the KG via MCP, reads pages, and synthesizes a structured
    response grounded in knowledge base evidence.

    The agent has read-only KG access only — no Write, Edit, Bash,
    or terminal tools. Uses the gated MCP server with only the kg
    gate enabled.

    Configuration is loaded from knowledge_search.yaml
    (searches.agentic_kg_search) and can be overridden at runtime
    via agent_config.
    """

    def __init__(self, agent_config: Optional[Dict[str, Any]] = None):
        """
        Initialize AgenticKGSearch.

        Loads defaults from knowledge_search.yaml, then applies
        any overrides from agent_config.

        Args:
            agent_config: Runtime overrides. Supports:
                - kg_index_path: Path to .index file
                - timeout: Agent timeout in seconds
                - use_bedrock: Use AWS Bedrock
                - aws_region: AWS region for Bedrock
                - model: Model ID override
        """
        self._yaml_config = _load_yaml_config()
        self._agent_config = agent_config or {}
        self._agent = None
        self._kg_search = None
        self._workspace = Path(tempfile.mkdtemp(prefix="agentic_search_"))
        self._wiki_structure_description = self._load_wiki_structure_description()
        self._initialize_agent(self._workspace)

    def _get(self, key: str, *yaml_path: str, default: Any = None) -> Any:
        """
        Resolve a config value: agent_config override > YAML path > default.

        Args:
            key: Key to look up in agent_config
            *yaml_path: Nested keys into self._yaml_config
            default: Fallback value
        """
        # Runtime override takes priority
        val = self._agent_config.get(key)
        if val is not None:
            return val

        # Walk into YAML config
        node = self._yaml_config
        for segment in yaml_path:
            if isinstance(node, dict):
                node = node.get(segment)
            else:
                node = None
            if node is None:
                return default
        return node if node is not None else default

    def _load_wiki_structure_description(self) -> str:
        """Load the wiki structure description for embedding in prompts."""
        desc_path = (
            Path(__file__).parent.parent
            / "wiki_structure"
            / "wiki_structure_description.md"
        )
        try:
            return desc_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning(f"Wiki structure description not found: {desc_path}")
            return "# Knowledge Base Structure\n\nNo structure description available."

    def _initialize_agent(self, workspace: Path) -> None:
        """
        Initialize Claude Code agent with read-only KG MCP tools.

        Reads agent, MCP server, and defaults config from YAML,
        then applies any runtime overrides from agent_config.
        """
        # project_root is the repo root (e.g. /home/ubuntu/kapso),
        # NOT the src/ directory.  agentic_kg_search.py lives at
        # src/kapso/knowledge_base/search/, so 5 parents up = repo root.
        project_root = Path(__file__).parent.parent.parent.parent.parent

        # --- MCP server env ---
        yaml_mcp_env = self._yaml_config.get("mcp_server", {}).get("env", {})
        mcp_env = {
            "PYTHONPATH": str(project_root / "src"),
            **yaml_mcp_env,
        }

        # Pass OPENAI_API_KEY explicitly so the MCP subprocess can
        # generate query embeddings even when env inheritance is incomplete.
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            mcp_env["OPENAI_API_KEY"] = openai_key

        # Resolve KG_INDEX_PATH to an absolute path so the MCP subprocess
        # can find it regardless of its working directory.
        kg_index_path = self._get("kg_index_path", "defaults", "kg_index_path")
        if kg_index_path:
            kg_abs = Path(kg_index_path).expanduser()
            if not kg_abs.is_absolute():
                kg_abs = project_root / kg_abs
            mcp_env["KG_INDEX_PATH"] = str(kg_abs)

        # --- MCP server command ---
        mcp_cfg = self._yaml_config.get("mcp_server", {})
        mcp_name = mcp_cfg.get("name", "kg-graph-search")
        mcp_servers = {
            mcp_name: {
                "command": mcp_cfg.get("command", "python"),
                "args": mcp_cfg.get("args", ["-m", "kapso.gated_mcp.server"]),
                "cwd": str(project_root),
                "env": mcp_env,
            }
        }

        # --- Allowed tools ---
        allowed_tools = self._get(
            "allowed_tools", "agent", "allowed_tools",
            default=[
                "Read",
                f"mcp__{mcp_name}__search_knowledge",
                f"mcp__{mcp_name}__get_wiki_page",
                f"mcp__{mcp_name}__get_page_structure",
            ],
        )

        # --- Agent specific ---
        timeout = self._get("timeout", "agent", "timeout", default=300)
        use_bedrock = self._get("use_bedrock", "agent", "use_bedrock", default=True)
        aws_region = self._get("aws_region", "agent", "aws_region")
        model = self._get("model", "agent", "model")

        agent_specific = {
            "allowed_tools": allowed_tools,
            "timeout": timeout,
            "mcp_servers": mcp_servers,
        }

        if use_bedrock:
            agent_specific["use_bedrock"] = True
            if aws_region:
                agent_specific["aws_region"] = aws_region

        config = CodingAgentFactory.build_config(
            agent_type=self._get(
                "agent_type", "agent", "agent_type", default="claude_code"
            ),
            model=model,
            agent_specific=agent_specific,
        )

        self._agent = CodingAgentFactory.create(config)
        self._agent.initialize(str(workspace))
        logger.info(
            f"Initialized AgenticKGSearch agent "
            f"(bedrock={use_bedrock}, model={model})"
        )

    def _build_prompt(self, task_instructions: str, user_message: str) -> str:
        """
        Assemble a full prompt from task instructions, wiki structure,
        MCP tool descriptions, and the user's message.
        """
        return f"""{task_instructions}

---

{self._wiki_structure_description}

---

## Available MCP Tools
- `search_knowledge(query, page_types?, domains?, top_k?)` -- Semantic KG search
- `get_wiki_page(title)` -- Retrieve full page by title
- `get_page_structure(page_type)` -- Get section definitions for a page type

## Search Strategy
- Search 2-3 different angles in PARALLEL in your first turn (call multiple search_knowledge tools at once, use top_k=3)
- Read the top 3-5 most relevant full pages in PARALLEL in your second turn
- Synthesize your response in your third turn — do NOT search further
- Always cite page IDs using [PageID] format
- You have a budget of 3 turns. Be decisive.

---

{user_message}"""

    def _run_agent(self, task_instructions: str, user_message: str) -> str:
        """Build prompt, run agent, return output text."""
        prompt = self._build_prompt(task_instructions, user_message)
        result = self._agent.generate_code(prompt)
        if not result.success:
            error_msg = getattr(result, "error", "Agent execution failed")
            logger.error(f"Agent failed: {error_msg}")
            return f"Error: {error_msg}"
        return result.output

    # =========================================================================
    # 7 Public Agentic Tool Methods
    # =========================================================================

    def search_knowledge(self, query: str, context: Optional[str] = None) -> str:
        """
        Search the knowledge base: search multiple angles, read relevant pages,
        synthesize a grounded answer with citations.

        Args:
            query: Question or topic to search for
            context: Optional additional context for the search

        Returns:
            Synthesized summary with [PageID] citations
        """
        task = (
            "You are a knowledge base search agent. "
            "Your job is to search from multiple angles, read the most "
            "relevant pages, and synthesize a grounded answer.\n\n"
            "Output format:\n"
            "- Synthesized summary addressing the query\n"
            "- Citations using [PageID] format\n"
            "- Distinguish between established consensus and emerging ideas"
        )
        msg = f"Query: {query}"
        if context:
            msg += f"\n\nAdditional context: {context}"
        return self._run_agent(task, msg)

    def build_plan(self, goal: str, constraints: Optional[str] = None) -> str:
        """
        ML execution planner: search Workflows, Principles, Implementations,
        and Heuristics to build a step-by-step plan.

        Args:
            goal: What the user wants to accomplish
            constraints: Optional constraints or requirements

        Returns:
            Structured plan with overview, specs, steps, and validation
        """
        task = (
            "You are an ML execution planner. Search the knowledge base "
            "for Workflows, Principles, Implementations, and Heuristics "
            "relevant to the goal.\n\n"
            "Output format:\n"
            "- Overview of the approach\n"
            "- Key specs and requirements\n"
            "- Numbered step-by-step plan\n"
            "- Tests and validation criteria"
        )
        msg = f"Goal: {goal}"
        if constraints:
            msg += f"\n\nConstraints: {constraints}"
        return self._run_agent(task, msg)

    def review_plan(self, proposal: str, goal: str) -> str:
        """
        ML plan reviewer: search for best practices and known pitfalls
        to evaluate a proposed plan.

        Args:
            proposal: The plan or proposal to review
            goal: The intended goal of the plan

        Returns:
            Review with approvals, risks, and suggestions
        """
        task = (
            "You are an ML plan reviewer. Search the knowledge base for "
            "best practices, known pitfalls, and relevant heuristics to "
            "evaluate the proposed plan.\n\n"
            "Output format:\n"
            "- Approvals: what looks good\n"
            "- Risks: potential issues or pitfalls\n"
            "- Suggestions: improvements based on KB evidence"
        )
        msg = f"Goal: {goal}\n\nProposal:\n{proposal}"
        return self._run_agent(task, msg)

    def verify_code_math(self, code_snippet: str, concept_name: str) -> str:
        """
        Math verification specialist: search for authoritative concept
        descriptions and reference implementations, then verify code.

        Args:
            code_snippet: Code to verify
            concept_name: The mathematical/ML concept being implemented

        Returns:
            Verdict (Pass/Fail) with analysis of discrepancies
        """
        task = (
            "You are a math verification specialist. Search for authoritative "
            "concept descriptions and reference implementations in the "
            "knowledge base, then verify the provided code.\n\n"
            "Output format:\n"
            "- Verdict: Pass or Fail\n"
            "- Analysis of any discrepancies between the code and KB references\n"
            "- Specific line-by-line issues if any"
        )
        msg = f"Concept: {concept_name}\n\nCode:\n```\n{code_snippet}\n```"
        return self._run_agent(task, msg)

    def diagnose_failure(self, symptoms: str, logs: str) -> str:
        """
        ML debugging specialist: search Heuristics for failure patterns
        and Environment pages for dependency issues.

        Args:
            symptoms: Description of the failure
            logs: Relevant log output or error messages

        Returns:
            Diagnosis, fix, and prevention advice
        """
        task = (
            "You are an ML debugging specialist. Search the knowledge base "
            "Heuristics for known failure patterns and Environment pages for "
            "dependency or configuration issues.\n\n"
            "Output format:\n"
            "- Diagnosis: what is likely going wrong\n"
            "- Fix: concrete steps to resolve\n"
            "- Prevention: how to avoid this in the future"
        )
        msg = f"Symptoms: {symptoms}\n\nLogs:\n```\n{logs}\n```"
        return self._run_agent(task, msg)

    def propose_hypothesis(
        self, current_status: str, recent_experiments: Optional[str] = None
    ) -> str:
        """
        ML research advisor: search for alternative approaches and
        strategies, then propose ranked hypotheses.

        Args:
            current_status: Where the project stands now
            recent_experiments: Optional description of recent experiments

        Returns:
            Ranked hypotheses with rationale from KB
        """
        task = (
            "You are an ML research advisor. Search the knowledge base for "
            "alternative approaches, strategies, and relevant principles.\n\n"
            "Output format:\n"
            "- Ranked hypotheses (most promising first)\n"
            "- Rationale grounded in KB evidence for each\n"
            "- Suggested experiments to test each hypothesis"
        )
        msg = f"Current status: {current_status}"
        if recent_experiments:
            msg += f"\n\nRecent experiments: {recent_experiments}"
        return self._run_agent(task, msg)

    def query_hyperparameter_priors(self, query: str) -> str:
        """
        Hyperparameter specialist: search for documented values, ranges,
        and tuning heuristics.

        Args:
            query: Hyperparameter question or topic

        Returns:
            Suggestion table with justification from KB
        """
        task = (
            "You are a hyperparameter specialist. Search the knowledge base "
            "for documented values, recommended ranges, and tuning heuristics.\n\n"
            "Output format:\n"
            "- Table of suggested hyperparameter values with ranges\n"
            "- Justification citing KB sources\n"
            "- Tuning strategy recommendations"
        )
        msg = f"Query: {query}"
        return self._run_agent(task, msg)

    # =========================================================================
    # Direct Passthrough Tool (no agent)
    # =========================================================================

    def _get_kg_search(self):
        """
        Lazily initialize a KGGraphSearch instance for direct page lookups.

        Uses the same kg_index_path from YAML config / agent_config.
        """
        if self._kg_search is not None:
            return self._kg_search

        from kapso.knowledge_base.search.factory import KnowledgeSearchFactory
        from kapso.knowledge_base.search.base import KGIndexMetadata

        kg_index_path = self._get("kg_index_path", "defaults", "kg_index_path")
        backend_type = "kg_graph_search"
        backend_refs = {}

        if kg_index_path:
            try:
                index_path = Path(kg_index_path).expanduser().resolve()
                if index_path.exists():
                    import json
                    index_data = json.loads(index_path.read_text(encoding="utf-8"))
                    metadata = KGIndexMetadata.from_dict(index_data)
                    backend_type = (metadata.search_backend or "").strip() or "kg_graph_search"
                    backend_refs = metadata.backend_refs or {}
                    logger.info(f"KGGraphSearch for get_page from index: {index_path}")
            except Exception as e:
                logger.warning(f"Failed to read kg_index_path for get_page: {e}")

        self._kg_search = KnowledgeSearchFactory.create(backend_type, params=backend_refs)
        return self._kg_search

    def _format_page(self, page) -> str:
        """Format a WikiPage as markdown."""
        parts = [
            f"# {page.id}\n",
            f"**Type:** {page.page_type}\n",
        ]

        if page.domains:
            parts.append(f"**Domains:** {', '.join(page.domains)}\n")

        if page.last_updated:
            parts.append(f"**Last Updated:** {page.last_updated}\n")

        parts.append(f"\n---\n")
        parts.append(f"\n## Overview\n{page.overview}\n")
        parts.append(f"\n## Content\n{page.content}\n")

        if page.sources:
            parts.append("\n## Sources\n")
            for src in page.sources:
                src_type = src.get('type', 'Link')
                src_title = src.get('title', 'Reference')
                src_url = src.get('url', '')
                if src_url:
                    parts.append(f"- **{src_type}:** [{src_title}]({src_url})\n")
                else:
                    parts.append(f"- **{src_type}:** {src_title}\n")

        if page.outgoing_links:
            parts.append("\n## Related Pages\n")
            for link in page.outgoing_links[:10]:
                edge_type = link.get('edge_type', 'related')
                target = link.get('target_id', '')
                target_type = link.get('target_type', '')
                parts.append(f"- {edge_type} → {target_type}: {target}\n")

        return "".join(parts)

    def get_page(self, page_id: str) -> str:
        """
        Retrieve a specific knowledge base page by its exact ID.

        Direct passthrough — no agent invocation. Calls KGGraphSearch.get_page()
        and formats the WikiPage as markdown.

        Args:
            page_id: Exact page ID (e.g., "Workflow/QLoRA_Finetuning")

        Returns:
            Markdown-formatted page content, or error message if not found
        """
        try:
            search = self._get_kg_search()
            page = search.get_page(page_id)

            if page is None:
                return (
                    f"Page not found: '{page_id}'\n\n"
                    "Tip: Use search_knowledge to search for pages by topic."
                )

            return self._format_page(page)

        except Exception as e:
            logger.error(f"get_page failed for '{page_id}': {e}", exc_info=True)
            return f"Error retrieving page: {str(e)}"

    # =========================================================================
    # Cleanup
    # =========================================================================

    def close(self) -> None:
        """Clean up resources."""
        self._agent = None
        if self._workspace and self._workspace.exists():
            import shutil
            try:
                shutil.rmtree(self._workspace)
            except Exception as e:
                logger.warning(f"Failed to clean up workspace: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
