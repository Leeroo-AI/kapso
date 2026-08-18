"""
Bank Gate
=========

Knowledge-bank pull tools for campaign sessions (learn-from-trajectories
design §5.1, pull mode): `bank_search` returns the whole eligible hero
shortlist (the calling agent is the reranker), `bank_get` renders full
served projections with the co-serving guard. Eligibility and quarantine
are law on both. Every call appends to the campaign's pull log — the
serving record's second exposure level (attribution binds to `got`).

The gate is a thin subprocess wrapper over the pure functions in
`kapso.learning.retriever`; its per-campaign parameters arrive through the
same env-injection channel the sibling gates use (set by the launching
runner, never read by core kapso code):

- KAPSO_BANK_DIR          pinned read-only bank checkout
- KAPSO_BANK_HEAD         the stamped head the checkout is pinned at
- KAPSO_SERVING_PULL_LOG  JSONL pull-log path inside the campaign workspace
- KAPSO_TASK_FAMILY       task coordinate: family
- KAPSO_TASK_DATASET      task coordinate: dataset (optional)

The feedback judge never gets this gate — a judge reading the bank couples
the evaluator to the thing under evaluation (§6).
"""

import logging
import os
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


class BankGate(ToolGate):
    """Gate for knowledge-bank pull tools (bank_search / bank_get)."""

    name = "bank"
    description = "Knowledge-bank pull tools (hero shortlist + full cards)"

    def __init__(self, config: Optional[GateConfig] = None):
        super().__init__(config)
        self._bank = None

    def _get_bank(self):
        if self._bank is None:
            from kapso.learning.bank import Bank
            self._bank = Bank(os.environ["KAPSO_BANK_DIR"])
        return self._bank

    def _task_coords(self) -> Dict[str, str]:
        coords = {"family": os.environ["KAPSO_TASK_FAMILY"]}
        dataset = os.environ.get("KAPSO_TASK_DATASET")
        if dataset:
            coords["dataset"] = dataset
        return coords

    def get_tools(self) -> List["Tool"]:
        if not HAS_MCP:
            return []
        return [
            Tool(
                name="bank_search",
                description=(
                    "List every knowledge-bank card eligible for this "
                    "campaign's task as one hero line each, reliability-"
                    "ordered. This is the bank's whole answer for the "
                    "task's scope — you are the reranker: read the hero "
                    "lines, then bank_get the cards worth reading in full."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "What you are looking for (logged for "
                                "attribution; the shortlist itself is the "
                                "whole eligible set)"
                            ),
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="bank_get",
                description=(
                    "Render the full served projection of named bank cards "
                    "(fact, reliability, scope, evidence digest, probe). "
                    "Cards outside this task's scope are refused by name. "
                    "A spec that uses a card cites [card:<name>]."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "card_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Card names from bank_search",
                        },
                    },
                    "required": ["card_names"],
                },
            ),
        ]

    async def handle_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Optional[List["TextContent"]]:
        if not HAS_MCP:
            return None
        from kapso.learning.retriever import (
            append_pull_event,
            pull_projections,
            pull_shortlist,
        )

        bank = self._get_bank()
        coords = self._task_coords()
        log_path = os.environ["KAPSO_SERVING_PULL_LOG"]
        bank_head = os.environ["KAPSO_BANK_HEAD"]

        if tool_name == "bank_search":
            query = str(arguments.get("query", ""))
            result = pull_shortlist(bank, coords, query)
            append_pull_event(log_path, {
                "tool": "bank_search",
                "query": query,
                "task": coords,
                "bank_head": bank_head,
                "eligible": result["eligible"],
                "shown": result["shown"],
            })
            return [TextContent(type="text", text=result["text"])]
        if tool_name == "bank_get":
            card_names = [str(n) for n in arguments.get("card_names", [])]
            result = pull_projections(bank, coords, card_names)
            append_pull_event(log_path, {
                "tool": "bank_get",
                "requested": card_names,
                "task": coords,
                "bank_head": bank_head,
                "got": result["got"],
                "refused": result["refused"],
                "tensions": result["tensions"],
            })
            return [TextContent(type="text", text=result["text"])]
        return None
