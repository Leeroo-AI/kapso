"""
Bank Gate
=========

Knowledge-bank tools for campaign sessions (serving-agentic-redesign.md):
`bank_index` returns the whole eligible set as book-index lines (the
calling agent is the selector), `bank_get_card` renders full card bodies,
and `bank_get_card_with_evidence` adds the reliability block and the full
evidence trail for due diligence. Eligibility and quarantine are law on
all three. Every call appends to the campaign's pull log — the serving
record's exposure ladder source (attribution binds at `read` and above).

The gate is a thin subprocess wrapper over the pure functions in
`kapso.learning.retriever`; its per-campaign parameters arrive through the
same env-injection channel the sibling gates use (set by the launching
runner, never read by core kapso code):

- KAPSO_BANK_DIR          pinned read-only bank checkout
- KAPSO_BANK_HEAD         the stamped head the checkout is pinned at
- KAPSO_SERVING_PULL_LOG  JSONL pull-log path inside the campaign workspace
- KAPSO_TASK_FAMILY       task coordinate: family
- KAPSO_TASK_DATASET      task coordinate: dataset (optional)
- KAPSO_PROBE_BUDGET      max probe offers riding card reads this campaign

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
    """Gate for the knowledge-bank tools (index / get-card / get-card-with-evidence)."""

    name = "bank"
    description = "Knowledge-bank tools (book index + full cards at two depths)"

    def __init__(self, config: Optional[GateConfig] = None):
        super().__init__(config)
        self._bank = None
        self._offers = None

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

    def _probe_offers(self) -> Dict[str, str]:
        if self._offers is None:
            from kapso.learning.retriever import probe_offers
            self._offers = probe_offers(
                self._get_bank(),
                self._task_coords(),
                int(os.environ["KAPSO_PROBE_BUDGET"]),
            )
        return self._offers

    def get_tools(self) -> List["Tool"]:
        if not HAS_MCP:
            return []
        return [
            Tool(
                name="bank_index",
                description=(
                    "The knowledge bank's index page: every card eligible "
                    "for this campaign's task — name, one-liner, score, and "
                    "when it applies — like the index of a book. This is "
                    "the bank's whole answer for the task's scope; you are "
                    "the selector: scan it against your plan, then open "
                    "the cards worth reading with bank_get_card."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "section": {
                            "type": "string",
                            "enum": ["insights", "procedures"],
                            "description": (
                                "Optional: restrict to one section; omit "
                                "for the whole index"
                            ),
                        },
                    },
                },
            ),
            Tool(
                name="bank_get_card",
                description=(
                    "Full card bodies for named bank cards (title, rule, "
                    "situation, what to do, why believe it). Procedures "
                    "include their runnable code location. Cards outside "
                    "this task's scope are refused by name."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "cards": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Card names from bank_index",
                        },
                    },
                    "required": ["cards"],
                },
            ),
            Tool(
                name="bank_get_card_with_evidence",
                description=(
                    "Named cards at due-diligence depth: the full body "
                    "plus the reliability block and the complete evidence "
                    "trail (per-entry trajectory, verdict, effect). Use "
                    "before staking real budget on a card's advice."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "cards": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Card names from bank_index",
                        },
                    },
                    "required": ["cards"],
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
            render_cards,
            render_index,
        )

        bank = self._get_bank()
        coords = self._task_coords()
        log_path = os.environ["KAPSO_SERVING_PULL_LOG"]
        bank_head = os.environ["KAPSO_BANK_HEAD"]

        if tool_name == "bank_index":
            section = arguments.get("section") or None
            result = render_index(bank, coords, section)
            append_pull_event(log_path, {
                "tool": "bank_index",
                "section": section,
                "task": coords,
                "bank_head": bank_head,
                "eligible": result["eligible"],
                "listed": result["listed"],
            })
            return [TextContent(type="text", text=result["text"])]
        if tool_name in ("bank_get_card", "bank_get_card_with_evidence"):
            with_evidence = tool_name == "bank_get_card_with_evidence"
            card_names = [str(n) for n in arguments.get("cards", [])]
            result = render_cards(
                bank, coords, card_names, with_evidence, self._probe_offers()
            )
            append_pull_event(log_path, {
                "tool": tool_name,
                "requested": card_names,
                "task": coords,
                "bank_head": bank_head,
                "served": result["served"],
                "refused": result["refused"],
                "tensions": result["tensions"],
            })
            return [TextContent(type="text", text=result["text"])]
        return None
