"""The generic strategy's parameters for direct construction in tests: the
packaged `search_strategy.params` (the one home of every default, Rule 1)
with a test's overrides merged over them, and the packaged hosts table;
and the line of every prompt slot, for a test of a text that must reach a
prompt as it was written."""

import re
from pathlib import Path
from typing import Any, Dict

import kapso
from kapso.core.config import deep_merge, load_config

GENERIC_STRATEGY = Path(kapso.__file__).parent / "execution" / "search_strategies" / "generic"


def every_prompt_slot() -> str:
    """Every `{{slot}}` of the generic strategy's prompt templates, on one
    line. A text that must reach a prompt unaltered carries it in a test:
    `render_prompt` substitutes the slots in the order of its variables,
    so any variable substituted after that text would rewrite the line."""
    slots = sorted({
        slot
        for template in GENERIC_STRATEGY.rglob("*.md")
        for slot in re.findall(r"\{\{\w+\}\}", template.read_text(encoding="utf-8"))
    })
    # A floor: templates that moved would leave the line empty and the tests that lean on it guarding nothing.
    assert {"{{problem}}", "{{goal}}", "{{solution}}", "{{candidates}}", "{{campaign_notes}}"} <= set(slots), slots
    return " ".join(slots)


def generic_params(**overrides: Any) -> Dict[str, Any]:
    """The packaged parameters with ``overrides`` deep-merged over them
    (a role block given whole replaces its keys one by one)."""
    return deep_merge(load_config()["search_strategy"]["params"], overrides)


def packaged_hosts() -> Dict[str, Dict[str, Any]]:
    return dict(load_config()["hosts"])
