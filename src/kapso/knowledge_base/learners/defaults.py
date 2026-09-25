"""Learner session settings for direct construction (tests, the module CLI).

`learn_knowledge()` threads the live config's `learner.ingestor` / `learner.merger`
blocks into the learners. When a learner is built directly with partial params, the
missing keys come from the packaged config — the single source (Rule 1) — so a
default change can never leave the code and the config out of step.
"""

from typing import Any, Dict

from kapso.core.config import PLATFORM_CONFIG_PATH, load_config


def learner_defaults(role: str) -> Dict[str, Any]:
    """The packaged `learner.<role>` block of the default mode ("ingestor" or "merger")."""
    config = load_config(str(PLATFORM_CONFIG_PATH))
    return dict(config["modes"][config["default_mode"]]["learner"][role])
