"""The single, lightweight loader for the packaged agent registry."""

from pathlib import Path

from kapso.core.config import load_config

AGENTS_YAML_PATH = (
    Path(__file__).resolve().parent.parent
    / "execution"
    / "coding_agents"
    / "agents.yaml"
)


def load_agent_manifest():
    """Load the required registry; missing or malformed manifests raise."""
    manifest = load_config(str(AGENTS_YAML_PATH))
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("agents"), dict
    ):
        raise ValueError("agents.yaml must contain an agents mapping")
    default = manifest.get("default_agent")
    if not isinstance(default, str) or default not in manifest["agents"]:
        raise ValueError(
            "agents.yaml default_agent must name a declared agent"
        )
    return manifest
