"""Config-only API endpoints shared by clients and preflight."""

from ipaddress import ip_address
from urllib.parse import urlsplit

from kapso.core.agent_manifest import load_agent_manifest


def openai_compatible_options(overrides=None):
    """Resolve adapter options from their single home in agents.yaml."""
    manifest = load_agent_manifest()
    defaults = manifest["agents"]["openai_compatible"]["agent_specific"]
    return {**defaults, **(overrides or {})}


def validate_api_base_url(base_url):
    """Require HTTPS, allowing plaintext only for literal loopback hosts."""
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be a non-empty URL from config")
    base_url = base_url.strip()
    parsed = urlsplit(base_url)
    host = parsed.hostname
    if (
        parsed.scheme not in ("http", "https")
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or any(character.isspace() for character in base_url)
        or "\\" in base_url
    ):
        raise ValueError("base_url must be an HTTP(S) URL without credentials")
    if parsed.port is not None and parsed.port <= 0:
        raise ValueError("base_url must use a positive port")
    if parsed.scheme == "http":
        loopback = host == "localhost"
        if ":" in host or all(c in "0123456789." for c in host):
            loopback = ip_address(host).is_loopback
        if not loopback:
            raise ValueError(
                "base_url requires HTTPS except on loopback hosts"
            )
    return base_url
