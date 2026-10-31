"""The build's sessions by role, over the config's hosts table.

`search_strategy.params` names three roles, `ideation`, `implementation`
and `judge`, each with an optional `cli`, `model`, `effort` and `timeout`;
what a role leaves out comes from `params.host` and the host's entry in
`hosts` (its model, its effort, and its other keys as the adapter's own
settings). The ensemble members, the selector and the lens planner are
specs of the same shape under `ideation`. `--coding-agent <cli>` sets
`params.host`, so every role without a `cli` of its own follows it. The
parameters a config file may set are a known set, refused by name
otherwise. Design: docs/plans/redesign/wp4-config.md.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from kapso.core.config import ConfigError

ROLES = ("ideation", "implementation", "judge")
# The selector reads the worktree to verify candidates: the two first-party
# CLIs.
SELECTOR_CLIS = ("claude_code", "codex")
SPEC_KEYS = frozenset({"cli", "model", "effort", "timeout"})
MEMBER_KEYS = SPEC_KEYS | {"lens"}
ROLE_KEYS: Dict[str, frozenset] = {
    "ideation": SPEC_KEYS | {
        "gates", "web", "candidates_per_member", "ensemble", "selector", "lens_planner", "ensemble_time_split",
    },
    "implementation": SPEC_KEYS | {"gates", "web", "fallback_model"},
    "judge": SPEC_KEYS,
}
GATE_FAILURE_POLICIES = ("skip", "warn", "error")
# The keys every host entry may set beside the adapter's own settings
# (`settings` in agents.yaml).
HOST_KEYS = frozenset({"model", "effort"})
# What a config file may set under search_strategy.params.
CONFIG_PARAMS = frozenset({
    "host", *ROLES, "gate_failure_policy", "parent_policy", "node_expansion_value",
    "expansion_lane_env", "env_strip", "session_env_defaults", "shared_cache_dir",
    "registered_evaluation_archive_glob", "repo_memory_failure_policy",
    "repo_memory_max_retries", "repo_memory_update_payload_max_chars", "parent_sampling", "cost",
    "honor_agent_stop",
})
# What the engine threads in at launch, never a file's.
ENGINE_PARAMS = frozenset({
    "bank_serving", "experience_intro_path", "kg_index_path", "hosted", "inbox",
    "experiment_history_path",
})
ORIGIN = "search_strategy.params"


@dataclass(frozen=True)
class SessionRole:
    """One session's host and what it is asked for."""

    name: str
    cli: str
    model: str
    effort: Optional[str]
    timeout: Optional[float]
    # The host entry's other keys: the adapter's own settings (auth_mode,
    # vertex_project, …), merged into the session's agent settings.
    settings: Dict[str, Any] = field(default_factory=dict)
    # The native web tools (the engine's permission boundary on both
    # hosts): the role's own switch, the ideation role's for its members,
    # selector and lens planner.
    web: bool = True
    lens: Optional[str] = None

    @property
    def label(self) -> str:
        return f"{self.cli}:{self.model}"


def _unknown(keys: Iterable[str], known: frozenset, origin: str) -> None:
    unknown = sorted(set(keys) - known)
    if unknown:
        raise ConfigError(
            f"{origin}: unknown key{'s' if len(unknown) > 1 else ''} {', '.join(unknown)}; "
            f"the keys are {', '.join(sorted(known))}"
        )


def check_params(params: Dict[str, Any], *, engine: bool = False) -> None:
    """Refuse a parameter the generic strategy does not read; with
    ``engine`` the keys the engine threads at launch are allowed too."""
    known = CONFIG_PARAMS | ENGINE_PARAMS if engine else CONFIG_PARAMS
    _unknown(params, known, ORIGIN)
    for role in ROLES:
        block = params.get(role)
        if block is None:
            continue
        if not isinstance(block, dict):
            raise ConfigError(f"{ORIGIN}.{role} must be a mapping, got {type(block).__name__}")
        _unknown(block, ROLE_KEYS[role], f"{ORIGIN}.{role}")
        if "web" in block and not isinstance(block["web"], bool):
            raise ConfigError(f"{ORIGIN}.{role}.web must be true or false")
    policy = params.get("gate_failure_policy")
    if policy is not None and policy not in GATE_FAILURE_POLICIES:
        raise ConfigError(
            f"{ORIGIN}.gate_failure_policy must be one of {', '.join(GATE_FAILURE_POLICIES)}, got {policy!r}"
        )


# The evaluation maintainer's block: its session spec, the switch and its
# own knobs (docs/reference/configuration.mdx).
MAINTAINER_KEYS = SPEC_KEYS | {
    "enabled", "subsample_seed", "calibration_fraction", "calibration_timeout_seconds",
    "fast_variant_threshold_minutes", "overhead_factor", "max_change_requests",
    "transition_freeze_fraction", "protected_data_paths",
}
# The build's identity in the config: the blocks a build reads, which the
# checkpoint fingerprints; the budget is an operator dial, the rest
# other verbs' or a runner's.
BUILD_CONFIG_BLOCKS = ("hosts", "search_strategy", "inference", "retry", "models", "inbox", "evaluation_maintainer")


def check_maintainer(block: Any) -> None:
    if not isinstance(block, dict):
        raise ConfigError(f"evaluation_maintainer must be a mapping, got {type(block).__name__}")
    _unknown(block, MAINTAINER_KEYS, "evaluation_maintainer")
    if not isinstance(block.get("enabled"), bool):
        raise ConfigError("evaluation_maintainer.enabled must be true or false")


# The deployment block (docs/plans/redesign/wp5-deploy.md): a session spec,
# the target the call defaults to, the deadlines and the ports.
DEPLOYMENT_KEYS = SPEC_KEYS | {"target", "call_timeout", "script_timeout", "port_base", "port_span"}
DEPLOYMENT = "deployment"
MAX_PORT = 65535


def deployment_role(config: Dict[str, Any], coding_agent: Optional[str] = None) -> SessionRole:
    """The deployment session's spec over the hosts table: the call's
    host, else the block's `cli`, else the build's host
    (`search_strategy.params.host`); the block's model and effort, else
    the host entry's."""
    block = config[DEPLOYMENT]
    spec = {key: block[key] for key in SPEC_KEYS if block.get(key) is not None}
    if coding_agent is not None:
        spec["cli"] = coding_agent
    host = spec.get("cli") or config["search_strategy"]["params"]["host"]
    return resolve_spec(config["hosts"], host, spec, name=DEPLOYMENT, origin=DEPLOYMENT)


def check_deployment(config: Dict[str, Any]) -> None:
    block = config[DEPLOYMENT]
    if not isinstance(block, dict):
        raise ConfigError(f"{DEPLOYMENT} must be a mapping, got {type(block).__name__}")
    _unknown(block, DEPLOYMENT_KEYS, DEPLOYMENT)
    target = block.get("target")
    if not isinstance(target, str) or not target.strip():
        raise ConfigError(f"{DEPLOYMENT}.target must be a non-empty string: auto, a bundled target or a harness's")
    for key in ("timeout", "call_timeout", "script_timeout"):
        value = block.get(key)
        if key == "timeout" and value is None:
            continue   # no deadline on the session, as a role's null timeout means
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ConfigError(f"{DEPLOYMENT}.{key} must be a positive number of seconds")
    for key in ("port_base", "port_span"):
        value = block.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigError(f"{DEPLOYMENT}.{key} must be a positive integer")
    if block["port_base"] + block["port_span"] - 1 > MAX_PORT:
        raise ConfigError(f"{DEPLOYMENT}.port_base + port_span must stay within the ports (1..{MAX_PORT})")
    deployment_role(config)


def validate_config(config: Dict[str, Any], manifest: Dict[str, Any]) -> None:
    """What the walk against the packaged file cannot check: the hosts'
    entries against the adapters, the strategy's parameters and roles,
    the maintainer's block, the deployment's. Called before any launch,
    by the facade, the doctor and the orchestrator."""
    check_hosts(config["hosts"], manifest)
    check_params(config["search_strategy"]["params"])
    check_maintainer(config["evaluation_maintainer"])
    check_deployment(config)


def build_hosts(manifest: Dict[str, Any]) -> List[str]:
    """The adapters a build session can run on: those with a staging
    (`host` in the registry), which is what every build session needs."""
    return sorted(name for name, entry in manifest.items() if entry.get("host"))


def check_hosts(hosts: Any, manifest: Dict[str, Any]) -> None:
    """Every host entry names an adapter a build can run on, a model, and
    only `effort` and the adapter's own settings (`settings` in the
    registry); an adapter the registry does not know, or one with no
    staging (gemini, openhands, aider, openai_compatible), is refused."""
    if not isinstance(hosts, dict) or not hosts:
        raise ConfigError("hosts must be a mapping of at least one CLI to its entry")
    for cli, entry in hosts.items():
        if cli not in manifest:
            raise ConfigError(f"hosts.{cli}: no adapter of that name; the adapters are {', '.join(sorted(manifest))}")
        if not manifest[cli].get("host"):
            raise ConfigError(
                f"hosts.{cli}: the {cli} adapter cannot host a build session (it has no staging); "
                f"the hosts a build runs on are {', '.join(build_hosts(manifest))}"
            )
        if not isinstance(entry, dict) or not isinstance(entry.get("model"), str) or not entry["model"].strip():
            raise ConfigError(f"hosts.{cli} needs a model")
        allowed = HOST_KEYS | set(manifest[cli].get("settings") or [])
        _unknown(entry, allowed, f"hosts.{cli}")


def resolve_spec(
    hosts: Dict[str, Dict[str, Any]], host: str, spec: Dict[str, Any], *, name: str, origin: str,
    keys: frozenset = SPEC_KEYS, web: bool = True,
) -> SessionRole:
    """One session spec over the hosts table: its own `cli`, else ``host``;
    its own `model` and `effort`, else the host entry's; ``web`` the
    switch the session follows."""
    if not isinstance(spec, dict):
        raise ConfigError(f"{origin} must be a mapping, got {type(spec).__name__}")
    _unknown(spec, keys, origin)
    for key in ("cli", "model", "effort"):
        if spec.get(key) is not None and (not isinstance(spec[key], str) or not spec[key].strip()):
            raise ConfigError(f"{origin}.{key} must be a non-empty string, got {spec[key]!r}")
    cli = spec.get("cli") or host
    if cli not in hosts:
        where = f"{origin}.cli" if spec.get("cli") else f"{ORIGIN}.host (what --coding-agent sets)"
        raise ConfigError(
            f"{where}: {cli} is not in hosts (the hosts are {', '.join(sorted(hosts))}); "
            f"add hosts.{cli}: {{model: …}} to the config to use it"
        )
    entry = hosts[cli]
    if not isinstance(entry, dict) or not isinstance(entry.get("model"), str) or not entry["model"].strip():
        raise ConfigError(f"hosts.{cli} needs a model")
    model = spec.get("model") or entry["model"]
    if not isinstance(model, str) or not model.strip():
        raise ConfigError(f"{origin}.model must be a non-empty string")
    timeout = spec.get("timeout")
    if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0):
        raise ConfigError(f"{origin}.timeout must be a positive number of seconds")
    settings = {key: value for key, value in entry.items() if key not in HOST_KEYS}
    effort = spec.get("effort")
    if effort is None:
        effort = entry.get("effort")
    return SessionRole(
        name=name, cli=cli, model=model,
        effort=effort,
        timeout=None if timeout is None else float(timeout),
        settings=settings, web=web, lens=spec.get("lens"),
    )


def resolve_role(hosts: Dict[str, Dict[str, Any]], params: Dict[str, Any], role: str) -> SessionRole:
    """The role's session: the role block's own keys over the host's."""
    if "host" not in params:
        raise ConfigError(f"{ORIGIN}.host is required")
    block = params.get(role) or {}
    spec = {key: block[key] for key in SPEC_KEYS if key in block}
    return resolve_spec(
        hosts, params["host"], spec, name=role, origin=f"{ORIGIN}.{role}", web=bool(block.get("web", True)),
    )


def resolve_members(
    hosts: Dict[str, Dict[str, Any]], params: Dict[str, Any], members: Any,
) -> Optional[List[SessionRole]]:
    """The ideation ensemble's members, each a spec of its own."""
    if members is None:
        return None
    if not isinstance(members, list) or not members:
        raise ConfigError(f"{ORIGIN}.ideation.ensemble must be a non-empty list of member specs")
    web = bool((params.get("ideation") or {}).get("web", True))
    return [
        resolve_spec(
            hosts, params["host"], member, name=f"member {index + 1}",
            origin=f"{ORIGIN}.ideation.ensemble[{index}]", keys=MEMBER_KEYS, web=web,
        )
        for index, member in enumerate(members)
    ]


def resolve_optional(
    hosts: Dict[str, Dict[str, Any]], params: Dict[str, Any], spec: Any, *, name: str,
) -> Optional[SessionRole]:
    """The selector's or the lens planner's spec, or None when unset."""
    if spec is None:
        return None
    web = bool((params.get("ideation") or {}).get("web", True))
    return resolve_spec(hosts, params["host"], spec, name=name, origin=f"{ORIGIN}.ideation.{name}", web=web)


def session_labels(roles: Sequence[SessionRole]) -> str:
    return ", ".join(f"{role.name} on {role.label}" for role in roles)
