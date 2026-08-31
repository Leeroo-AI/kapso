"""Per-verb preflight: what THIS config actually needs, checked up front.

Every requirement is DERIVED from the active config, never hardcoded — an
all-codex config is never asked for `claude`, a `skip_merge=True` ingest is
never asked for Neo4j, and a `learn()` whose codify target is `local` is
never asked for `gcloud`. The same resolver serves two consumers:

  - the facade verbs, which run the static tier before doing any work, so
    a missing binary costs seconds instead of surfacing four hours into a
    crew session (onboarding E2E findings #1 and #5), and
  - `kapso doctor [verb]`, which renders the same rows and can add the
    live tier.

Two tiers, because they cost different amounts. STATIC is milliseconds —
a binary on PATH, a credential present, a port open, a git remote
reachable. LIVE spends one throwaway token per {cli, model} pair to prove
the account can actually serve it; it is config-gated
(`preflight.live_model_probe`) because it costs real quota.

Every failed row carries the config key that asked for it and one
copy-pasteable fix. That pairing is the point: "install claude" is not
actionable, "install claude — modes.GENERIC.coding_agent.model wants
claude-opus-5" is.
"""

import json
import os
import shutil
import socket
import subprocess
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from kapso.core.config import load_config
from kapso.gated_mcp.presets import GATES, resolve_gates
from kapso.learning.bank_remote import bank_origin, bank_remote_error

# The packaged platform config is the single source for preflight's own
# defaults (Rule 1) — a user config that predates the `preflight:` block
# inherits them rather than a duplicated literal.
_PACKAGED_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

# The coding-agent manifest already declares each agent's env_vars and
# install_command — the fix lines below come from it rather than from a
# second copy here. Read directly (a structural path, not a knob) so
# preflight stays free of the execution stack's import weight; the other
# reader is execution/coding_agents/factory.py.
_AGENTS_YAML_PATH = (
    Path(__file__).resolve().parent.parent
    / "execution" / "coding_agents" / "agents.yaml"
)

# Agent types that run as an external CLI binary. Everything else in the
# manifest is a Python SDK adapter, whose requirement is the package its
# install_command names — surfaced from the manifest, not invented here.
CLI_BINARIES = {
    "claude_code": "claude",
    "oss_claude_code": "claude",
    "codex": "codex",
    "aider": "aider",
}

# Deployment targets and the tool each one needs on the box. LOCAL runs in
# process; AUTO is resolved by a coding-agent session, so its requirement
# is that session's agent (added separately).
DEPLOY_TOOLS = {
    "docker": ("docker", "install Docker: https://docs.docker.com/get-docker/"),
    "modal": ("modal", "pip install modal && modal token new"),
    "bentoml": ("bentoml", "pip install bentoml"),
}

VERBS = ("research", "learn_knowledge", "evolve", "learn", "deploy")


# =============================================================================
# RECORDS
# =============================================================================

@dataclass(frozen=True)
class Requirement:
    """One thing the active config needs, and how to get it.

    `origin` names the config key (or call argument) that asked for it —
    without that, a failed row tells the user what is missing but not why
    Kapso wants it. `required=False` rows are advisory: the run proceeds
    degraded, which is what a `gate_failure_policy: warn` gate does.
    """

    label: str
    ok: bool
    fix: str
    origin: str
    required: bool = True
    detail: str = ""


@dataclass(frozen=True)
class SessionSpec:
    """One coding-agent session the config can spawn."""

    cli: str
    model: str
    origin: str
    auth_mode: str = "auto"


class PreflightError(RuntimeError):
    """Raised when a verb's required checks fail. The message IS the
    rendered report — the caller prints nothing extra."""

    def __init__(self, verb: str, requirements: Sequence[Requirement]):
        self.verb = verb
        self.requirements = tuple(requirements)
        super().__init__(render(verb, requirements))


# =============================================================================
# PROBES — each returns a plain bool, never raises for the "absent" case
# =============================================================================

def agent_manifest() -> Dict[str, Any]:
    """The coding-agent manifest's `agents:` mapping."""
    if not _AGENTS_YAML_PATH.is_file():
        return {}
    return (yaml.safe_load(_AGENTS_YAML_PATH.read_text()) or {}).get("agents", {})


def default_preflight_config() -> Dict[str, Any]:
    """The platform `preflight:` block from the packaged config."""
    return load_config(str(_PACKAGED_CONFIG_PATH))["preflight"]


def preflight_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """The active preflight settings: the caller's `preflight:` block over
    the packaged defaults, so a config written before the block existed
    still gets the shipped behaviour."""
    return {**default_preflight_config(), **(config.get("preflight") or {})}


def port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    """connect_ex, not connect: a closed port is an ordinary answer here,
    not an exception to swallow (Rule 2)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def claude_logged_in() -> bool:
    """A stored Claude CLI login, or the OAuth token in the environment."""
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return True
    if shutil.which("claude") is None:
        return False
    result = subprocess.run(
        ["claude", "auth", "status"], capture_output=True, text=True, timeout=15,
    )
    if result.returncode != 0:
        return False
    # A zero exit with unparseable output is a real anomaly — let it raise.
    return bool(json.loads(result.stdout).get("loggedIn"))


def codex_authenticated() -> bool:
    """The ChatGPT login on disk, or an API key in the environment."""
    if os.environ.get("CODEX_API_KEY"):
        return True
    return (Path.home() / ".codex" / "auth.json").is_file()


def probe_model_access(cli_name: str, model: str) -> Tuple[bool, str]:
    """One-token live probe: can this CLI actually serve this model on the
    current subscription? Returns (ok, the CLI's own answer when not) — a
    capped model fails here in seconds instead of deep inside a crew
    session (onboarding E2E finding #5)."""
    prompt = "Reply with exactly: ok"
    if cli_name in ("claude_code", "oss_claude_code"):
        cmd = ["claude", "-p", "--model", model, prompt]
    else:
        # --skip-git-repo-check: codex refuses untrusted (non-git) CWDs,
        # and preflight runs wherever the user is; a read-only one-token
        # probe needs no trust. stdin closed — codex reads a non-tty
        # stdin as extra prompt input. (Both caught by the live smoke.)
        cmd = ["codex", "exec", "--model", model, "--sandbox", "read-only",
               "--skip-git-repo-check", prompt]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180,
                            stdin=subprocess.DEVNULL)
    if result.returncode == 0:
        return True, ""
    # stdout's first line carries claude's cap message; codex prefixes
    # stderr with noise, so its real reason is the LAST line.
    stdout_lines = [line for line in result.stdout.splitlines() if line.strip()]
    stderr_lines = [line for line in result.stderr.splitlines() if line.strip()]
    if stdout_lines:
        return False, stdout_lines[0]
    if stderr_lines:
        return False, stderr_lines[-1]
    return False, f"exit {result.returncode}"


# =============================================================================
# CONFIG -> SESSIONS
# =============================================================================

def session_specs(node: Any, prefix: str = "") -> List[SessionSpec]:
    """Every coding-agent session the given config subtree can spawn.

    A model key (`model` or `*_model`) names the session; the CLI that runs
    it resolves from, in order: an explicit `cli` sibling, an
    `implementation_cli` sibling (for `implementation_model`), a `type`
    sibling naming a known agent, or the model-name prefix (claude-* runs
    on the claude CLI, everything else on codex). Embedding models are not
    sessions — their check is the OPENAI_API_KEY row.
    """
    manifest = agent_manifest()
    specs: List[SessionSpec] = []

    def resolve_cli(block: Dict[str, Any], key: str, model_name: str) -> str:
        if block.get("cli"):
            return str(block["cli"])
        if key == "implementation_model" and block.get("implementation_cli"):
            return str(block["implementation_cli"])
        # `coding_agent` / `feedback_generator` name their CLI as `type`;
        # guard against unrelated `type` keys (search_strategy.type is a
        # strategy name, not an agent).
        block_type = block.get("type")
        if isinstance(block_type, str) and block_type.lower() in manifest:
            return block_type.lower()
        return "claude_code" if model_name.startswith("claude") else "codex"

    def resolve_auth_mode(block: Dict[str, Any]) -> str:
        agent_specific = block.get("agent_specific")
        if isinstance(agent_specific, dict) and agent_specific.get("auth_mode"):
            return str(agent_specific["auth_mode"])
        return str(block.get("auth_mode") or "auto")

    def walk(current: Any, path: str) -> None:
        if isinstance(current, dict):
            for key, value in current.items():
                child = f"{path}.{key}" if path else key
                if key == "model" or key.endswith("_model"):
                    if isinstance(value, str) and value:
                        specs.append(SessionSpec(
                            cli=resolve_cli(current, key, value),
                            model=value,
                            origin=f"{child} = {value}",
                            auth_mode=resolve_auth_mode(current),
                        ))
                    continue
                walk(value, child)
        elif isinstance(current, list):
            for index, item in enumerate(current):
                walk(item, f"{path}[{index}]")

    walk(node, prefix)
    return specs


def summarize_origins(specs: Sequence[SessionSpec]) -> str:
    """What asked for this, in one readable line. Six config keys all naming
    claude-opus-5 is a wall of near-identical text; "the first key (+5 more)
    = claude-opus-5" says the same thing and can be acted on."""
    by_model: Dict[str, List[str]] = {}
    for spec in specs:
        by_model.setdefault(spec.model, []).append(spec.origin.split(" = ")[0])
    parts = []
    for model, keys in by_model.items():
        unique = list(dict.fromkeys(keys))
        extra = f" (+{len(unique) - 1} more)" if len(unique) > 1 else ""
        parts.append(f"{unique[0]}{extra} = {model}")
    return "; ".join(parts)


def cli_requirements(specs: Sequence[SessionSpec]) -> List[Requirement]:
    """The binaries and credentials the given sessions need — one row per
    distinct requirement, listing every config key that wants it."""
    manifest = agent_manifest()
    by_cli: Dict[str, List[SessionSpec]] = {}
    for spec in specs:
        by_cli.setdefault(spec.cli, []).append(spec)

    requirements: List[Requirement] = []
    if not by_cli:
        return requirements

    binaries = {
        CLI_BINARIES[cli] for cli in by_cli if cli in CLI_BINARIES
    }
    if binaries & {"claude", "codex"}:
        requirements.append(Requirement(
            label="node",
            ok=shutil.which("node") is not None,
            fix="install Node.js from https://nodejs.org — the agent CLIs ship via npm",
            origin="required by every claude/codex session below",
        ))

    for cli, cli_specs in sorted(by_cli.items()):
        origin = summarize_origins(cli_specs)
        info = manifest.get(cli, {})
        install = info.get("install_command") or f"install the {cli} agent"

        binary = CLI_BINARIES.get(cli)
        if binary is None:
            # An SDK adapter: its requirement is the package, plus whatever
            # credentials the manifest declares.
            requirements.append(Requirement(
                label=f"{cli} agent installed",
                ok=False if not info else True,
                fix=install,
                origin=origin,
            ))
            for name in info.get("env_vars") or []:
                requirements.append(Requirement(
                    label=name,
                    ok=bool(os.environ.get(name)),
                    fix=f"add {name}=... to .env in this directory",
                    origin=origin,
                ))
            continue

        present = shutil.which(binary) is not None
        requirements.append(Requirement(
            label=f"{binary} CLI",
            ok=present,
            fix=install,
            origin=origin,
        ))
        if not present:
            continue

        requirements.extend(_auth_requirements(cli, cli_specs, origin))

    return requirements


def _auth_requirements(
    cli: str, specs: Sequence[SessionSpec], origin: str,
) -> List[Requirement]:
    """The credential a CLI needs, which depends on how the config asked
    it to authenticate — `auth_mode: api_key` wants a key in the
    environment, `oauth` wants the CLI's own stored login, `auto` takes
    either."""
    if cli == "codex":
        return [Requirement(
            label="codex authenticated",
            ok=codex_authenticated(),
            fix="codex login   (or set CODEX_API_KEY in .env)",
            origin=origin,
        )]

    if cli not in ("claude_code", "oss_claude_code"):
        return []

    modes = {spec.auth_mode for spec in specs}
    if modes == {"api_key"}:
        return [Requirement(
            label="ANTHROPIC_API_KEY",
            ok=bool(os.environ.get("ANTHROPIC_API_KEY")),
            fix="add ANTHROPIC_API_KEY=sk-ant-... to .env in this directory",
            origin=f"{origin}  (auth_mode: api_key)",
        )]

    logged_in = claude_logged_in()
    ok = logged_in or (
        "api_key" in modes or "auto" in modes
    ) and bool(os.environ.get("ANTHROPIC_API_KEY"))
    return [Requirement(
        label="claude authenticated",
        ok=bool(ok),
        fix="claude auth login   (or set CLAUDE_CODE_OAUTH_TOKEN in .env)",
        origin=origin,
    )]


def _mode_block(config: Dict[str, Any], mode: Optional[str]) -> Tuple[str, Dict]:
    """The active mode's name and config block."""
    name = mode or config.get("default_mode", "GENERIC")
    return name, (config.get("modes") or {}).get(name) or {}


def _embedding_requirement(origin: str) -> Requirement:
    return Requirement(
        label="OPENAI_API_KEY (embeddings)",
        ok=bool(os.environ.get("OPENAI_API_KEY")),
        fix="add OPENAI_API_KEY=sk-... to .env in this directory",
        origin=origin,
    )


def _kg_backend_requirements(origin: str) -> List[Requirement]:
    """The two stores a knowledge-graph read or write touches."""
    return [
        Requirement(
            label="Weaviate (localhost:8080)",
            ok=port_open("localhost", 8080),
            fix="bash scripts/start_infra.sh   (starts Weaviate + Neo4j in docker)",
            origin=origin,
        ),
        Requirement(
            label="Neo4j (localhost:7687)",
            ok=port_open("localhost", 7687),
            fix="bash scripts/start_infra.sh   (starts Weaviate + Neo4j in docker)",
            origin=origin,
        ),
    ]


def _git_requirement(origin: str) -> Requirement:
    return Requirement(
        label="git",
        ok=shutil.which("git") is not None,
        fix="install git: https://git-scm.com/downloads",
        origin=origin,
    )


# =============================================================================
# PER-VERB REQUIREMENTS
# =============================================================================

def research_requirements(config: Dict[str, Any]) -> List[Requirement]:
    """`research()` runs one web-search session — the `inference` block's
    default spec with the `research` role's overrides on top."""
    inference = config.get("inference") or {}
    spec = {**(inference.get("default") or {}),
            **((inference.get("roles") or {}).get("research") or {})}
    if not spec.get("model"):
        return []
    return cli_requirements([SessionSpec(
        cli=str(spec.get("cli") or "codex"),
        model=str(spec["model"]),
        origin=f"inference.roles.research = {spec['model']}",
        auth_mode=str(spec.get("auth_mode") or "auto"),
    )])


def learn_knowledge_requirements(
    config: Dict[str, Any],
    *,
    mode: Optional[str] = None,
    skip_merge: bool = False,
) -> List[Requirement]:
    """`learn_knowledge()` ingests sources into wiki pages, then merges
    them into the KG. `skip_merge=True` stops after extraction, which
    drops the merger session and both backend stores."""
    mode_name, block = _mode_block(config, mode)
    learner = block.get("learner") or {}
    prefix = f"modes.{mode_name}.learner"

    specs = session_specs(learner.get("ingestor") or {}, f"{prefix}.ingestor")
    if not skip_merge:
        specs += session_specs(learner.get("merger") or {}, f"{prefix}.merger")

    requirements = cli_requirements(specs)
    requirements.append(_git_requirement(
        "Source.Repo ingestion clones the repository"
    ))
    requirements.append(Requirement(
        label="GITHUB_PAT",
        ok=bool(os.environ.get("GITHUB_PAT")),
        fix="add GITHUB_PAT=ghp_... to .env — or expect the workflow-repo "
            "phase to fail or push to the wrong account",
        origin=f"{prefix}.ingestor publishes each extracted workflow as a repo",
        required=False,
    ))
    if not skip_merge:
        requirements.append(_embedding_requirement(
            "the merge embeds every wiki page"
        ))
        requirements.extend(_kg_backend_requirements(
            "the merge writes pages into the KG stores "
            "(pass skip_merge=True to extract only)"
        ))
    return requirements


def evolve_requirements(
    config: Dict[str, Any],
    *,
    mode: Optional[str] = None,
    coding_agent: Optional[str] = None,
    kg_index: Optional[str] = None,
    bank_home: Optional[Path] = None,
) -> List[Requirement]:
    """`evolve()` runs a campaign: ideation, implementation, evaluation and
    feedback sessions, plus whatever MCP gates the strategy names."""
    mode_name, block = _mode_block(config, mode)
    prefix = f"modes.{mode_name}"
    strategy = block.get("search_strategy") or {}
    params = strategy.get("params") or {}

    # A caller-passed coding_agent overrides the mode's `type`, so the
    # requirement must follow the agent that will actually run.
    agent_block = dict(block.get("coding_agent") or {})
    if coding_agent:
        agent_block["type"] = coding_agent

    specs = session_specs(params, f"{prefix}.search_strategy.params")
    specs += session_specs(agent_block, f"{prefix}.coding_agent")
    specs += session_specs(
        block.get("feedback_generator") or {}, f"{prefix}.feedback_generator"
    )

    requirements = cli_requirements(specs)
    requirements.append(_git_requirement(
        "the campaign workspace is a git repository"
    ))
    requirements.extend(_gate_requirements(params, prefix))

    if kg_index:
        requirements.append(_embedding_requirement(
            f"knowledge search — Kapso(kg_index={kg_index!r})"
        ))
        requirements.extend(_kg_backend_requirements(
            f"knowledge search — Kapso(kg_index={kg_index!r})"
        ))

    serving = ((config.get("learning") or {}).get("serving") or {})
    if serving.get("enabled") and bank_home is not None:
        requirements.append(Requirement(
            label=f"lesson bank at {bank_home}",
            ok=Path(bank_home).exists(),
            fix="run kapso.learn(...) once to create it, or "
                "set learning.serving.enabled: false",
            origin="learning.serving.enabled: true — campaigns are served bank cards",
            required=False,
        ))
    return requirements


def _gate_requirements(
    params: Dict[str, Any], prefix: str,
) -> List[Requirement]:
    """The MCP gates the strategy names, checked through the gate
    resolver that the campaign itself uses. Whether a missing gate BLOCKS
    is the config's own call: `gate_failure_policy: error` makes it
    required, `warn`/`skip` leave it advisory."""
    gates = list(dict.fromkeys(
        list(params.get("ideation_gates") or [])
        + list(params.get("implementation_gates") or [])
    ))
    if not gates:
        return []
    policy = str(params.get("gate_failure_policy") or "warn").strip().lower()
    # Resolve in skip mode and decide required-ness here, so a policy of
    # `error` yields a readable row rather than a bare capability raise.
    resolution = resolve_gates(gates, policy="skip")
    requirements = []
    for diagnostic in resolution.diagnostics:
        if diagnostic.enabled:
            continue
        definition = GATES[diagnostic.gate_name]
        # Vars Kapso sets when it launches the session are not the user's
        # to provide — reporting them here would be a guaranteed false
        # positive on every clean machine.
        missing_env = tuple(
            name for name in diagnostic.missing_env
            if name not in definition.injected_env
        )
        if not missing_env and not diagnostic.missing_commands:
            continue
        # The gate's own setup_hint when it has one; otherwise name exactly
        # what is missing. Under `warn`/`skip` the [warn] mark already says
        # the campaign proceeds, so only `error` — where the user may want
        # to downgrade the policy instead — earns an extra clause.
        remedy = definition.setup_hint
        if not remedy:
            parts = []
            if diagnostic.missing_commands:
                parts.append(f"install {', '.join(diagnostic.missing_commands)}")
            if missing_env:
                parts.append(f"set {', '.join(missing_env)} in .env")
            remedy = "; ".join(parts)
        if policy == "error":
            downgrade = "set gate_failure_policy: warn to run without it"
            fix = f"{remedy} — or {downgrade}" if remedy else downgrade
        else:
            fix = remedy
        reason = []
        if missing_env:
            reason.append(f"missing environment: {', '.join(missing_env)}")
        if diagnostic.missing_commands:
            reason.append(
                f"missing commands: {', '.join(diagnostic.missing_commands)}"
            )
        requirements.append(Requirement(
            label=f"MCP gate '{diagnostic.gate_name}'",
            ok=False,
            fix=fix,
            origin=f"{prefix}.search_strategy.params gates",
            required=policy == "error",
            detail="; ".join(reason),
        ))
    return requirements


def learn_requirements(
    config: Dict[str, Any],
    *,
    bank_home: Optional[Path] = None,
    push: Optional[bool] = None,
) -> List[Requirement]:
    """`learn()` runs four crews — mining, grading, the update crew, and
    (when a procedure is codified) the codify run. The grading crew is
    deliberately cross-model, so a default config genuinely needs both
    CLIs here even though `research()` needs one."""
    learning = config.get("learning") or {}

    specs = session_specs(learning.get("mining") or {}, "learning.mining")
    specs += session_specs(
        (learning.get("graders") or {}).get("crew") or {}, "learning.graders.crew"
    )
    specs += session_specs(learning.get("update_crew") or {}, "learning.update_crew")

    codify = learning.get("codify") or {}
    specs += session_specs(
        {k: codify[k] for k in ("implementor", "judge") if k in codify},
        "learning.codify",
    )

    requirements = cli_requirements(specs)
    requirements.append(_git_requirement(
        "every bank update clones and pushes the bank repo"
    ))

    if str(codify.get("target") or "") == "gcp_ephemeral":
        requirements.append(Requirement(
            label="gcloud CLI",
            ok=shutil.which("gcloud") is not None,
            fix="install gcloud (https://cloud.google.com/sdk/docs/install) "
                "and run `gcloud auth login`, or set "
                "learning.codify.target: local to evaluate on this machine",
            origin="learning.codify.target: gcp_ephemeral",
            required=False,
        ))

    requirements.extend(_bank_push_requirements(bank_home, push))
    return requirements


def _bank_push_requirements(
    bank_home: Optional[Path], push: Optional[bool],
) -> List[Requirement]:
    """The bank's share remote, verified before the crews start rather
    than after them (onboarding E2E finding #1: auth problems then cost
    seconds at the start, never hours at the end)."""
    if bank_home is None or not Path(bank_home).exists():
        return []
    origin_url = bank_origin(Path(bank_home))
    if not origin_url:
        if push:
            return [Requirement(
                label="bank origin remote",
                ok=False,
                fix="kapso bank connect <url>   (or kapso bank create <org>/<name>)",
                origin="push=True was requested",
            )]
        if push is None:
            # No caller decision — the doctor's view. Local-only is a valid
            # state, so this nudges rather than fails; learn() itself passes
            # a resolved bool and never reaches here, which keeps the nudge
            # out of every run's output.
            return [Requirement(
                label="bank remote (sharing)",
                ok=False,
                fix="kapso bank connect <url> — shares lessons across "
                    "machines and teammates",
                origin=f"{bank_home} has no origin remote — lessons stay local",
                required=False,
            )]
        return []
    if push is False:
        # An explicit push=False runs local-only, so an attached remote's
        # reachability is irrelevant — probing it here would block a run
        # that was never going to touch it.
        return []
    error = bank_remote_error(Path(bank_home), origin_url)
    return [Requirement(
        label=f"bank remote reachable ({origin_url})",
        ok=error is None,
        fix="fix git access (ssh key / gh auth / credential helper), or "
            f"detach it: git --git-dir {bank_home} remote remove origin",
        origin="the lesson is pushed here when learn() finishes",
        detail=error or "",
    )]


def deploy_requirements(
    config: Dict[str, Any],
    *,
    strategy: Optional[str] = None,
    coding_agent: str = "claude_code",
) -> List[Requirement]:
    """`deploy()` adapts the solution with a coding-agent session, then
    hands it to the target's runner.

    Three cases. A named target checks that target's tool. AUTO checks
    nothing extra — the target is chosen by the selector session, so
    demanding every tool up front would fail deploys that were only ever
    going to run locally. `strategy=None` is the doctor's view, with no
    call in flight: it maps which targets this machine could serve.
    """
    manifest = agent_manifest()
    model = (manifest.get(coding_agent) or {}).get("default_model") or coding_agent
    requirements = cli_requirements([SessionSpec(
        cli=coding_agent,
        model=str(model),
        origin=f"deploy(coding_agent={coding_agent!r}) adapts the solution",
    )])

    if strategy is None:
        for name, (binary, fix) in sorted(DEPLOY_TOOLS.items()):
            requirements.append(Requirement(
                label=f"deploy target {name}",
                ok=shutil.which(binary) is not None,
                fix=fix,
                origin=f"only needed for deploy(strategy=DeployStrategy."
                       f"{name.upper()})",
                required=False,
            ))
        return requirements

    target = str(strategy).lower()
    if target not in DEPLOY_TOOLS:
        return requirements

    binary, fix = DEPLOY_TOOLS[target]
    present = shutil.which(binary) is not None
    requirements.append(Requirement(
        label=f"{binary} (deploy target)",
        ok=present,
        fix=fix,
        origin=f"deploy(strategy={strategy})",
    ))
    if target == "docker" and present:
        daemon = subprocess.run(
            ["docker", "info"], capture_output=True, text=True, timeout=30,
        )
        requirements.append(Requirement(
            label="docker daemon running",
            ok=daemon.returncode == 0,
            fix="start Docker Desktop, or: sudo systemctl start docker",
            origin=f"deploy(strategy={strategy})",
        ))
    return requirements


_BUILDERS = {
    "research": research_requirements,
    "learn_knowledge": learn_knowledge_requirements,
    "evolve": evolve_requirements,
    "learn": learn_requirements,
    "deploy": deploy_requirements,
}


def configured_bank_home(config: Dict[str, Any]) -> Optional[Path]:
    """The bank home the config names, if any. The doctor resolves it from
    the config it was handed — never from the packaged default, which is
    how `--config` used to be ignored for the bank rows."""
    local_path = (
        ((config.get("learning") or {}).get("bank") or {}).get("local_path")
    )
    return Path(local_path).expanduser() if local_path else None


def requirements_for(
    verb: Optional[str], config: Dict[str, Any], **context: Any,
) -> List[Requirement]:
    """Static requirements for one verb, or — with `verb=None` — the union
    across all five, which is what a bare `kapso doctor` reports."""
    if verb is None:
        bank_home = configured_bank_home(config)
        merged: List[Requirement] = []
        for name in VERBS:
            merged.extend(
                _BUILDERS[name](config, bank_home=bank_home)
                if name in ("evolve", "learn")
                else _BUILDERS[name](config)
            )
        return dedupe(merged)
    if verb not in _BUILDERS:
        raise ValueError(
            f"unknown verb {verb!r}: expected one of {', '.join(VERBS)}"
        )
    return dedupe(_BUILDERS[verb](config, **context))


def dedupe(requirements: Sequence[Requirement]) -> List[Requirement]:
    """One row per label, keeping every origin that asked for it."""
    merged: Dict[str, Requirement] = {}
    for item in requirements:
        seen = merged.get(item.label)
        if seen is None:
            merged[item.label] = item
            continue
        origins = dict.fromkeys(seen.origin.split("; ") + item.origin.split("; "))
        merged[item.label] = Requirement(
            label=seen.label,
            ok=seen.ok and item.ok,
            fix=seen.fix,
            origin="; ".join(origins),
            required=seen.required or item.required,
            detail=seen.detail or item.detail,
        )
    return list(merged.values())


def live_model_requirements(
    verb: Optional[str], config: Dict[str, Any], **context: Any,
) -> List[Requirement]:
    """The live tier: one throwaway token per distinct {cli, model} pair
    the verb can spawn. Pairs whose CLI is missing are skipped — the
    static tier already reported that."""
    specs = _specs_for(verb, config, **context)
    probed: Dict[Tuple[str, str], List[str]] = {}
    for spec in specs:
        probed.setdefault((spec.cli, spec.model), []).append(spec.origin)

    requirements = []
    for (cli, model), origins in sorted(probed.items()):
        binary = CLI_BINARIES.get(cli)
        if binary is None or shutil.which(binary) is None:
            continue
        ok, detail = probe_model_access(cli, model)
        requirements.append(Requirement(
            label=f"{cli} can serve {model}",
            ok=ok,
            fix="copy the packaged config, change the model, and pass "
                "Kapso(config_path=...) — see README \"Choosing models\"",
            origin="; ".join(dict.fromkeys(origins)),
            detail=detail,
        ))
    return requirements


def _specs_for(
    verb: Optional[str], config: Dict[str, Any], **context: Any,
) -> List[SessionSpec]:
    """Every session spec a verb (or the whole config) can spawn. Built by
    re-walking the same subtrees the requirement builders use, so the live
    tier can never probe a model the static tier does not know about."""
    mode_name, block = _mode_block(config, context.get("mode"))
    learning = config.get("learning") or {}
    inference = config.get("inference") or {}
    research_spec = {**(inference.get("default") or {}),
                     **((inference.get("roles") or {}).get("research") or {})}
    agent_block = dict(block.get("coding_agent") or {})
    if context.get("coding_agent"):
        agent_block["type"] = context["coding_agent"]

    per_verb: Dict[str, List[SessionSpec]] = {
        "research": (
            [SessionSpec(
                cli=str(research_spec.get("cli") or "codex"),
                model=str(research_spec["model"]),
                origin=f"inference.roles.research = {research_spec['model']}",
            )] if research_spec.get("model") else []
        ),
        "learn_knowledge": session_specs(
            block.get("learner") or {}, f"modes.{mode_name}.learner"
        ),
        "evolve": (
            session_specs(
                (block.get("search_strategy") or {}).get("params") or {},
                f"modes.{mode_name}.search_strategy.params",
            )
            + session_specs(
                agent_block, f"modes.{mode_name}.coding_agent"
            )
            + session_specs(
                block.get("feedback_generator") or {},
                f"modes.{mode_name}.feedback_generator",
            )
        ),
        "learn": (
            session_specs(learning.get("mining") or {}, "learning.mining")
            + session_specs(
                (learning.get("graders") or {}).get("crew") or {},
                "learning.graders.crew",
            )
            + session_specs(learning.get("update_crew") or {}, "learning.update_crew")
            + session_specs(
                {k: v for k, v in (learning.get("codify") or {}).items()
                 if k in ("implementor", "judge")},
                "learning.codify",
            )
        ),
        "deploy": [],
    }
    if verb is None:
        return [spec for name in VERBS for spec in per_verb[name]]
    return per_verb[verb]


# =============================================================================
# RENDER + ENFORCE
# =============================================================================

def render(verb: Optional[str], requirements: Sequence[Requirement]) -> str:
    """The report a user reads. Failed rows first, each with the config key
    that asked for it and one copy-pasteable fix."""
    scope = f"kapso {verb}" if verb else "kapso"
    failures = [item for item in requirements if not item.ok and item.required]
    advisories = [item for item in requirements if not item.ok and not item.required]

    if not failures and not advisories:
        return (f"{scope} — preflight passed "
                f"({len(requirements)} requirements checked)")

    lines = []
    if failures:
        lines.append(
            f"{scope} — preflight failed "
            f"({len(failures)} of {len(requirements)} requirements missing)"
        )
    else:
        lines.append(
            f"{scope} — preflight passed with {len(advisories)} advisory item(s)"
        )
    for item in failures + advisories:
        mark = "FAIL" if item.required else "warn"
        lines.append("")
        lines.append(f"  [{mark}] {item.label}")
        if item.detail:
            lines.append(f"         reason     {item.detail}")
        lines.append(f"         needed by  {item.origin}")
        if item.fix:
            lines.append(f"         fix        {item.fix}")
    if verb:
        lines.append("")
        lines.append(f"Full report: kapso doctor {verb}")
    return "\n".join(lines)


def run_preflight(
    verb: str, config: Dict[str, Any], **context: Any,
) -> List[Requirement]:
    """Check a verb's requirements before it does any work; raise
    PreflightError when a required one is missing. Advisory rows print
    once and the verb continues."""
    settings = preflight_settings(config)
    if not settings.get("enabled"):
        return []

    requirements = requirements_for(verb, config, **context)
    if settings.get("live_model_probe"):
        requirements = dedupe(
            requirements + live_model_requirements(verb, config, **context)
        )

    if any(not item.ok and item.required for item in requirements):
        raise PreflightError(verb, requirements)
    if any(not item.ok for item in requirements):
        print(render(verb, requirements))
    return requirements
