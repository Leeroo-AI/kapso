"""The build's sessions by role over the hosts table (kapso.core.roles,
docs/plans/redesign/wp4-config.md): a role's own cli, model, effort and
timeout win over the host's; a role on a host the table lacks is refused
by name; `--coding-agent` sets the host every role without a cli of its
own follows; the unknown-parameter refusal; the preflight derives the
build's sessions from the same resolver and refuses a model its host
cannot serve."""

import pytest

from kapso.core.agent_manifest import load_agent_manifest
from kapso.core.config import ConfigError, load_config
from kapso.core.preflight import SessionSpec, build_params, build_session_specs, host_model_requirements, requirements_for
from kapso.core.roles import (
    CONFIG_PARAMS,
    ENGINE_PARAMS,
    ROLES,
    SessionRole,
    check_deployment,
    check_hosts,
    check_params,
    deployment_role,
    resolve_members,
    resolve_optional,
    resolve_role,
    resolve_spec,
)

HOSTS = {
    "claude_code": {"model": "claude-opus-5", "auth_mode": "oauth"},
    "codex": {"model": "gpt-5.6-sol", "effort": "xhigh"},
    "vertex_claude_code": {"model": "zai-org/glm-5.2-maas", "vertex_project": "p", "effort": "high"},
}


def params(**overrides):
    base = dict(load_config()["search_strategy"]["params"])
    base.update(overrides)
    return base


def test_a_role_takes_the_host_then_its_own_keys():
    p = params(host="claude_code", judge={"cli": "codex", "timeout": 900}, implementation={"gates": [], "effort": "low"})
    ideation = resolve_role(HOSTS, p, "ideation")
    implementation = resolve_role(HOSTS, p, "implementation")
    judge = resolve_role(HOSTS, p, "judge")
    assert (ideation.cli, ideation.model, ideation.effort, ideation.timeout) == ("claude_code", "claude-opus-5", None, None)
    assert ideation.settings == {"auth_mode": "oauth"}
    assert (implementation.cli, implementation.effort) == ("claude_code", "low")
    assert (judge.cli, judge.model, judge.effort, judge.timeout, judge.settings) == ("codex", "gpt-5.6-sol", "xhigh", 900.0, {})
    assert judge.label == "codex:gpt-5.6-sol"


def test_the_host_setting_routes_every_role_without_a_cli():
    p = params(host="codex", ideation={"gates": [], "cli": "claude_code"})
    assert resolve_role(HOSTS, p, "implementation").cli == "codex"
    assert resolve_role(HOSTS, p, "judge").cli == "codex"
    assert resolve_role(HOSTS, p, "ideation").cli == "claude_code"
    # The host entry's other keys are the adapter's own settings.
    vertex = resolve_role(HOSTS, params(host="vertex_claude_code"), "implementation")
    assert vertex.settings == {"vertex_project": "p"} and vertex.effort == "high"


def test_a_missing_host_or_model_is_refused_by_name():
    with pytest.raises(ValueError, match=r"search_strategy.params.judge.cli: gemini is not in hosts \(the hosts are claude_code, codex, vertex_claude_code\); add hosts.gemini"):
        resolve_role(HOSTS, params(judge={"cli": "gemini"}), "judge")
    # The default host is named by where it is set: the config's key, or
    # the flag that sets it.
    with pytest.raises(ValueError, match=r"search_strategy.params.host \(what --coding-agent sets\): gemini is not in hosts"):
        resolve_role(HOSTS, params(host="gemini"), "judge")
    with pytest.raises(ValueError, match="hosts.codex needs a model"):
        resolve_role({"codex": {"effort": "xhigh"}}, params(host="codex"), "judge")
    with pytest.raises(ValueError, match="search_strategy.params.host is required"):
        resolve_role(HOSTS, {"judge": {}}, "judge")
    with pytest.raises(ValueError, match="timeout must be a positive number"):
        resolve_role(HOSTS, params(judge={"timeout": 0}), "judge")
    # A nan lexically passes `<= 0` but would poison the session's deadline
    # arithmetic; infinity has no deadline meaning either.
    for timeout in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ConfigError, match=r"ideation\.timeout must be a positive number of seconds"):
            resolve_role(HOSTS, params(ideation={"timeout": timeout}), "ideation")
    assert resolve_role(HOSTS, params(ideation={"timeout": 900}), "ideation").timeout == 900.0


def test_members_selector_and_planner_are_specs_of_their_own():
    members = resolve_members(HOSTS, params(host="claude_code"), [{"cli": "codex"}, {"model": "claude-fable-5", "lens": "data"}])
    assert [m.label for m in members] == ["codex:gpt-5.6-sol", "claude_code:claude-fable-5"]
    assert members[1].lens == "data" and members[0].lens is None and members[0].name == "member 1"
    assert resolve_members(HOSTS, params(), None) is None
    with pytest.raises(ValueError, match="ensemble must be a non-empty list"):
        resolve_members(HOSTS, params(), [])
    with pytest.raises(ValueError, match=r"ideation.ensemble\[0\]: unknown key base_url"):
        resolve_members(HOSTS, params(), [{"cli": "codex", "base_url": "x"}])
    planner = resolve_optional(HOSTS, params(), {"cli": "claude_code", "effort": "max", "timeout": 600}, name="lens_planner")
    assert (planner.name, planner.effort, planner.timeout) == ("lens_planner", "max", 600.0)
    assert resolve_optional(HOSTS, params(), None, name="selector") is None


def test_the_parameters_are_a_known_set():
    check_params(params())
    with pytest.raises(ValueError, match="search_strategy.params: unknown key implementation_web; the keys are"):
        check_params(params(implementation_web=False))
    with pytest.raises(ValueError, match="unknown keys design_axes, idea_generation_model"):
        check_params(params(idea_generation_model="m", design_axes=["a"]))
    with pytest.raises(ValueError, match="search_strategy.params.ideation: unknown key web_search"):
        check_params(params(ideation={"gates": [], "web_search": False}))
    with pytest.raises(ValueError, match="search_strategy.params.judge must be a mapping"):
        check_params(params(judge="codex"))
    # What the engine threads at launch is allowed only for the engine.
    with pytest.raises(ValueError, match="unknown key bank_serving"):
        check_params(params(bank_serving={}))
    check_params(params(bank_serving={}), engine=True)
    assert not (CONFIG_PARAMS & ENGINE_PARAMS)


def test_the_preflight_names_the_sessions_a_build_runs():
    config = load_config()
    specs = build_session_specs(config, build_params(config, "codex"))
    assert [spec.origin for spec in specs] == [
        f"search_strategy.params.{role} = codex:gpt-5.6-sol" for role in ROLES
    ]
    config = dict(config, evaluation_maintainer={**config["evaluation_maintainer"], "enabled": True, "cli": "claude_code"})
    p = build_params(config, None)
    p["ideation"] = dict(p["ideation"], ensemble=[{"cli": "codex"}], selector={"cli": "claude_code"})
    origins = [spec.origin for spec in build_session_specs(config, p)]
    assert "search_strategy.params.ideation.ensemble[0] = codex:gpt-5.6-sol" in origins
    # The members' sessions replace the ideation role's own: no row asks
    # for its host.
    assert not [origin for origin in origins if origin.startswith("search_strategy.params.ideation =")]
    assert "search_strategy.params.ideation.selector = claude_code:claude-opus-5" in origins
    assert origins[-1] == "evaluation_maintainer = claude_code:claude-opus-5"


def test_a_model_its_host_cannot_serve_fails_the_preflight_by_key():
    config = load_config()
    config = dict(config, hosts={**config["hosts"], "codex": {"model": "claude-opus-5"}})
    rows = {row.label: row for row in host_model_requirements(build_session_specs(config, build_params(config, "codex")))}
    row = rows["codex is asked for claude-opus-5"]
    assert row.ok is False and "search_strategy.params.ideation" in row.fix and row.detail == "codex cannot serve claude-opus-5"
    assert row.origin.startswith("search_strategy.params.ideation; search_strategy.params.implementation")
    ok = host_model_requirements([SessionSpec(cli="claude_code", model="claude-opus-5", origin="search_strategy.params.judge = x")])
    assert ok[0].ok is True
    # The other adapters serve what their endpoint does: no row.
    assert host_model_requirements([SessionSpec(cli="vertex_claude_code", model="zai-org/glm-5.2-maas", origin="o = x")]) == []
    labels = [row.label for row in requirements_for("build", load_config(), coding_agent="codex")]
    assert "codex is asked for gpt-5.6-sol" in labels and "claude_code is asked for claude-opus-5" not in labels


def test_resolve_spec_keeps_a_members_own_model_over_the_hosts():
    spec = resolve_spec(HOSTS, "codex", {"model": "gpt-5.5", "effort": "low"}, name="x", origin="o")
    assert (spec.cli, spec.model, spec.effort) == ("codex", "gpt-5.5", "low")


def test_a_host_with_no_staging_and_a_spec_of_the_wrong_shape_are_refused():
    """Only the adapters with a staging can host a build session (the
    registry's `host`): `hosts.gemini` is refused with the hosts a build
    runs on. A spec's cli, model and effort are strings: a list or a
    `false` is refused rather than read as unset."""
    adapters = load_agent_manifest()["agents"]
    check_hosts(HOSTS, adapters)
    with pytest.raises(ConfigError, match="hosts.gemini: the gemini adapter cannot host a build session .* claude_code, codex, oss_claude_code, vertex_claude_code"):
        check_hosts({**HOSTS, "gemini": {"model": "gemini-2.5-flash"}}, adapters)
    with pytest.raises(ConfigError, match=r"judge.cli must be a non-empty string, got \['codex'\]"):
        resolve_role(HOSTS, params(judge={"cli": ["codex"]}), "judge")
    with pytest.raises(ConfigError, match="implementation.model must be a non-empty string, got False"):
        resolve_role(HOSTS, params(implementation={"model": False}), "implementation")


def test_the_deployment_block_is_a_spec_over_the_hosts_following_the_builds_host():
    """`deployment.cli: null` follows search_strategy.params.host, so a
    project that moves to Codex changes one key; the block's own model
    wins over the host's; the call's host wins over both."""
    config = load_config()
    role = deployment_role(config)
    assert (role.cli, role.model, role.effort, role.timeout) == ("claude_code", "claude-opus-5", None, 1800.0)
    assert deployment_role(config, "codex").cli == "codex"
    config["search_strategy"]["params"]["host"] = "codex"
    assert deployment_role(config).label == "codex:gpt-5.6-sol"
    config["deployment"].update(cli="claude_code", model="claude-sonnet-5", effort="low")
    assert (deployment_role(config).model, deployment_role(config).effort) == ("claude-sonnet-5", "low")
    check_deployment(config)


@pytest.mark.parametrize("block, message", [
    ({"cli": "gemini"}, r"deployment\.cli: gemini is not in hosts"),
    ({"target": ""}, r"deployment\.target must be a non-empty string"),
    ({"timeout": -1}, r"deployment\.timeout must be a positive number of seconds"),
    ({"call_timeout": "soon"}, r"deployment\.call_timeout must be a positive number of seconds"),
    ({"call_timeout": None}, r"deployment\.call_timeout must be a positive number of seconds"),
    ({"script_timeout": float("inf")}, r"deployment\.script_timeout must be a positive number of seconds"),
    ({"port_span": 0}, r"deployment\.port_span must be a positive integer"),
    ({"port_base": 65000, "port_span": 1000}, r"deployment\.port_base \+ port_span must stay within the ports \(1\.\.65535\)"),
    ({"coding_agent": "claude_code"}, r"deployment: unknown key coding_agent; the keys are"),
])
def test_the_deployment_block_is_refused_by_key(block, message):
    config = load_config()
    config["deployment"].update(block)
    with pytest.raises(ConfigError, match=message):
        check_deployment(config)
