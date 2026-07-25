import copy

import pytest

from kapso.core.config import (
    load_config,
    load_effective_config,
    load_mode_config,
)
from kapso.cross_run.canonical import content_id
from kapso.cross_run.contracts import (
    ContractValidationError,
    CrossRunTaskBindingSettings,
    IdentityConflictError,
)
from kapso.cross_run.settings import (
    CrossRunConfigurationError,
    CrossRunSettings,
    compose_runtime_config,
    validate_runtime_registry,
)

CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"


def test_shipped_cross_run_config_is_strict_and_single_sourced():
    raw = load_config(CANONICAL_CONFIG_PATH)
    settings = CrossRunSettings.from_dict(raw["cross_run"])
    repositories = settings.scopes.resolve("ml_ai")

    assert repositories.expert_repository == "Leeroo-AI/kapso-expert"
    assert repositories.knowledge_repository == "Leeroo-AI/kapso-knowledge"
    assert repositories.security_repository == "Leeroo-AI/kapso-security"
    assert settings.to_dict() == raw["cross_run"]
    assert "api_key" not in str(settings.to_dict()).lower()
    assert "oauth_token" not in str(settings.to_dict()).lower()


def test_execution_journal_bound_contains_its_complete_spawn_authority():
    raw = copy.deepcopy(load_config(CANONICAL_CONFIG_PATH)["cross_run"])
    raw["expert"]["validation"]["policy"][
        "task_evaluation_journal_event_byte_limit"
    ] = 1

    with pytest.raises(CrossRunConfigurationError, match="cannot contain"):
        CrossRunSettings.from_dict(raw)


def test_catalog_agents_and_admission_policy_are_fully_typed():
    settings = CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    ).catalog

    assert settings.claim_proposer.cli == "codex"
    assert settings.claim_proposer.model == "gpt-5.6-sol"
    assert settings.claim_proposer_id == "catalog_claim_proposer"
    assert tuple(reviewer.agent.effort for reviewer in settings.reviewers) == (
        "xhigh",
        "xhigh",
    )
    assert settings.admission.required_approvals == len(settings.reviewers)
    assert settings.configuration_fingerprint.startswith("sha256:")


def test_expert_proposers_and_trigger_policy_are_fully_typed():
    cross_run_settings = CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    )
    settings = cross_run_settings.expert

    assert settings.architect.cli == "claude_code"
    assert settings.architect.model == "fable"
    assert settings.architect.effort == "xhigh"
    assert settings.generalizer.cli == "codex"
    assert settings.generalizer.model == "gpt-5.6-sol"
    assert settings.composition_policy_version == "kapso.expert_composition.v1"
    assert settings.composition_source_limit == 16
    assert settings.recovery_lineage_limit == 1000
    assert settings.workspace_path == ".kapso/cross_run/expert_workspaces"
    assert settings.triggers.inspector_id == "expert_trigger_inspector"
    assert settings.triggers.inspection_policy_version == "kapso.expert_inspection.v1"
    assert settings.triggers.minimum_success_contexts == 2
    assert (
        settings.task_adapters.active_authority.authority_id
        == "kapso_task_adapter_authority"
    )
    assert settings.task_adapters.zstd_window_size_bytes == (
        cross_run_settings.github.zstd_window_size_bytes
    )
    replay_provider = settings.validation.task_evaluation_provider
    docker_runtime = cross_run_settings.docker
    assert replay_provider.runtime is docker_runtime
    assert replay_provider.to_dict()["runtime"] == docker_runtime.to_dict()
    assert (
        "runtime"
        not in cross_run_settings.to_dict()["expert"]["validation"][
            "task_evaluation_provider"
        ]
    )
    assert docker_runtime.runtime_executable_path == "/usr/bin/docker"
    assert docker_runtime.runtime_socket_path == "/run/docker.sock"
    assert docker_runtime.helper_executable_path == "/usr/bin/busybox"
    assert docker_runtime.helper_executable_digest == (
        "sha256:dbac288c29ba568459550a2da9e7ae0ded6b1fc728ee9fad3044c44e62d6ac14"
    )
    assert docker_runtime.runtime_server_version == "29.1.3"
    assert docker_runtime.runtime_root_directory == "/var/lib/docker"
    assert docker_runtime.runtime_cgroup_driver == "systemd"
    assert docker_runtime.required_security_options == (
        "name=apparmor",
        "name=cgroupns",
        "name=seccomp,profile=builtin",
    )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda expert: expert.__setitem__("workspace_path", expert["candidate_path"]),
        lambda expert: expert.__setitem__(
            "workspace_path", expert["agent_artifact_path"] + "/nested"
        ),
        lambda expert: expert["task_adapters"].__setitem__(
            "state_path", expert["validation"]["state_path"]
        ),
        lambda expert: expert["validation"]["task_evaluation_provider"].__setitem__(
            "workspace_path", expert["validation"]["state_path"]
        ),
    ),
)
def test_expert_state_paths_must_be_disjoint(mutate):
    raw = copy.deepcopy(load_config(CANONICAL_CONFIG_PATH)["cross_run"])
    mutate(raw["expert"])

    with pytest.raises(CrossRunConfigurationError, match="must be disjoint"):
        CrossRunSettings.from_dict(raw)


@pytest.mark.parametrize(
    ("field_path", "invalid_value", "message"),
    (
        (("docker", "runtime_executable_path"), "usr/bin/docker", "must be absolute"),
        (("docker", "helper_executable_path"), "usr/bin/busybox", "must be absolute"),
        (
            (
                "docker",
                "helper_executable_digest",
            ),
            "sha256:wrong",
            "sha256 digest",
        ),
        (
            ("docker", "required_security_options"),
            ["name=seccomp", "name=apparmor"],
            "sorted and unique",
        ),
        (("docker", "cleanup_timeout_seconds"), 61, "exceeds"),
    ),
)
def test_docker_and_task_evaluation_provider_authorities_are_strict(
    field_path,
    invalid_value,
    message,
):
    raw = copy.deepcopy(load_config(CANONICAL_CONFIG_PATH)["cross_run"])
    target = raw
    for part in field_path[:-1]:
        target = target[part]
    target[field_path[-1]] = invalid_value

    with pytest.raises(CrossRunConfigurationError, match=message):
        CrossRunSettings.from_dict(raw)


def test_raw_provider_cannot_override_derived_docker_runtime():
    raw = copy.deepcopy(load_config(CANONICAL_CONFIG_PATH)["cross_run"])
    raw["expert"]["validation"]["task_evaluation_provider"]["runtime"] = copy.deepcopy(
        raw["docker"]
    )

    with pytest.raises(CrossRunConfigurationError, match="derived"):
        CrossRunSettings.from_dict(raw)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda validation: validation.__setitem__(
            "source_replay_provider",
            validation.pop("task_evaluation_provider"),
        ),
        lambda validation: validation["policy"].__setitem__(
            "source_replay_cpu_millicore_limit",
            validation["policy"].pop("task_evaluation_cpu_millicore_limit"),
        ),
    ),
)
def test_legacy_source_replay_execution_config_is_rejected(mutate):
    raw = copy.deepcopy(load_config(CANONICAL_CONFIG_PATH)["cross_run"])
    mutate(raw["expert"]["validation"])

    with pytest.raises((ContractValidationError, CrossRunConfigurationError)):
        CrossRunSettings.from_dict(raw)


def test_task_evaluation_cpu_quota_must_be_exact():
    raw = copy.deepcopy(load_config(CANONICAL_CONFIG_PATH)["cross_run"])
    raw["expert"]["validation"]["task_evaluation_provider"][
        "cpu_period_microseconds"
    ] = 99999
    raw["expert"]["validation"]["policy"]["task_evaluation_cpu_millicore_limit"] = 8001

    with pytest.raises(CrossRunConfigurationError, match="exact runtime quota"):
        CrossRunSettings.from_dict(raw)


def test_task_evaluation_provider_command_must_contain_graceful_stop():
    raw = copy.deepcopy(load_config(CANONICAL_CONFIG_PATH)["cross_run"])
    raw["docker"]["command_timeout_seconds"] = raw["expert"]["validation"]["policy"][
        "task_evaluation_termination_grace_seconds"
    ]
    raw["docker"]["cleanup_timeout_seconds"] = raw["expert"]["validation"]["policy"][
        "task_evaluation_termination_grace_seconds"
    ]

    with pytest.raises(CrossRunConfigurationError, match="contain graceful stop"):
        CrossRunSettings.from_dict(raw)


@pytest.mark.parametrize("stage", ("source_run_replay", "release_matrix"))
def test_task_evaluation_grace_must_fit_each_task_evaluator(stage):
    raw = copy.deepcopy(load_config(CANONICAL_CONFIG_PATH)["cross_run"])
    policy = raw["expert"]["validation"]["policy"]
    evaluator = next(item for item in policy["evaluators"] if item["stage"] == stage)
    evaluator["timeout_seconds"] = (
        policy["task_evaluation_termination_grace_seconds"] - 1
    )

    with pytest.raises(CrossRunConfigurationError, match="exceeds a leg timeout"):
        CrossRunSettings.from_dict(raw)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda catalog: catalog["claim_proposer"].__setitem__("effort", "max"),
        lambda catalog: catalog["claim_proposer"].__setitem__(
            "allowed_tools", ["Write"]
        ),
        lambda catalog: catalog.__setitem__(
            "agent_artifact_path", catalog["state_path"] + "/agent_calls"
        ),
        lambda catalog: catalog["reviewers"].__setitem__(
            1, copy.deepcopy(catalog["reviewers"][0])
        ),
        lambda catalog: catalog["admission"].__setitem__(
            "required_approvals", len(catalog["reviewers"]) + 1
        ),
        lambda catalog: catalog["admission"].__setitem__(
            "rejection_judgment", catalog["admission"]["approval_judgment"]
        ),
        lambda catalog: catalog.__setitem__(
            "claim_proposer_id", catalog["reviewers"][0]["reviewer_id"]
        ),
    ],
)
def test_invalid_catalog_agent_or_admission_configuration_fails(mutate):
    raw = copy.deepcopy(load_config(CANONICAL_CONFIG_PATH)["cross_run"])
    mutate(raw["catalog"])

    with pytest.raises((ContractValidationError, CrossRunConfigurationError)):
        CrossRunSettings.from_dict(raw)


def test_effective_config_retains_registry_without_polluting_workload_mode():
    effective = load_effective_config(CANONICAL_CONFIG_PATH, "GENERIC")
    mode = load_mode_config(CANONICAL_CONFIG_PATH, "GENERIC")

    assert effective.cross_run is not None
    assert (
        effective.registry_source_fingerprint == effective.cross_run.scopes.fingerprint
    )
    assert mode["search_strategy"] == effective.mode["search_strategy"]
    assert "cross_run" not in mode
    assert "cross_run_registry_fingerprint" not in mode


def test_runtime_composition_copies_registry_and_fingerprint_once():
    canonical = load_config(CANONICAL_CONFIG_PATH)
    workload = {
        "default_mode": "POSTTRAIN",
        "modes": {
            "POSTTRAIN": {
                "cross_run_binding": {
                    "scope_id": "ml_ai",
                    "task_family_id": "language_model_post_training",
                    "task_adapter_id": "posttrain",
                },
                "search_strategy": {"type": "generic", "params": {}},
            }
        },
    }

    runtime = compose_runtime_config(canonical, workload)
    canonical_settings = CrossRunSettings.from_dict(canonical["cross_run"])
    validate_runtime_registry(runtime, canonical_settings)

    assert runtime["cross_run"] == canonical["cross_run"]
    assert runtime["cross_run_registry_fingerprint"] == (
        canonical_settings.scopes.fingerprint
    )
    assert "repositories" not in runtime["modes"]["POSTTRAIN"]


def test_semantic_retrieval_weight_is_derived_from_its_single_configured_knob():
    settings = CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    )

    assert settings.knowledge.retrieval.lexical_weight == 0.35
    assert settings.knowledge.retrieval.semantic_weight == 0.65


@pytest.mark.parametrize(
    "binding",
    [
        {
            "scope_id": "ml_ai",
            "task_family_id": "language_model_post_training",
            "task_adapter_id": "posttrain",
            "expert_repository": "Leeroo-AI/kapso-expert",
        },
        {
            "scope_id": "unknown_scope",
            "task_family_id": "language_model_post_training",
            "task_adapter_id": "posttrain",
        },
    ],
)
def test_runtime_composition_validates_each_task_binding(binding):
    canonical = load_config(CANONICAL_CONFIG_PATH)
    workload = {
        "default_mode": "X",
        "modes": {"X": {"cross_run_binding": binding}},
    }

    with pytest.raises((ContractValidationError, CrossRunConfigurationError)):
        compose_runtime_config(canonical, workload)


@pytest.mark.parametrize(
    "override",
    [
        {"cross_run": {}},
        {"cross_run_registry_fingerprint": "sha256:" + "0" * 64},
        {"repositories": {"expert": "Leeroo-AI/kapso-expert"}},
        {"expert_repository": "Leeroo-AI/kapso-expert"},
        {"security_repository": "Leeroo-AI/kapso-security"},
        {"some_value": "Leeroo-AI/kapso-knowledge"},
        {"some_value": "Leeroo-AI/kapso-security"},
    ],
)
def test_workload_cannot_override_or_duplicate_repository_routing(override):
    canonical = load_config(CANONICAL_CONFIG_PATH)
    workload = {"default_mode": "X", "modes": {"X": override}}

    with pytest.raises(CrossRunConfigurationError):
        compose_runtime_config(canonical, workload)


def test_runtime_registry_rejects_stale_source_fingerprint_and_modified_copy():
    canonical = load_config(CANONICAL_CONFIG_PATH)
    canonical_settings = CrossRunSettings.from_dict(canonical["cross_run"])
    workload = {"default_mode": "X", "modes": {"X": {}}}
    runtime = compose_runtime_config(canonical, workload)

    stale = copy.deepcopy(runtime)
    stale["cross_run_registry_fingerprint"] = "sha256:" + "0" * 64
    with pytest.raises(CrossRunConfigurationError):
        validate_runtime_registry(stale, canonical_settings)

    modified = copy.deepcopy(runtime)
    modified["cross_run"]["scopes"]["ml_ai"]["repositories"][
        "expert"
    ] = "Leeroo-AI/different-expert"
    with pytest.raises(CrossRunConfigurationError):
        validate_runtime_registry(modified, canonical_settings)


@pytest.mark.parametrize(
    "runtime",
    [
        {"cross_run": {}},
        {"cross_run_registry_fingerprint": "sha256:" + "0" * 64},
    ],
)
def test_runtime_registry_requires_both_registry_fields(runtime):
    canonical = load_config(CANONICAL_CONFIG_PATH)
    settings = CrossRunSettings.from_dict(canonical["cross_run"])

    with pytest.raises(CrossRunConfigurationError):
        validate_runtime_registry(runtime, settings)


def test_registry_rejects_aliases_duplicate_pairs_and_duplicate_ownership():
    raw = load_config(CANONICAL_CONFIG_PATH)["cross_run"]

    alias = copy.deepcopy(raw)
    alias["scopes"]["ml_ai"]["repositories"]["knowledge"] = alias["scopes"]["ml_ai"][
        "repositories"
    ]["expert"]
    with pytest.raises(IdentityConflictError):
        CrossRunSettings.from_dict(alias)

    duplicate = copy.deepcopy(raw)
    duplicate["scopes"]["second"] = copy.deepcopy(duplicate["scopes"]["ml_ai"])
    with pytest.raises(IdentityConflictError):
        CrossRunSettings.from_dict(duplicate)

    shared_repository = copy.deepcopy(raw)
    shared_repository["scopes"]["second"] = {
        "repositories": {
            "expert": "Leeroo-AI/second-expert",
            "knowledge": "Leeroo-AI/kapso-knowledge",
            "security": "Leeroo-AI/second-security",
        }
    }
    with pytest.raises(IdentityConflictError):
        CrossRunSettings.from_dict(shared_repository)


def test_task_binding_has_exact_three_fields_and_unknown_scope_fails():
    binding = CrossRunTaskBindingSettings.from_dict(
        {
            "scope_id": "ml_ai",
            "task_family_id": "language_model_post_training",
            "task_adapter_id": "posttrain",
        }
    )
    assert binding.scope_id == "ml_ai"

    with pytest.raises(ContractValidationError):
        CrossRunTaskBindingSettings.from_dict(
            {
                **binding.to_dict(),
                "expert_repository": "Leeroo-AI/kapso-expert",
            }
        )

    settings = CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    )
    with pytest.raises(CrossRunConfigurationError):
        settings.scopes.resolve("unknown")


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("github", "command_timeout_seconds"), 0),
        (("github", "control_blob_size_bytes"), 0),
        (("github", "content_write_budget_per_minute"), 1),
        (("github", "content_write_budget_per_minute"), 34),
        (("github", "git_tree_request_size_bytes"), 0),
        (("github", "release_asset_count_limit"), 256),
        (("github", "request_point_budget_per_minute"), 1),
        (("github", "request_point_budget_per_minute"), 747),
        (("github", "source_entry_limit"), 0),
        (("github", "source_entry_limit"), 100000),
        (("github", "source_tree_size_bytes"), 0),
        (("github", "zstd_window_size_bytes"), 1),
        (("github", "zstd_window_size_bytes"), 1023),
        (("launch", "knowledge_snapshot_file_size_bytes"), 0),
        (("capture", "git_command_timeout_seconds"), 0),
        (("capture", "git_command_output_bytes"), 1),
        (("capture", "bundle_lineage_limit"), 0),
        (("capture", "score_comparison_tolerance"), 0.0),
        (("capture", "state_path"), "/tmp/absolute-capture"),
        (("capture", "quarantine_path"), "/tmp/absolute-quarantine"),
        (("capture", "checkpoint_path"), "/tmp/run-state.json"),
        (("capture", "experiment_history_path"), "../history.json"),
        (("capture", "journal_filename"), "nested/events.jsonl"),
        (("knowledge", "retrieval", "lexical_weight"), 1.1),
        (("knowledge", "embeddings", "dimensions"), True),
        (("expert", "validation", "reviewer_count"), 0),
        (
            (
                "expert",
                "validation",
                "policy",
                "task_evaluation_termination_grace_seconds",
            ),
            0,
        ),
        (
            (
                "expert",
                "validation",
                "policy",
                "task_evaluation_cpu_millicore_limit",
            ),
            True,
        ),
        (
            (
                "expert",
                "validation",
                "policy",
                "task_evaluation_memory_byte_limit",
            ),
            0,
        ),
        (
            (
                "expert",
                "validation",
                "policy",
                "task_evaluation_shared_memory_byte_limit",
            ),
            0,
        ),
        (
            (
                "expert",
                "validation",
                "policy",
                "task_evaluation_process_limit",
            ),
            0,
        ),
        (
            (
                "expert",
                "validation",
                "policy",
                "task_evaluation_open_file_limit",
            ),
            0,
        ),
        (
            (
                "expert",
                "validation",
                "policy",
                "task_evaluation_writable_inode_limit",
            ),
            0,
        ),
        (
            (
                "expert",
                "validation",
                "policy",
                "task_evaluation_writable_storage_byte_limit",
            ),
            0,
        ),
        (
            (
                "expert",
                "validation",
                "policy",
                "task_evaluation_stdout_byte_limit",
            ),
            0,
        ),
        (
            (
                "expert",
                "validation",
                "policy",
                "task_evaluation_stderr_byte_limit",
            ),
            0,
        ),
        (
            (
                "expert",
                "validation",
                "policy",
                "task_evaluation_accelerator_count",
            ),
            True,
        ),
        (("expert", "validation", "policy", "source_replay_episode_limit"), 0),
        (("expert", "validation", "policy", "source_replay_bundle_limit"), 0),
        (
            (
                "expert",
                "validation",
                "policy",
                "task_evaluation_materialization_entry_limit",
            ),
            0,
        ),
        (
            (
                "expert",
                "validation",
                "policy",
                "task_evaluation_aggregate_tolerance",
            ),
            0,
        ),
        (
            (
                "expert",
                "validation",
                "policy",
                "task_evaluation_materialization_byte_limit",
            ),
            0,
        ),
        (
            (
                "expert",
                "validation",
                "policy",
                "task_evaluation_materialization_timeout_seconds",
            ),
            0,
        ),
        (("expert", "task_adapters", "zstd_window_size_bytes"), 0),
        (("launch", "cache_path"), "../escape"),
        (("launch", "compatibility_policy_version"), "invalid policy"),
        (("launch", "security_denylist_checkpoint_size_bytes"), 0),
        (("launch", "security_denylist_checked_subject_limit"), 0),
        (("launch", "security_denylist_checked_subject_size_bytes"), 0),
    ],
)
def test_invalid_operational_values_fail_before_external_work(path, value):
    raw = copy.deepcopy(load_config(CANONICAL_CONFIG_PATH)["cross_run"])
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises((ContractValidationError, CrossRunConfigurationError)):
        CrossRunSettings.from_dict(raw)


def test_zstd_window_configuration_uses_decoder_byte_units():
    raw = copy.deepcopy(load_config(CANONICAL_CONFIG_PATH)["cross_run"])
    raw["github"]["zstd_window_size_bytes"] = 1024

    settings = CrossRunSettings.from_dict(raw)

    assert settings.github.zstd_window_size_bytes == 1024


@pytest.mark.parametrize(
    "mutate",
    (
        lambda policy: policy.__setitem__(
            "task_evaluation_shared_memory_byte_limit",
            policy["task_evaluation_memory_byte_limit"] + 1,
        ),
        lambda policy: policy.__setitem__(
            "artifact_entry_limit",
            policy["task_evaluation_writable_inode_limit"],
        ),
        lambda policy: policy.__setitem__(
            "task_evaluation_result_byte_limit",
            policy["artifact_byte_limit"] + 1,
        ),
        lambda policy: policy.__setitem__(
            "task_evaluation_accelerator_count",
            1,
        ),
        lambda policy: policy.__setitem__(
            "task_evaluation_accelerator_class_id",
            "h100",
        ),
        lambda policy: policy.__setitem__(
            "task_evaluation_termination_grace_seconds",
            next(
                evaluator["timeout_seconds"]
                for evaluator in policy["evaluators"]
                if evaluator["stage"] == "source_run_replay"
            )
            + 1,
        ),
    ),
)
def test_source_replay_compute_policy_rejects_inconsistent_limits(mutate):
    raw = copy.deepcopy(load_config(CANONICAL_CONFIG_PATH)["cross_run"])
    policy = raw["expert"]["validation"]["policy"]
    mutate(policy)

    with pytest.raises(CrossRunConfigurationError):
        CrossRunSettings.from_dict(raw)


def test_source_replay_must_use_the_capture_projection_tolerance():
    raw = copy.deepcopy(load_config(CANONICAL_CONFIG_PATH)["cross_run"])
    raw["expert"]["validation"]["policy"]["task_evaluation_aggregate_tolerance"] *= 2

    with pytest.raises(
        CrossRunConfigurationError,
        match="aggregate tolerances must match",
    ):
        CrossRunSettings.from_dict(raw)


@pytest.mark.parametrize(
    ("state_path", "quarantine_path"),
    [
        (".kapso/cross_run", ".kapso/cross_run"),
        (".kapso/cross_run", ".kapso/cross_run/quarantine"),
        (".kapso/cross_run/capture", ".kapso/cross_run"),
    ],
)
def test_capture_state_and_quarantine_paths_are_disjoint(state_path, quarantine_path):
    raw = copy.deepcopy(load_config(CANONICAL_CONFIG_PATH)["cross_run"])
    raw["capture"]["state_path"] = state_path
    raw["capture"]["quarantine_path"] = quarantine_path

    with pytest.raises(CrossRunConfigurationError):
        CrossRunSettings.from_dict(raw)


@pytest.mark.parametrize(
    ("checkpoint_path", "history_path", "quarantine_path"),
    [
        (".kapso/state.json", ".kapso/state.json", ".kapso/quarantine"),
        (
            ".kapso/quarantine/state.json",
            ".kapso/history.json",
            ".kapso/quarantine",
        ),
        (
            ".kapso/state.json",
            ".kapso/quarantine/history.json",
            ".kapso/quarantine",
        ),
    ],
)
def test_capture_authority_paths_are_distinct_and_outside_quarantine(
    checkpoint_path,
    history_path,
    quarantine_path,
):
    raw = copy.deepcopy(load_config(CANONICAL_CONFIG_PATH)["cross_run"])
    raw["capture"]["checkpoint_path"] = checkpoint_path
    raw["capture"]["experiment_history_path"] = history_path
    raw["capture"]["quarantine_path"] = quarantine_path

    with pytest.raises(CrossRunConfigurationError):
        CrossRunSettings.from_dict(raw)


def test_github_rate_budget_scales_with_failed_upload_recovery():
    raw = copy.deepcopy(load_config(CANONICAL_CONFIG_PATH)["cross_run"])
    raw["github"]["release_asset_count_limit"] = 17
    raw["github"]["content_write_budget_per_minute"] = 43

    with pytest.raises(CrossRunConfigurationError, match="content-write budget"):
        CrossRunSettings.from_dict(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("commit_author_name", "unsafe\nname"),
        ("commit_author_name", "unsafe <name>"),
        ("commit_author_email", "unsafe<author@example.com"),
    ],
)
def test_github_commit_identity_rejects_git_header_injection(field, value):
    raw = copy.deepcopy(load_config(CANONICAL_CONFIG_PATH)["cross_run"])
    raw["github"][field] = value

    with pytest.raises(CrossRunConfigurationError):
        CrossRunSettings.from_dict(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("workspace_path", "/absolute/workspace"),
        ("knowledge_snapshot_path", "workspace/knowledge"),
        ("task_adapter_path", "readonly"),
        ("launch_manifest_path", "workspace/launch.json"),
        ("bootstrap_pin_path", "readonly/knowledge/pin.json"),
        (
            "bootstrap_pin_path",
            ".kapso/launch_manifest.json/pin.json",
        ),
        (
            "launch_manifest_path",
            ".kapso/bootstrap_pin.json/manifest.json",
        ),
        ("run_checkpoint_path", "workspace/checkpoint.json"),
        ("run_checkpoint_journal_path", ".other/checkpoint.journal"),
        ("run_checkpoint_lock_path", ".other/checkpoint.lock"),
        (
            "run_checkpoint_staging_path",
            ".kapso/run_checkpoint.json/staging",
        ),
        ("run_idea_archive_path", "workspace/idea_archive.json"),
        ("run_experiment_history_path", ".other/experiment_history.json"),
        ("run_execution_journal_path", "readonly/execution_events.jsonl"),
        (
            "run_derived_state_store_path",
            ".kapso/run_derived_state_staging",
        ),
        (
            "run_derived_state_staging_path",
            ".kapso/idea_archive.json/staging",
        ),
        (
            "run_action_workspace_staging_path",
            ".kapso/run_action_store",
        ),
        ("workspace_git_branch", "../unsafe"),
    ],
)
def test_launch_workspace_layout_is_relative_and_prefix_disjoint(field, value):
    raw = copy.deepcopy(load_config(CANONICAL_CONFIG_PATH)["cross_run"])
    raw["launch"][field] = value

    with pytest.raises(CrossRunConfigurationError):
        CrossRunSettings.from_dict(raw)


def test_launch_derived_generation_bound_covers_all_projection_authorities():
    raw = copy.deepcopy(load_config(CANONICAL_CONFIG_PATH)["cross_run"])
    authority_bound = raw["launch"]["run_checkpoint_size_bytes"] + sum(
        raw["launch"][field]
        for field in (
            "run_idea_archive_size_bytes",
            "run_experiment_history_size_bytes",
            "run_execution_journal_size_bytes",
        )
    )
    raw["launch"]["run_derived_generation_size_bytes"] = authority_bound

    with pytest.raises(
        CrossRunConfigurationError,
        match="generation bound",
    ):
        CrossRunSettings.from_dict(raw)


@pytest.mark.parametrize(
    "field",
    [
        "run_idea_archive_size_bytes",
        "run_experiment_history_size_bytes",
        "run_execution_journal_size_bytes",
        "run_derived_generation_size_bytes",
        "run_derived_state_store_entry_limit",
        "run_derived_state_staging_entry_limit",
        "run_workspace_entry_limit",
        "run_workspace_size_bytes",
        "run_workspace_git_entry_limit",
        "run_workspace_git_metadata_size_bytes",
    ],
)
def test_launch_derived_state_bounds_are_positive(field):
    raw = copy.deepcopy(load_config(CANONICAL_CONFIG_PATH)["cross_run"])
    raw["launch"][field] = 0

    with pytest.raises(CrossRunConfigurationError):
        CrossRunSettings.from_dict(raw)


@pytest.mark.parametrize(
    "secret_key", ["api_key", "token", "credentials", "github_token"]
)
def test_secret_bearing_config_keys_are_rejected(secret_key):
    raw = copy.deepcopy(load_config(CANONICAL_CONFIG_PATH)["cross_run"])
    raw["knowledge"]["embeddings"][secret_key] = "must-never-be-configured"

    with pytest.raises(CrossRunConfigurationError):
        CrossRunSettings.from_dict(raw)


def test_repository_relocation_changes_only_registry_location_identity():
    raw = copy.deepcopy(load_config(CANONICAL_CONFIG_PATH)["cross_run"])
    original = CrossRunSettings.from_dict(raw)
    scientific_payload = {
        "scope_contract_id": content_id("fixture", {"scope": "ml_ai"}),
        "statement": "A scientific fact independent of transport.",
    }
    scientific_id = content_id("scientific", scientific_payload)

    raw["scopes"]["ml_ai"]["repositories"]["expert"] = "Leeroo-AI/relocated-expert"
    relocated = CrossRunSettings.from_dict(raw)

    assert relocated.scopes.fingerprint != original.scopes.fingerprint
    assert content_id("scientific", scientific_payload) == scientific_id


def test_duplicate_yaml_keys_fail_instead_of_last_value_winning(tmp_path):
    config = tmp_path / "duplicate.yaml"
    config.write_text("default_mode: A\ndefault_mode: B\nmodes: {}\n", encoding="utf-8")

    with pytest.raises(CrossRunConfigurationError):
        load_config(str(config))


def test_orphan_runtime_registry_fingerprint_fails_loud(tmp_path):
    config = tmp_path / "orphan-fingerprint.yaml"
    config.write_text(
        "default_mode: X\n"
        "modes:\n"
        "  X: {}\n"
        f"cross_run_registry_fingerprint: {'sha256:' + '0' * 64}\n",
        encoding="utf-8",
    )

    with pytest.raises(CrossRunConfigurationError):
        load_effective_config(str(config))


@pytest.mark.parametrize(
    "forbidden_field",
    ["cross_run", "cross_run_registry_fingerprint"],
)
def test_mode_cannot_override_global_cross_run_configuration(tmp_path, forbidden_field):
    config = tmp_path / "mode-override.yaml"
    config.write_text(
        "default_mode: X\n" "modes:\n" "  X:\n" f"    {forbidden_field}: {{}}\n",
        encoding="utf-8",
    )

    with pytest.raises(CrossRunConfigurationError):
        load_effective_config(str(config))
