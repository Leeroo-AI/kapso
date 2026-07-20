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
    assert settings.to_dict() == raw["cross_run"]
    assert "api_key" not in str(settings.to_dict()).lower()
    assert "oauth_token" not in str(settings.to_dict()).lower()


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
        {"some_value": "Leeroo-AI/kapso-knowledge"},
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
        (("knowledge", "retrieval", "lexical_weight"), 1.1),
        (("knowledge", "embeddings", "dimensions"), True),
        (("expert", "validation", "reviewer_count"), 0),
        (("launch", "cache_path"), "../escape"),
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


def test_github_rate_budget_scales_with_failed_upload_recovery():
    raw = copy.deepcopy(load_config(CANONICAL_CONFIG_PATH)["cross_run"])
    raw["github"]["release_asset_count_limit"] = 17
    raw["github"]["content_write_budget_per_minute"] = 43

    with pytest.raises(CrossRunConfigurationError, match="content-write budget"):
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
