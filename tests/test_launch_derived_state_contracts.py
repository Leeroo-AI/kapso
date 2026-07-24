from dataclasses import replace

import pytest

from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.contracts import ContractValidationError
from kapso.cross_run.launch.derived_state_contracts import (
    DerivedStateContractError,
    RunDerivedStateGeneration,
    RunStateAuthority,
    RunStateAuthorityBinding,
    RunStateLayout,
    RunStatePayloadFormat,
    RunStatePayloadTransition,
)


def _authority_paths(strategy_kind="generic"):
    paths = {
        RunStateAuthority.ACTION_LEDGER: "state/action_ledger.json",
        RunStateAuthority.EXPERIMENT_HISTORY: "state/experiment_history.json",
        RunStateAuthority.EXECUTION_JOURNAL: "state/execution_journal.jsonl",
    }
    if strategy_kind == "generic":
        paths[RunStateAuthority.IDEA_ARCHIVE] = "state/idea_archive.json"
    return paths


def _layout(strategy_kind="generic"):
    return RunStateLayout.build(
        strategy_kind=strategy_kind,
        authority_paths=_authority_paths(strategy_kind),
    )


def _digest(label):
    return tree_or_blob_digest(label.encode("utf-8"))


def _transitions(layout, *, successor=False):
    transitions = []
    for position, binding in enumerate(layout.bindings):
        predecessor_digest = _digest(f"before-{position}") if successor else None
        predecessor_revision = position if successor else None
        predecessor_size_bytes = position + 9 if successor else None
        transitions.append(
            RunStatePayloadTransition.mint(
                authority_binding_id=binding.authority_binding_id,
                predecessor_digest=predecessor_digest,
                predecessor_revision=predecessor_revision,
                predecessor_size_bytes=predecessor_size_bytes,
                target_digest=(
                    _digest(f"after-{position}")
                    if successor
                    else _digest(f"genesis-{position}")
                ),
                target_revision=position + 1 if successor else 0,
                target_size_bytes=position + 10,
            )
        )
    return tuple(transitions)


def _generation(*, successor=False):
    layout = _layout()
    return RunDerivedStateGeneration.build(
        bootstrap_pin_id=content_id("bootstrap-pin", {"run": "contract-test"}),
        run_state_layout=layout,
        predecessor_checkpoint_head_id=content_id(
            "run-checkpoint-head",
            {"successor": successor},
        ),
        predecessor_checkpoint_id=(
            content_id("run-checkpoint", {"sequence": 0}) if successor else None
        ),
        predecessor_evidence_id=(
            content_id("run-derivative-evidence", {"revision": 0})
            if successor
            else None
        ),
        target_evidence_id=content_id(
            "run-derivative-evidence",
            {"revision": 1 if successor else 0},
        ),
        payload_transitions=_transitions(layout, successor=successor),
    )


def _generation_values(generation, **changes):
    values = {
        "bootstrap_pin_id": generation.bootstrap_pin_id,
        "run_state_layout": generation.run_state_layout,
        "predecessor_checkpoint_head_id": (generation.predecessor_checkpoint_head_id),
        "predecessor_checkpoint_id": generation.predecessor_checkpoint_id,
        "predecessor_evidence_id": generation.predecessor_evidence_id,
        "target_evidence_id": generation.target_evidence_id,
        "payload_transitions": generation.payload_transitions,
    }
    values.update(changes)
    return values


def test_layout_builds_complete_typed_generic_authority_map():
    layout = _layout()

    assert tuple(binding.authority for binding in layout.bindings) == tuple(
        sorted(RunStateAuthority, key=lambda authority: authority.value)
    )
    assert tuple(binding.payload_format for binding in layout.bindings) == (
        RunStatePayloadFormat.CANONICAL_JSON,
        RunStatePayloadFormat.CANONICAL_JSONL,
        RunStatePayloadFormat.CANONICAL_JSON,
        RunStatePayloadFormat.CANONICAL_JSON,
    )
    assert RunStateLayout.from_json_bytes(layout.to_json_bytes()) == layout
    assert layout.layout_id.startswith("run-state-layout:sha256:")


def test_layout_builds_complete_tree_authority_map_without_idea_archive():
    layout = _layout("benchmark_tree_search")

    assert tuple(binding.authority for binding in layout.bindings) == (
        RunStateAuthority.ACTION_LEDGER,
        RunStateAuthority.EXECUTION_JOURNAL,
        RunStateAuthority.EXPERIMENT_HISTORY,
    )


@pytest.mark.parametrize("strategy_kind", ("unknown", "", "tree"))
def test_layout_rejects_unsupported_strategy(strategy_kind):
    with pytest.raises(DerivedStateContractError, match="unsupported"):
        RunStateLayout.build(
            strategy_kind=strategy_kind,
            authority_paths=_authority_paths("benchmark_tree_search"),
        )


def test_layout_requires_exact_strategy_authority_set():
    generic_paths = _authority_paths()
    del generic_paths[RunStateAuthority.IDEA_ARCHIVE]
    with pytest.raises(DerivedStateContractError, match="incomplete"):
        RunStateLayout.build(
            strategy_kind="generic",
            authority_paths=generic_paths,
        )

    tree_paths = _authority_paths("benchmark_tree_search")
    tree_paths[RunStateAuthority.IDEA_ARCHIVE] = "state/ideas.json"
    with pytest.raises(DerivedStateContractError, match="incomplete"):
        RunStateLayout.build(
            strategy_kind="benchmark_tree_search",
            authority_paths=tree_paths,
        )


def test_layout_requires_explicit_typed_authority_mapping():
    with pytest.raises(DerivedStateContractError, match="typed"):
        RunStateLayout.build(
            strategy_kind="generic",
            authority_paths={
                authority.value: path for authority, path in _authority_paths().items()
            },
        )


def test_layout_rejects_authority_order_changes():
    layout = _layout()
    with pytest.raises(DerivedStateContractError, match="authority order"):
        RunStateLayout.mint(
            strategy_kind=layout.strategy_kind,
            bindings=tuple(reversed(layout.bindings)),
        )


@pytest.mark.parametrize(
    "relative_path",
    (
        "",
        ".",
        "/state/ideas.json",
        "../state/ideas.json",
        "state/../ideas.json",
        "state//ideas.json",
        "state/ideas.json/",
        "./state/ideas.json",
        "state/\x00ideas.json",
    ),
)
def test_authority_binding_rejects_unsafe_or_unnormalized_paths(relative_path):
    with pytest.raises(DerivedStateContractError, match="normalized"):
        RunStateAuthorityBinding.build(
            authority=RunStateAuthority.IDEA_ARCHIVE,
            relative_path=relative_path,
        )


@pytest.mark.parametrize(
    "paths",
    (
        (
            "state/shared",
            "state/shared",
            "state/journal.jsonl",
        ),
        (
            "state/history",
            "state/history/records.json",
            "state/journal.jsonl",
        ),
    ),
)
def test_layout_rejects_equal_and_ancestor_path_collisions(paths):
    with pytest.raises(DerivedStateContractError, match="overlap"):
        RunStateLayout.build(
            strategy_kind="generic",
            authority_paths={
                RunStateAuthority.ACTION_LEDGER: "state/action_ledger.json",
                RunStateAuthority.IDEA_ARCHIVE: paths[0],
                RunStateAuthority.EXPERIMENT_HISTORY: paths[1],
                RunStateAuthority.EXECUTION_JOURNAL: paths[2],
            },
        )


def test_binding_rejects_payload_format_incompatible_with_authority():
    with pytest.raises(DerivedStateContractError, match="wrong payload format"):
        RunStateAuthorityBinding.mint(
            authority=RunStateAuthority.EXECUTION_JOURNAL,
            relative_path="state/journal.jsonl",
            payload_format=RunStatePayloadFormat.CANONICAL_JSON,
        )


def test_content_ids_change_with_layout_and_transition_content():
    original_layout = _layout()
    changed_paths = _authority_paths()
    changed_paths[RunStateAuthority.IDEA_ARCHIVE] = "state/new_ideas.json"
    changed_layout = RunStateLayout.build(
        strategy_kind="generic",
        authority_paths=changed_paths,
    )
    original_transition = _transitions(original_layout)[0]
    changed_transition = RunStatePayloadTransition.mint(
        authority_binding_id=original_transition.authority_binding_id,
        predecessor_digest=None,
        predecessor_revision=None,
        predecessor_size_bytes=None,
        target_digest=_digest("different"),
        target_revision=0,
        target_size_bytes=original_transition.target_size_bytes,
    )

    assert changed_layout.layout_id != original_layout.layout_id
    assert (
        changed_transition.payload_transition_id
        != original_transition.payload_transition_id
    )


def test_payload_transition_accepts_genesis_changed_successor_and_noop_successor():
    binding_id = _layout().bindings[0].authority_binding_id
    genesis = RunStatePayloadTransition.mint(
        authority_binding_id=binding_id,
        predecessor_digest=None,
        predecessor_revision=None,
        predecessor_size_bytes=None,
        target_digest=_digest("genesis"),
        target_revision=0,
        target_size_bytes=20,
    )
    changed = RunStatePayloadTransition.mint(
        authority_binding_id=binding_id,
        predecessor_digest=genesis.target_digest,
        predecessor_revision=genesis.target_revision,
        predecessor_size_bytes=genesis.target_size_bytes,
        target_digest=_digest("changed"),
        target_revision=4,
        target_size_bytes=30,
    )
    unchanged = RunStatePayloadTransition.mint(
        authority_binding_id=binding_id,
        predecessor_digest=changed.target_digest,
        predecessor_revision=changed.target_revision,
        predecessor_size_bytes=changed.target_size_bytes,
        target_digest=changed.target_digest,
        target_revision=changed.target_revision,
        target_size_bytes=changed.target_size_bytes,
    )

    assert genesis.predecessor_digest is None
    assert changed.target_revision == 4
    assert unchanged.target_digest == unchanged.predecessor_digest


def test_noop_payload_transition_requires_exact_predecessor_size():
    binding_id = _layout().bindings[0].authority_binding_id
    digest = _digest("unchanged")
    with pytest.raises(DerivedStateContractError, match="exact size"):
        RunStatePayloadTransition.mint(
            authority_binding_id=binding_id,
            predecessor_digest=digest,
            predecessor_revision=3,
            predecessor_size_bytes=20,
            target_digest=digest,
            target_revision=3,
            target_size_bytes=21,
        )


def test_payload_transition_requires_joint_predecessor_fields():
    binding_id = _layout().bindings[0].authority_binding_id
    with pytest.raises(DerivedStateContractError, match="predecessor fields"):
        RunStatePayloadTransition.mint(
            authority_binding_id=binding_id,
            predecessor_digest=_digest("before"),
            predecessor_revision=None,
            predecessor_size_bytes=1,
            target_digest=_digest("after"),
            target_revision=0,
            target_size_bytes=1,
        )


def test_payload_transition_requires_genesis_revision_zero():
    binding_id = _layout().bindings[0].authority_binding_id
    with pytest.raises(DerivedStateContractError, match="revision zero"):
        RunStatePayloadTransition.mint(
            authority_binding_id=binding_id,
            predecessor_digest=None,
            predecessor_revision=None,
            predecessor_size_bytes=None,
            target_digest=_digest("genesis"),
            target_revision=1,
            target_size_bytes=1,
        )


@pytest.mark.parametrize(
    ("predecessor_digest", "predecessor_revision", "target_digest", "target_revision"),
    (
        (_digest("before"), 2, _digest("after"), 1),
        (_digest("same"), 2, _digest("same"), 3),
        (_digest("before"), 2, _digest("after"), 2),
    ),
)
def test_payload_transition_rejects_rollback_and_revision_digest_disagreement(
    predecessor_digest,
    predecessor_revision,
    target_digest,
    target_revision,
):
    binding_id = _layout().bindings[0].authority_binding_id
    with pytest.raises(DerivedStateContractError):
        RunStatePayloadTransition.mint(
            authority_binding_id=binding_id,
            predecessor_digest=predecessor_digest,
            predecessor_revision=predecessor_revision,
            predecessor_size_bytes=1,
            target_digest=target_digest,
            target_revision=target_revision,
            target_size_bytes=1,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("target_digest", "not-a-digest", "sha256"),
        ("predecessor_digest", "sha256:abc", "sha256"),
        ("predecessor_revision", -1, "non-negative"),
        ("predecessor_size_bytes", -1, "non-negative"),
        ("target_revision", -1, "non-negative"),
        ("target_size_bytes", -1, "non-negative"),
    ),
)
def test_payload_transition_rejects_invalid_digest_revision_and_size(
    field,
    value,
    message,
):
    binding_id = _layout().bindings[0].authority_binding_id
    values = {
        "authority_binding_id": binding_id,
        "predecessor_digest": _digest("before"),
        "predecessor_revision": 0,
        "predecessor_size_bytes": 1,
        "target_digest": _digest("after"),
        "target_revision": 1,
        "target_size_bytes": 1,
    }
    values[field] = value
    with pytest.raises(DerivedStateContractError, match=message):
        RunStatePayloadTransition.mint(**values)


def test_generation_builds_exact_genesis_dependency_closure_and_round_trips():
    generation = _generation()
    expected_dependencies = {
        generation.bootstrap_pin_id,
        generation.run_state_layout.layout_id,
        generation.predecessor_checkpoint_head_id,
        generation.target_evidence_id,
        *(
            transition.payload_transition_id
            for transition in generation.payload_transitions
        ),
    }

    assert generation.exact_dependency_ids == tuple(sorted(expected_dependencies))
    assert (
        RunDerivedStateGeneration.from_json_bytes(generation.to_json_bytes())
        == generation
    )
    assert generation.generation_id.startswith("run-derived-state-generation:sha256:")


def test_generation_builds_exact_successor_dependency_closure():
    generation = _generation(successor=True)

    assert generation.predecessor_checkpoint_id in generation.exact_dependency_ids
    assert generation.predecessor_evidence_id in generation.exact_dependency_ids
    assert generation.predecessor_checkpoint_head_id in (
        generation.exact_dependency_ids
    )


def test_generation_requires_joint_predecessor_checkpoint_and_evidence():
    generation = _generation()
    with pytest.raises(DerivedStateContractError, match="predecessor fields"):
        RunDerivedStateGeneration.build(
            **_generation_values(
                generation,
                predecessor_checkpoint_id=content_id(
                    "run-checkpoint",
                    {"sequence": 0},
                ),
            )
        )


def test_generation_requires_genesis_or_successor_transitions_for_its_frontier():
    genesis = _generation()
    successor = _generation(successor=True)
    with pytest.raises(DerivedStateContractError, match="frontier"):
        RunDerivedStateGeneration.build(
            **_generation_values(
                genesis,
                payload_transitions=_transitions(
                    genesis.run_state_layout,
                    successor=True,
                ),
            )
        )
    with pytest.raises(DerivedStateContractError, match="frontier"):
        RunDerivedStateGeneration.build(
            **_generation_values(
                successor,
                payload_transitions=_transitions(
                    successor.run_state_layout,
                    successor=False,
                ),
            )
        )


def test_generation_requires_complete_layout_ordered_payload_transitions():
    generation = _generation()
    with pytest.raises(DerivedStateContractError, match="incomplete or unordered"):
        RunDerivedStateGeneration.build(
            **_generation_values(
                generation,
                payload_transitions=generation.payload_transitions[:-1],
            )
        )
    with pytest.raises(DerivedStateContractError, match="incomplete or unordered"):
        RunDerivedStateGeneration.build(
            **_generation_values(
                generation,
                payload_transitions=tuple(reversed(generation.payload_transitions)),
            )
        )


@pytest.mark.parametrize(
    ("field", "namespace"),
    (
        ("bootstrap_pin_id", "wrong-bootstrap"),
        ("predecessor_checkpoint_head_id", "wrong-head"),
        ("target_evidence_id", "wrong-evidence"),
    ),
)
def test_generation_rejects_wrong_dependency_namespaces(field, namespace):
    generation = _generation()
    with pytest.raises(DerivedStateContractError, match="wrong namespace"):
        RunDerivedStateGeneration.build(
            **_generation_values(
                generation,
                **{field: content_id(namespace, {"field": field})},
            )
        )


def test_successor_generation_rejects_wrong_predecessor_namespaces():
    generation = _generation(successor=True)
    for field in ("predecessor_checkpoint_id", "predecessor_evidence_id"):
        with pytest.raises(DerivedStateContractError, match="wrong namespace"):
            RunDerivedStateGeneration.build(
                **_generation_values(
                    generation,
                    **{field: content_id("wrong", {"field": field})},
                )
            )


def test_generation_rejects_missing_extra_unsorted_and_duplicate_dependencies():
    generation = _generation()
    extra = content_id("unexpected", {"dependency": True})
    dependency_cases = (
        generation.exact_dependency_ids[:-1],
        tuple(sorted((*generation.exact_dependency_ids, extra))),
        tuple(reversed(generation.exact_dependency_ids)),
        (
            generation.exact_dependency_ids[0],
            *generation.exact_dependency_ids,
        ),
    )
    for dependencies in dependency_cases:
        with pytest.raises(DerivedStateContractError):
            RunDerivedStateGeneration.mint(
                **_generation_values(
                    generation,
                ),
                exact_dependency_ids=dependencies,
            )


def test_contracts_reject_unknown_missing_and_forged_content_identity():
    layout = _layout()
    payload = layout.to_dict()
    payload["unknown"] = "field"
    with pytest.raises(ContractValidationError, match="unknown"):
        RunStateLayout.from_dict(payload)

    payload = layout.to_dict()
    del payload["bindings"]
    with pytest.raises(ContractValidationError, match="missing"):
        RunStateLayout.from_dict(payload)

    with pytest.raises(ValueError):
        replace(layout, strategy_kind="benchmark_tree_search")
