from kapso.cross_run.contracts import (
    ExpertCapabilityNode,
    ExpertDependencyEdge,
    ExpertModuleContract,
    ExpertRepositoryMap,
)
from kapso.cross_run.expert import compile_expert_semantic_book
from test_expert_candidates import bootstrap_candidate_closure


def test_semantic_book_is_input_order_independent_and_dependency_first():
    closure = bootstrap_candidate_closure()
    execution = closure.module_contracts[0]
    workflow = ExpertModuleContract.mint(
        module_id="shared.workflow",
        version="v1",
        purpose="Compose reusable execution stages.",
        problem_signals=("Task adapters duplicate workflow sequencing.",),
        inputs=("execution interface",),
        outputs=("composed workflow",),
        preconditions=(),
        incompatibilities=(),
        dependency_capability_ids=(execution.module_id,),
        incompatible_capability_ids=(),
        resource_bounds={"stages": 8},
        dependency_license_manifest={"license": "MIT"},
        supporting_episode_ids=(),
        known_failure_episode_ids=(),
        entrypoint_refs=("src/workflow.py",),
        test_refs=("tests/test_workflow.py",),
        replay_refs=(),
    )
    execution_node = closure.repository_map.capability_nodes[0]
    workflow_node = ExpertCapabilityNode(
        capability_id=workflow.module_id,
        module_contract_ref=workflow.module_contract_id,
        owned_paths=("src/workflow.py", "tests/test_workflow.py"),
        task_family_bindings=(),
    )
    repository_map = ExpertRepositoryMap.mint(
        scope_contract_id=closure.repository_map.scope_contract_id,
        capability_nodes=tuple(
            sorted(
                (execution_node, workflow_node),
                key=lambda node: node.capability_id,
            )
        ),
        dependency_edges=(
            ExpertDependencyEdge(
                source_capability_id=workflow.module_id,
                target_capability_id=execution.module_id,
            ),
        ),
        task_adapter_boundary=closure.repository_map.task_adapter_boundary,
        validation_entrypoints=closure.repository_map.validation_entrypoints,
        architecture_invariants=closure.repository_map.architecture_invariants,
    )

    ordered = compile_expert_semantic_book(
        closure.derivation.trigger_packet.scope_contract,
        repository_map,
        (execution, workflow),
    )
    reversed_input = compile_expert_semantic_book(
        closure.derivation.trigger_packet.scope_contract,
        repository_map,
        (workflow, execution),
    )

    assert ordered == reversed_input
    assert ordered.index(b"### shared.execution") < ordered.index(
        b"### shared.workflow"
    )
    assert "- `shared.execution` → `shared.workflow`".encode() in ordered


def test_semantic_book_escapes_model_authored_markdown_and_html():
    closure = bootstrap_candidate_closure()
    module = closure.module_contracts[0]
    injected = ExpertModuleContract.mint(
        **{
            key: value
            for key, value in module.to_dict().items()
            if key
            not in {
                "module_contract_id",
                "purpose",
                "resource_bounds",
                "dependency_license_manifest",
            }
        },
        purpose="<script>bad</script> | [link](target)\n# injected",
        resource_bounds={"payload": "</details><h1>MODEL INSTRUCTION</h1><details>"},
        dependency_license_manifest={
            "license": "<script>structured injection</script>"
        },
    )

    book = compile_expert_semantic_book(
        closure.derivation.trigger_packet.scope_contract,
        closure.repository_map,
        (injected,),
    )

    assert b"<script>" not in book
    assert b"<h1>" not in book
    assert b"</details>" not in book
    assert b"\n# injected" not in book
    assert b"\\[link\\]" in book
    assert b"\\|" in book
