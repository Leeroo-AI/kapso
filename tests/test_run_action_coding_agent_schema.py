import math
import pytest

from kapso.cross_run.contracts import ExpertCandidateOperationKind
from kapso.cross_run.expert.proposal_contract import (
    expert_proposal_response_schema,
)
from kapso.cross_run.launch.run_action_coding_agent_schema import (
    CODING_AGENT_JSON_SCHEMA_DIALECT,
    RunActionCodingAgentSchemaError,
    validate_run_action_coding_agent_output,
    validate_run_action_coding_agent_provider_schema,
    validate_run_action_coding_agent_schema,
)
from kapso.execution.search_strategies.generic.ideation.evidence_author import (
    EVIDENCE_AUTHOR_RESPONSE_SCHEMA,
)
from kapso.execution.search_strategies.generic.ideation.generator import (
    CANDIDATE_RESPONSE_SCHEMA,
)
from kapso.execution.search_strategies.generic.ideation.selector import (
    SELECTOR_RESPONSE_SCHEMA,
)


@pytest.mark.parametrize(
    "schema",
    (
        CANDIDATE_RESPONSE_SCHEMA,
        EVIDENCE_AUTHOR_RESPONSE_SCHEMA,
        SELECTOR_RESPONSE_SCHEMA,
        expert_proposal_response_schema(ExpertCandidateOperationKind.BOOTSTRAP),
        expert_proposal_response_schema(ExpertCandidateOperationKind.GENERALIZE),
    ),
)
def test_current_coding_agent_schema_shapes_are_in_the_closed_subset(schema) -> None:
    validate_run_action_coding_agent_schema(schema)


@pytest.mark.parametrize(
    "schema",
    (
        CANDIDATE_RESPONSE_SCHEMA,
        EVIDENCE_AUTHOR_RESPONSE_SCHEMA,
        SELECTOR_RESPONSE_SCHEMA,
    ),
)
def test_current_ideation_schemas_are_provider_portable(schema) -> None:
    validate_run_action_coding_agent_provider_schema(schema)


@pytest.mark.parametrize(
    "operation_kind",
    (
        ExpertCandidateOperationKind.BOOTSTRAP,
        ExpertCandidateOperationKind.GENERALIZE,
    ),
)
def test_expert_free_form_maps_are_rejected_from_provider_strict_requests(
    operation_kind,
) -> None:
    with pytest.raises(
        RunActionCodingAgentSchemaError,
        match="additionalProperties",
    ):
        validate_run_action_coding_agent_provider_schema(
            expert_proposal_response_schema(operation_kind)
        )


@pytest.mark.parametrize(
    "schema",
    (
        {"type": "object"},
        {"type": "object", "additionalProperties": False},
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": [],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }
            },
            "required": ["nested"],
            "additionalProperties": False,
        },
    ),
)
def test_provider_schema_requires_every_object_to_be_closed_and_fully_required(
    schema,
) -> None:
    validate_run_action_coding_agent_schema(schema)
    with pytest.raises(
        RunActionCodingAgentSchemaError,
        match="additionalProperties|require every",
    ):
        validate_run_action_coding_agent_provider_schema(schema)


def test_provider_schema_accepts_nullable_properties_instead_of_optional_keys() -> None:
    validate_run_action_coding_agent_provider_schema(
        {
            "type": "object",
            "properties": {
                "answer": {
                    "type": ["string", "null"],
                }
            },
            "required": ["answer"],
            "additionalProperties": False,
        }
    )


def test_current_ideation_candidate_output_validates() -> None:
    output = {
        "proposal": "Change the sampling policy.",
        "directive_rationale": "The evidence indicates under-exploration.",
        "descriptor": {
            "approach_family": "sampling",
            "intervention_target": "candidate selection",
            "mechanism": "uncertainty weighting",
            "expected_effect": "higher novelty",
        },
        "assumptions": ["Scores are calibrated."],
        "evidence_refs": ["evidence-1"],
        "claim_ids": ["claim-1"],
        "resolves_claim_ids": [],
        "expected_observations": ["More distinct mechanisms."],
        "evaluation_method": "Compare novelty at equal cost.",
        "resource_request": "one experiment",
        "predicted_gain": 0.2,
        "predicted_cost": 1,
        "confidence": None,
        "claimed_nearest_idea_id": None,
        "claimed_nearest_experiment_node_id": 4,
        "prior_knowledge_refs": [],
        "prior_adaptation_rationale": None,
    }

    validate_run_action_coding_agent_output(CANDIDATE_RESPONSE_SCHEMA, output)


def test_nested_unknown_keyword_and_remote_reference_are_rejected() -> None:
    with pytest.raises(
        RunActionCodingAgentSchemaError,
        match="unsupported keywords",
    ):
        validate_run_action_coding_agent_schema(
            {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "$ref": "https://example.com/schema.json",
                    }
                },
            }
        )


def test_only_the_exact_offline_dialect_is_admitted() -> None:
    validate_run_action_coding_agent_schema(
        {
            "$schema": CODING_AGENT_JSON_SCHEMA_DIALECT,
            "type": "object",
        }
    )
    with pytest.raises(
        RunActionCodingAgentSchemaError,
        match="unsupported JSON Schema dialect",
    ):
        validate_run_action_coding_agent_schema(
            {
                "$schema": "https://json-schema.org/draft/2019-09/schema",
                "type": "object",
            }
        )


@pytest.mark.parametrize(
    "schema",
    (
        {},
        {"type": "unknown"},
        {"type": []},
        {"type": ["string", "string"]},
        {"type": "object", "properties": []},
        {"type": "object", "properties": {"answer": []}},
        {"type": "object", "required": "answer"},
        {"type": "object", "required": ["answer", "answer"]},
        {"type": "object", "additionalProperties": {}},
        {"type": "array", "items": []},
        {"type": "string", "enum": []},
        {"type": "string", "enum": ["same", "same"]},
        {"type": "string", "minLength": True},
        {"type": "array", "minItems": -1},
        {"type": "object", "minProperties": 1.0},
        {"type": "string", "pattern": 1},
        {"type": "string", "properties": {}},
        {"type": "object", "items": {"type": "string"}},
        {"type": "array", "minLength": 1},
    ),
)
def test_malformed_schema_shapes_are_rejected(schema) -> None:
    with pytest.raises(RunActionCodingAgentSchemaError):
        validate_run_action_coding_agent_schema(schema)


@pytest.mark.parametrize(
    "pattern",
    (
        "",
        "[",
        r"\d+",
        "(?P<name>a)",
        "(?:a)",
        "a|b",
        "[9-1]",
        "[^0-9]",
        "é",
    ),
)
def test_nonportable_regular_expression_is_rejected_before_execution(pattern) -> None:
    with pytest.raises(RunActionCodingAgentSchemaError, match="portable"):
        validate_run_action_coding_agent_schema(
            {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        "pattern": pattern,
                    }
                },
            }
        )


@pytest.mark.parametrize("value", (math.inf, -math.inf, math.nan))
def test_nonfinite_output_and_enum_values_are_rejected(value) -> None:
    with pytest.raises(
        RunActionCodingAgentSchemaError,
        match="non-finite",
    ):
        validate_run_action_coding_agent_output(
            {
                "type": "object",
                "properties": {"score": {"type": "number"}},
                "required": ["score"],
                "additionalProperties": False,
            },
            {"score": value},
        )
    with pytest.raises(
        RunActionCodingAgentSchemaError,
        match="non-finite",
    ):
        validate_run_action_coding_agent_schema({"type": "number", "enum": [value]})


@pytest.mark.parametrize("declared_type", ("number", "integer"))
def test_boolean_is_not_a_number_or_integer(declared_type) -> None:
    with pytest.raises(
        RunActionCodingAgentSchemaError,
        match="declared schema type",
    ):
        validate_run_action_coding_agent_output(
            {
                "type": "object",
                "properties": {"value": {"type": declared_type}},
                "required": ["value"],
            },
            {"value": True},
        )


def test_number_and_integer_follow_json_schema_numeric_semantics() -> None:
    validate_run_action_coding_agent_output(
        {
            "type": "object",
            "properties": {"value": {"type": "number"}},
            "required": ["value"],
        },
        {"value": 1},
    )
    validate_run_action_coding_agent_output(
        {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        },
        {"value": 1.0},
    )
    with pytest.raises(RunActionCodingAgentSchemaError):
        validate_run_action_coding_agent_output(
            {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
            {"value": 1.5},
        )


def test_enum_uses_json_schema_numeric_equality() -> None:
    schema = {
        "type": "object",
        "properties": {"value": {"type": "number", "enum": [1]}},
        "required": ["value"],
    }

    validate_run_action_coding_agent_output(schema, {"value": 1.0})
    with pytest.raises(RunActionCodingAgentSchemaError, match="must be unique"):
        validate_run_action_coding_agent_schema(
            {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "enum": [1, 1.0],
                    }
                },
            }
        )


def test_object_closure_required_and_minimum_properties_are_enforced() -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
        "minProperties": 1,
    }
    for output in ({}, {"answer": "yes", "extra": False}):
        with pytest.raises(RunActionCodingAgentSchemaError):
            validate_run_action_coding_agent_output(schema, output)
    validate_run_action_coding_agent_output(schema, {"answer": "yes"})


def test_nested_array_string_enum_union_and_pattern_constraints_are_enforced() -> None:
    schema = {
        "type": "object",
        "properties": {
            "values": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": ["string", "null"],
                    "enum": ["v1", None],
                    "minLength": 2,
                    "pattern": "^v[0-9]+$",
                },
            }
        },
        "required": ["values"],
        "additionalProperties": False,
    }
    validate_run_action_coding_agent_output(schema, {"values": ["v1", None]})
    for values in ([], ["x"], ["v2", "other"]):
        with pytest.raises(RunActionCodingAgentSchemaError):
            validate_run_action_coding_agent_output(schema, {"values": values})


@pytest.mark.parametrize(
    ("schema", "output"),
    (
        ({"type": "array"}, []),
        ({"type": "string"}, "text"),
        ({"type": "number"}, 1),
        ({"type": "boolean"}, True),
        ({"type": "null"}, None),
        ({"type": ["object", "null"]}, {}),
    ),
)
def test_non_object_root_schema_is_rejected(schema, output) -> None:
    with pytest.raises(
        RunActionCodingAgentSchemaError,
        match="root type must be object",
    ):
        validate_run_action_coding_agent_output(schema, output)


def test_unconstrained_additional_object_values_still_require_json_data() -> None:
    with pytest.raises(
        RunActionCodingAgentSchemaError,
        match="outside the JSON data model",
    ):
        validate_run_action_coding_agent_output(
            {"type": "object", "minProperties": 1},
            {"opaque": object()},
        )
