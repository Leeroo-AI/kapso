"""Pure validation for the closed coding-agent response-schema subset."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

CODING_AGENT_JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"

_ALLOWED_KEYWORDS = frozenset(
    {
        "$schema",
        "additionalProperties",
        "enum",
        "items",
        "minItems",
        "minLength",
        "minProperties",
        "pattern",
        "properties",
        "required",
        "type",
    }
)
_JSON_TYPES = frozenset(
    {
        "array",
        "boolean",
        "integer",
        "null",
        "number",
        "object",
        "string",
    }
)
_OBJECT_KEYWORDS = frozenset(
    {"additionalProperties", "minProperties", "properties", "required"}
)
_ARRAY_KEYWORDS = frozenset({"items", "minItems"})
_STRING_KEYWORDS = frozenset({"minLength", "pattern"})
_PORTABLE_PATTERN_LITERAL_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-/:"
)
_PORTABLE_PATTERN_CLASS_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
)


class RunActionCodingAgentSchemaError(ValueError):
    """A coding-agent schema or structured output is outside the closed subset."""


def validate_run_action_coding_agent_schema(schema: Mapping[str, Any]) -> None:
    """Validate one complete schema without resolving external references."""

    if not isinstance(schema, Mapping):
        raise RunActionCodingAgentSchemaError(
            "coding-agent response schema must be an object"
        )
    _validate_schema_node(schema, "$")
    if _declared_types(schema["type"], "$") != ("object",):
        raise RunActionCodingAgentSchemaError(
            "coding-agent response schema root type must be object"
        )


def validate_run_action_coding_agent_provider_schema(
    schema: Mapping[str, Any],
) -> None:
    """Require the provider-portable strict subset used before model spend."""

    validate_run_action_coding_agent_schema(schema)
    _require_strict_object_nodes(schema, "$")


def validate_run_action_coding_agent_output(
    schema: Mapping[str, Any],
    output: Any,
) -> None:
    """Apply one validated schema to a complete in-memory JSON output."""

    validate_run_action_coding_agent_schema(schema)
    if not _is_object(output):
        raise RunActionCodingAgentSchemaError(
            "coding-agent structured output must be an object"
        )
    _validate_json_value(output, "$")
    _apply_schema_node(schema, output, "$")


def _require_strict_object_nodes(schema: Mapping[str, Any], path: str) -> None:
    declared_types = _declared_types(schema["type"], path)
    if "object" in declared_types:
        properties = schema.get("properties", {})
        required = schema.get("required")
        if schema.get("additionalProperties") is not False:
            raise RunActionCodingAgentSchemaError(
                f"{path} object schema must set additionalProperties to false"
            )
        if (
            not isinstance(required, (list, tuple))
            or len(required) != len(properties)
            or set(required) != set(properties)
        ):
            raise RunActionCodingAgentSchemaError(
                f"{path} object schema must require every declared property"
            )
        for name, child in properties.items():
            _require_strict_object_nodes(child, f"{path}.properties[{name!r}]")
    if "array" in declared_types and "items" in schema:
        _require_strict_object_nodes(schema["items"], f"{path}.items")


def _validate_schema_node(schema: Mapping[str, Any], path: str) -> None:
    if not isinstance(schema, Mapping):
        raise RunActionCodingAgentSchemaError(f"{path} schema must be an object")
    if any(not isinstance(keyword, str) for keyword in schema):
        raise RunActionCodingAgentSchemaError(f"{path} schema keywords must be strings")
    unknown = set(schema) - _ALLOWED_KEYWORDS
    if unknown:
        raise RunActionCodingAgentSchemaError(
            f"{path} schema uses unsupported keywords: {tuple(sorted(unknown))}"
        )
    if "type" not in schema:
        raise RunActionCodingAgentSchemaError(
            f"{path} schema must declare an explicit type"
        )
    declared_types = _declared_types(schema["type"], path)
    if "$schema" in schema and schema["$schema"] != CODING_AGENT_JSON_SCHEMA_DIALECT:
        raise RunActionCodingAgentSchemaError(
            f"{path} schema uses an unsupported JSON Schema dialect"
        )
    _require_keyword_type_compatibility(schema, declared_types, path)

    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, Mapping) or any(
            not isinstance(name, str) for name in properties
        ):
            raise RunActionCodingAgentSchemaError(
                f"{path}.properties must be an object with string keys"
            )
        for name, child in properties.items():
            if not isinstance(child, Mapping):
                raise RunActionCodingAgentSchemaError(
                    f"{path}.properties[{name!r}] must be a schema object"
                )
            _validate_schema_node(child, f"{path}.properties[{name!r}]")

    required = schema.get("required")
    if required is not None:
        if (
            not isinstance(required, (list, tuple))
            or any(not isinstance(name, str) for name in required)
            or len(required) != len(set(required))
        ):
            raise RunActionCodingAgentSchemaError(
                f"{path}.required must be a unique string array"
            )

    additional_properties = schema.get("additionalProperties")
    if additional_properties is not None and type(additional_properties) is not bool:
        raise RunActionCodingAgentSchemaError(
            f"{path}.additionalProperties must be a boolean"
        )

    items = schema.get("items")
    if items is not None:
        if not isinstance(items, Mapping):
            raise RunActionCodingAgentSchemaError(
                f"{path}.items must be one schema object"
            )
        _validate_schema_node(items, f"{path}.items")

    enum = schema.get("enum")
    if enum is not None:
        if not isinstance(enum, (list, tuple)) or not enum:
            raise RunActionCodingAgentSchemaError(
                f"{path}.enum must be a non-empty array"
            )
        members = []
        for position, value in enumerate(enum):
            _validate_json_value(value, f"{path}.enum[{position}]")
            if not any(_matches_type(value, name) for name in declared_types):
                raise RunActionCodingAgentSchemaError(
                    f"{path}.enum[{position}] differs from its declared type"
                )
            if any(_json_values_equal(value, member) for member in members):
                raise RunActionCodingAgentSchemaError(
                    f"{path}.enum values must be unique"
                )
            members.append(value)

    for keyword in ("minLength", "minItems", "minProperties"):
        if keyword in schema and (
            type(schema[keyword]) is not int or schema[keyword] < 0
        ):
            raise RunActionCodingAgentSchemaError(
                f"{path}.{keyword} must be a non-negative integer"
            )

    pattern = schema.get("pattern")
    if pattern is not None:
        _validate_portable_pattern(pattern, f"{path}.pattern")


def _declared_types(value: Any, path: str) -> tuple[str, ...]:
    if isinstance(value, str):
        declared = (value,)
    elif isinstance(value, (list, tuple)):
        declared = tuple(value)
    else:
        raise RunActionCodingAgentSchemaError(
            f"{path}.type must be a string or string array"
        )
    if (
        not declared
        or any(
            not isinstance(name, str) or name not in _JSON_TYPES for name in declared
        )
        or len(declared) != len(set(declared))
    ):
        raise RunActionCodingAgentSchemaError(
            f"{path}.type contains duplicate or unsupported types"
        )
    return declared


def _validate_portable_pattern(pattern: Any, path: str) -> None:
    if not isinstance(pattern, str) or not pattern or not pattern.isascii():
        raise RunActionCodingAgentSchemaError(
            f"{path} must use the portable ASCII pattern subset"
        )
    position = 1 if pattern.startswith("^") else 0
    terminal = len(pattern) - 1 if pattern.endswith("$") else len(pattern)
    atom_count = 0
    while position < terminal:
        character = pattern[position]
        if character in _PORTABLE_PATTERN_LITERAL_CHARACTERS:
            position += 1
        elif character == "[":
            position = _validate_portable_character_class(
                pattern,
                position,
                terminal,
                path,
            )
        else:
            raise RunActionCodingAgentSchemaError(
                f"{path} is outside the portable pattern subset"
            )
        atom_count += 1
        if position < terminal and pattern[position] in "*+?":
            position += 1
    if atom_count == 0 or position != terminal:
        raise RunActionCodingAgentSchemaError(
            f"{path} is outside the portable pattern subset"
        )
    re.compile(pattern)


def _validate_portable_character_class(
    pattern: str,
    position: int,
    terminal: int,
    path: str,
) -> int:
    position += 1
    member_count = 0
    while position < terminal and pattern[position] != "]":
        start = pattern[position]
        if start not in _PORTABLE_PATTERN_CLASS_CHARACTERS:
            raise RunActionCodingAgentSchemaError(
                f"{path} character class is outside the portable subset"
            )
        position += 1
        member_count += 1
        if position < terminal and pattern[position] == "-":
            if (
                position + 1 >= terminal
                or pattern[position + 1] not in _PORTABLE_PATTERN_CLASS_CHARACTERS
                or ord(start) > ord(pattern[position + 1])
            ):
                raise RunActionCodingAgentSchemaError(
                    f"{path} character range is outside the portable subset"
                )
            position += 2
    if member_count == 0 or position >= terminal or pattern[position] != "]":
        raise RunActionCodingAgentSchemaError(
            f"{path} character class is outside the portable subset"
        )
    return position + 1


def _require_keyword_type_compatibility(
    schema: Mapping[str, Any],
    declared_types: tuple[str, ...],
    path: str,
) -> None:
    present = set(schema)
    for keywords, required_type in (
        (_OBJECT_KEYWORDS, "object"),
        (_ARRAY_KEYWORDS, "array"),
        (_STRING_KEYWORDS, "string"),
    ):
        if present & keywords and required_type not in declared_types:
            raise RunActionCodingAgentSchemaError(
                f"{path} uses {required_type} keywords without declaring "
                f"{required_type}"
            )


def _apply_schema_node(
    schema: Mapping[str, Any],
    value: Any,
    path: str,
) -> None:
    declared_types = _declared_types(schema["type"], path)
    if not any(_matches_type(value, name) for name in declared_types):
        raise RunActionCodingAgentSchemaError(
            f"{path} differs from its declared schema type"
        )

    enum = schema.get("enum")
    if enum is not None:
        if not any(_json_values_equal(value, member) for member in enum):
            raise RunActionCodingAgentSchemaError(
                f"{path} is not one of its allowed enum values"
            )

    if _is_object(value):
        _apply_object_schema(schema, value, path)
    elif _is_array(value):
        _apply_array_schema(schema, value, path)
    elif isinstance(value, str):
        _apply_string_schema(schema, value, path)


def _apply_object_schema(
    schema: Mapping[str, Any],
    value: Mapping[str, Any],
    path: str,
) -> None:
    required = schema.get("required", ())
    missing = set(required) - set(value)
    if missing:
        raise RunActionCodingAgentSchemaError(
            f"{path} is missing required properties: {tuple(sorted(missing))}"
        )
    minimum = schema.get("minProperties")
    if minimum is not None and len(value) < minimum:
        raise RunActionCodingAgentSchemaError(
            f"{path} has fewer than {minimum} properties"
        )
    properties = schema.get("properties", {})
    unknown = set(value) - set(properties)
    if schema.get("additionalProperties", True) is False and unknown:
        raise RunActionCodingAgentSchemaError(
            f"{path} has additional properties: {tuple(sorted(unknown))}"
        )
    for name, child_schema in properties.items():
        if name in value:
            _apply_schema_node(child_schema, value[name], f"{path}[{name!r}]")


def _apply_array_schema(
    schema: Mapping[str, Any],
    value: list[Any] | tuple[Any, ...],
    path: str,
) -> None:
    minimum = schema.get("minItems")
    if minimum is not None and len(value) < minimum:
        raise RunActionCodingAgentSchemaError(f"{path} has fewer than {minimum} items")
    items = schema.get("items")
    if items is not None:
        for position, child in enumerate(value):
            _apply_schema_node(items, child, f"{path}[{position}]")


def _apply_string_schema(
    schema: Mapping[str, Any],
    value: str,
    path: str,
) -> None:
    minimum = schema.get("minLength")
    if minimum is not None and len(value) < minimum:
        raise RunActionCodingAgentSchemaError(
            f"{path} has fewer than {minimum} characters"
        )
    pattern = schema.get("pattern")
    if pattern is not None and re.search(pattern, value) is None:
        raise RunActionCodingAgentSchemaError(
            f"{path} does not match its required pattern"
        )


def _validate_json_value(value: Any, path: str) -> None:
    if value is None or type(value) is bool or isinstance(value, str):
        return
    if type(value) is int:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise RunActionCodingAgentSchemaError(
                f"{path} contains a non-finite number"
            )
        return
    if _is_object(value):
        if any(not isinstance(name, str) for name in value):
            raise RunActionCodingAgentSchemaError(f"{path} object keys must be strings")
        for name, child in value.items():
            _validate_json_value(child, f"{path}[{name!r}]")
        return
    if _is_array(value):
        for position, child in enumerate(value):
            _validate_json_value(child, f"{path}[{position}]")
        return
    raise RunActionCodingAgentSchemaError(
        f"{path} contains a value outside the JSON data model"
    )


def _matches_type(value: Any, declared_type: str) -> bool:
    if declared_type == "object":
        return _is_object(value)
    if declared_type == "array":
        return _is_array(value)
    if declared_type == "string":
        return isinstance(value, str)
    if declared_type == "number":
        return type(value) in {int, float} and (
            type(value) is int or math.isfinite(value)
        )
    if declared_type == "integer":
        return type(value) is int or (
            type(value) is float and math.isfinite(value) and value.is_integer()
        )
    if declared_type == "boolean":
        return type(value) is bool
    if declared_type == "null":
        return value is None
    raise RunActionCodingAgentSchemaError(
        f"unsupported coding-agent schema type: {declared_type}"
    )


def _is_object(value: Any) -> bool:
    return isinstance(value, Mapping)


def _is_array(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _json_values_equal(left: Any, right: Any) -> bool:
    if type(left) in {int, float} and type(right) in {int, float}:
        return left == right
    if _is_object(left) and _is_object(right):
        return set(left) == set(right) and all(
            _json_values_equal(left[key], right[key]) for key in left
        )
    if _is_array(left) and _is_array(right):
        return len(left) == len(right) and all(
            _json_values_equal(left_child, right_child)
            for left_child, right_child in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right


__all__ = [
    "CODING_AGENT_JSON_SCHEMA_DIALECT",
    "RunActionCodingAgentSchemaError",
    "validate_run_action_coding_agent_output",
    "validate_run_action_coding_agent_schema",
]
