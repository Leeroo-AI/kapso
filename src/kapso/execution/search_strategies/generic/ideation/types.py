"""Strict domain contracts for evidence-directed ideation."""

import math
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, Optional, Tuple, Type, TypeVar

_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*_[0-9a-f]{32}$")
_EnumType = TypeVar("_EnumType", bound=Enum)


def utc_now() -> str:
    """Return a stable UTC timestamp representation."""
    return datetime.now(timezone.utc).isoformat()


def new_identifier(prefix: str) -> str:
    """Create a descriptive opaque identifier."""
    _require_nonempty_string(prefix, "identifier prefix")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", prefix):
        raise ValueError("Identifier prefix must be lowercase snake case")
    return f"{prefix}_{uuid.uuid4().hex}"


def content_identifier(prefix: str, digest: str) -> str:
    """Create a typed identifier from canonical content."""
    _require_sha256(digest, "content identifier digest")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", prefix):
        raise ValueError("Identifier prefix must be lowercase snake case")
    return f"{prefix}_{digest[:32]}"


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_optional_string(value: Any, name: str) -> Optional[str]:
    if value is None:
        return None
    return _require_nonempty_string(value, name)


def _require_identifier(value: Any, name: str) -> str:
    identifier = _require_nonempty_string(value, name)
    if not _IDENTIFIER_PATTERN.fullmatch(identifier):
        raise ValueError(f"{name} must be an opaque Kapso identifier")
    return identifier


def _require_optional_identifier(value: Any, name: str) -> Optional[str]:
    if value is None:
        return None
    return _require_identifier(value, name)


def _require_typed_identifier(value: Any, name: str, prefix: str) -> str:
    identifier = _require_identifier(value, name)
    if not identifier.startswith(prefix + "_"):
        raise ValueError(f"{name} must use the {prefix} identifier prefix")
    return identifier


def _require_optional_typed_identifier(
    value: Any,
    name: str,
    prefix: str,
) -> Optional[str]:
    if value is None:
        return None
    return _require_typed_identifier(value, name, prefix)


def _require_integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _require_optional_integer(
    value: Any,
    name: str,
    minimum: int = 0,
) -> Optional[int]:
    if value is None:
        return None
    return _require_integer(value, name, minimum)


def _require_number(
    value: Any,
    name: str,
    minimum: Optional[float] = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite")
    numeric = float(value)
    if minimum is not None and numeric < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return numeric


def _require_optional_number(
    value: Any,
    name: str,
    minimum: Optional[float] = None,
) -> Optional[float]:
    if value is None:
        return None
    return _require_number(value, name, minimum)


def _require_strings(values: Any, name: str) -> Tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list of strings")
    result = tuple(_require_nonempty_string(value, name) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _require_identifiers(values: Any, name: str) -> Tuple[str, ...]:
    identifiers = _require_strings(values, name)
    for identifier in identifiers:
        _require_identifier(identifier, name)
    return identifiers


def _require_typed_identifiers(
    values: Any,
    name: str,
    prefix: str,
) -> Tuple[str, ...]:
    identifiers = _require_strings(values, name)
    for identifier in identifiers:
        _require_typed_identifier(identifier, name, prefix)
    return identifiers


def _require_timestamp(value: Any, name: str) -> str:
    timestamp = _require_nonempty_string(value, name)
    parsed = datetime.fromisoformat(timestamp)
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{name} must include a UTC offset")
    return timestamp


def _require_sha256(value: Any, name: str) -> str:
    digest = _require_nonempty_string(value, name)
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _require_exact_keys(data: Any, expected: Iterable[str], name: str) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be an object")
    expected_keys = set(expected)
    actual_keys = set(data)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unknown = sorted(actual_keys - expected_keys)
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        raise ValueError(f"{name} fields are invalid: {'; '.join(details)}")


def _parse_enum(enum_type: Type[_EnumType], value: Any, name: str) -> _EnumType:
    if not isinstance(value, str) or value not in {
        member.value for member in enum_type
    }:
        raise ValueError(f"{name} is invalid")
    return enum_type(value)


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class JsonRecord:
    """Mixin for deterministic dataclass serialization."""

    def to_dict(self) -> Dict[str, Any]:
        return _json_value(asdict(self))


class BatchStatus(str, Enum):
    PLANNED = "planned"
    GENERATED = "generated"
    ANALYZED = "analyzed"
    SELECTED = "selected"
    BRIDGED = "bridged"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class IdeaStatus(str, Enum):
    GENERATED = "generated"
    INVALID = "invalid"
    DEFERRED = "deferred"
    REJECTED = "rejected"
    SELECTED = "selected"
    IMPLEMENTING = "implementing"
    EVALUATED = "evaluated"
    FAILED_TECHNICAL = "failed_technical"
    ABANDONED = "abandoned"


class CandidateDispositionKind(str, Enum):
    SELECTED = "selected"
    DEFERRED = "deferred"
    REJECTED = "rejected"
    INVALID = "invalid"


class ClaimKind(str, Enum):
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    CONSTRAINT = "constraint"


class EvidenceStatus(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT = "insufficient"


class EvidenceSignal(str, Enum):
    NO_COMPARABLE_EXPERIMENT = "no_comparable_experiment"
    RECOVERABLE_TECHNICAL_FAILURE = "recoverable_technical_failure"
    FIDELITY_PROMOTION_REQUIRED = "fidelity_promotion_required"
    PROXY_FULL_DIVERGENCE = "proxy_full_divergence"
    SURPRISING_GAIN = "surprising_gain"
    CREDIBLE_IMPROVEMENT = "credible_improvement"
    PLATEAU = "plateau"
    SUPPORTED_LEVER = "supported_lever"
    CONTRADICTED_LEVER = "contradicted_lever"
    DIVERSITY_COLLAPSE = "diversity_collapse"
    GAP_DEBT = "gap_debt"
    DELIVERY_INCUMBENT = "delivery_incumbent"
    PROVISIONAL_NOISE = "provisional_noise"


class ObjectiveDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class GapState(str, Enum):
    OPEN = "open"
    INCONCLUSIVE = "inconclusive"
    CLOSED = "closed"


class CampaignAction(str, Enum):
    IDEATE = "ideate"
    RECOVER = "recover"
    FINALIZE = "finalize"


class IdeationMode(str, Enum):
    BOOTSTRAP = "bootstrap"
    RECOVER = "recover"
    VERIFY = "verify"
    EXPLOIT = "exploit"
    EXPLORE = "explore"


class OperatorKind(str, Enum):
    INDEPENDENT_DRAFT = "independent_draft"
    TARGET_GAP = "target_gap"
    ATOMIC_REFINE = "atomic_refine"
    ABLATE = "ablate"
    MECHANISM_SHIFT = "mechanism_shift"
    CROSSOVER = "crossover"
    VERIFY = "verify"
    RECOVER = "recover"


class ParentPlanKind(str, Enum):
    BEST_VALID = "best_valid"
    BASELINE = "baseline"
    SPECIFIC_EXPERIMENT = "specific_experiment"
    RECOVER_BRANCH = "recover_branch"


class EvaluationStatus(str, Enum):
    NOT_RUN = "not_run"
    VALID = "valid"
    INVALID = "invalid"
    INCONCLUSIVE = "inconclusive"


class ImplementationStatus(str, Enum):
    NOT_STARTED = "not_started"
    COMPLETED = "completed"
    FAILED_TECHNICAL = "failed_technical"
    ABANDONED = "abandoned"


BATCH_TRANSITIONS = {
    BatchStatus.PLANNED: frozenset({BatchStatus.GENERATED, BatchStatus.ABANDONED}),
    BatchStatus.GENERATED: frozenset({BatchStatus.ANALYZED, BatchStatus.ABANDONED}),
    BatchStatus.ANALYZED: frozenset({BatchStatus.SELECTED, BatchStatus.ABANDONED}),
    BatchStatus.SELECTED: frozenset({BatchStatus.BRIDGED, BatchStatus.ABANDONED}),
    BatchStatus.BRIDGED: frozenset({BatchStatus.COMPLETED, BatchStatus.ABANDONED}),
    BatchStatus.COMPLETED: frozenset(),
    BatchStatus.ABANDONED: frozenset(),
}

IDEA_TRANSITIONS = {
    IdeaStatus.GENERATED: frozenset(
        {
            IdeaStatus.INVALID,
            IdeaStatus.DEFERRED,
            IdeaStatus.REJECTED,
            IdeaStatus.SELECTED,
            IdeaStatus.ABANDONED,
        }
    ),
    IdeaStatus.DEFERRED: frozenset(
        {
            IdeaStatus.INVALID,
            IdeaStatus.REJECTED,
            IdeaStatus.SELECTED,
            IdeaStatus.ABANDONED,
        }
    ),
    IdeaStatus.SELECTED: frozenset({IdeaStatus.IMPLEMENTING, IdeaStatus.ABANDONED}),
    IdeaStatus.IMPLEMENTING: frozenset(
        {
            IdeaStatus.EVALUATED,
            IdeaStatus.FAILED_TECHNICAL,
            IdeaStatus.ABANDONED,
        }
    ),
    IdeaStatus.INVALID: frozenset(),
    IdeaStatus.REJECTED: frozenset(),
    IdeaStatus.EVALUATED: frozenset(),
    IdeaStatus.FAILED_TECHNICAL: frozenset(),
    IdeaStatus.ABANDONED: frozenset(),
}

GAP_TRANSITIONS = {
    GapState.OPEN: frozenset({GapState.INCONCLUSIVE, GapState.CLOSED}),
    GapState.INCONCLUSIVE: frozenset({GapState.CLOSED}),
    GapState.CLOSED: frozenset(),
}


def require_batch_transition(current: BatchStatus, target: BatchStatus) -> None:
    if target not in BATCH_TRANSITIONS[current]:
        raise ValueError(f"illegal batch transition: {current.value} -> {target.value}")


def require_idea_transition(current: IdeaStatus, target: IdeaStatus) -> None:
    if target not in IDEA_TRANSITIONS[current]:
        raise ValueError(f"illegal idea transition: {current.value} -> {target.value}")


def require_gap_transition(current: GapState, target: GapState) -> None:
    if target not in GAP_TRANSITIONS[current]:
        raise ValueError(f"illegal gap transition: {current.value} -> {target.value}")


@dataclass(frozen=True)
class CodingAgentCallRequest(JsonRecord):
    role: str
    cli: str
    model: str
    prompt: str
    workspace: str
    timeout_seconds: float
    effort: Optional[str] = None
    allowed_tools: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty_string(self.role, "coding-agent role")
        if self.cli not in {"codex", "claude_code"}:
            raise ValueError("coding-agent cli must be codex or claude_code")
        _require_nonempty_string(self.model, "coding-agent model")
        _require_nonempty_string(self.prompt, "coding-agent prompt")
        _require_nonempty_string(self.workspace, "coding-agent workspace")
        timeout = _require_number(
            self.timeout_seconds,
            "coding-agent timeout",
            0.0,
        )
        if timeout == 0.0:
            raise ValueError("coding-agent timeout must be greater than zero")
        _require_optional_string(self.effort, "coding-agent effort")
        object.__setattr__(
            self,
            "allowed_tools",
            _require_strings(self.allowed_tools, "coding-agent allowed tools"),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodingAgentCallRequest":
        _require_exact_keys(
            data,
            {
                "role",
                "cli",
                "model",
                "prompt",
                "workspace",
                "timeout_seconds",
                "effort",
                "allowed_tools",
            },
            "coding-agent request",
        )
        return cls(**data)


@dataclass(frozen=True)
class CodingAgentCallResult(JsonRecord):
    output: str
    duration_seconds: float
    cost_usd: Optional[float]
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    artifacts: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.output, str):
            raise ValueError("coding-agent output must be a string")
        _require_number(self.duration_seconds, "coding-agent duration", 0.0)
        _require_optional_number(self.cost_usd, "coding-agent cost", 0.0)
        _require_optional_integer(self.input_tokens, "coding-agent input tokens")
        _require_optional_integer(self.output_tokens, "coding-agent output tokens")
        object.__setattr__(
            self,
            "artifacts",
            _require_strings(self.artifacts, "coding-agent artifacts"),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodingAgentCallResult":
        _require_exact_keys(
            data,
            {
                "output",
                "duration_seconds",
                "cost_usd",
                "input_tokens",
                "output_tokens",
                "artifacts",
            },
            "coding-agent result",
        )
        return cls(**data)


@dataclass(frozen=True)
class EmbeddingTelemetry(JsonRecord):
    provider: str
    model: str
    call_count: int
    input_tokens: Optional[int]
    duration_seconds: float

    def __post_init__(self) -> None:
        _require_nonempty_string(self.provider, "embedding telemetry provider")
        _require_nonempty_string(self.model, "embedding telemetry model")
        _require_integer(self.call_count, "embedding call count", 1)
        _require_optional_integer(self.input_tokens, "embedding input tokens")
        _require_number(self.duration_seconds, "embedding duration", 0.0)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingTelemetry":
        _require_exact_keys(
            data,
            {
                "provider",
                "model",
                "call_count",
                "input_tokens",
                "duration_seconds",
            },
            "embedding telemetry",
        )
        return cls(**data)


@dataclass(frozen=True)
class EmbeddingRecord(JsonRecord):
    provider: str
    model: str
    dimensions: int
    input_hash: str
    vector: Tuple[float, ...]

    def __post_init__(self) -> None:
        _require_nonempty_string(self.provider, "embedding provider")
        _require_nonempty_string(self.model, "embedding model")
        _require_integer(self.dimensions, "embedding dimensions", 1)
        _require_sha256(self.input_hash, "embedding input hash")
        if not isinstance(self.vector, (list, tuple)):
            raise ValueError("embedding vector must be a list")
        vector = tuple(
            _require_number(value, "embedding vector value") for value in self.vector
        )
        if len(vector) != self.dimensions:
            raise ValueError("embedding dimensions must match vector length")
        object.__setattr__(self, "vector", vector)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingRecord":
        _require_exact_keys(
            data,
            {"provider", "model", "dimensions", "input_hash", "vector"},
            "embedding record",
        )
        return cls(**data)


@dataclass(frozen=True)
class IdeaDescriptor(JsonRecord):
    approach_family: str
    intervention_target: str
    mechanism: str
    expected_effect: str

    def __post_init__(self) -> None:
        _require_nonempty_string(self.approach_family, "approach family")
        _require_nonempty_string(self.intervention_target, "intervention target")
        _require_nonempty_string(self.mechanism, "mechanism")
        _require_nonempty_string(self.expected_effect, "expected effect")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IdeaDescriptor":
        _require_exact_keys(
            data,
            {"approach_family", "intervention_target", "mechanism", "expected_effect"},
            "idea descriptor",
        )
        return cls(**data)


@dataclass(frozen=True)
class ParentPlan(JsonRecord):
    kind: ParentPlanKind
    experiment_node_id: Optional[int] = None
    source_idea_ids: Tuple[str, ...] = ()
    source_experiment_node_ids: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ParentPlanKind):
            raise ValueError("parent plan kind is invalid")
        _require_optional_integer(self.experiment_node_id, "parent experiment node")
        object.__setattr__(
            self,
            "source_idea_ids",
            _require_typed_identifiers(
                self.source_idea_ids,
                "parent source idea ids",
                "idea",
            ),
        )
        if not isinstance(self.source_experiment_node_ids, (list, tuple)):
            raise ValueError("parent source experiment ids must be a list")
        source_nodes = tuple(
            _require_integer(value, "parent source experiment node")
            for value in self.source_experiment_node_ids
        )
        if len(set(source_nodes)) != len(source_nodes):
            raise ValueError("parent source experiment ids must be unique")
        object.__setattr__(self, "source_experiment_node_ids", source_nodes)
        if (
            self.kind
            in {
                ParentPlanKind.SPECIFIC_EXPERIMENT,
                ParentPlanKind.RECOVER_BRANCH,
            }
            and self.experiment_node_id is None
        ):
            raise ValueError("specific and recovery parent plans require a node")
        if (
            self.kind
            in {
                ParentPlanKind.BEST_VALID,
                ParentPlanKind.BASELINE,
            }
            and self.experiment_node_id is not None
        ):
            raise ValueError("best and baseline parent plans resolve nodes later")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParentPlan":
        _require_exact_keys(
            data,
            {
                "kind",
                "experiment_node_id",
                "source_idea_ids",
                "source_experiment_node_ids",
            },
            "parent plan",
        )
        return cls(
            kind=_parse_enum(ParentPlanKind, data["kind"], "parent plan kind"),
            experiment_node_id=data["experiment_node_id"],
            source_idea_ids=data["source_idea_ids"],
            source_experiment_node_ids=data["source_experiment_node_ids"],
        )


@dataclass(frozen=True)
class ResolvedParentSnapshot(JsonRecord):
    node_id: int
    branch_name: str
    git_ref: str
    materialized_ref: str
    diff_base_ref: str
    feedback_base_ref: str

    def __post_init__(self) -> None:
        _require_integer(self.node_id, "resolved parent node id")
        _require_nonempty_string(self.branch_name, "resolved parent branch")
        _require_nonempty_string(self.git_ref, "resolved parent git ref")
        _require_nonempty_string(
            self.materialized_ref,
            "resolved parent materialized ref",
        )
        _require_nonempty_string(self.diff_base_ref, "resolved parent diff base")
        _require_nonempty_string(
            self.feedback_base_ref,
            "resolved parent feedback base",
        )
        references = {
            self.git_ref,
            self.materialized_ref,
            self.diff_base_ref,
            self.feedback_base_ref,
        }
        if len(references) != 1:
            raise ValueError("resolved parent refs must identify one immutable base")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResolvedParentSnapshot":
        _require_exact_keys(
            data,
            {
                "node_id",
                "branch_name",
                "git_ref",
                "materialized_ref",
                "diff_base_ref",
                "feedback_base_ref",
            },
            "resolved parent snapshot",
        )
        return cls(**data)


@dataclass(frozen=True)
class OperatorBrief(JsonRecord):
    operator: OperatorKind
    rationale: str
    descriptor_target: IdeaDescriptor
    parent_plan: ParentPlan
    target_gap_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.operator, OperatorKind):
            raise ValueError("operator brief kind is invalid")
        _require_nonempty_string(self.rationale, "operator rationale")
        if not isinstance(self.descriptor_target, IdeaDescriptor):
            raise ValueError("operator descriptor target is invalid")
        if not isinstance(self.parent_plan, ParentPlan):
            raise ValueError("operator parent plan is invalid")
        _require_optional_typed_identifier(
            self.target_gap_id,
            "operator target gap",
            "gap",
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OperatorBrief":
        _require_exact_keys(
            data,
            {
                "operator",
                "rationale",
                "descriptor_target",
                "parent_plan",
                "target_gap_id",
            },
            "operator brief",
        )
        return cls(
            operator=_parse_enum(OperatorKind, data["operator"], "operator kind"),
            rationale=data["rationale"],
            descriptor_target=IdeaDescriptor.from_dict(data["descriptor_target"]),
            parent_plan=ParentPlan.from_dict(data["parent_plan"]),
            target_gap_id=data["target_gap_id"],
        )


@dataclass(frozen=True)
class PolicyReason(JsonRecord):
    code: str
    statement: str
    evidence_refs: Tuple[str, ...]

    def __post_init__(self) -> None:
        _require_nonempty_string(self.code, "policy reason code")
        _require_nonempty_string(self.statement, "policy reason statement")
        object.__setattr__(
            self,
            "evidence_refs",
            _require_strings(self.evidence_refs, "policy reason evidence refs"),
        )
        if not self.evidence_refs:
            raise ValueError("policy reasons require evidence references")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyReason":
        _require_exact_keys(
            data,
            {"code", "statement", "evidence_refs"},
            "policy reason",
        )
        return cls(**data)


@dataclass(frozen=True)
class PolicyDecision(JsonRecord):
    action: CampaignAction
    mode: Optional[IdeationMode]
    reasons: Tuple[PolicyReason, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.action, CampaignAction):
            raise ValueError("campaign action is invalid")
        if self.mode is not None and not isinstance(self.mode, IdeationMode):
            raise ValueError("ideation mode is invalid")
        if not isinstance(self.reasons, (list, tuple)) or not all(
            isinstance(reason, PolicyReason) for reason in self.reasons
        ):
            raise ValueError("policy reasons are invalid")
        object.__setattr__(self, "reasons", tuple(self.reasons))
        if not self.reasons:
            raise ValueError("policy decision requires at least one reason")
        if self.action == CampaignAction.FINALIZE and self.mode is not None:
            raise ValueError("finalize decisions must not have an ideation mode")
        if self.action == CampaignAction.IDEATE and self.mode is None:
            raise ValueError("ideate decisions require an ideation mode")
        if self.action == CampaignAction.IDEATE and self.mode == IdeationMode.RECOVER:
            raise ValueError("recovery is an execution action, not ideation")
        if self.action == CampaignAction.RECOVER and self.mode != IdeationMode.RECOVER:
            raise ValueError("recover actions require recover mode")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyDecision":
        _require_exact_keys(data, {"action", "mode", "reasons"}, "policy decision")
        return cls(
            action=_parse_enum(CampaignAction, data["action"], "campaign action"),
            mode=(
                None
                if data["mode"] is None
                else _parse_enum(IdeationMode, data["mode"], "ideation mode")
            ),
            reasons=tuple(PolicyReason.from_dict(reason) for reason in data["reasons"]),
        )


@dataclass(frozen=True)
class SearchDirective(JsonRecord):
    decision: PolicyDecision
    evidence_snapshot_id: str
    capacity_snapshot_id: str
    operator_briefs: Tuple[OperatorBrief, ...]
    candidate_quota: int
    repair_quota: int
    validation_requirements: Tuple[str, ...] = ()
    allowed_parent_plan_kinds: Tuple[ParentPlanKind, ...] = ()
    terminal_constraints: Tuple[str, ...] = ()
    reserved_gap_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision, PolicyDecision):
            raise ValueError("search directive decision is invalid")
        _require_typed_identifier(
            self.evidence_snapshot_id,
            "evidence snapshot id",
            "evidence_snapshot",
        )
        _require_typed_identifier(
            self.capacity_snapshot_id,
            "capacity snapshot id",
            "capacity_snapshot",
        )
        if not isinstance(self.operator_briefs, (list, tuple)) or not all(
            isinstance(brief, OperatorBrief) for brief in self.operator_briefs
        ):
            raise ValueError("search directive operator briefs are invalid")
        object.__setattr__(self, "operator_briefs", tuple(self.operator_briefs))
        _require_integer(self.candidate_quota, "candidate quota")
        _require_integer(self.repair_quota, "repair quota")
        if self.repair_quota > 1:
            raise ValueError("search directive permits at most one repair round")
        object.__setattr__(
            self,
            "validation_requirements",
            _require_strings(
                self.validation_requirements,
                "validation requirements",
            ),
        )
        if not isinstance(self.allowed_parent_plan_kinds, (list, tuple)) or not all(
            isinstance(kind, ParentPlanKind) for kind in self.allowed_parent_plan_kinds
        ):
            raise ValueError("allowed parent plan kinds are invalid")
        if len(set(self.allowed_parent_plan_kinds)) != len(
            self.allowed_parent_plan_kinds
        ):
            raise ValueError("allowed parent plan kinds must be unique")
        object.__setattr__(
            self,
            "allowed_parent_plan_kinds",
            tuple(self.allowed_parent_plan_kinds),
        )
        object.__setattr__(
            self,
            "terminal_constraints",
            _require_strings(self.terminal_constraints, "terminal constraints"),
        )
        _require_optional_typed_identifier(
            self.reserved_gap_id,
            "reserved gap id",
            "gap",
        )
        if self.decision.action == CampaignAction.FINALIZE and self.operator_briefs:
            raise ValueError("finalize directives must not contain operators")
        if (
            self.decision.action == CampaignAction.FINALIZE
            and self.candidate_quota != 0
        ):
            raise ValueError("finalize directives must have zero candidate quota")
        if self.decision.action == CampaignAction.FINALIZE and self.repair_quota != 0:
            raise ValueError("finalize directives must have zero repair quota")
        if (
            self.decision.action == CampaignAction.FINALIZE
            and self.reserved_gap_id is not None
        ):
            raise ValueError("finalize directives cannot reserve a gap")
        if self.decision.action == CampaignAction.RECOVER:
            if self.candidate_quota != 0 or self.repair_quota != 0:
                raise ValueError("recover directives cannot generate candidates")
            if len(self.operator_briefs) != 1 or (
                self.operator_briefs[0].operator != OperatorKind.RECOVER
            ):
                raise ValueError("recover directives require one recovery brief")
            if self.reserved_gap_id is not None:
                raise ValueError("recover directives cannot reserve a gap")
        if self.decision.action == CampaignAction.IDEATE and not self.operator_briefs:
            raise ValueError("ideate directives require operators")
        if self.decision.action == CampaignAction.IDEATE and self.candidate_quota == 0:
            raise ValueError("ideate directives require a candidate quota")
        if (
            self.decision.action == CampaignAction.IDEATE
            and self.candidate_quota != len(self.operator_briefs)
        ):
            raise ValueError("candidate quota must match operator brief count")
        if self.decision.action == CampaignAction.IDEATE and not (
            self.allowed_parent_plan_kinds
        ):
            raise ValueError("ideate directives require allowed parent plans")
        if any(
            brief.parent_plan.kind not in self.allowed_parent_plan_kinds
            for brief in self.operator_briefs
        ):
            raise ValueError("operator brief uses a disallowed parent plan")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchDirective":
        _require_exact_keys(
            data,
            {
                "decision",
                "evidence_snapshot_id",
                "capacity_snapshot_id",
                "operator_briefs",
                "candidate_quota",
                "repair_quota",
                "validation_requirements",
                "allowed_parent_plan_kinds",
                "terminal_constraints",
                "reserved_gap_id",
            },
            "search directive",
        )
        return cls(
            decision=PolicyDecision.from_dict(data["decision"]),
            evidence_snapshot_id=data["evidence_snapshot_id"],
            capacity_snapshot_id=data["capacity_snapshot_id"],
            operator_briefs=tuple(
                OperatorBrief.from_dict(brief) for brief in data["operator_briefs"]
            ),
            candidate_quota=data["candidate_quota"],
            repair_quota=data["repair_quota"],
            validation_requirements=data["validation_requirements"],
            allowed_parent_plan_kinds=tuple(
                _parse_enum(
                    ParentPlanKind,
                    kind,
                    "allowed parent plan kind",
                )
                for kind in data["allowed_parent_plan_kinds"]
            ),
            terminal_constraints=data["terminal_constraints"],
            reserved_gap_id=data["reserved_gap_id"],
        )


@dataclass(frozen=True)
class SimilarityMatch(JsonRecord):
    idea_id: str
    similarity: float

    def __post_init__(self) -> None:
        _require_typed_identifier(self.idea_id, "similar idea id", "idea")
        similarity = _require_number(self.similarity, "idea similarity", -1.0)
        if similarity > 1.0:
            raise ValueError("idea similarity must be <= 1")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimilarityMatch":
        _require_exact_keys(data, {"idea_id", "similarity"}, "similarity match")
        return cls(**data)


@dataclass(frozen=True)
class CandidateAnalysis(JsonRecord):
    idea_id: str
    eligible: bool
    hard_failures: Tuple[str, ...] = ()
    unsupported_claims: Tuple[str, ...] = ()
    exact_duplicate_of: Optional[str] = None
    exact_duplicate_changed_conditions: Tuple[str, ...] = ()
    semantic_neighbors: Tuple[SimilarityMatch, ...] = ()

    def __post_init__(self) -> None:
        _require_typed_identifier(self.idea_id, "analyzed idea id", "idea")
        if not isinstance(self.eligible, bool):
            raise ValueError("candidate eligibility must be boolean")
        object.__setattr__(
            self,
            "hard_failures",
            _require_strings(self.hard_failures, "candidate hard failures"),
        )
        object.__setattr__(
            self,
            "unsupported_claims",
            _require_strings(
                self.unsupported_claims,
                "candidate unsupported claims",
            ),
        )
        _require_optional_typed_identifier(
            self.exact_duplicate_of,
            "exact duplicate id",
            "idea",
        )
        object.__setattr__(
            self,
            "exact_duplicate_changed_conditions",
            _require_strings(
                self.exact_duplicate_changed_conditions,
                "exact duplicate changed conditions",
            ),
        )
        if not isinstance(self.semantic_neighbors, (list, tuple)) or not all(
            isinstance(match, SimilarityMatch) for match in self.semantic_neighbors
        ):
            raise ValueError("candidate semantic neighbors are invalid")
        object.__setattr__(self, "semantic_neighbors", tuple(self.semantic_neighbors))
        neighbor_ids = tuple(match.idea_id for match in self.semantic_neighbors)
        if len(set(neighbor_ids)) != len(neighbor_ids):
            raise ValueError("candidate semantic neighbors must be unique")
        if self.idea_id in neighbor_ids:
            raise ValueError("candidate cannot be its own semantic neighbor")
        if (
            tuple(
                sorted(
                    self.semantic_neighbors,
                    key=lambda match: (-match.similarity, match.idea_id),
                )
            )
            != self.semantic_neighbors
        ):
            raise ValueError(
                "candidate semantic neighbors must be deterministically ordered"
            )
        if self.eligible and self.hard_failures:
            raise ValueError("eligible candidates cannot have hard failures")
        if self.exact_duplicate_of == self.idea_id:
            raise ValueError("candidate cannot duplicate itself")
        if (
            self.exact_duplicate_of is not None
            and self.eligible
            and not self.exact_duplicate_changed_conditions
        ):
            raise ValueError(
                "eligible exact duplicates require materially changed conditions"
            )
        if self.exact_duplicate_of is None and self.exact_duplicate_changed_conditions:
            raise ValueError("changed duplicate conditions require a duplicate")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateAnalysis":
        _require_exact_keys(
            data,
            {
                "idea_id",
                "eligible",
                "hard_failures",
                "unsupported_claims",
                "exact_duplicate_of",
                "exact_duplicate_changed_conditions",
                "semantic_neighbors",
            },
            "candidate analysis",
        )
        return cls(
            idea_id=data["idea_id"],
            eligible=data["eligible"],
            hard_failures=data["hard_failures"],
            unsupported_claims=data["unsupported_claims"],
            exact_duplicate_of=data["exact_duplicate_of"],
            exact_duplicate_changed_conditions=data[
                "exact_duplicate_changed_conditions"
            ],
            semantic_neighbors=tuple(
                SimilarityMatch.from_dict(match) for match in data["semantic_neighbors"]
            ),
        )


@dataclass(frozen=True)
class DiagnosisAudit(JsonRecord):
    claim_id: str
    status: EvidenceStatus
    evidence_refs: Tuple[str, ...]

    def __post_init__(self) -> None:
        _require_typed_identifier(self.claim_id, "diagnosis claim id", "claim")
        if not isinstance(self.status, EvidenceStatus):
            raise ValueError("diagnosis status is invalid")
        object.__setattr__(
            self,
            "evidence_refs",
            _require_strings(self.evidence_refs, "diagnosis evidence refs"),
        )
        if self.status != EvidenceStatus.INSUFFICIENT and not self.evidence_refs:
            raise ValueError("supported diagnosis entries require evidence")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiagnosisAudit":
        _require_exact_keys(
            data, {"claim_id", "status", "evidence_refs"}, "diagnosis audit"
        )
        return cls(
            claim_id=data["claim_id"],
            status=_parse_enum(EvidenceStatus, data["status"], "evidence status"),
            evidence_refs=data["evidence_refs"],
        )


@dataclass(frozen=True)
class CandidateDisposition(JsonRecord):
    idea_id: str
    disposition: CandidateDispositionKind
    reason: str

    def __post_init__(self) -> None:
        _require_typed_identifier(self.idea_id, "candidate disposition idea id", "idea")
        if not isinstance(self.disposition, CandidateDispositionKind):
            raise ValueError("candidate disposition is invalid")
        _require_nonempty_string(self.reason, "candidate disposition reason")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateDisposition":
        _require_exact_keys(
            data,
            {"idea_id", "disposition", "reason"},
            "candidate disposition",
        )
        return cls(
            idea_id=data["idea_id"],
            disposition=_parse_enum(
                CandidateDispositionKind,
                data["disposition"],
                "candidate disposition",
            ),
            reason=data["reason"],
        )


@dataclass(frozen=True)
class SelectionDecision(JsonRecord):
    selected_idea_id: str
    fallback_idea_ids: Tuple[str, ...]
    dispositions: Tuple[CandidateDisposition, ...]
    diagnosis_audit: Tuple[DiagnosisAudit, ...]
    hard_rule_results: Tuple[str, ...]
    gap_decisions: Tuple[str, ...]
    duplicate_overrides: Tuple[str, ...]
    decision_summary: str
    expected_benefit: float
    expected_cost: float

    def __post_init__(self) -> None:
        _require_typed_identifier(self.selected_idea_id, "selected idea id", "idea")
        object.__setattr__(
            self,
            "fallback_idea_ids",
            _require_typed_identifiers(
                self.fallback_idea_ids,
                "fallback idea ids",
                "idea",
            ),
        )
        if self.selected_idea_id in self.fallback_idea_ids:
            raise ValueError("selected idea cannot also be a fallback")
        if not isinstance(self.dispositions, (list, tuple)) or not all(
            isinstance(disposition, CandidateDisposition)
            for disposition in self.dispositions
        ):
            raise ValueError("candidate dispositions are invalid")
        object.__setattr__(self, "dispositions", tuple(self.dispositions))
        disposition_ids = tuple(item.idea_id for item in self.dispositions)
        if len(set(disposition_ids)) != len(disposition_ids):
            raise ValueError("candidate dispositions must be unique")
        disposition_by_id = {
            item.idea_id: item.disposition for item in self.dispositions
        }
        if (
            disposition_by_id.get(self.selected_idea_id)
            != CandidateDispositionKind.SELECTED
        ):
            raise ValueError("selected idea requires a selected disposition")
        if any(
            disposition_by_id.get(idea_id) != CandidateDispositionKind.DEFERRED
            for idea_id in self.fallback_idea_ids
        ):
            raise ValueError("fallback ideas require deferred dispositions")
        if not isinstance(self.diagnosis_audit, (list, tuple)) or not all(
            isinstance(audit, DiagnosisAudit) for audit in self.diagnosis_audit
        ):
            raise ValueError("diagnosis audit is invalid")
        object.__setattr__(self, "diagnosis_audit", tuple(self.diagnosis_audit))
        object.__setattr__(
            self,
            "hard_rule_results",
            _require_strings(self.hard_rule_results, "hard rule results"),
        )
        object.__setattr__(
            self,
            "gap_decisions",
            _require_strings(self.gap_decisions, "gap decisions"),
        )
        object.__setattr__(
            self,
            "duplicate_overrides",
            _require_strings(self.duplicate_overrides, "duplicate overrides"),
        )
        _require_nonempty_string(self.decision_summary, "selection summary")
        object.__setattr__(
            self,
            "expected_benefit",
            _require_number(self.expected_benefit, "expected benefit"),
        )
        object.__setattr__(
            self,
            "expected_cost",
            _require_number(self.expected_cost, "expected cost", 0.0),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SelectionDecision":
        _require_exact_keys(
            data,
            {
                "selected_idea_id",
                "fallback_idea_ids",
                "dispositions",
                "diagnosis_audit",
                "hard_rule_results",
                "gap_decisions",
                "duplicate_overrides",
                "decision_summary",
                "expected_benefit",
                "expected_cost",
            },
            "selection decision",
        )
        return cls(
            selected_idea_id=data["selected_idea_id"],
            fallback_idea_ids=data["fallback_idea_ids"],
            dispositions=tuple(
                CandidateDisposition.from_dict(disposition)
                for disposition in data["dispositions"]
            ),
            diagnosis_audit=tuple(
                DiagnosisAudit.from_dict(audit) for audit in data["diagnosis_audit"]
            ),
            hard_rule_results=data["hard_rule_results"],
            gap_decisions=data["gap_decisions"],
            duplicate_overrides=data["duplicate_overrides"],
            decision_summary=data["decision_summary"],
            expected_benefit=data["expected_benefit"],
            expected_cost=data["expected_cost"],
        )


@dataclass(frozen=True)
class IdeaOutcome(JsonRecord):
    evaluation_status: EvaluationStatus
    implementation_status: ImplementationStatus
    normalized_delta: Optional[float]
    validation_tier: Optional[str]
    actual_cost: Optional[float]
    actual_duration: Optional[float]
    gap_effects: Tuple[str, ...] = ()
    supported_claim_ids: Tuple[str, ...] = ()
    contradicted_claim_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation_status, EvaluationStatus):
            raise ValueError("outcome evaluation status is invalid")
        if not isinstance(self.implementation_status, ImplementationStatus):
            raise ValueError("outcome implementation status is invalid")
        normalized_delta = _require_optional_number(
            self.normalized_delta,
            "normalized delta",
        )
        object.__setattr__(self, "normalized_delta", normalized_delta)
        _require_optional_string(self.validation_tier, "validation tier")
        object.__setattr__(
            self,
            "actual_cost",
            _require_optional_number(self.actual_cost, "actual cost", 0.0),
        )
        object.__setattr__(
            self,
            "actual_duration",
            _require_optional_number(
                self.actual_duration,
                "actual duration",
                0.0,
            ),
        )
        object.__setattr__(
            self,
            "gap_effects",
            _require_strings(self.gap_effects, "gap effects"),
        )
        object.__setattr__(
            self,
            "supported_claim_ids",
            _require_typed_identifiers(
                self.supported_claim_ids,
                "supported claim ids",
                "claim",
            ),
        )
        object.__setattr__(
            self,
            "contradicted_claim_ids",
            _require_typed_identifiers(
                self.contradicted_claim_ids,
                "contradicted claim ids",
                "claim",
            ),
        )
        if set(self.supported_claim_ids) & set(self.contradicted_claim_ids):
            raise ValueError("a claim cannot be supported and contradicted")
        if (
            self.evaluation_status != EvaluationStatus.VALID
            and self.normalized_delta is not None
        ):
            raise ValueError("only valid evaluations may have normalized delta")
        if (
            self.evaluation_status == EvaluationStatus.VALID
            and self.normalized_delta is None
        ):
            raise ValueError("valid evaluations require normalized delta")
        if (
            self.implementation_status == ImplementationStatus.FAILED_TECHNICAL
            and self.evaluation_status != EvaluationStatus.NOT_RUN
        ):
            raise ValueError("technical failures cannot have evaluation evidence")
        if (
            self.evaluation_status != EvaluationStatus.NOT_RUN
            and self.implementation_status != ImplementationStatus.COMPLETED
        ):
            raise ValueError("evaluation evidence requires completed implementation")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IdeaOutcome":
        _require_exact_keys(
            data,
            {
                "evaluation_status",
                "implementation_status",
                "normalized_delta",
                "validation_tier",
                "actual_cost",
                "actual_duration",
                "gap_effects",
                "supported_claim_ids",
                "contradicted_claim_ids",
            },
            "idea outcome",
        )
        return cls(
            evaluation_status=_parse_enum(
                EvaluationStatus,
                data["evaluation_status"],
                "evaluation status",
            ),
            implementation_status=_parse_enum(
                ImplementationStatus,
                data["implementation_status"],
                "implementation status",
            ),
            normalized_delta=data["normalized_delta"],
            validation_tier=data["validation_tier"],
            actual_cost=data["actual_cost"],
            actual_duration=data["actual_duration"],
            gap_effects=data["gap_effects"],
            supported_claim_ids=data["supported_claim_ids"],
            contradicted_claim_ids=data["contradicted_claim_ids"],
        )


@dataclass(frozen=True)
class IdeaRecord(JsonRecord):
    idea_id: str
    origin_batch_id: str
    proposal: str
    operator: OperatorKind
    descriptor: IdeaDescriptor
    parent_plan: ParentPlan
    resolved_parent: ResolvedParentSnapshot
    assumptions: Tuple[str, ...]
    evidence_refs: Tuple[str, ...]
    directive_rationale: str
    evaluation_method: str
    resource_request: str
    created_at: str
    status: IdeaStatus = IdeaStatus.GENERATED
    selected_in_batch_id: Optional[str] = None
    parent_idea_ids: Tuple[str, ...] = ()
    parent_experiment_node_ids: Tuple[int, ...] = ()
    target_gap_ids: Tuple[str, ...] = ()
    claim_ids: Tuple[str, ...] = ()
    resolves_claim_ids: Tuple[str, ...] = ()
    expected_observations: Tuple[str, ...] = ()
    predicted_gain: Optional[float] = None
    predicted_cost: Optional[float] = None
    confidence: Optional[float] = None
    embedding: Optional[EmbeddingRecord] = None
    exact_duplicate_of: Optional[str] = None
    claimed_nearest_idea_id: Optional[str] = None
    claimed_nearest_experiment_node_id: Optional[int] = None
    nearest_experiment_node_ids: Tuple[int, ...] = ()
    similarity_flags: Tuple[str, ...] = ()
    generation_artifacts: Tuple[str, ...] = ()
    selection_reason: Optional[str] = None
    deferral_reason: Optional[str] = None
    rejection_reason: Optional[str] = None
    experiment_node_id: Optional[int] = None
    outcome: Optional[IdeaOutcome] = None

    def __post_init__(self) -> None:
        _require_typed_identifier(self.idea_id, "idea id", "idea")
        _require_typed_identifier(self.origin_batch_id, "origin batch id", "batch")
        _require_nonempty_string(self.proposal, "idea proposal")
        if not isinstance(self.operator, OperatorKind):
            raise ValueError("idea operator is invalid")
        if not isinstance(self.descriptor, IdeaDescriptor):
            raise ValueError("idea descriptor is invalid")
        if not isinstance(self.parent_plan, ParentPlan):
            raise ValueError("idea parent plan is invalid")
        if not isinstance(self.resolved_parent, ResolvedParentSnapshot):
            raise ValueError("idea resolved parent is invalid")
        object.__setattr__(
            self,
            "assumptions",
            _require_strings(self.assumptions, "idea assumptions"),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _require_strings(self.evidence_refs, "idea evidence refs"),
        )
        _require_nonempty_string(self.directive_rationale, "idea directive rationale")
        _require_nonempty_string(self.evaluation_method, "idea evaluation method")
        _require_nonempty_string(self.resource_request, "idea resource request")
        _require_timestamp(self.created_at, "idea created timestamp")
        if not isinstance(self.status, IdeaStatus):
            raise ValueError("idea status is invalid")
        _require_optional_typed_identifier(
            self.selected_in_batch_id,
            "selection batch id",
            "batch",
        )
        object.__setattr__(
            self,
            "parent_idea_ids",
            _require_typed_identifiers(
                self.parent_idea_ids,
                "parent idea ids",
                "idea",
            ),
        )
        if self.idea_id in self.parent_idea_ids:
            raise ValueError("idea cannot be its own parent")
        if not isinstance(self.parent_experiment_node_ids, (list, tuple)):
            raise ValueError("parent experiment node ids must be a list")
        parent_nodes = tuple(
            _require_integer(node_id, "parent experiment node id")
            for node_id in self.parent_experiment_node_ids
        )
        object.__setattr__(self, "parent_experiment_node_ids", parent_nodes)
        if len(set(parent_nodes)) != len(parent_nodes):
            raise ValueError("parent experiment node ids must be unique")
        object.__setattr__(
            self,
            "target_gap_ids",
            _require_typed_identifiers(
                self.target_gap_ids,
                "target gap ids",
                "gap",
            ),
        )
        object.__setattr__(
            self,
            "claim_ids",
            _require_typed_identifiers(
                self.claim_ids,
                "idea claim ids",
                "claim",
            ),
        )
        object.__setattr__(
            self,
            "resolves_claim_ids",
            _require_typed_identifiers(
                self.resolves_claim_ids,
                "idea resolving claim ids",
                "claim",
            ),
        )
        if not set(self.resolves_claim_ids).issubset(self.claim_ids):
            raise ValueError("resolving claim ids must be cited by the idea")
        object.__setattr__(
            self,
            "expected_observations",
            _require_strings(
                self.expected_observations,
                "expected observations",
            ),
        )
        object.__setattr__(
            self,
            "predicted_gain",
            _require_optional_number(self.predicted_gain, "predicted gain"),
        )
        object.__setattr__(
            self,
            "predicted_cost",
            _require_optional_number(self.predicted_cost, "predicted cost", 0.0),
        )
        confidence = _require_optional_number(self.confidence, "idea confidence", 0.0)
        if confidence is not None and confidence > 1.0:
            raise ValueError("idea confidence must be <= 1")
        if self.embedding is not None and not isinstance(
            self.embedding, EmbeddingRecord
        ):
            raise ValueError("idea embedding is invalid")
        _require_optional_typed_identifier(
            self.exact_duplicate_of,
            "exact duplicate idea id",
            "idea",
        )
        if self.exact_duplicate_of == self.idea_id:
            raise ValueError("idea cannot duplicate itself")
        _require_optional_typed_identifier(
            self.claimed_nearest_idea_id,
            "claimed nearest idea id",
            "idea",
        )
        if self.claimed_nearest_idea_id == self.idea_id:
            raise ValueError("idea cannot claim itself as its nearest idea")
        _require_optional_integer(
            self.claimed_nearest_experiment_node_id,
            "claimed nearest experiment node id",
        )
        if not isinstance(self.nearest_experiment_node_ids, (list, tuple)):
            raise ValueError("nearest experiment node ids must be a list")
        nearest_nodes = tuple(
            _require_integer(node_id, "nearest experiment node id")
            for node_id in self.nearest_experiment_node_ids
        )
        if len(set(nearest_nodes)) != len(nearest_nodes):
            raise ValueError("nearest experiment node ids must be unique")
        object.__setattr__(self, "nearest_experiment_node_ids", nearest_nodes)
        object.__setattr__(
            self,
            "similarity_flags",
            _require_strings(self.similarity_flags, "idea similarity flags"),
        )
        object.__setattr__(
            self,
            "generation_artifacts",
            _require_strings(self.generation_artifacts, "generation artifacts"),
        )
        _require_optional_string(self.selection_reason, "selection reason")
        _require_optional_string(self.deferral_reason, "deferral reason")
        _require_optional_string(self.rejection_reason, "rejection reason")
        _require_optional_integer(self.experiment_node_id, "experiment node id")
        if self.outcome is not None and not isinstance(self.outcome, IdeaOutcome):
            raise ValueError("idea outcome is invalid")
        if self.selected_in_batch_id is None and self.status in {
            IdeaStatus.SELECTED,
            IdeaStatus.IMPLEMENTING,
            IdeaStatus.EVALUATED,
            IdeaStatus.FAILED_TECHNICAL,
        }:
            raise ValueError("selected idea states require a selection batch")
        if self.experiment_node_id is None and self.status in {
            IdeaStatus.IMPLEMENTING,
            IdeaStatus.EVALUATED,
            IdeaStatus.FAILED_TECHNICAL,
        }:
            raise ValueError("executing idea states require an experiment node")
        if self.outcome is not None and self.status not in {
            IdeaStatus.EVALUATED,
            IdeaStatus.FAILED_TECHNICAL,
        }:
            raise ValueError("only completed ideas may have outcomes")
        unselected_statuses = {
            IdeaStatus.GENERATED,
            IdeaStatus.INVALID,
            IdeaStatus.DEFERRED,
            IdeaStatus.REJECTED,
        }
        if self.status in unselected_statuses and self.selected_in_batch_id is not None:
            raise ValueError("unselected idea states cannot have a selection batch")
        if self.status in unselected_statuses and self.experiment_node_id is not None:
            raise ValueError("unselected idea states cannot have an experiment node")
        if (
            self.status
            in {
                IdeaStatus.SELECTED,
                IdeaStatus.IMPLEMENTING,
                IdeaStatus.EVALUATED,
                IdeaStatus.FAILED_TECHNICAL,
            }
            and self.selection_reason is None
        ):
            raise ValueError("selected idea states require a selection reason")
        if self.status == IdeaStatus.DEFERRED and self.deferral_reason is None:
            raise ValueError("deferred ideas require a deferral reason")
        if self.status in {IdeaStatus.INVALID, IdeaStatus.REJECTED} and (
            self.rejection_reason is None
        ):
            raise ValueError("invalid and rejected ideas require a reason")
        if self.status in {IdeaStatus.EVALUATED, IdeaStatus.FAILED_TECHNICAL} and (
            self.outcome is None
        ):
            raise ValueError("completed idea states require an outcome")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IdeaRecord":
        fields = {
            "idea_id",
            "origin_batch_id",
            "proposal",
            "operator",
            "descriptor",
            "parent_plan",
            "resolved_parent",
            "assumptions",
            "evidence_refs",
            "directive_rationale",
            "evaluation_method",
            "resource_request",
            "created_at",
            "status",
            "selected_in_batch_id",
            "parent_idea_ids",
            "parent_experiment_node_ids",
            "target_gap_ids",
            "claim_ids",
            "resolves_claim_ids",
            "expected_observations",
            "predicted_gain",
            "predicted_cost",
            "confidence",
            "embedding",
            "exact_duplicate_of",
            "claimed_nearest_idea_id",
            "claimed_nearest_experiment_node_id",
            "nearest_experiment_node_ids",
            "similarity_flags",
            "generation_artifacts",
            "selection_reason",
            "deferral_reason",
            "rejection_reason",
            "experiment_node_id",
            "outcome",
        }
        _require_exact_keys(data, fields, "idea record")
        return cls(
            idea_id=data["idea_id"],
            origin_batch_id=data["origin_batch_id"],
            proposal=data["proposal"],
            operator=_parse_enum(OperatorKind, data["operator"], "idea operator"),
            descriptor=IdeaDescriptor.from_dict(data["descriptor"]),
            parent_plan=ParentPlan.from_dict(data["parent_plan"]),
            resolved_parent=ResolvedParentSnapshot.from_dict(data["resolved_parent"]),
            assumptions=data["assumptions"],
            evidence_refs=data["evidence_refs"],
            directive_rationale=data["directive_rationale"],
            evaluation_method=data["evaluation_method"],
            resource_request=data["resource_request"],
            created_at=data["created_at"],
            status=_parse_enum(IdeaStatus, data["status"], "idea status"),
            selected_in_batch_id=data["selected_in_batch_id"],
            parent_idea_ids=data["parent_idea_ids"],
            parent_experiment_node_ids=data["parent_experiment_node_ids"],
            target_gap_ids=data["target_gap_ids"],
            claim_ids=data["claim_ids"],
            resolves_claim_ids=data["resolves_claim_ids"],
            expected_observations=data["expected_observations"],
            predicted_gain=data["predicted_gain"],
            predicted_cost=data["predicted_cost"],
            confidence=data["confidence"],
            embedding=(
                None
                if data["embedding"] is None
                else EmbeddingRecord.from_dict(data["embedding"])
            ),
            exact_duplicate_of=data["exact_duplicate_of"],
            claimed_nearest_idea_id=data["claimed_nearest_idea_id"],
            claimed_nearest_experiment_node_id=data[
                "claimed_nearest_experiment_node_id"
            ],
            nearest_experiment_node_ids=data["nearest_experiment_node_ids"],
            similarity_flags=data["similarity_flags"],
            generation_artifacts=data["generation_artifacts"],
            selection_reason=data["selection_reason"],
            deferral_reason=data["deferral_reason"],
            rejection_reason=data["rejection_reason"],
            experiment_node_id=data["experiment_node_id"],
            outcome=(
                None
                if data["outcome"] is None
                else IdeaOutcome.from_dict(data["outcome"])
            ),
        )


@dataclass(frozen=True)
class ResurfacedIdea(JsonRecord):
    idea_id: str
    changed_conditions: Tuple[str, ...]

    def __post_init__(self) -> None:
        _require_typed_identifier(self.idea_id, "resurfaced idea id", "idea")
        object.__setattr__(
            self,
            "changed_conditions",
            _require_strings(
                self.changed_conditions,
                "resurfaced idea changed conditions",
            ),
        )
        if not self.changed_conditions:
            raise ValueError("resurfaced ideas require changed conditions")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResurfacedIdea":
        _require_exact_keys(
            data,
            {"idea_id", "changed_conditions"},
            "resurfaced idea",
        )
        return cls(**data)


@dataclass(frozen=True)
class IdeaBatch(JsonRecord):
    batch_id: str
    campaign_id: str
    iteration_index: int
    context_hash: str
    evidence_snapshot_id: str
    directive: SearchDirective
    created_at: str
    updated_at: str
    status: BatchStatus = BatchStatus.PLANNED
    generated_idea_ids: Tuple[str, ...] = ()
    resurfaced_ideas: Tuple[ResurfacedIdea, ...] = ()
    considered_idea_ids: Tuple[str, ...] = ()
    analyses: Tuple[CandidateAnalysis, ...] = ()
    selection: Optional[SelectionDecision] = None
    abandoned_reason: Optional[str] = None

    def __post_init__(self) -> None:
        _require_typed_identifier(self.batch_id, "batch id", "batch")
        _require_nonempty_string(self.campaign_id, "campaign id")
        _require_integer(self.iteration_index, "iteration index")
        _require_sha256(self.context_hash, "context hash")
        _require_typed_identifier(
            self.evidence_snapshot_id,
            "batch evidence snapshot id",
            "evidence_snapshot",
        )
        if not isinstance(self.directive, SearchDirective):
            raise ValueError("batch directive is invalid")
        if self.evidence_snapshot_id != self.directive.evidence_snapshot_id:
            raise ValueError("batch and directive evidence snapshots must match")
        _require_timestamp(self.created_at, "batch created timestamp")
        _require_timestamp(self.updated_at, "batch updated timestamp")
        if datetime.fromisoformat(self.updated_at) < datetime.fromisoformat(
            self.created_at
        ):
            raise ValueError("batch updated timestamp cannot precede creation")
        if not isinstance(self.status, BatchStatus):
            raise ValueError("batch status is invalid")
        object.__setattr__(
            self,
            "generated_idea_ids",
            _require_typed_identifiers(
                self.generated_idea_ids,
                "generated idea ids",
                "idea",
            ),
        )
        if not isinstance(self.resurfaced_ideas, (list, tuple)) or not all(
            isinstance(resurfaced, ResurfacedIdea)
            for resurfaced in self.resurfaced_ideas
        ):
            raise ValueError("batch resurfaced ideas are invalid")
        object.__setattr__(self, "resurfaced_ideas", tuple(self.resurfaced_ideas))
        resurfaced_ids = tuple(
            resurfaced.idea_id for resurfaced in self.resurfaced_ideas
        )
        if len(set(resurfaced_ids)) != len(resurfaced_ids):
            raise ValueError("batch resurfaced ideas must be unique")
        object.__setattr__(
            self,
            "considered_idea_ids",
            _require_typed_identifiers(
                self.considered_idea_ids,
                "considered idea ids",
                "idea",
            ),
        )
        if not set(self.generated_idea_ids).issubset(self.considered_idea_ids):
            raise ValueError("generated ideas must be considered")
        if set(self.generated_idea_ids) & set(resurfaced_ids):
            raise ValueError("generated ideas cannot also be resurfaced")
        if set(self.considered_idea_ids) != set(self.generated_idea_ids) | set(
            resurfaced_ids
        ):
            raise ValueError(
                "considered ideas must be generated or explicitly resurfaced"
            )
        if not isinstance(self.analyses, (list, tuple)) or not all(
            isinstance(analysis, CandidateAnalysis) for analysis in self.analyses
        ):
            raise ValueError("batch analyses are invalid")
        object.__setattr__(self, "analyses", tuple(self.analyses))
        analysis_ids = tuple(analysis.idea_id for analysis in self.analyses)
        if len(set(analysis_ids)) != len(analysis_ids):
            raise ValueError("batch analyses must be unique by idea")
        if not set(analysis_ids).issubset(self.considered_idea_ids):
            raise ValueError("batch analyses must reference considered ideas")
        if len({analysis.idea_id for analysis in self.analyses}) != len(self.analyses):
            raise ValueError("batch analyses must have unique idea ids")
        if self.selection is not None and not isinstance(
            self.selection, SelectionDecision
        ):
            raise ValueError("batch selection is invalid")
        _require_optional_string(self.abandoned_reason, "batch abandoned reason")
        if self.status == BatchStatus.ABANDONED and self.abandoned_reason is None:
            raise ValueError("abandoned batches require a reason")
        if self.selection is not None and (
            self.selection.selected_idea_id not in self.considered_idea_ids
        ):
            raise ValueError("batch selection must reference a considered idea")
        if self.selection is not None and not set(
            self.selection.fallback_idea_ids
        ).issubset(self.considered_idea_ids):
            raise ValueError("batch fallbacks must reference considered ideas")
        if self.selection is not None:
            disposition_ids = {
                disposition.idea_id for disposition in self.selection.dispositions
            }
            if disposition_ids != set(self.considered_idea_ids):
                raise ValueError(
                    "selection dispositions must cover every considered idea"
                )
            analysis_by_id = {analysis.idea_id: analysis for analysis in self.analyses}
            eligible_dispositions = {
                CandidateDispositionKind.SELECTED,
                CandidateDispositionKind.DEFERRED,
                CandidateDispositionKind.REJECTED,
            }
            for disposition in self.selection.dispositions:
                analysis = analysis_by_id.get(disposition.idea_id)
                if analysis is None:
                    raise ValueError("selection disposition requires analysis")
                if (
                    disposition.disposition in eligible_dispositions
                    and not analysis.eligible
                ):
                    raise ValueError("ineligible ideas must have invalid disposition")
                if (
                    disposition.disposition == CandidateDispositionKind.INVALID
                    and analysis.eligible
                ):
                    raise ValueError("eligible ideas cannot have invalid disposition")
        if self.status == BatchStatus.PLANNED and (
            self.generated_idea_ids
            or self.resurfaced_ideas
            or self.considered_idea_ids
            or self.analyses
            or self.selection is not None
        ):
            raise ValueError("planned batches cannot contain candidate results")
        if (
            self.status
            in {
                BatchStatus.GENERATED,
                BatchStatus.ANALYZED,
                BatchStatus.SELECTED,
                BatchStatus.BRIDGED,
                BatchStatus.COMPLETED,
            }
            and not self.considered_idea_ids
        ):
            raise ValueError("active batches require considered ideas")
        if self.status in {
            BatchStatus.ANALYZED,
            BatchStatus.SELECTED,
            BatchStatus.BRIDGED,
            BatchStatus.COMPLETED,
        } and set(analysis_ids) != set(self.considered_idea_ids):
            raise ValueError("analyzed batches require complete candidate coverage")
        if (
            self.status
            in {
                BatchStatus.SELECTED,
                BatchStatus.BRIDGED,
                BatchStatus.COMPLETED,
            }
            and self.selection is None
        ):
            raise ValueError("selected batch states require a selection")
        if (
            self.status
            in {
                BatchStatus.PLANNED,
                BatchStatus.GENERATED,
                BatchStatus.ANALYZED,
            }
            and self.selection is not None
        ):
            raise ValueError("pre-selection batch states cannot contain a selection")
        if self.status == BatchStatus.ABANDONED and self.abandoned_reason is None:
            raise ValueError("abandoned batches require a reason")
        if self.status != BatchStatus.ABANDONED and self.abandoned_reason is not None:
            raise ValueError("only abandoned batches may have an abandonment reason")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IdeaBatch":
        fields = {
            "batch_id",
            "campaign_id",
            "iteration_index",
            "context_hash",
            "evidence_snapshot_id",
            "directive",
            "created_at",
            "updated_at",
            "status",
            "generated_idea_ids",
            "resurfaced_ideas",
            "considered_idea_ids",
            "analyses",
            "selection",
            "abandoned_reason",
        }
        _require_exact_keys(data, fields, "idea batch")
        return cls(
            batch_id=data["batch_id"],
            campaign_id=data["campaign_id"],
            iteration_index=data["iteration_index"],
            context_hash=data["context_hash"],
            evidence_snapshot_id=data["evidence_snapshot_id"],
            directive=SearchDirective.from_dict(data["directive"]),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            status=_parse_enum(BatchStatus, data["status"], "batch status"),
            generated_idea_ids=data["generated_idea_ids"],
            resurfaced_ideas=tuple(
                ResurfacedIdea.from_dict(resurfaced)
                for resurfaced in data["resurfaced_ideas"]
            ),
            considered_idea_ids=data["considered_idea_ids"],
            analyses=tuple(
                CandidateAnalysis.from_dict(analysis) for analysis in data["analyses"]
            ),
            selection=(
                None
                if data["selection"] is None
                else SelectionDecision.from_dict(data["selection"])
            ),
            abandoned_reason=data["abandoned_reason"],
        )


@dataclass(frozen=True)
class EvaluationGap(JsonRecord):
    gap_id: str
    axis: str
    description: str
    state: GapState
    evidence_refs: Tuple[str, ...]
    impact: float
    uncertainty: float
    estimated_cost: Optional[float]
    deferral_count: int
    opened_at: str
    last_considered_at: Optional[str] = None
    closure_reason: Optional[str] = None
    resolution_idea_id: Optional[str] = None
    resolution_experiment_node_id: Optional[int] = None

    def __post_init__(self) -> None:
        _require_typed_identifier(self.gap_id, "gap id", "gap")
        _require_nonempty_string(self.axis, "gap axis")
        _require_nonempty_string(self.description, "gap description")
        if not isinstance(self.state, GapState):
            raise ValueError("gap state is invalid")
        object.__setattr__(
            self,
            "evidence_refs",
            _require_strings(self.evidence_refs, "gap evidence refs"),
        )
        impact = _require_number(self.impact, "gap impact", 0.0)
        uncertainty = _require_number(self.uncertainty, "gap uncertainty", 0.0)
        if impact > 1.0 or uncertainty > 1.0:
            raise ValueError("gap impact and uncertainty must be <= 1")
        _require_optional_number(self.estimated_cost, "gap estimated cost", 0.0)
        _require_integer(self.deferral_count, "gap deferral count")
        _require_timestamp(self.opened_at, "gap opened timestamp")
        if self.last_considered_at is not None:
            _require_timestamp(
                self.last_considered_at,
                "gap last considered timestamp",
            )
            if datetime.fromisoformat(self.last_considered_at) < datetime.fromisoformat(
                self.opened_at
            ):
                raise ValueError("gap consideration cannot precede opening")
        _require_optional_string(self.closure_reason, "gap closure reason")
        if self.state == GapState.CLOSED and self.closure_reason is None:
            raise ValueError("closed gaps require a closure reason")
        if self.state != GapState.CLOSED and self.closure_reason is not None:
            raise ValueError("only closed gaps may have a closure reason")
        if self.deferral_count > 0 and self.last_considered_at is None:
            raise ValueError("deferred gaps require a consideration timestamp")
        _require_optional_typed_identifier(
            self.resolution_idea_id,
            "gap resolution idea id",
            "idea",
        )
        _require_optional_integer(
            self.resolution_experiment_node_id,
            "gap resolution experiment node id",
        )
        if self.state in {GapState.INCONCLUSIVE, GapState.CLOSED} and (
            self.resolution_idea_id is None
            or self.resolution_experiment_node_id is None
        ):
            raise ValueError("resolved gaps require outcome provenance")
        if self.state == GapState.OPEN and (
            self.resolution_idea_id is not None
            or self.resolution_experiment_node_id is not None
        ):
            raise ValueError("open gaps cannot have outcome provenance")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationGap":
        _require_exact_keys(
            data,
            {
                "gap_id",
                "axis",
                "description",
                "state",
                "evidence_refs",
                "impact",
                "uncertainty",
                "estimated_cost",
                "deferral_count",
                "opened_at",
                "last_considered_at",
                "closure_reason",
                "resolution_idea_id",
                "resolution_experiment_node_id",
            },
            "evaluation gap",
        )
        return cls(
            gap_id=data["gap_id"],
            axis=data["axis"],
            description=data["description"],
            state=_parse_enum(GapState, data["state"], "gap state"),
            evidence_refs=data["evidence_refs"],
            impact=data["impact"],
            uncertainty=data["uncertainty"],
            estimated_cost=data["estimated_cost"],
            deferral_count=data["deferral_count"],
            opened_at=data["opened_at"],
            last_considered_at=data["last_considered_at"],
            closure_reason=data["closure_reason"],
            resolution_idea_id=data["resolution_idea_id"],
            resolution_experiment_node_id=data["resolution_experiment_node_id"],
        )


@dataclass(frozen=True)
class EvidenceClaim(JsonRecord):
    claim_id: str
    statement: str
    kind: ClaimKind
    status: EvidenceStatus
    source_refs: Tuple[str, ...]
    affected_idea_ids: Tuple[str, ...]
    affected_experiment_node_ids: Tuple[int, ...]
    updated_at: str

    def __post_init__(self) -> None:
        _require_typed_identifier(self.claim_id, "evidence claim id", "claim")
        _require_nonempty_string(self.statement, "evidence claim statement")
        if not isinstance(self.kind, ClaimKind):
            raise ValueError("evidence claim kind is invalid")
        if not isinstance(self.status, EvidenceStatus):
            raise ValueError("evidence claim status is invalid")
        object.__setattr__(
            self,
            "source_refs",
            _require_strings(self.source_refs, "evidence claim source refs"),
        )
        if self.status != EvidenceStatus.INSUFFICIENT and not self.source_refs:
            raise ValueError("supported and contradicted claims require sources")
        object.__setattr__(
            self,
            "affected_idea_ids",
            _require_typed_identifiers(
                self.affected_idea_ids,
                "evidence claim affected idea ids",
                "idea",
            ),
        )
        if not isinstance(self.affected_experiment_node_ids, (list, tuple)):
            raise ValueError("evidence claim experiment ids must be a list")
        experiment_ids = tuple(
            _require_integer(node_id, "evidence claim experiment node id")
            for node_id in self.affected_experiment_node_ids
        )
        if len(set(experiment_ids)) != len(experiment_ids):
            raise ValueError("evidence claim experiment ids must be unique")
        object.__setattr__(
            self,
            "affected_experiment_node_ids",
            experiment_ids,
        )
        _require_timestamp(self.updated_at, "evidence claim timestamp")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceClaim":
        _require_exact_keys(
            data,
            {
                "claim_id",
                "statement",
                "kind",
                "status",
                "source_refs",
                "affected_idea_ids",
                "affected_experiment_node_ids",
                "updated_at",
            },
            "evidence claim",
        )
        return cls(
            claim_id=data["claim_id"],
            statement=data["statement"],
            kind=_parse_enum(ClaimKind, data["kind"], "evidence claim kind"),
            status=_parse_enum(
                EvidenceStatus,
                data["status"],
                "evidence claim status",
            ),
            source_refs=data["source_refs"],
            affected_idea_ids=data["affected_idea_ids"],
            affected_experiment_node_ids=data["affected_experiment_node_ids"],
            updated_at=data["updated_at"],
        )


@dataclass(frozen=True)
class ExperimentEvidence(JsonRecord):
    node_id: int
    idea_id: str
    selection_batch_id: str
    parent_node_id: Optional[int]
    proposal: str
    raw_score: Optional[float]
    normalized_utility: Optional[float]
    evaluation_status: EvaluationStatus
    implementation_status: ImplementationStatus
    evaluator_id: Optional[str]
    build_fidelity: str
    eval_fidelity: Optional[str]
    eval_fraction: Optional[float]
    seed: Optional[int]
    comparable: bool
    feedback: str
    technical_difficulty: Optional[str]
    created_at: str

    def __post_init__(self) -> None:
        _require_integer(self.node_id, "experiment evidence node id")
        _require_typed_identifier(
            self.idea_id,
            "experiment evidence idea id",
            "idea",
        )
        _require_typed_identifier(
            self.selection_batch_id,
            "experiment evidence selection batch id",
            "batch",
        )
        _require_optional_integer(self.parent_node_id, "experiment evidence parent")
        _require_nonempty_string(self.proposal, "experiment evidence proposal")
        _require_optional_number(self.raw_score, "experiment evidence raw score")
        _require_optional_number(
            self.normalized_utility,
            "experiment evidence normalized utility",
        )
        if not isinstance(self.evaluation_status, EvaluationStatus):
            raise ValueError("experiment evidence evaluation status is invalid")
        if not isinstance(self.implementation_status, ImplementationStatus):
            raise ValueError("experiment evidence implementation status is invalid")
        _require_optional_string(self.evaluator_id, "experiment evidence evaluator")
        _require_nonempty_string(
            self.build_fidelity,
            "experiment evidence build fidelity",
        )
        _require_optional_string(
            self.eval_fidelity,
            "experiment evidence eval fidelity",
        )
        fraction = _require_optional_number(
            self.eval_fraction,
            "experiment evidence eval fraction",
            0.0,
        )
        if fraction is not None and fraction > 1.0:
            raise ValueError("experiment evidence eval fraction must be <= 1")
        _require_optional_integer(self.seed, "experiment evidence seed")
        if not isinstance(self.comparable, bool):
            raise ValueError("experiment evidence comparable must be boolean")
        if not isinstance(self.feedback, str):
            raise ValueError("experiment evidence feedback must be a string")
        _require_optional_string(
            self.technical_difficulty,
            "experiment evidence technical difficulty",
        )
        _require_timestamp(self.created_at, "experiment evidence timestamp")
        if self.comparable and self.normalized_utility is None:
            raise ValueError("comparable evidence requires normalized utility")
        if self.comparable and self.evaluation_status != EvaluationStatus.VALID:
            raise ValueError("only valid evidence may be comparable")
        if self.raw_score is None and self.normalized_utility is not None:
            raise ValueError("normalized utility requires a raw score")
        if (
            self.evaluation_status != EvaluationStatus.VALID
            and self.normalized_utility is not None
        ):
            raise ValueError("only valid evidence may have normalized utility")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentEvidence":
        _require_exact_keys(
            data,
            {
                "node_id",
                "idea_id",
                "selection_batch_id",
                "parent_node_id",
                "proposal",
                "raw_score",
                "normalized_utility",
                "evaluation_status",
                "implementation_status",
                "evaluator_id",
                "build_fidelity",
                "eval_fidelity",
                "eval_fraction",
                "seed",
                "comparable",
                "feedback",
                "technical_difficulty",
                "created_at",
            },
            "experiment evidence",
        )
        return cls(
            node_id=data["node_id"],
            idea_id=data["idea_id"],
            selection_batch_id=data["selection_batch_id"],
            parent_node_id=data["parent_node_id"],
            proposal=data["proposal"],
            raw_score=data["raw_score"],
            normalized_utility=data["normalized_utility"],
            evaluation_status=_parse_enum(
                EvaluationStatus,
                data["evaluation_status"],
                "experiment evidence evaluation status",
            ),
            implementation_status=_parse_enum(
                ImplementationStatus,
                data["implementation_status"],
                "experiment evidence implementation status",
            ),
            evaluator_id=data["evaluator_id"],
            build_fidelity=data["build_fidelity"],
            eval_fidelity=data["eval_fidelity"],
            eval_fraction=data["eval_fraction"],
            seed=data["seed"],
            comparable=data["comparable"],
            feedback=data["feedback"],
            technical_difficulty=data["technical_difficulty"],
            created_at=data["created_at"],
        )


@dataclass(frozen=True)
class CampaignEvidenceSnapshot(JsonRecord):
    snapshot_id: str
    campaign_id: str
    objective_direction: ObjectiveDirection
    generated_at: str
    content_hash: str
    experiments: Tuple[ExperimentEvidence, ...]
    claims: Tuple[EvidenceClaim, ...]
    gaps: Tuple[EvaluationGap, ...]
    relevant_idea_ids: Tuple[str, ...]
    incumbent_node_id: Optional[int]
    latest_node_id: Optional[int]
    noise_floor: Optional[float]
    signals: Tuple[EvidenceSignal, ...]

    def __post_init__(self) -> None:
        _require_typed_identifier(
            self.snapshot_id,
            "evidence snapshot id",
            "evidence_snapshot",
        )
        _require_nonempty_string(self.campaign_id, "evidence campaign id")
        if not isinstance(self.objective_direction, ObjectiveDirection):
            raise ValueError("objective direction is invalid")
        _require_timestamp(self.generated_at, "evidence snapshot timestamp")
        _require_sha256(self.content_hash, "evidence content hash")
        if self.snapshot_id != content_identifier(
            "evidence_snapshot",
            self.content_hash,
        ):
            raise ValueError("evidence snapshot id must be content-addressed")
        if not isinstance(self.experiments, (list, tuple)) or not all(
            isinstance(experiment, ExperimentEvidence)
            for experiment in self.experiments
        ):
            raise ValueError("evidence snapshot experiments are invalid")
        object.__setattr__(self, "experiments", tuple(self.experiments))
        if not isinstance(self.claims, (list, tuple)) or not all(
            isinstance(claim, EvidenceClaim) for claim in self.claims
        ):
            raise ValueError("evidence snapshot claims are invalid")
        object.__setattr__(self, "claims", tuple(self.claims))
        if not isinstance(self.gaps, (list, tuple)) or not all(
            isinstance(gap, EvaluationGap) for gap in self.gaps
        ):
            raise ValueError("evidence snapshot gaps are invalid")
        object.__setattr__(self, "gaps", tuple(self.gaps))
        for records, name, identifier in (
            (self.experiments, "experiment", "node_id"),
            (self.claims, "claim", "claim_id"),
            (self.gaps, "gap", "gap_id"),
        ):
            identifiers = tuple(getattr(record, identifier) for record in records)
            if len(set(identifiers)) != len(identifiers):
                raise ValueError(f"evidence snapshot {name} ids must be unique")
        object.__setattr__(
            self,
            "relevant_idea_ids",
            _require_typed_identifiers(
                self.relevant_idea_ids,
                "relevant idea ids",
                "idea",
            ),
        )
        _require_optional_integer(self.incumbent_node_id, "incumbent node id")
        _require_optional_integer(self.latest_node_id, "latest node id")
        object.__setattr__(
            self,
            "noise_floor",
            _require_optional_number(
                self.noise_floor,
                "evidence noise floor",
                0.0,
            ),
        )
        if not isinstance(self.signals, (list, tuple)) or not all(
            isinstance(signal, EvidenceSignal) for signal in self.signals
        ):
            raise ValueError("evidence signals are invalid")
        if len(set(self.signals)) != len(self.signals):
            raise ValueError("evidence signals must be unique")
        object.__setattr__(self, "signals", tuple(self.signals))
        node_ids = {experiment.node_id for experiment in self.experiments}
        if (
            self.incumbent_node_id is not None
            and self.incumbent_node_id not in node_ids
        ):
            raise ValueError("evidence incumbent must reference an experiment")
        if self.latest_node_id is not None and self.latest_node_id not in node_ids:
            raise ValueError("latest node must reference an experiment")
        if self.incumbent_node_id is not None:
            incumbent = next(
                experiment
                for experiment in self.experiments
                if experiment.node_id == self.incumbent_node_id
            )
            if not incumbent.comparable:
                raise ValueError("evidence incumbent must be valid and comparable")
        sign = 1.0 if self.objective_direction == ObjectiveDirection.MAXIMIZE else -1.0
        for experiment in self.experiments:
            if experiment.normalized_utility is not None and (
                experiment.normalized_utility != sign * experiment.raw_score
            ):
                raise ValueError(
                    "normalized utility does not match objective direction"
                )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CampaignEvidenceSnapshot":
        _require_exact_keys(
            data,
            {
                "snapshot_id",
                "campaign_id",
                "objective_direction",
                "generated_at",
                "content_hash",
                "experiments",
                "claims",
                "gaps",
                "relevant_idea_ids",
                "incumbent_node_id",
                "latest_node_id",
                "noise_floor",
                "signals",
            },
            "campaign evidence snapshot",
        )
        return cls(
            snapshot_id=data["snapshot_id"],
            campaign_id=data["campaign_id"],
            objective_direction=_parse_enum(
                ObjectiveDirection,
                data["objective_direction"],
                "objective direction",
            ),
            generated_at=data["generated_at"],
            content_hash=data["content_hash"],
            experiments=tuple(
                ExperimentEvidence.from_dict(experiment)
                for experiment in data["experiments"]
            ),
            claims=tuple(EvidenceClaim.from_dict(claim) for claim in data["claims"]),
            gaps=tuple(EvaluationGap.from_dict(gap) for gap in data["gaps"]),
            relevant_idea_ids=data["relevant_idea_ids"],
            incumbent_node_id=data["incumbent_node_id"],
            latest_node_id=data["latest_node_id"],
            noise_floor=data["noise_floor"],
            signals=tuple(
                _parse_enum(EvidenceSignal, signal, "evidence signal")
                for signal in data["signals"]
            ),
        )


@dataclass(frozen=True)
class IdeationCapacityView(JsonRecord):
    capacity_snapshot_id: str
    iteration_index: int
    max_iterations: int
    remaining_seconds: Optional[float]
    remaining_after_reserve_seconds: Optional[float]
    remaining_usd: Optional[float]
    fidelity_profile: str
    build_fidelity: str
    eval_fidelity: str
    eval_fraction: float
    target_node_id: Optional[int]
    reserve_run: bool
    deadline_seconds: Optional[float]
    can_start_complete_action: bool
    can_run_comparable_evaluation: bool
    preserves_finalization_reserve: bool
    opportunity_probe_required: bool
    opportunity_probe_admissible: bool

    def __post_init__(self) -> None:
        _require_typed_identifier(
            self.capacity_snapshot_id,
            "capacity snapshot id",
            "capacity_snapshot",
        )
        _require_integer(self.iteration_index, "capacity iteration index")
        _require_integer(self.max_iterations, "capacity max iterations", 1)
        _require_optional_number(
            self.remaining_seconds,
            "capacity remaining seconds",
        )
        _require_optional_number(
            self.remaining_after_reserve_seconds,
            "capacity remaining after reserve",
        )
        _require_optional_number(self.remaining_usd, "capacity remaining usd")
        _require_nonempty_string(self.fidelity_profile, "capacity fidelity profile")
        _require_nonempty_string(self.build_fidelity, "capacity build fidelity")
        _require_nonempty_string(self.eval_fidelity, "capacity eval fidelity")
        fraction = _require_number(
            self.eval_fraction,
            "capacity eval fraction",
            0.0,
        )
        if fraction > 1.0:
            raise ValueError("capacity eval fraction must be <= 1")
        _require_optional_integer(self.target_node_id, "capacity target node id")
        if not isinstance(self.reserve_run, bool):
            raise ValueError("capacity reserve run must be boolean")
        _require_optional_number(
            self.deadline_seconds,
            "capacity deadline seconds",
            0.0,
        )
        for value, name in (
            (self.can_start_complete_action, "complete action admission"),
            (
                self.can_run_comparable_evaluation,
                "comparable evaluation admission",
            ),
            (
                self.preserves_finalization_reserve,
                "finalization reserve admission",
            ),
            (self.opportunity_probe_required, "opportunity probe requirement"),
            (self.opportunity_probe_admissible, "opportunity probe admission"),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"capacity {name} must be boolean")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IdeationCapacityView":
        _require_exact_keys(
            data,
            {
                "capacity_snapshot_id",
                "iteration_index",
                "max_iterations",
                "remaining_seconds",
                "remaining_after_reserve_seconds",
                "remaining_usd",
                "fidelity_profile",
                "build_fidelity",
                "eval_fidelity",
                "eval_fraction",
                "target_node_id",
                "reserve_run",
                "deadline_seconds",
                "can_start_complete_action",
                "can_run_comparable_evaluation",
                "preserves_finalization_reserve",
                "opportunity_probe_required",
                "opportunity_probe_admissible",
            },
            "ideation capacity view",
        )
        return cls(**data)
