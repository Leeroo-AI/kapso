"""The handles a build takes and hands back: `Budget`, `Brief`, `Solution`.

Design: docs/research/harness-design.md section 8 and
docs/plans/redesign/wp1-build.md. A `Brief` is both the Python object and
the `kapso-brief.md` file, one grammar; `build()` renders the whole brief
into the engine's goal text, so nothing an engineer wrote is lost on the
way to the model. A `Solution` is written to `.kapso/solution.json` in the
campaign directory every time `build()` returns, and `Solution.open()`
reads that file and nothing else.
"""

import json
import math
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class SolutionError(FileNotFoundError):
    """A path with no solution to open: not a campaign directory, or a
    build that has not returned. The CLI prints the message as one line,
    with the other engine errors."""


class BriefError(ValueError):
    """A brief that does not follow the grammar, with the line it failed on."""


# =============================================================================
# LAUNCH
# =============================================================================


@dataclass(frozen=True)
class Launch:
    """What a detached `build(detach=True)` or `learn(detach=True)` hands
    back at once: the session's pid, the log it writes, and the status to
    follow (`kapso watch <status>`). The work goes on after this process
    ends; `Solution.open(output)` reads a build's result when it has one."""

    pid: int
    log: str
    status: str


# =============================================================================
# BUDGET
# =============================================================================

_BUDGET_FIELDS = ("minutes", "experiments", "usd", "reserve_minutes")


@dataclass
class Budget:
    """What a build may spend. A field left `None` keeps the config's value
    (or, on a resume, the launch's). `usd` is model spend; the
    platform-credit budget is a config value, not a handle field."""

    minutes: Optional[float] = None
    experiments: Optional[int] = None
    usd: Optional[float] = None
    reserve_minutes: Optional[float] = None   # wall clock escrowed for the final checkout and evaluation

    def __post_init__(self) -> None:
        for name in ("minutes", "usd", "reserve_minutes"):
            value = getattr(self, name)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    f"Budget.{name} must be a positive finite number, got {value!r}"
                )
        if self.experiments is not None and (
            isinstance(self.experiments, bool)
            or not isinstance(self.experiments, int)
            or self.experiments < 1
        ):
            raise ValueError(
                f"Budget.experiments must be a positive integer, got {self.experiments!r}"
            )
        if (
            self.minutes is not None
            and self.reserve_minutes is not None
            and self.reserve_minutes >= self.minutes
        ):
            raise ValueError(
                f"Budget.reserve_minutes ({self.reserve_minutes}) must be smaller "
                f"than Budget.minutes ({self.minutes})"
            )

    @property
    def empty(self) -> bool:
        return all(getattr(self, name) is None for name in _BUDGET_FIELDS)

    def merged_over(self, other: Optional["Budget"]) -> "Budget":
        """This budget's set fields, and `other`'s where this one is unset."""
        values = {}
        for name in _BUDGET_FIELDS:
            value = getattr(self, name)
            if value is None and other is not None:
                value = getattr(other, name)
            values[name] = value
        return Budget(**values)

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in _BUDGET_FIELDS}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Budget":
        return cls(**{name: data[name] for name in _BUDGET_FIELDS})

    def tokens(self) -> str:
        """The `name=value` form the brief file uses."""
        parts = []
        for name in _BUDGET_FIELDS:
            value = getattr(self, name)
            if value is not None:
                parts.append(f"{name}={_number(value)}")
        return " ".join(parts)


def _number(value: Union[int, float]) -> str:
    """A number as the brief writes it: plain digits, a decimal point when
    needed, never an exponent, so the parser reads back what was written."""
    if float(value).is_integer():
        return str(int(value))
    return format(value, "f").rstrip("0")


# =============================================================================
# BRIEF
# =============================================================================

# Field name in the file -> attribute. The names are matched as whole words
# at the start of an unindented line, case-insensitively.
_FIELDS = {
    "objective": "objective",
    "measured by": "measured_by",
    "baseline": "baseline",
    "target": "target",
    "rules": "rules",
    "budget": "budget",
    "harness": "harness",
    "experience": "experience",
}
_FIELD_TITLES = {
    "objective": "Objective",
    "measured_by": "Measured by",
    "baseline": "Baseline",
    "target": "Target",
    "rules": "Rules",
    "budget": "Budget",
    "harness": "Harness",
    "experience": "Experience",
}
_LIST_FIELDS = ("rules", "experience")
_SCALAR_FIELDS = ("objective", "measured_by", "baseline", "target", "harness")
_FIELD_LINE = re.compile(
    r"^(objective|measured by|baseline|target|rules|budget|harness|experience)"
    r"(?:[ \t]+(.*))?$",
    re.IGNORECASE,
)
# Values start at column 13: the longest name, `Measured by`, padded to 12
# and one space. Continuation lines carry the same indent.
_VALUE_COLUMN = 13
_CONTINUATION = " " * _VALUE_COLUMN
_ITEM = "  - "
_ITEM_CONTINUATION = "    "


def _dedent(lines: List[str]) -> List[str]:
    """Remove the whitespace the non-blank lines share, keeping relative
    indentation (a pasted code block in an objective stays a code block)."""
    filled = [line for line in lines if line.strip()]
    if not filled:
        return ["" for _ in lines]
    common = min(len(line) - len(line.lstrip()) for line in filled)
    return [line[common:] if line.strip() else "" for line in lines]


def _normalize_text(value: str) -> str:
    """The canonical form of a field's text, the one the file holds: the
    first line stripped (it sits after the field name in the file), the
    lines after it dedented together so their relative indentation
    survives, trailing whitespace off every line, blank lines off both
    ends. A brief built in Python and one read from its own rendering are
    then equal, which is what a resume's fingerprint relies on."""
    first, *rest = value.strip().splitlines() or [""]
    rest = _dedent([line.rstrip() for line in rest])
    return "\n".join([first.strip(), *rest])


@dataclass
class Brief:
    """What to build and how success is measured. `objective` may span
    paragraphs and is never clipped; the other fields are optional. The
    `harness` and `experience` references are kept verbatim; the packages
    that resolve them (WP2, WP3) read them from here."""

    objective: str
    measured_by: Optional[str] = None
    baseline: Optional[str] = None
    target: Optional[str] = None
    rules: List[str] = field(default_factory=list)
    budget: Optional[Budget] = None
    harness: Optional[str] = None
    experience: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.objective, str) or not self.objective.strip():
            raise ValueError("Brief.objective must be a non-empty string")
        self.objective = _normalize_text(self.objective)
        for name in ("measured_by", "baseline", "target", "harness"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Brief.{name} must be a non-empty string or None")
            setattr(self, name, _normalize_text(value))
        for name in _LIST_FIELDS:
            items = getattr(self, name)
            if not isinstance(items, list) or not all(
                isinstance(item, str) and item.strip() for item in items
            ):
                raise ValueError(f"Brief.{name} must be a list of non-empty strings")
            setattr(self, name, [_normalize_text(item) for item in items])
        if self.budget is not None:
            if not isinstance(self.budget, Budget):
                raise ValueError("Brief.budget must be a Budget or None")
            if self.budget.empty:
                self.budget = None

    # ------------------------------------------------------------ the file

    @classmethod
    def read(cls, path: Union[str, Path]) -> "Brief":
        """Parse a brief file. The whole file is read; an unknown or repeated
        field, a missing objective, or text that belongs to no field raises
        `BriefError` with the line number."""
        path = Path(path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"brief not found: {path}")
        return cls.parse(path.read_text(encoding="utf-8"), source=str(path))

    @classmethod
    def parse(cls, text: str, *, source: str = "<brief>") -> "Brief":
        parser = _BriefParser(source)
        for number, raw in enumerate(text.splitlines(), start=1):
            parser.feed(number, raw)
        return cls(**parser.finish())

    def render(self) -> str:
        """The brief as text: the file's grammar, deterministic, and the
        text the engine receives as its goal, whole."""
        lines: List[str] = []

        def scalar(title: str, value: str) -> None:
            first, *rest = value.split("\n")
            lines.append(f"{title:<12} {first}".rstrip())
            for part in rest:
                lines.append(f"{_CONTINUATION}{part}" if part else "")

        scalar("Objective", self.objective)
        if self.measured_by is not None:
            scalar("Measured by", self.measured_by)
        if self.baseline is not None:
            scalar("Baseline", self.baseline)
        if self.target is not None:
            scalar("Target", self.target)
        if self.rules:
            lines.append("Rules")
            lines.extend(_render_items(self.rules))
        if self.budget is not None:
            lines.append(f"{'Budget':<12} {self.budget.tokens()}")
        if self.harness is not None:
            scalar("Harness", self.harness)
        if self.experience:
            if any("," in item or "\n" in item for item in self.experience):
                lines.append("Experience")
                lines.extend(_render_items(self.experience))
            else:
                lines.append(f"{'Experience':<12} {', '.join(self.experience)}")
        return "\n".join(lines) + "\n"

    def write(self, path: Union[str, Path]) -> Path:
        path = Path(path).expanduser()
        path.write_text(self.render(), encoding="utf-8")
        return path

    # ---------------------------------------------------------- the record

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective": self.objective,
            "measured_by": self.measured_by,
            "baseline": self.baseline,
            "target": self.target,
            "rules": list(self.rules),
            "budget": self.budget.to_dict() if self.budget else None,
            "harness": self.harness,
            "experience": list(self.experience),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Brief":
        return cls(
            objective=data["objective"],
            measured_by=data["measured_by"],
            baseline=data["baseline"],
            target=data["target"],
            rules=list(data["rules"]),
            budget=Budget.from_dict(data["budget"]) if data["budget"] else None,
            harness=data["harness"],
            experience=list(data["experience"]),
        )


def _render_items(items: List[str]) -> List[str]:
    lines = []
    for item in items:
        first, *rest = item.split("\n")
        lines.append(f"{_ITEM}{first}")
        for part in rest:
            lines.append(f"{_ITEM_CONTINUATION}{part}" if part else "")
    return lines


class _BriefParser:
    """One pass over the lines. A field starts an unindented line; an
    indented line continues the field above, text fields taking any text
    (a bullet in an objective is just text); blank lines are ignored except
    between two lines of the same field, where they stay as paragraph
    breaks. `Rules` and `Experience` take `- ` items; `Experience` also
    takes one comma-separated line. `Budget` takes `name=value` tokens."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.current: Optional[str] = None
        self.seen: List[str] = []
        self.field_lines: Dict[str, int] = {}
        self.first_line: Dict[str, Optional[str]] = {}
        self.continuations: Dict[str, List[str]] = {}
        self.items: Dict[str, List[List[str]]] = {}
        self.inline_items: Dict[str, List[str]] = {}
        self.budget_tokens: List[tuple] = []
        self.pending_blanks = 0

    def fail(self, number: int, message: str) -> BriefError:
        return BriefError(f"{self.source}:{number}: {message}")

    def feed(self, number: int, raw: str) -> None:
        line = raw.rstrip()
        if not line.strip():
            self.pending_blanks += 1
            return
        if line[0] not in " \t":
            self._field_line(number, line)
            return
        if self.current is None:
            raise self.fail(number, "indented line before any field")
        stripped = line.strip()
        if self.current in _LIST_FIELDS:
            self._list_line(number, line, stripped)
        elif self.current == "budget":
            self.budget_tokens.extend((number, token) for token in stripped.split())
        else:
            lines = self.continuations[self.current]
            if self.pending_blanks and (lines or self.first_line[self.current]):
                lines.extend([""] * self.pending_blanks)
            lines.append(line)
        self.pending_blanks = 0

    def _field_line(self, number: int, line: str) -> None:
        match = _FIELD_LINE.match(line)
        if match is None:
            raise self.fail(
                number,
                f"not a field: {line!r}. A line starts with one of "
                f"{', '.join(_FIELD_TITLES.values())}, or is indented to continue "
                "the field above",
            )
        attribute = _FIELDS[match.group(1).lower()]
        title = _FIELD_TITLES[attribute]
        if attribute in self.seen:
            raise self.fail(number, f"{title} given twice")
        self.seen.append(attribute)
        self.field_lines[attribute] = number
        self.current = attribute
        self.pending_blanks = 0
        value = (match.group(2) or "").strip()
        if attribute in _LIST_FIELDS:
            self.items[attribute] = []
            if value and attribute == "rules":
                raise self.fail(number, "Rules takes `- ` items on the lines below it")
            if value:
                parts = [part.strip() for part in value.split(",")]
                if not all(parts):
                    raise self.fail(number, "Experience has an empty item")
                self.inline_items[attribute] = parts
        elif attribute == "budget":
            self.budget_tokens.extend((number, token) for token in value.split())
        else:
            self.first_line[attribute] = value or None
            self.continuations[attribute] = []

    def _list_line(self, number: int, line: str, stripped: str) -> None:
        title = _FIELD_TITLES[self.current]
        if stripped == "-":
            raise self.fail(number, f"empty item under {title}")
        if stripped.startswith("- "):
            if self.current in self.inline_items:
                raise self.fail(
                    number,
                    f"{title} is either one comma-separated line or `- ` items, not both",
                )
            self.items[self.current].append([stripped[2:].strip()])
            return
        if not self.items[self.current]:
            raise self.fail(number, f"expected a `- ` item under {title}")
        item = self.items[self.current][-1]
        item.extend([""] * self.pending_blanks)
        item.append(line)

    def finish(self) -> Dict[str, Any]:
        if "objective" not in self.seen:
            raise BriefError(f"{self.source}: Objective is required")
        values: Dict[str, Any] = {}
        for attribute in _SCALAR_FIELDS:
            if attribute not in self.seen:
                continue
            parts = []
            if self.first_line[attribute]:
                parts.append(self.first_line[attribute])
            parts.extend(_dedent(self.continuations[attribute]))
            text = "\n".join(parts).strip()
            if not text:
                raise self.fail(
                    self.field_lines[attribute], f"{_FIELD_TITLES[attribute]} has no value"
                )
            values[attribute] = text
        for attribute in _LIST_FIELDS:
            if attribute not in self.seen:
                continue
            if attribute in self.inline_items:
                values[attribute] = self.inline_items[attribute]
                continue
            items = []
            for lines in self.items[attribute]:
                first, *rest = lines
                items.append("\n".join([first, *_dedent(rest)]).strip())
            if not items:
                raise self.fail(
                    self.field_lines[attribute], f"{_FIELD_TITLES[attribute]} has no items"
                )
            values[attribute] = items
        if "budget" in self.seen:
            values["budget"] = self._budget()
        return values

    def _budget(self) -> Budget:
        if not self.budget_tokens:
            raise self.fail(
                self.field_lines["budget"],
                f"Budget needs name=value tokens, from {', '.join(_BUDGET_FIELDS)}",
            )
        values: Dict[str, Any] = {}
        for number, token in self.budget_tokens:
            name, equals, text = token.partition("=")
            if not equals or name not in _BUDGET_FIELDS or not text:
                raise self.fail(
                    number,
                    f"Budget token {token!r} is not name=value with a name from "
                    f"{', '.join(_BUDGET_FIELDS)}",
                )
            if name in values:
                raise self.fail(number, f"Budget {name} given twice")
            if name == "experiments":
                if not re.fullmatch(r"[0-9]+", text):
                    raise self.fail(
                        number, f"Budget experiments must be an integer, got {text!r}"
                    )
                values[name] = int(text)
            else:
                if not re.fullmatch(r"[0-9]+(\.[0-9]+)?", text):
                    raise self.fail(number, f"Budget {name} must be a number, got {text!r}")
                values[name] = float(text)
        return Budget(**values)


# =============================================================================
# SOLUTION
# =============================================================================

@dataclass
class EvaluatorRef:
    """The registered evaluator at delivery: the version the delivered
    candidate's score was measured under."""
    version: int
    evaluator_id: str
    provenance: str   # provided | maintainer_built | derived_fast


@dataclass
class HarnessRef:
    """The harness a build ran with (filled by the harness package)."""
    name: str
    version: str
    hash: str
    source: str       # the reference it was loaded from


@dataclass
class ExperienceRef:
    """One experience source a build was served (filled by the experience
    package)."""
    reference: str
    kind: str
    resolved: str
    head: Optional[str]
    provenance: str
    scope: str          # "harness" or "all": what of the source the campaign was served


# The record's six fields, the shape every record carries (the serving
# record adds the count served beside them).
EXPERIENCE_RECORD_FIELDS = ("reference", "kind", "resolved", "head", "provenance", "scope")


SOLUTION_RECORD = Path(".kapso") / "solution.json"
SOLUTION_SCHEMA = 5
_LABEL = 17   # the column explain()'s values start at: the longest label, "Sessions failed:", and a space
# A failed session's error on the summary line is its first line, clipped
# for display; the record keeps the whole text.
_ERROR_HEADLINE = 80


def _failed_sessions_line(failures: List[Dict[str, Any]]) -> str:
    """`implementation on codex ×2 (Quota exceeded …)`: each kind of
    session that failed, how many times, and the last error's first
    line."""
    kinds: Dict[tuple, List[Dict[str, Any]]] = {}
    for failure in failures:
        kinds.setdefault((failure["module"], failure["host"]), []).append(failure)
    parts = []
    for (module, host), group in kinds.items():
        error = (group[-1]["error"] or "").strip().splitlines()
        headline = error[0] if error else group[-1]["exit"]
        if len(headline) > _ERROR_HEADLINE:
            headline = headline[: _ERROR_HEADLINE - 1] + "…"
        parts.append(f"{module} on {host} ×{len(group)} ({headline})")
    return "; ".join(parts)


@dataclass
class Solution:
    """What a build delivered, and the account of how. `campaign` is the
    campaign directory: the workspace, checked out at the delivered branch,
    with the record under `.kapso/`. `branch` and `commit` are `None` when
    no candidate succeeded; `evaluator` is `None` when no evaluator was
    registered (the agent built its own evaluation)."""

    brief: Brief
    campaign: Path
    branch: Optional[str]
    commit: Optional[str]
    base_commit: str              # the workspace's seed commit, diff()'s base
    score: Optional[float]        # the delivered candidate's authoritative measurement
    goal_achieved: bool           # the judge's verdict that the objective was met
    stopped_reason: str           # goal_achieved | max_iterations | budget_exhausted | waiting_for_user | ...
    stop_detail: Optional[str]    # which budget dial stopped it, when one did
    experiments: int              # experiments this run
    experiments_total: int        # experiments across every resume
    cost_usd: float
    requests: List[Dict[str, Any]]   # what the person must answer when waiting_for_user
    feedback: Optional[str]       # the judge's words on the delivered candidate, in full
    evaluator: Optional[EvaluatorRef]
    harness: Optional[HarnessRef]
    verification: Optional[Dict[str, Any]]   # the harness's verdict on the delivered candidate: {verdict, summary, findings}
    experience: List[ExperienceRef]          # what the campaign was served, each source at its pinned head and scope
    rejected_candidates: int      # candidates rejected for touching the evaluator
    callback_metrics: Dict[str, float]      # the caller's own iteration_evaluator, delivered candidate
    callback_primary_metric: Optional[str]
    # The sessions' run entries that ended error, timeout or refused:
    # {label, module, host, exit, error} each (kapso.execution.trace).
    session_failures: List[Dict[str, Any]]

    @property
    def record_path(self) -> Path:
        return self.campaign / SOLUTION_RECORD

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SOLUTION_SCHEMA,
            "brief": self.brief.to_dict(),
            "branch": self.branch,
            "commit": self.commit,
            "base_commit": self.base_commit,
            "score": self.score,
            "goal_achieved": self.goal_achieved,
            "stopped_reason": self.stopped_reason,
            "stop_detail": self.stop_detail,
            "experiments": self.experiments,
            "experiments_total": self.experiments_total,
            "cost_usd": self.cost_usd,
            "requests": list(self.requests),
            "feedback": self.feedback,
            "evaluator": (
                {
                    "version": self.evaluator.version,
                    "evaluator_id": self.evaluator.evaluator_id,
                    "provenance": self.evaluator.provenance,
                }
                if self.evaluator else None
            ),
            "harness": (
                {
                    "name": self.harness.name,
                    "version": self.harness.version,
                    "hash": self.harness.hash,
                    "source": self.harness.source,
                }
                if self.harness else None
            ),
            "experience": [
                {
                    "reference": source.reference,
                    "kind": source.kind,
                    "resolved": source.resolved,
                    "head": source.head,
                    "provenance": source.provenance,
                    "scope": source.scope,
                }
                for source in self.experience
            ],
            "verification": self.verification,
            "rejected_candidates": self.rejected_candidates,
            "callback_metrics": dict(self.callback_metrics),
            "callback_primary_metric": self.callback_primary_metric,
            "session_failures": [dict(failure) for failure in self.session_failures],
        }

    def save(self) -> Path:
        """Write the record; `build()` does this every time it returns."""
        path = self.record_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        return path

    @classmethod
    def open(cls, path: Union[str, Path]) -> "Solution":
        """The solution a build wrote in `path`. A campaign that has not
        returned yet has no solution: follow it with `kapso watch`."""
        campaign = Path(path).expanduser().resolve()
        record = campaign / SOLUTION_RECORD
        if not record.is_file():
            if (campaign / ".kapso").is_dir():
                raise SolutionError(
                    f"no solution in {campaign} yet: the build has not returned "
                    f"(follow it with `kapso watch {campaign}`; a build that just "
                    "ended is writing its record), or it stopped before delivering "
                    "(its log says where)"
                )
            raise SolutionError(f"{campaign} is not a campaign directory")
        data = json.loads(record.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema_version") != SOLUTION_SCHEMA:
            raise ValueError(
                f"{record}: not a solution record of schema {SOLUTION_SCHEMA}"
            )
        evaluator = data["evaluator"]
        harness = data["harness"]
        return cls(
            brief=Brief.from_dict(data["brief"]),
            campaign=campaign,
            branch=data["branch"],
            commit=data["commit"],
            base_commit=data["base_commit"],
            score=data["score"],
            goal_achieved=data["goal_achieved"],
            stopped_reason=data["stopped_reason"],
            stop_detail=data["stop_detail"],
            experiments=data["experiments"],
            experiments_total=data["experiments_total"],
            cost_usd=data["cost_usd"],
            requests=list(data["requests"]),
            feedback=data["feedback"],
            evaluator=EvaluatorRef(**evaluator) if evaluator else None,
            harness=HarnessRef(**harness) if harness else None,
            experience=[ExperienceRef(**source) for source in data["experience"]],
            verification=data["verification"],
            rejected_candidates=data["rejected_candidates"],
            callback_metrics=dict(data["callback_metrics"]),
            callback_primary_metric=data["callback_primary_metric"],
            session_failures=[dict(failure) for failure in data["session_failures"]],
        )

    def diff(self) -> str:
        """The delivered code against the seed, as `git diff`; empty when
        no candidate succeeded."""
        if self.commit is None:
            return ""
        return subprocess.run(
            ["git", "-C", str(self.campaign), "diff", f"{self.base_commit}..{self.commit}"],
            check=True, capture_output=True, text=True,
        ).stdout

    def explain(self) -> str:
        """The summary block: what was delivered, how it measured, why the
        build stopped and what it cost."""
        objective = self.brief.objective.split("\n")[0]
        lines = []

        def row(label: str, value: Any) -> None:
            lines.append(f"{label + ':':<{_LABEL}}{value}")

        row("Solution for", objective)
        row("Campaign", self.campaign)
        if self.branch is None:
            row("Candidate", "none succeeded")
        else:
            row("Candidate", f"{self.branch} @ {self.commit}")
        if self.score is not None:
            measured = str(self.score)
            if self.evaluator is not None:
                measured += (
                    f" (evaluator v{self.evaluator.version}, "
                    f"{self.evaluator.provenance})"
                )
            row("Score", measured)
        row("Goal achieved", self.goal_achieved)
        stopped = self.stopped_reason
        if self.stop_detail:
            stopped += f" ({self.stop_detail})"
        row("Stopped", stopped)
        experiments = f"{self.experiments} this run, {self.experiments_total} in all"
        if self.rejected_candidates:
            experiments += f", {self.rejected_candidates} rejected"
        row("Experiments", experiments)
        if self.session_failures:
            row("Sessions failed", _failed_sessions_line(self.session_failures))
        row("Cost", f"${self.cost_usd:.3f}")
        if self.callback_primary_metric:
            row(
                "Your metric",
                f"{self.callback_primary_metric} = "
                f"{self.callback_metrics.get(self.callback_primary_metric)}",
            )
        if self.harness is not None:
            # The hash's first twelve characters, as every print shows it;
            # the record keeps the full hex.
            row("Harness", f"{self.harness.name} {self.harness.version} ({self.harness.hash[:12]})")
        if self.verification is not None:
            row("Verified", f"{self.verification['verdict']} — {self.verification['summary']}")
        for source in self.experience:
            # The head's first twelve characters, as every print shows it;
            # the record keeps the full value.
            pinned = f" at {source.head[:12]}" if source.head else ""
            every = " (every scope)" if source.kind == "bank" and source.scope == "all" else ""
            row("Experience", f"{source.reference}{pinned}, {source.provenance}{every}")
        if self.requests:
            lines.append("Waiting on you:")
            for request in self.requests:
                lines.append(f"  #{request['id']}  {request['key']}")
        if self.feedback:
            lines.append("")
            lines.append("Feedback:")
            lines.append(self.feedback)
        return "\n".join(lines)
