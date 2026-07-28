"""Strict executed-experiment memory for evidence-directed campaigns."""

import fcntl
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional

from kapso.core.llm import LLMBackend
from kapso.cross_run.canonical import (
    canonical_json_bytes,
    require_identifier,
    to_json_value,
)
from kapso.cross_run.capture.journal import (
    ExecutionRevisionJournal,
    JournalConflictError,
)
from kapso.cross_run.contracts import EpisodeEvaluationStatus, ExecutionStatus
from kapso.cross_run.git_command import BoundedGitCommand
from kapso.cross_run.git_refs import require_git_ref_name
from kapso.cross_run.github.command import CommandOutputKind, CommandRunner
from kapso.execution.memories.experiment_memory.record import (
    EXPERIMENT_HISTORY_SCHEMA as _EXPERIMENT_HISTORY_SCHEMA,
    ExperimentRecord as _ExperimentRecord,
    cosine_similarity as _cosine_similarity,
)


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate experiment-history key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str):
    raise ValueError(f"experiment history contains non-finite value: {value}")


class ExperimentHistoryStore:
    """Atomic, objective-aware storage for executed nodes only."""

    def __init__(
        self,
        json_path: str,
        objective_direction: Optional[str] = None,
        require_idea_links: Optional[bool] = None,
        goal: Optional[str] = None,
        llm: Optional[LLMBackend] = None,
        run_id: Optional[str] = None,
        campaign_id: Optional[str] = None,
        journal_path: Optional[str] = None,
        git_command_timeout_seconds: Optional[int] = None,
        git_command_output_bytes: Optional[int] = None,
        git_command_runner: Optional[CommandRunner] = None,
    ):
        self.path = Path(json_path)
        self.goal = goal
        self._llm = llm
        self.experiments: List[_ExperimentRecord] = []
        self.revision = 0
        self.run_id = run_id
        self.campaign_id = campaign_id
        self.objective_direction = objective_direction
        self.require_idea_links = require_idea_links
        if (git_command_timeout_seconds is None) != (git_command_output_bytes is None):
            raise ValueError("Git command bounds must be provided together")
        self._git_command = (
            BoundedGitCommand(
                timeout_seconds=git_command_timeout_seconds,
                maximum_output_bytes=git_command_output_bytes,
                runner=git_command_runner,
            )
            if git_command_timeout_seconds is not None
            else None
        )
        if self.path.exists():
            self._load()
            if (
                objective_direction is not None
                and objective_direction != self.objective_direction
            ):
                raise ValueError("experiment-history objective direction changed")
            if (
                require_idea_links is not None
                and require_idea_links != self.require_idea_links
            ):
                raise ValueError("experiment-history idea-link policy changed")
            if run_id is not None and run_id != self.run_id:
                raise ValueError("experiment-history run identity changed")
            if campaign_id is not None and campaign_id != self.campaign_id:
                raise ValueError("experiment-history campaign identity changed")
        elif objective_direction not in {"maximize", "minimize"}:
            raise ValueError(
                "new experiment history requires maximize or minimize direction"
            )
        elif not isinstance(require_idea_links, bool):
            raise ValueError("new experiment history requires an idea-link policy")
        elif journal_path is not None:
            self.run_id = require_identifier(run_id, "run_id")
            self.campaign_id = require_identifier(campaign_id, "campaign_id")
        self.revision_journal = None
        if journal_path is not None:
            if self.run_id is None or self.campaign_id is None:
                raise ValueError("journaled experiment history requires run identities")
            self.revision_journal = ExecutionRevisionJournal(
                journal_path,
                run_id=self.run_id,
                campaign_id=self.campaign_id,
            )
            journal_file = Path(journal_path)
            self._transaction_lock_path = journal_file.with_name(
                journal_file.name + ".transaction.lock"
            )
            for component in (
                self._transaction_lock_path,
                *self._transaction_lock_path.parents,
            ):
                if component.is_symlink():
                    raise ValueError("experiment-history transaction path is a symlink")
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with self._transaction_lock_path.open("a+b") as lock:
                self._transaction_lock_path.chmod(0o600)
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                self._bind_transaction_lock(lock)
                self._refresh_journaled_authorities()
                if not self.path.exists():
                    self._save(self.experiments, self.revision)
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def add_experiment(self, node: Any) -> _ExperimentRecord:
        if self.revision_journal is None:
            raise ValueError("experiment-history writes require a revision journal")
        with self._transaction_lock_path.open("a+b") as lock:
            self._transaction_lock_path.chmod(0o600)
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            self._bind_transaction_lock(lock)
            self._refresh_journaled_authorities()
            record = self._add_experiment_locked(node)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        return record

    def _add_experiment_locked(self, node: Any) -> _ExperimentRecord:
        existing = tuple(
            item for item in self.experiments if item.node_id == node.node_id
        )
        solution_embedding: Iterable[float] = ()
        if existing and existing[0].solution == node.solution:
            solution_embedding = existing[0].solution_embedding
        elif self._llm is not None and node.solution.strip():
            solution_embedding = self._llm.create_embedding(node.solution)
        record = _ExperimentRecord.from_node(
            node,
            self.objective_direction,
            self.require_idea_links,
            solution_embedding,
        )
        if existing:
            prior = existing[0]
            if prior != record:
                if record.execution_revision < prior.execution_revision:
                    raise ValueError("experiment node revision moved backwards")
                stable_identity = (
                    prior.idea_id,
                    prior.selection_batch_id,
                    prior.parent_node_id,
                    prior.solution,
                    prior.objective_direction,
                )
                next_identity = (
                    record.idea_id,
                    record.selection_batch_id,
                    record.parent_node_id,
                    record.solution,
                    record.objective_direction,
                )
                if record.execution_revision > prior.execution_revision and (
                    stable_identity != next_identity
                    or record.execution_revision != prior.execution_revision + 1
                ):
                    raise ValueError("experiment node identity or revision changed")
                proposed = (
                    [
                        record if item.node_id == record.node_id else item
                        for item in self.experiments
                    ]
                    if record.execution_revision > prior.execution_revision
                    else self.experiments
                )
            else:
                proposed = self.experiments
        else:
            if record.node_id != len(self.experiments):
                raise ValueError("experiment node ids must be contiguous")
            proposed = self.experiments + [record]
        execution_status = (
            ExecutionStatus.INTERRUPTED
            if record.recoverable_error
            else (
                ExecutionStatus.FAILED_TECHNICAL
                if record.had_error
                else ExecutionStatus.COMPLETED
            )
        )
        evaluation_status = (
            EpisodeEvaluationStatus.NOT_RUN
            if record.had_error
            else (
                EpisodeEvaluationStatus.INVALID
                if not record.evaluation_valid
                else (
                    EpisodeEvaluationStatus.VALID
                    if record.raw_score is not None and record.evaluation_attempts
                    else EpisodeEvaluationStatus.PARTIAL
                )
            )
        )
        measurements = dict(record.metrics)
        if record.raw_score is not None:
            measurements["raw_score"] = record.raw_score
        artifact_refs = {
            name: value
            for name, value in {
                "branch": record.branch_name,
                "parent_branch": getattr(node, "parent_branch_name", ""),
                "implementation_base": getattr(node, "implementation_base_ref", ""),
                "diff_base": getattr(node, "diff_base_ref", ""),
                "feedback_base": getattr(node, "feedback_base_ref", ""),
            }.items()
            if value
        }
        candidate_commit, candidate_ref = self._pin_revision_commit(node, record)
        if candidate_commit is not None:
            artifact_refs["candidate_commit"] = candidate_commit
            for name, commit in self._resolve_revision_base_commits(
                Path(node.workspace_dir),
                candidate_commit,
                {
                    "implementation": getattr(node, "implementation_base_ref", ""),
                    "diff": getattr(node, "diff_base_ref", ""),
                    "feedback": getattr(node, "feedback_base_ref", ""),
                },
            ).items():
                artifact_refs[f"{name}_base_commit"] = commit
        if candidate_ref:
            artifact_refs["candidate_ref"] = candidate_ref
        for position, attempt in enumerate(record.evaluation_attempts):
            artifact_refs[f"evaluation_commit_{position}"] = attempt.commit_sha
        self.revision_journal.append_projection(
            projection=record.to_dict(),
            execution_status=execution_status,
            evaluation_status=evaluation_status,
            evaluator_fingerprint_ids=tuple(
                sorted({attempt.evaluator_id for attempt in record.evaluation_attempts})
            ),
            measurements=measurements,
            artifact_refs=artifact_refs,
        )
        if existing:
            prior = existing[0]
            if prior == record:
                return prior
            if record.execution_revision == prior.execution_revision:
                raise JournalConflictError(
                    "execution journal revision conflicts with prior content"
                )
        self._save(proposed, self.revision + 1)
        self.experiments = proposed
        self.revision += 1
        return record

    def _resolve_revision_base_commits(
        self,
        workspace: Path,
        candidate_commit: str,
        base_refs: Mapping[str, str],
    ) -> dict[str, str]:
        resolved: dict[str, str] = {}
        for name, base_ref in base_refs.items():
            if not base_ref:
                continue
            git_command = self._require_git_command()
            result = git_command.run(
                workspace,
                (
                    "rev-parse",
                    "--verify",
                    f"{base_ref}^{{commit}}",
                ),
                output_kind=CommandOutputKind.TEXT,
            )
            if result.returncode != 0:
                raise ValueError(
                    result.stderr.decode("utf-8").strip()
                    or f"could not resolve {name} base commit"
                )
            commit = result.output.strip()
            if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
                raise ValueError(f"resolved {name} base commit is invalid")
            ancestry = git_command.run(
                workspace,
                (
                    "merge-base",
                    "--is-ancestor",
                    commit,
                    candidate_commit,
                ),
                output_kind=CommandOutputKind.BINARY,
            )
            if ancestry.returncode == 1:
                raise ValueError(f"{name} base is not a candidate ancestor")
            if ancestry.returncode != 0:
                raise ValueError(ancestry.stderr.decode("utf-8", errors="strict"))
            resolved[name] = commit
        return resolved

    def _pin_revision_commit(
        self,
        node: Any,
        record: _ExperimentRecord,
    ) -> tuple[str | None, str]:
        attempt_commits = tuple(
            sorted({attempt.commit_sha for attempt in record.evaluation_attempts})
        )
        if len(attempt_commits) > 1:
            raise ValueError("one execution revision cannot evaluate multiple commits")
        workspace_text = getattr(node, "workspace_dir", "")
        if not workspace_text:
            return (attempt_commits[0], "") if attempt_commits else (None, "")
        workspace = Path(workspace_text)
        if not workspace.is_dir() or not (workspace / ".git").exists():
            raise ValueError("execution revision workspace is not a Git repository")
        branch_name = record.branch_name
        require_git_ref_name(
            branch_name,
            "experiment branch",
            qualified=False,
            error_type=ValueError,
        )
        pinned_ref = (
            f"refs/kapso/execution-revisions/{self.run_id}/"
            f"node-{record.node_id}/revision-{record.execution_revision}"
        )
        git_command = self._require_git_command()
        pinned = git_command.run(
            workspace,
            (
                "for-each-ref",
                "--format=%(objectname)",
                pinned_ref,
            ),
            output_kind=CommandOutputKind.TEXT,
        )
        if pinned.returncode != 0:
            raise ValueError(
                pinned.stderr.decode("utf-8").strip() or "could not read revision pin"
            )
        commit = pinned.output.strip()
        if commit:
            if not re.fullmatch(r"[0-9a-f]{40}", commit):
                raise ValueError("pinned execution revision commit is invalid")
            if attempt_commits and attempt_commits[0] != commit:
                raise ValueError("evaluation commit conflicts with pinned revision")
            return commit, pinned_ref
        branch_ref = f"refs/heads/{branch_name}"
        branch = git_command.run(
            workspace,
            (
                "for-each-ref",
                "--format=%(objectname)",
                branch_ref,
            ),
            output_kind=CommandOutputKind.TEXT,
        )
        if branch.returncode != 0:
            raise ValueError(
                branch.stderr.decode("utf-8").strip()
                or "could not read experiment branch"
            )
        branch_commit = branch.output.strip()
        if not branch_commit:
            if record.had_error and not attempt_commits:
                return None, ""
            raise ValueError("executed revision branch is missing")
        if not re.fullmatch(r"[0-9a-f]{40}", branch_commit):
            raise ValueError("experiment branch commit is invalid")
        commit = attempt_commits[0] if attempt_commits else branch_commit
        if attempt_commits and branch_commit != commit:
            raise ValueError("evaluation commit differs from experiment branch")
        verify = git_command.run(
            workspace,
            ("cat-file", "-e", f"{commit}^{{commit}}"),
            output_kind=CommandOutputKind.BINARY,
        )
        if verify.returncode != 0:
            raise ValueError("execution revision commit object is unavailable")
        create = git_command.run(
            workspace,
            ("update-ref", pinned_ref, commit, "0" * 40),
            output_kind=CommandOutputKind.TEXT,
        )
        if create.returncode != 0:
            raise ValueError(
                create.stderr.decode("utf-8").strip()
                or "could not pin execution revision"
            )
        return commit, pinned_ref

    def _require_git_command(self) -> BoundedGitCommand:
        if self._git_command is None:
            raise ValueError(
                "journaled Git evidence requires configured command bounds"
            )
        return self._git_command

    def get_top_experiments(self, k: int = 5) -> List[_ExperimentRecord]:
        self._require_limit(k)
        eligible = [
            record
            for record in self.experiments
            if not record.had_error
            and record.evaluation_valid
            and record.normalized_utility is not None
        ]
        return sorted(
            eligible,
            key=lambda record: (record.normalized_utility, -record.node_id),
            reverse=True,
        )[:k]

    def get_recent_experiments(self, k: int = 5) -> List[_ExperimentRecord]:
        self._require_limit(k)
        return self.experiments[-k:]

    def search_similar(self, query: str, k: int = 3) -> List[_ExperimentRecord]:
        self._require_limit(k)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("experiment similarity query must be non-empty")
        embedded = [record for record in self.experiments if record.solution_embedding]
        if self._llm is None or not embedded:
            return self.get_recent_experiments(k)
        query_embedding = self._llm.create_embedding(query)
        ranked = sorted(
            embedded,
            key=lambda record: (
                _cosine_similarity(query_embedding, record.solution_embedding),
                record.node_id,
            ),
            reverse=True,
        )
        return ranked[:k]

    def get_experiment_count(self) -> int:
        return len(self.experiments)

    def close(self) -> None:
        return None

    @staticmethod
    def _require_limit(k: int) -> None:
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("experiment retrieval limit must be positive")

    def _load(self) -> None:
        data = json.loads(
            self.path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
        if not isinstance(data, dict) or set(data) != {
            "schema",
            "run_id",
            "campaign_id",
            "revision",
            "objective_direction",
            "require_idea_links",
            "records",
        }:
            raise ValueError("experiment-history document fields are invalid")
        if data["schema"] != _EXPERIMENT_HISTORY_SCHEMA:
            raise ValueError("experiment-history schema is incompatible")
        if data["objective_direction"] not in {"maximize", "minimize"}:
            raise ValueError("experiment-history objective direction is invalid")
        if not isinstance(data["require_idea_links"], bool):
            raise ValueError("experiment-history idea-link policy is invalid")
        if not isinstance(data["records"], list):
            raise ValueError("experiment-history records must be a list")
        require_identifier(data["run_id"], "experiment-history run_id")
        require_identifier(data["campaign_id"], "experiment-history campaign_id")
        if type(data["revision"]) is not int or data["revision"] < 0:
            raise ValueError("experiment-history revision must be non-negative")
        records = [_ExperimentRecord.from_dict(item) for item in data["records"]]
        if [record.node_id for record in records] != list(range(len(records))):
            raise ValueError("experiment-history node ids must be contiguous")
        self.objective_direction = data["objective_direction"]
        self.require_idea_links = data["require_idea_links"]
        self.run_id = data["run_id"]
        self.campaign_id = data["campaign_id"]
        self.revision = data["revision"]
        self.experiments = records

    def _recover_from_journal(self) -> None:
        if self.revision_journal is None:
            raise ValueError("journal recovery requires a revision journal")
        events = self.revision_journal.read_events()
        if self.revision not in {len(events), len(events) - 1}:
            raise ValueError(
                "experiment history is not at a recoverable journal boundary"
            )
        projected_prefix: dict[int, _ExperimentRecord] = {}
        for event in events[: self.revision]:
            projected_prefix[event.node_id] = _ExperimentRecord.from_dict(
                to_json_value(event.projection)
            )
        expected_history = [
            projected_prefix[node_id] for node_id in sorted(projected_prefix)
        ]
        if self.experiments != expected_history:
            raise ValueError("experiment history conflicts with its journal prefix")
        terminal_by_node = {}
        for event in events:
            terminal_by_node[event.node_id] = _ExperimentRecord.from_dict(
                to_json_value(event.projection)
            )
        terminal_records = [
            terminal_by_node[node_id] for node_id in sorted(terminal_by_node)
        ]
        if [record.node_id for record in terminal_records] != list(
            range(len(terminal_records))
        ):
            raise ValueError("journal terminal projections are not contiguous")
        for prior, terminal in zip(self.experiments, terminal_records):
            if prior.execution_revision > terminal.execution_revision:
                raise ValueError("experiment history revision exceeds journal terminal")
        if self.experiments != terminal_records or self.revision != len(events):
            self._save(terminal_records, len(events))
            self.experiments = terminal_records
            self.revision = len(events)

    def _refresh_journaled_authorities(self) -> None:
        expected_identity = (
            self.run_id,
            self.campaign_id,
            self.objective_direction,
            self.require_idea_links,
        )
        if self.path.exists():
            self._load()
        actual_identity = (
            self.run_id,
            self.campaign_id,
            self.objective_direction,
            self.require_idea_links,
        )
        if actual_identity != expected_identity:
            raise ValueError("experiment-history transaction identity changed")
        self._recover_from_journal()

    def _bind_transaction_lock(self, lock: Any) -> None:
        identity = canonical_json_bytes(
            {
                "campaign_id": self.campaign_id,
                "history_path": str(self.path.resolve()),
                "run_id": self.run_id,
            }
        )
        lock.seek(0)
        existing = lock.read()
        if existing and existing != identity:
            raise ValueError("journal is bound to another experiment history")
        if not existing:
            lock.write(identity)
            lock.flush()
            os.fsync(lock.fileno())

    def reconcile_revision_journal(self) -> None:
        """Require exact agreement with the journal terminal frontier."""
        if self.revision_journal is None:
            raise ValueError("experiment history has no revision journal")
        with self._transaction_lock_path.open("a+b") as lock:
            self._transaction_lock_path.chmod(0o600)
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            self._bind_transaction_lock(lock)
            self._refresh_journaled_authorities()
            events = self.revision_journal.read_events()
            terminal_by_node = {}
            for event in events:
                terminal_by_node[event.node_id] = _ExperimentRecord.from_dict(
                    to_json_value(event.projection)
                )
            terminal = tuple(
                terminal_by_node[node_id] for node_id in sorted(terminal_by_node)
            )
            if tuple(self.experiments) != terminal or self.revision != len(events):
                raise ValueError("experiment history and revision journal diverged")
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _save(self, records: Iterable[_ExperimentRecord], revision: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        persisted_records = list(records)
        document = {
            "schema": _EXPERIMENT_HISTORY_SCHEMA,
            "run_id": self.run_id,
            "campaign_id": self.campaign_id,
            "revision": revision,
            "objective_direction": self.objective_direction,
            "require_idea_links": self.require_idea_links,
            "records": [record.to_dict() for record in persisted_records],
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.path.parent,
            prefix=self.path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.chmod(0o600)
        os.replace(temporary, self.path)
        self.path.chmod(0o600)
        directory_descriptor = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        os.fsync(directory_descriptor)
        os.close(directory_descriptor)


def format_experiments(experiments: Iterable[_ExperimentRecord]) -> str:
    """Render complete executed content without exposing caller-owned metrics."""
    records = tuple(experiments)
    if not records:
        return "No experiments found."
    lines = []
    for record in records:
        status = (
            "FAILED"
            if record.had_error
            else (
                "INVALID EVALUATION"
                if not record.evaluation_valid
                else f"raw_score={record.raw_score}; utility={record.normalized_utility}"
            )
        )
        lines.append(f"""
## Experiment {record.node_id} ({status})

**Idea:** `{record.idea_id or 'not_applicable'}`

**Selection batch:** `{record.selection_batch_id or 'not_applicable'}`

**Parent node:** `{record.parent_node_id}`

**Fidelity:** `{record.build_fidelity}` build / `{record.eval_fidelity}` eval ({record.validation_tier})

**Solution:**
{record.solution}

**Feedback:**
{record.feedback}""")
        if record.technical_difficulties:
            lines.append(f"""
**Technical difficulties:**
{record.technical_difficulties}""")
    return "\n".join(lines)
