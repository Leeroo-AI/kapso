"""Immutable GitHub exchange for externally signed expert evaluation."""

from __future__ import annotations

import base64
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from kapso.cross_run.canonical import (
    canonical_json_bytes,
    content_id,
    parse_json_bytes,
    require_content_id,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    ExpertEvaluatorOutcome,
    ExpertEvaluatorResultRecord,
    ExpertValidationAttempt,
    ExpertValidationStage,
)
from kapso.cross_run.expert.attestation import ConfiguredExpertAttestationVerifier
from kapso.cross_run.expert.store import StoredExpertCandidate
from kapso.cross_run.github.command import GitHubCommandClient
from kapso.cross_run.settings import (
    ExpertValidationSettings,
    GitHubSettings,
    SanitationSettings,
)


class GitHubExpertEvaluatorError(ValueError):
    """The remote evaluator exchange is conflicting, incomplete, or untrusted."""


_PROTOCOL_VERSION = "kapso.github_expert_evaluator.v1"
_WORKFLOW_FILE = "kapso-expert-evaluator.yml"
_REQUEST_ASSET = "request.json"
_EVALUATION_ASSET = "evaluation.json"
_RESULT_ASSET = "result.json"
_RELEASE_PAGE_SIZE = 100
_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_EXTERNAL_STAGES = {
    ExpertValidationStage.CONTRACT_SCHEMA,
    ExpertValidationStage.IDENTITY_SECRETS_LICENSE_DEPENDENCY,
    ExpertValidationStage.STATIC_UNIT_SECURITY_RESOURCE,
    ExpertValidationStage.SYNTHETIC_FRESH_TASK,
}


def build_github_expert_evaluator_request(
    *,
    stored_candidate: StoredExpertCandidate,
    attempt: ExpertValidationAttempt,
    stage: ExpertValidationStage,
    expected_transition_id: str,
    evaluator_revision: str,
    validation_settings: ExpertValidationSettings,
    sanitation_settings: SanitationSettings,
) -> Mapping[str, Any]:
    """Bind candidate bytes, active transition, policy, and evaluator revision."""

    if type(stored_candidate) is not StoredExpertCandidate:
        raise GitHubExpertEvaluatorError(
            "evaluator request requires a stored candidate"
        )
    if type(attempt) is not ExpertValidationAttempt:
        raise GitHubExpertEvaluatorError(
            "evaluator request requires a validation attempt"
        )
    if stage not in _EXTERNAL_STAGES or stage not in attempt.required_stages:
        raise GitHubExpertEvaluatorError("evaluator request stage is not external")
    require_content_id(expected_transition_id, "expected_transition_id")
    if _SHA_PATTERN.fullmatch(evaluator_revision) is None:
        raise GitHubExpertEvaluatorError("evaluator revision must be a commit SHA")
    closure = stored_candidate.closure
    if (
        closure.manifest.candidate_id != attempt.candidate_id
        or closure.candidate_tree.tree_hash != attempt.candidate_tree_hash
        or closure.manifest.scope_contract_id != attempt.scope_contract_id
    ):
        raise GitHubExpertEvaluatorError("stored candidate differs from its attempt")
    exchange_input_id = content_id(
        "expert-evaluator-exchange-input",
        {
            "evaluator_revision": evaluator_revision,
            "expected_transition_id": expected_transition_id,
            "stage": stage.value,
            "validation_attempt_id": attempt.validation_attempt_id,
        },
    )
    content = {
        "candidate_contents_base64": {
            path: base64.b64encode(payload).decode("ascii")
            for path, payload in sorted(closure.candidate_contents.items())
        },
        "candidate_sanitation_report": closure.sanitation_report.to_dict(),
        "candidate_tree": closure.candidate_tree.to_dict(),
        "evaluator_revision": evaluator_revision,
        "exact_additional_input_ids": [exchange_input_id],
        "expected_transition_id": expected_transition_id,
        "module_contracts": [module.to_dict() for module in closure.module_contracts],
        "protocol_version": _PROTOCOL_VERSION,
        "repository_map": closure.repository_map.to_dict(),
        "sanitation_settings": sanitation_settings.to_dict(),
        "scope_contract_id": attempt.scope_contract_id,
        "stage": stage.value,
        "validation_attempt": attempt.to_dict(),
        "validation_settings": validation_settings.to_dict(),
    }
    return {
        "request_id": content_id("expert-evaluator-request", content),
        **content,
    }


class GitHubExpertEvaluatorExchange:
    """Publish one request and consume one immutable signed response."""

    def __init__(
        self,
        *,
        client: GitHubCommandClient,
        github_settings: GitHubSettings,
        validation_settings: ExpertValidationSettings,
        sanitation_settings: SanitationSettings,
        security_repository: str,
    ) -> None:
        if type(client) is not GitHubCommandClient:
            raise GitHubExpertEvaluatorError("evaluator exchange requires GitHub")
        if type(github_settings) is not GitHubSettings:
            raise GitHubExpertEvaluatorError(
                "evaluator exchange requires GitHub settings"
            )
        if type(validation_settings) is not ExpertValidationSettings:
            raise GitHubExpertEvaluatorError(
                "evaluator exchange requires validation settings"
            )
        if type(sanitation_settings) is not SanitationSettings:
            raise GitHubExpertEvaluatorError(
                "evaluator exchange requires sanitation settings"
            )
        if (
            re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", security_repository)
            is None
        ):
            raise GitHubExpertEvaluatorError("security repository is invalid")
        self.client = client
        self.github_settings = github_settings
        self.validation_settings = validation_settings
        self.sanitation_settings = sanitation_settings
        self.security_repository = security_repository

    def evaluate(
        self,
        *,
        stored_candidate: StoredExpertCandidate,
        attempt: ExpertValidationAttempt,
        stage: ExpertValidationStage,
        expected_transition_id: str,
    ) -> ExpertEvaluatorResultRecord:
        evaluator = tuple(
            configured
            for configured in self.validation_settings.policy.evaluators
            if configured.stage is stage
        )
        if len(evaluator) != 1:
            raise GitHubExpertEvaluatorError(
                "external evaluator authority is ambiguous"
            )
        deadline = time.monotonic() + evaluator[0].timeout_seconds
        evaluator_revision = self.client.read_ref_commit(
            self.security_repository,
            f"refs/heads/{self.github_settings.default_branch}",
            allow_missing=False,
        )
        if evaluator_revision is None:
            raise GitHubExpertEvaluatorError("evaluator revision is unavailable")
        request = build_github_expert_evaluator_request(
            stored_candidate=stored_candidate,
            attempt=attempt,
            stage=stage,
            expected_transition_id=expected_transition_id,
            evaluator_revision=evaluator_revision,
            validation_settings=self.validation_settings,
            sanitation_settings=self.sanitation_settings,
        )
        request_bytes = canonical_json_bytes(request)
        if len(request_bytes) > self.github_settings.release_asset_size_bytes:
            raise GitHubExpertEvaluatorError(
                "evaluator request exceeds its asset bound"
            )
        request_suffix = request["request_id"].split(":sha256:", 1)[1]
        request_tag = f"kapso-evaluator-request-{request_suffix}"
        response_tag = f"kapso-evaluator-response-{request_suffix}"
        with tempfile.TemporaryDirectory(
            prefix="expert-evaluator-exchange-",
            dir=self.client.working_directory,
        ) as temporary:
            root = Path(temporary)
            request_path = root / _REQUEST_ASSET
            request_path.write_bytes(request_bytes)
            self._ensure_request_release(
                tag=request_tag,
                evaluator_revision=evaluator_revision,
                request_path=request_path,
                request_bytes=request_bytes,
                deadline=deadline,
                root=root,
            )
            response = self._ensure_response_draft(
                tag=response_tag,
                evaluator_revision=evaluator_revision,
            )
            if response.get("draft") is True:
                self.client.dispatch_workflow(
                    self.security_repository,
                    _WORKFLOW_FILE,
                    self.github_settings.default_branch,
                    {
                        "evaluator_revision": evaluator_revision,
                        "request_id": request["request_id"],
                        "request_release_tag": request_tag,
                        "response_release_id": str(response["id"]),
                        "response_release_tag": response_tag,
                    },
                )
                response = self._wait_for_immutable_response(
                    response_id=response["id"],
                    response_tag=response_tag,
                    deadline=deadline,
                )
            result_bytes = self._download_response(response, root)
        result = ExpertEvaluatorResultRecord.from_json_bytes(result_bytes)
        if result.to_json_bytes() != result_bytes:
            raise GitHubExpertEvaluatorError("evaluator result is not canonical")
        self._verify_result(
            result=result,
            request=request,
            attempt=attempt,
            stage=stage,
        )
        return result

    def _ensure_request_release(
        self,
        *,
        tag: str,
        evaluator_revision: str,
        request_path: Path,
        request_bytes: bytes,
        deadline: float,
        root: Path,
    ) -> None:
        release = self._find_release(tag)
        if release is None:
            release = self._create_draft(tag, evaluator_revision)
        self._validate_release_identity(release, tag)
        release_id = release["id"]
        if release.get("draft") is True:
            assets = self._assets(release, allowed_names={_REQUEST_ASSET})
            if _REQUEST_ASSET not in assets:
                self.client.upload_release_asset(
                    self.security_repository,
                    release_id,
                    request_path,
                    _REQUEST_ASSET,
                    "application/json",
                    len(request_bytes),
                )
            release = self._release(release_id)
            self._validate_exact_asset_bytes(
                release=release,
                name=_REQUEST_ASSET,
                expected=request_bytes,
                root=root,
                verification_label="draft",
            )
            release = self.client.api_json(
                "PATCH",
                f"repos/{self.security_repository}/releases/{release_id}",
                {"draft": False},
            )
            release = self._wait_for_immutable_release(
                release=release,
                tag=tag,
                deadline=deadline,
            )
        self._validate_immutable_release(release, tag, {_REQUEST_ASSET})
        self._validate_exact_asset_bytes(
            release=release,
            name=_REQUEST_ASSET,
            expected=request_bytes,
            root=root,
            verification_label="immutable",
        )

    def _ensure_response_draft(
        self,
        *,
        tag: str,
        evaluator_revision: str,
    ) -> Mapping[str, Any]:
        release = self._find_release(tag)
        if release is None:
            release = self._create_draft(tag, evaluator_revision)
        self._validate_release_identity(release, tag)
        if release.get("draft") is True:
            self._assets(release, allowed_names={_EVALUATION_ASSET})
            return release
        self._validate_immutable_release(
            release,
            tag,
            {_EVALUATION_ASSET, _RESULT_ASSET},
        )
        return release

    def _create_draft(self, tag: str, evaluator_revision: str) -> Mapping[str, Any]:
        release = self.client.api_json(
            "POST",
            f"repos/{self.security_repository}/releases",
            {
                "draft": True,
                "name": tag,
                "prerelease": False,
                "tag_name": tag,
                "target_commitish": evaluator_revision,
            },
        )
        if not isinstance(release, Mapping):
            raise GitHubExpertEvaluatorError("created evaluator release is invalid")
        self._validate_release_identity(release, tag)
        if release.get("draft") is not True or release.get("immutable") is not False:
            raise GitHubExpertEvaluatorError("created evaluator release is not a draft")
        if self._assets(release, allowed_names=set()):
            raise GitHubExpertEvaluatorError("new evaluator draft contains assets")
        return release

    def _find_release(self, tag: str) -> Mapping[str, Any] | None:
        releases = self.client.api_json_pages(
            f"repos/{self.security_repository}/releases?per_page={_RELEASE_PAGE_SIZE}"
        )
        matches = tuple(
            release
            for release in releases
            if isinstance(release, Mapping) and release.get("tag_name") == tag
        )
        if len(matches) > 1:
            raise GitHubExpertEvaluatorError("evaluator release tag is ambiguous")
        return None if not matches else matches[0]

    def _wait_for_immutable_response(
        self,
        *,
        response_id: int,
        response_tag: str,
        deadline: float,
    ) -> Mapping[str, Any]:
        while True:
            release = self._release(response_id)
            self._validate_release_identity(release, response_tag)
            if release.get("draft") is False and release.get("immutable") is True:
                self._validate_immutable_release(
                    release,
                    response_tag,
                    {_EVALUATION_ASSET, _RESULT_ASSET},
                )
                return release
            if (
                release.get("draft") is not True
                or release.get("immutable") is not False
            ):
                raise GitHubExpertEvaluatorError(
                    "evaluator response left the draft/immutable protocol"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GitHubExpertEvaluatorError("evaluator workflow timed out")
            time.sleep(
                min(
                    self.github_settings.release_visibility_poll_interval_seconds,
                    remaining,
                )
            )

    def _wait_for_immutable_release(
        self,
        *,
        release: Any,
        tag: str,
        deadline: float,
    ) -> Mapping[str, Any]:
        if not isinstance(release, Mapping):
            raise GitHubExpertEvaluatorError("published evaluator release is invalid")
        self._validate_release_identity(release, tag)
        while release.get("immutable") is not True:
            if release.get("draft") is not False:
                raise GitHubExpertEvaluatorError("evaluator request did not publish")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GitHubExpertEvaluatorError(
                    "evaluator request did not become immutable"
                )
            time.sleep(
                min(
                    self.github_settings.release_visibility_poll_interval_seconds,
                    remaining,
                )
            )
            release = self._release(release["id"])
            self._validate_release_identity(release, tag)
        return release

    def _download_response(self, release: Mapping[str, Any], root: Path) -> bytes:
        assets = self._assets(
            release,
            allowed_names={_EVALUATION_ASSET, _RESULT_ASSET},
        )
        result = b""
        for name in (_EVALUATION_ASSET, _RESULT_ASSET):
            destination = root / name
            metadata = assets[name]
            self.client.download_release_asset(
                self.security_repository,
                str(metadata["id"]),
                destination,
                self.github_settings.release_asset_size_bytes,
            )
            payload = destination.read_bytes()
            if (
                len(payload) != metadata["size"]
                or tree_or_blob_digest(payload) != metadata["digest"]
            ):
                raise GitHubExpertEvaluatorError(
                    f"evaluator response bytes differ: {name}"
                )
            if name == _RESULT_ASSET:
                result = payload
        return result

    def _validate_exact_asset_bytes(
        self,
        *,
        release: Mapping[str, Any],
        name: str,
        expected: bytes,
        root: Path,
        verification_label: str,
    ) -> None:
        assets = self._assets(release, allowed_names={name})
        if set(assets) != {name}:
            raise GitHubExpertEvaluatorError("evaluator request asset is incomplete")
        destination = root / f"verified-{verification_label}-{name}"
        self.client.download_release_asset(
            self.security_repository,
            str(assets[name]["id"]),
            destination,
            self.github_settings.release_asset_size_bytes,
        )
        observed = destination.read_bytes()
        if (
            observed != expected
            or len(observed) != assets[name]["size"]
            or tree_or_blob_digest(observed) != assets[name]["digest"]
        ):
            raise GitHubExpertEvaluatorError("evaluator request bytes conflict")

    def _validate_immutable_release(
        self,
        release: Mapping[str, Any],
        tag: str,
        expected_assets: set[str],
    ) -> None:
        self._validate_release_identity(release, tag)
        if release.get("draft") is not False or release.get("immutable") is not True:
            raise GitHubExpertEvaluatorError("evaluator release is not immutable")
        if set(self._assets(release, allowed_names=expected_assets)) != expected_assets:
            raise GitHubExpertEvaluatorError("evaluator release asset closure differs")

    def _validate_release_identity(
        self,
        release: Mapping[str, Any],
        tag: str,
    ) -> None:
        release_id = release.get("id")
        author = release.get("author")
        if (
            type(release_id) is not int
            or release_id < 1
            or release.get("tag_name") != tag
            or not isinstance(author, Mapping)
            or author.get("login") != self.github_settings.publisher_login
        ):
            raise GitHubExpertEvaluatorError("evaluator release identity is invalid")

    def _assets(
        self,
        release: Mapping[str, Any],
        *,
        allowed_names: set[str],
    ) -> Mapping[str, Mapping[str, Any]]:
        raw_assets = release.get("assets")
        if (
            not isinstance(raw_assets, list)
            or len(raw_assets) > self.github_settings.release_asset_count_limit
        ):
            raise GitHubExpertEvaluatorError("evaluator release assets are invalid")
        assets: dict[str, Mapping[str, Any]] = {}
        for value in raw_assets:
            if not isinstance(value, Mapping):
                raise GitHubExpertEvaluatorError("evaluator release asset is invalid")
            name = value.get("name")
            asset_id = value.get("id")
            size = value.get("size")
            digest = value.get("digest")
            if (
                name not in allowed_names
                or name in assets
                or type(asset_id) is not int
                or asset_id < 1
                or type(size) is not int
                or size <= 0
                or size > self.github_settings.release_asset_size_bytes
                or not isinstance(digest, str)
                or _DIGEST_PATTERN.fullmatch(digest) is None
                or value.get("state") != "uploaded"
            ):
                raise GitHubExpertEvaluatorError(
                    "evaluator release asset metadata is invalid"
                )
            assets[name] = value
        return assets

    def _release(self, release_id: int) -> Mapping[str, Any]:
        release = self.client.api_json(
            "GET",
            f"repos/{self.security_repository}/releases/{release_id}",
        )
        if not isinstance(release, Mapping):
            raise GitHubExpertEvaluatorError("evaluator release response is invalid")
        return release

    def _verify_result(
        self,
        *,
        result: ExpertEvaluatorResultRecord,
        request: Mapping[str, Any],
        attempt: ExpertValidationAttempt,
        stage: ExpertValidationStage,
    ) -> None:
        ConfiguredExpertAttestationVerifier(self.validation_settings).verify(
            result.attestation_envelope
        )
        run = result.evaluator_run
        additional_ids = tuple(request["exact_additional_input_ids"])
        if (
            run.validation_attempt_id != attempt.validation_attempt_id
            or run.candidate_id != attempt.candidate_id
            or run.candidate_tree_hash != attempt.candidate_tree_hash
            or run.stage is not stage
            or run.outcome is not ExpertEvaluatorOutcome.PASSED
            or not set(additional_ids).issubset(run.exact_input_ids)
            or set(run.output_payloads_base64) != {"report.json"}
        ):
            raise GitHubExpertEvaluatorError(
                "signed evaluator result binds another run"
            )
        report = parse_json_bytes(
            base64.b64decode(
                run.output_payloads_base64["report.json"],
                validate=True,
            )
        )
        if (
            not isinstance(report, Mapping)
            or report.get("request_id") != request["request_id"]
            or report.get("evaluator_revision") != request["evaluator_revision"]
            or report.get("candidate_id") != attempt.candidate_id
            or report.get("candidate_tree_hash") != attempt.candidate_tree_hash
            or report.get("stage") != stage.value
            or report.get("status") != "passed"
        ):
            raise GitHubExpertEvaluatorError(
                "signed evaluator report binds another request"
            )
