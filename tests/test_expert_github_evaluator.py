"""The GitHub evaluator exchange binds bytes, transition, signer, and replay."""

from __future__ import annotations

import base64
from dataclasses import replace

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from kapso.core.config import load_effective_config
from kapso.cross_run.canonical import (
    canonical_json_bytes,
    content_id,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    ExpertEvaluatorOutcome,
    ExpertValidationStage,
)
from kapso.cross_run.expert.github_evaluator import (
    GitHubExpertEvaluatorExchange,
    GitHubExpertEvaluatorRevision,
    GitHubExpertEvaluatorRevisionInstaller,
    build_github_expert_evaluator_request,
)
from kapso.cross_run.expert.validation import ExpertEvaluatorRunBuilder
from kapso.cross_run.github.command import GitHubCommandClient
from kapso.cross_run.settings import ExpertEvaluatorTrustRootSettings
from test_expert_candidate_store import candidate_store
from test_expert_candidates import bootstrap_candidate_closure
from test_expert_validation import _attempt, _eligibility_evaluator, _task_adapter

_CONFIG_PATH = "src/kapso/config.yaml"
_REVISION = "a" * 40


class _UnusedRunner:
    def run(self, _request):
        raise AssertionError("exchange used the raw runner")


def _case(tmp_path):
    settings = load_effective_config(_CONFIG_PATH, "GENERIC").cross_run
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    validation = replace(
        settings.expert.validation,
        evaluator_trust_roots=(
            ExpertEvaluatorTrustRootSettings(
                trust_root_id="test_github_evaluator_root",
                issuer_ids=("expert_contract_evaluator",),
                public_key_base64=base64.b64encode(public_key).decode("ascii"),
            ),
        ),
    )
    candidate_state = tmp_path / "candidate-state"
    candidate_state.mkdir()
    store = candidate_store(candidate_state)
    closure = bootstrap_candidate_closure()
    stored = store.persist(closure)
    eligibility = _eligibility_evaluator(
        validation,
        store,
        _task_adapter(closure),
    ).decide(candidate_id=closure.manifest.candidate_id)
    attempt = _attempt(eligibility.decision)
    transition_id = content_id("expert-validation-transition", {"position": 1})
    request = build_github_expert_evaluator_request(
        stored_candidate=stored,
        attempt=attempt,
        stage=ExpertValidationStage.CONTRACT_SCHEMA,
        expected_transition_id=transition_id,
        evaluator_revision=_REVISION,
        validation_settings=validation,
        sanitation_settings=settings.sanitation,
    )
    report = canonical_json_bytes(
        {
            "candidate_id": attempt.candidate_id,
            "candidate_tree_hash": attempt.candidate_tree_hash,
            "evaluator_revision": _REVISION,
            "request_id": request["request_id"],
            "stage": ExpertValidationStage.CONTRACT_SCHEMA.value,
            "status": "passed",
        }
    )
    builder = ExpertEvaluatorRunBuilder(validation)
    unsigned = builder.build(
        attempt=attempt,
        stage=ExpertValidationStage.CONTRACT_SCHEMA,
        exact_additional_input_ids=tuple(request["exact_additional_input_ids"]),
        output_payloads={"report.json": report},
        measurements={"checks_passed": 1.0},
        costs={},
        duration_seconds=0.0,
        outcome=ExpertEvaluatorOutcome.PASSED,
        signature=base64.b64encode(bytes(64)).decode("ascii"),
    )
    signature = private_key.sign(
        unsigned.attestation_envelope.attestation.to_json_bytes()
    )
    result = builder.build(
        attempt=attempt,
        stage=ExpertValidationStage.CONTRACT_SCHEMA,
        exact_additional_input_ids=tuple(request["exact_additional_input_ids"]),
        output_payloads={"report.json": report},
        measurements={"checks_passed": 1.0},
        costs={},
        duration_seconds=0.0,
        outcome=ExpertEvaluatorOutcome.PASSED,
        signature=base64.b64encode(signature).decode("ascii"),
    )
    return settings, validation, stored, attempt, transition_id, request, result


def test_request_identity_changes_with_transition_candidate_bytes_and_revision(
    tmp_path,
):
    settings, validation, stored, attempt, transition_id, request, _result = _case(
        tmp_path
    )

    next_transition = build_github_expert_evaluator_request(
        stored_candidate=stored,
        attempt=attempt,
        stage=ExpertValidationStage.CONTRACT_SCHEMA,
        expected_transition_id=content_id(
            "expert-validation-transition",
            {"position": 2},
        ),
        evaluator_revision=_REVISION,
        validation_settings=validation,
        sanitation_settings=settings.sanitation,
    )

    assert request["expected_transition_id"] == transition_id
    assert request["candidate_contents_base64"]
    assert request["request_id"] != next_transition["request_id"]
    content = dict(request)
    request_id = content.pop("request_id")
    assert request_id == content_id("expert-evaluator-request", content)


def test_revision_installer_overlays_default_and_mints_immutable_dispatch_ref(
    tmp_path,
    monkeypatch,
):
    settings = load_effective_config(_CONFIG_PATH, "GENERIC").cross_run
    client_root = tmp_path / "github-installer"
    client_root.mkdir(mode=0o700)
    client = GitHubCommandClient(
        _UnusedRunner(),
        working_directory=client_root,
        timeout_seconds=settings.github.command_timeout_seconds,
        api_version=settings.github.api_version,
        minimum_cli_version=settings.github.minimum_cli_version,
        release_visibility_poll_interval_seconds=(
            settings.github.release_visibility_poll_interval_seconds
        ),
        control_blob_size_bytes=settings.github.control_blob_size_bytes,
    )
    repository = "Leeroo-AI/kapso-security"
    source_revision = "a" * 40
    current_revision = "b" * 40
    source_tree_sha = "c" * 40
    current_tree_sha = "d" * 40
    overlay_tree_sha = "e" * 40
    overlay_revision = "f" * 40
    source_entries = (
        (".github/workflows/kapso-expert-evaluator.yml", "1" * 40),
        ("evaluator/evaluate.py", "2" * 40),
        ("evaluator/sign.py", "3" * 40),
    )
    updated_refs = []
    created_refs = []
    active_workflows = []

    def read_ref(_repository, qualified_ref, *, allow_missing):
        assert _repository == repository
        assert not allow_missing
        return (
            source_revision
            if qualified_ref == "refs/heads/kapso-evaluator"
            else current_revision
        )

    def api_json(method, endpoint, body=None):
        if method == "GET" and endpoint.endswith(f"commits/{source_revision}"):
            return {"sha": source_revision, "tree": {"sha": source_tree_sha}}
        if method == "GET" and endpoint.endswith(f"commits/{current_revision}"):
            return {"sha": current_revision, "tree": {"sha": current_tree_sha}}
        if method == "GET" and f"trees/{source_tree_sha}" in endpoint:
            return {
                "tree": [
                    {
                        "mode": "100644",
                        "path": path,
                        "sha": sha,
                        "type": "blob",
                    }
                    for path, sha in source_entries
                ],
                "truncated": False,
            }
        if method == "GET" and f"trees/{current_tree_sha}" in endpoint:
            return {"tree": [], "truncated": False}
        if method == "POST" and endpoint.endswith("/git/trees"):
            assert body == {
                "base_tree": current_tree_sha,
                "tree": [
                    {
                        "mode": "100644",
                        "path": path,
                        "sha": sha,
                        "type": "blob",
                    }
                    for path, sha in source_entries
                ],
            }
            return {"sha": overlay_tree_sha}
        if method == "POST" and endpoint.endswith("/git/commits"):
            assert body == {
                "message": "Install Kapso expert evaluator",
                "parents": [current_revision],
                "tree": overlay_tree_sha,
            }
            return {
                "parents": [{"sha": current_revision}],
                "sha": overlay_revision,
                "tree": {"sha": overlay_tree_sha},
            }
        assert method == "GET" and endpoint == f"repos/{repository}"
        return {"full_name": repository, "node_id": "security-repository-node"}

    monkeypatch.setattr(client, "read_ref_commit", read_ref)
    monkeypatch.setattr(client, "api_json", api_json)
    monkeypatch.setattr(
        client,
        "update_ref_compare_and_swap",
        lambda *arguments: updated_refs.append(arguments),
    )
    monkeypatch.setattr(
        client,
        "wait_for_active_workflow",
        lambda *arguments: active_workflows.append(arguments),
    )
    monkeypatch.setattr(
        client,
        "create_ref_if_absent",
        lambda *arguments: created_refs.append(arguments),
    )

    installed = GitHubExpertEvaluatorRevisionInstaller(
        client,
        settings.github,
    ).install(repository)

    assert installed == GitHubExpertEvaluatorRevision(
        commit_sha=source_revision,
        dispatch_ref=f"kapso-evaluator-revisions/{source_revision}",
    )
    assert updated_refs == [
        (
            repository,
            "security-repository-node",
            settings.github.default_branch,
            current_revision,
            overlay_revision,
        )
    ]
    assert active_workflows == [(repository, "kapso-expert-evaluator.yml")]
    assert created_refs == [
        (
            repository,
            f"refs/heads/kapso-evaluator-revisions/{source_revision}",
            source_revision,
        )
    ]


def test_revision_identity_survives_default_branch_generations(
    tmp_path,
    monkeypatch,
):
    (
        settings,
        validation,
        stored,
        attempt,
        transition_id,
        _request,
        _result,
    ) = _case(tmp_path)
    client_root = tmp_path / "github-installer"
    client_root.mkdir(mode=0o700)
    client = GitHubCommandClient(
        _UnusedRunner(),
        working_directory=client_root,
        timeout_seconds=settings.github.command_timeout_seconds,
        api_version=settings.github.api_version,
        minimum_cli_version=settings.github.minimum_cli_version,
        release_visibility_poll_interval_seconds=(
            settings.github.release_visibility_poll_interval_seconds
        ),
        control_blob_size_bytes=settings.github.control_blob_size_bytes,
    )
    repository = "Leeroo-AI/kapso-security"
    source_revision = _REVISION
    current_revisions = ("b" * 40, "c" * 40)
    current_revision_iterator = iter(current_revisions)
    created_refs = []
    overlay_revisions = iter(("6" * 40, "7" * 40))

    def read_ref(_repository, qualified_ref, *, allow_missing):
        assert _repository == repository
        assert not allow_missing
        if qualified_ref == "refs/heads/kapso-evaluator":
            return source_revision
        return next(current_revision_iterator)

    monkeypatch.setattr(client, "read_ref_commit", read_ref)
    monkeypatch.setattr(client, "wait_for_active_workflow", lambda *args: None)
    monkeypatch.setattr(
        client,
        "create_ref_if_absent",
        lambda *arguments: created_refs.append(arguments),
    )
    installer = GitHubExpertEvaluatorRevisionInstaller(client, settings.github)
    monkeypatch.setattr(
        installer,
        "_source_files",
        lambda _repository, _revision: {
            ".github/workflows/kapso-expert-evaluator.yml": ("1" * 40, "100644"),
            "evaluator/evaluate.py": ("2" * 40, "100644"),
            "evaluator/sign.py": ("3" * 40, "100644"),
        },
    )
    monkeypatch.setattr(
        installer,
        "_commit_tree",
        lambda _repository, revision: (revision, {}),
    )
    monkeypatch.setattr(
        installer,
        "_commit_overlay",
        lambda **_arguments: next(overlay_revisions),
    )

    first_revision = installer.install(repository)
    second_revision = installer.install(repository)
    first_request = build_github_expert_evaluator_request(
        stored_candidate=stored,
        attempt=attempt,
        stage=ExpertValidationStage.CONTRACT_SCHEMA,
        expected_transition_id=transition_id,
        evaluator_revision=first_revision.commit_sha,
        validation_settings=validation,
        sanitation_settings=settings.sanitation,
    )
    second_request = build_github_expert_evaluator_request(
        stored_candidate=stored,
        attempt=attempt,
        stage=ExpertValidationStage.CONTRACT_SCHEMA,
        expected_transition_id=transition_id,
        evaluator_revision=second_revision.commit_sha,
        validation_settings=validation,
        sanitation_settings=settings.sanitation,
    )

    assert (
        first_revision
        == second_revision
        == GitHubExpertEvaluatorRevision(
            commit_sha=source_revision,
            dispatch_ref=f"kapso-evaluator-revisions/{source_revision}",
        )
    )
    assert created_refs == [
        (
            repository,
            f"refs/heads/kapso-evaluator-revisions/{source_revision}",
            source_revision,
        ),
        (
            repository,
            f"refs/heads/kapso-evaluator-revisions/{source_revision}",
            source_revision,
        ),
    ]
    assert first_request["request_id"] == second_request["request_id"]


def test_signed_immutable_response_replays_without_another_dispatch(
    tmp_path,
    monkeypatch,
):
    (
        settings,
        validation,
        stored,
        attempt,
        transition_id,
        _request,
        expected_result,
    ) = _case(tmp_path)
    client_root = tmp_path / "github"
    client_root.mkdir(mode=0o700)
    client = GitHubCommandClient(
        _UnusedRunner(),
        working_directory=client_root,
        timeout_seconds=settings.github.command_timeout_seconds,
        api_version=settings.github.api_version,
        minimum_cli_version=settings.github.minimum_cli_version,
        release_visibility_poll_interval_seconds=(
            settings.github.release_visibility_poll_interval_seconds
        ),
        control_blob_size_bytes=settings.github.control_blob_size_bytes,
    )
    releases = []
    payloads = {}
    dispatches = []

    def graphql(_query, variables):
        matches = tuple(
            release for release in releases if release["tag_name"] == variables["tag"]
        )
        assert variables["owner"] == "Leeroo-AI"
        assert variables["repository"] == "kapso-security"
        assert len(matches) <= 1
        return {
            "data": {
                "repository": {
                    "release": (
                        None if not matches else {"databaseId": matches[0]["id"]}
                    )
                }
            }
        }

    def api_json(method, endpoint, body=None):
        if method == "POST" and endpoint.endswith("/releases"):
            release = {
                "assets": [],
                "author": {"login": settings.github.publisher_login},
                "draft": True,
                "id": len(releases) + 1,
                "immutable": False,
                "tag_name": body["tag_name"],
            }
            releases.append(release)
            return release
        release_id = int(endpoint.rsplit("/", 1)[1])
        release = next(item for item in releases if item["id"] == release_id)
        if method == "PATCH":
            release["draft"] = False
            release["immutable"] = True
        return release

    def upload(_repository, release_id, asset_path, name, media_type, asset_size):
        payload = asset_path.read_bytes()
        assert len(payload) == asset_size
        asset_id = len(payloads) + 100
        payloads[asset_id] = payload
        release = next(item for item in releases if item["id"] == release_id)
        metadata = {
            "content_type": media_type,
            "digest": tree_or_blob_digest(payload),
            "id": asset_id,
            "name": name,
            "size": len(payload),
            "state": "uploaded",
        }
        release["assets"].append(metadata)
        return metadata

    def download(_repository, asset_id, destination, _maximum_bytes):
        destination.write_bytes(payloads[int(asset_id)])
        return destination

    def dispatch(repository, workflow_file, ref, inputs):
        assert repository == "Leeroo-AI/kapso-security"
        assert workflow_file == "kapso-expert-evaluator.yml"
        assert ref == f"kapso-evaluator-revisions/{_REVISION}"
        dispatches.append(inputs)
        response = next(
            release
            for release in releases
            if "kapso-evaluator-response" in release["tag_name"]
        )
        evaluation = canonical_json_bytes({"outcome": "passed"})
        for name, payload in (
            ("evaluation.json", evaluation),
            ("result.json", expected_result.to_json_bytes()),
        ):
            asset_id = len(payloads) + 100
            payloads[asset_id] = payload
            response["assets"].append(
                {
                    "digest": tree_or_blob_digest(payload),
                    "id": asset_id,
                    "name": name,
                    "size": len(payload),
                    "state": "uploaded",
                }
            )
        response["draft"] = False
        response["immutable"] = True
        return {
            "workflow_run_id": 1,
            "run_url": "https://api.github.com/run/1",
            "html_url": "https://github.com/run/1",
        }

    monkeypatch.setattr(client, "graphql", graphql)
    monkeypatch.setattr(client, "api_json", api_json)
    monkeypatch.setattr(client, "upload_release_asset", upload)
    monkeypatch.setattr(client, "download_release_asset", download)
    monkeypatch.setattr(client, "dispatch_workflow", dispatch)
    exchange = GitHubExpertEvaluatorExchange(
        client=client,
        github_settings=settings.github,
        validation_settings=validation,
        sanitation_settings=settings.sanitation,
        security_repository="Leeroo-AI/kapso-security",
    )
    monkeypatch.setattr(
        exchange.revision_installer,
        "install",
        lambda _repository: GitHubExpertEvaluatorRevision(
            commit_sha=_REVISION,
            dispatch_ref=f"kapso-evaluator-revisions/{_REVISION}",
        ),
    )

    first = exchange.evaluate(
        stored_candidate=stored,
        attempt=attempt,
        stage=ExpertValidationStage.CONTRACT_SCHEMA,
        expected_transition_id=transition_id,
    )
    second = exchange.evaluate(
        stored_candidate=stored,
        attempt=attempt,
        stage=ExpertValidationStage.CONTRACT_SCHEMA,
        expected_transition_id=transition_id,
    )

    assert first == expected_result
    assert second == expected_result
    assert len(dispatches) == 1
    assert dispatches[0]["evaluator_revision"] == _REVISION
    assert tuple(release["draft"] for release in releases) == (False, False)
