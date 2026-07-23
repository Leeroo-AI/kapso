import io
import tarfile
import base64
import hashlib
from dataclasses import dataclass, replace
from typing import Callable

import pytest

from kapso.core.config import load_config
from kapso.cross_run.canonical import (
    content_id,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    ExpertBaseReleaseManifest,
    KnowledgeSnapshotManifest,
    PublicationArtifactKind,
    ScopeRepositorySettings,
)
from kapso.cross_run.git_refs import git_tree_shas
from kapso.cross_run.github.command import GitHubCompareAndSwapError
from kapso.cross_run.github.materializer import GitHubArtifactMaterializer
from kapso.cross_run.github.materializer import MaterializationError
from kapso.cross_run.github.publisher import (
    AutonomousGitHubPublisher,
    GitHubPublicationError,
    PublicationEnvelope,
    ReleaseAssetInput,
)
from kapso.cross_run.github.resolver import (
    ArtifactPublicationIntent,
    CurrentArtifactPointer,
    CurrentPointerState,
    PublicationAssetIntent,
    PublicationSourceFile,
    RepositoryPolicyReport,
)
from kapso.cross_run.settings import CrossRunSettings
from tests.cross_run_github_fixtures import release_attestation

CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"
REPOSITORY = "Leeroo-AI/kapso-knowledge"
EXPECTED_PARENT = "a" * 40
SOURCE_COMMIT = "b" * 40
POINTER_COMMIT = "c" * 40
IDENTITY_COMMIT = "d" * 40
INTENT_COMMIT = "e" * 40


def cross_run_settings():
    return CrossRunSettings.from_dict(load_config(CANONICAL_CONFIG_PATH)["cross_run"])


def repositories():
    return ScopeRepositorySettings(
        scope_id="ml_ai",
        expert_repository="Leeroo-AI/kapso-expert",
        knowledge_repository=REPOSITORY,
        security_repository="Leeroo-AI/kapso-security",
    )


def snapshot_manifest(data, additional_checksums=None):
    checksums = {"data.txt": tree_or_blob_digest(data)}
    checksums.update(additional_checksums or {})
    return KnowledgeSnapshotManifest.mint(
        scope_contract_id=content_id("fixture", {"scope": "ml_ai"}),
        scope_id="ml_ai",
        parent_snapshot_ids=(),
        included_bundle_ids=(),
        admitted_episode_ids=(),
        admitted_prior_idea_ids=(),
        active_claim_revision_ids=(),
        catalog_generation=1,
        configuration_fingerprint=tree_or_blob_digest(b"config"),
        entry_state_refs=(),
        included_assertion_ids=(),
        included_revocation_ids=(),
        proof_dependency_closure_ids=(),
        sanitation_policy_version="kapso.sanitation.v1",
        retrieval_policy_version="kapso.retrieval.v1",
        embedding_sidecars=(),
        prompt_budget_policy={"maximum_records": 1},
        checksums=checksums,
        published_at="2026-07-20T15:00:00Z",
        publisher_attestation={"issuer": "fixture"},
    )


def build_envelope(tmp_path):
    data = b"scientific evidence"
    manifest = snapshot_manifest(data)
    source = tmp_path / "source"
    source.mkdir()
    (source / "snapshot.json").write_bytes(manifest.to_json_bytes())
    (source / "data.txt").write_bytes(data)
    archive = tmp_path / "snapshot.tar"
    with tarfile.open(archive, "w") as package:
        for name, payload in (
            ("data.txt", data),
            ("snapshot.json", manifest.to_json_bytes()),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            member.mtime = 0
            package.addfile(member, io.BytesIO(payload))
    asset_payload = archive.read_bytes()
    envelope = PublicationEnvelope(
        artifact_kind=PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
        artifact_id=manifest.snapshot_id,
        scope_id="ml_ai",
        expected_parent_sha=EXPECTED_PARENT,
        source_tree=source,
        manifest_relative_path="snapshot.json",
        assets=(
            ReleaseAssetInput(
                path=archive,
                name="snapshot.tar",
                media_type="application/x-tar",
                size=len(asset_payload),
                sha256=tree_or_blob_digest(asset_payload),
            ),
        ),
        tag="knowledge/S000001",
        committed_at="2026-07-20T15:00:00Z",
        validation_closure_ids=(content_id("fixture", {"review": 1}),),
    )
    return envelope, manifest


def build_expert_envelope(tmp_path):
    scope_contract_id = content_id("expert-scope-contract", {"scope": "ml_ai"})
    repository_map_id = content_id("expert-repository-map", {"scope": "ml_ai"})
    approval_assertion_id = content_id("fixture", {"approval": "expert"})
    release_ids = {
        namespace: content_id(namespace, {"fixture": "expert-publication"})
        for namespace in (
            "expert-candidate",
            "expert-candidate-commit",
            "expert-source-tree",
            "expert-agent-proposal-derivation",
            "expert-candidate-validation-context",
            "expert-candidate-patch",
            "expert-candidate-sanitation",
            "expert-module-contract",
            "expert-validation-attempt",
            "expert-validation-transition",
            "expert-candidate-validation-state",
            "expert-publication-eligibility-stage-result",
            "expert-release-matrix-stage-result",
            "expert-release-matrix-report",
            "expert-release-matrix-promotion-decision",
            "expert-validation-policy",
            "expert-release-evidence-manifest",
            "expert-release-matrix-summary",
        )
    }
    dependencies = tuple(
        sorted(
            {
                scope_contract_id,
                repository_map_id,
                approval_assertion_id,
                *release_ids.values(),
            }
        )
    )
    source_payload = b"expert source"
    evidence_payload = b"expert evidence"
    manifest = ExpertBaseReleaseManifest.mint(
        scope_contract_id=scope_contract_id,
        scope_id="ml_ai",
        parent_release_id=None,
        candidate_id=release_ids["expert-candidate"],
        candidate_commit_record_id=release_ids["expert-candidate-commit"],
        candidate_tree_ref=release_ids["expert-source-tree"],
        candidate_tree_hash=tree_or_blob_digest(source_payload),
        candidate_derivation_ref=release_ids["expert-agent-proposal-derivation"],
        candidate_validation_context_ref=release_ids[
            "expert-candidate-validation-context"
        ],
        candidate_patch_ref=release_ids["expert-candidate-patch"],
        candidate_sanitation_report_id=release_ids["expert-candidate-sanitation"],
        candidate_ancestor_ids=(),
        candidate_source_dependency_ids=(scope_contract_id,),
        repository_map_ref=repository_map_id,
        module_contract_refs=(release_ids["expert-module-contract"],),
        module_versions={"shared.runner": "v1"},
        semantic_book_digest=tree_or_blob_digest(b"semantic book"),
        validation_attempt_id=release_ids["expert-validation-attempt"],
        approval_transition_id=release_ids["expert-validation-transition"],
        approval_state_id=release_ids["expert-candidate-validation-state"],
        publication_eligibility_result_id=release_ids[
            "expert-publication-eligibility-stage-result"
        ],
        release_matrix_stage_result_id=release_ids[
            "expert-release-matrix-stage-result"
        ],
        release_matrix_report_id=release_ids["expert-release-matrix-report"],
        promotion_decision_id=release_ids["expert-release-matrix-promotion-decision"],
        approval_assertion_ids=(approval_assertion_id,),
        validation_policy_id=release_ids["expert-validation-policy"],
        configuration_fingerprint=tree_or_blob_digest(b"expert config"),
        source_archive_ref="expert-source.tar.zst",
        evidence_archive_ref="expert-evidence.tar.zst",
        evidence_manifest_ref=release_ids["expert-release-evidence-manifest"],
        test_matrix_summary_ref=release_ids["expert-release-matrix-summary"],
        evidence_dependency_ids=(release_ids["expert-release-evidence-manifest"],),
        dependency_closure_ids=dependencies,
        checksums={
            "expert-source.tar.zst": tree_or_blob_digest(source_payload),
            "expert-evidence.tar.zst": tree_or_blob_digest(evidence_payload),
        },
    )
    source = tmp_path / "expert-source"
    source.mkdir()
    (source / "expert-release.json").write_bytes(manifest.to_json_bytes())
    asset = tmp_path / "expert-source.tar.zst"
    asset.write_bytes(source_payload)
    return PublicationEnvelope(
        artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
        artifact_id=manifest.release_id,
        scope_id="ml_ai",
        expected_parent_sha=EXPECTED_PARENT,
        source_tree=source,
        manifest_relative_path="expert-release.json",
        assets=(
            ReleaseAssetInput(
                path=asset,
                name=asset.name,
                media_type="application/zstd",
                size=len(source_payload),
                sha256=tree_or_blob_digest(source_payload),
            ),
        ),
        tag="expert/E000000",
        committed_at="2026-07-20T15:00:00Z",
        validation_closure_ids=tuple(
            sorted({manifest.release_id, *manifest.dependency_closure_ids})
        ),
    )


def publication_intent(envelope, source_git_tree_sha, materialized_digest=None):
    source_files = tuple(
        PublicationSourceFile(
            relative_path=path.relative_to(envelope.source_tree).as_posix(),
            sha256=tree_or_blob_digest(path.read_bytes()),
            mode="100755" if path.stat().st_mode & 0o111 else "100644",
            size=path.stat().st_size,
            git_blob_sha=hashlib.sha1(
                f"blob {path.stat().st_size}\0".encode("ascii") + path.read_bytes(),
                usedforsecurity=False,
            ).hexdigest(),
        )
        for path in sorted(envelope.source_tree.rglob("*"))
        if path.is_file()
    )
    source_digest = source_tree_digest(
        {
            source.relative_path: (source.sha256, source.mode, source.size)
            for source in source_files
        }
    )
    return ArtifactPublicationIntent(
        scope_id=envelope.scope_id,
        artifact_kind=envelope.artifact_kind,
        artifact_id=envelope.artifact_id,
        repository_node_id="repository-node",
        repository_full_name=REPOSITORY,
        expected_parent_sha=envelope.expected_parent_sha,
        source_commit_sha=SOURCE_COMMIT,
        source_tree_digest=source_digest,
        source_git_tree_sha=source_git_tree_sha,
        source_files=source_files,
        preserved_current=None,
        materialized_tree_digest=materialized_digest or source_digest,
        manifest_relative_path=envelope.manifest_relative_path,
        manifest_digest=tree_or_blob_digest(
            (envelope.source_tree / envelope.manifest_relative_path).read_bytes()
        ),
        tag=envelope.tag,
        assets=tuple(
            PublicationAssetIntent(
                name=asset.name,
                media_type=asset.media_type,
                size=asset.size,
                sha256=asset.sha256,
            )
            for asset in envelope.assets
        ),
        validation_closure_ids=envelope.validation_closure_ids,
        publisher_identity="leeroo-coder",
        committed_at=envelope.committed_at,
    )


@dataclass
class FakeResolver:
    existing: CurrentArtifactPointer | None = None
    identity: CurrentArtifactPointer | None = None
    intent: ArtifactPublicationIntent | None = None
    release_id: int | None = None
    current_head: str = EXPECTED_PARENT
    artifact_kind: PublicationArtifactKind = PublicationArtifactKind.KNOWLEDGE_SNAPSHOT
    repository: str = REPOSITORY
    repository_node_id: str = "repository-node"
    current_state_observer: Callable[[], None] | None = None

    def __post_init__(self):
        self.verified = []
        self.verified_source_intents = []
        self.policy = RepositoryPolicyReport(
            repository_full_name=self.repository,
            repository_node_id=self.repository_node_id,
            private=True,
            default_branch="main",
            authenticated_actor="leeroo-coder",
            write_access=True,
            immutable_releases=True,
        )

    def diagnose_repository(self, repository_settings, artifact_kind):
        assert repository_settings == "ml_ai"
        assert artifact_kind is self.artifact_kind
        return self.policy

    def repositories_for_scope(self, scope_id):
        assert scope_id == "ml_ai"
        return repositories()

    def read_current_pointer_state(
        self, repository_settings, artifact_kind, allow_missing
    ):
        assert repository_settings == "ml_ai"
        assert allow_missing
        if self.current_state_observer is not None:
            self.current_state_observer()
        return CurrentPointerState(
            pointer=self.existing,
            head_commit_sha=self.current_head,
        )

    def read_artifact_pointer(self, scope_id, artifact_kind, artifact_id):
        assert scope_id == "ml_ai"
        if self.identity is not None:
            assert self.identity.publication_record.artifact_id == artifact_id
        return self.identity

    def read_artifact_intent(self, scope_id, artifact_kind, artifact_id):
        assert scope_id == "ml_ai"
        if self.intent is not None:
            assert self.intent.artifact_id == artifact_id
        return self.intent

    def verify_pointer(
        self, repository_settings, artifact_kind, policy, pointer, intent
    ):
        assert repository_settings == "ml_ai"
        assert intent.binds(pointer)
        self.verified.append(pointer)

    def verify_publication_intent_source(self, repository, intent):
        assert repository == self.repository
        assert intent.repository_full_name == repository
        self.verified_source_intents.append(intent)

    def find_release_id(self, repository, tag):
        assert repository == self.repository
        return self.release_id


class InjectedFailure(RuntimeError):
    pass


class FakePublisherClient:
    def __init__(
        self,
        asset,
        fail_event=None,
        *,
        repository=REPOSITORY,
        repository_node_id="repository-node",
        artifact_kind=PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
        tag="knowledge/S000001",
        head_observer=None,
    ):
        self.asset = asset
        self.fail_event = fail_event
        self.events = []
        self.head = EXPECTED_PARENT
        self.uploaded = False
        self.starter_asset = False
        self.starter_name = asset.name
        self.release_published = False
        self.blob_contents = []
        self.source_tree_contents = {}
        self.source_tree_sha = None
        self.intent_payload = None
        self.identity_payload = None
        self.tag_target = SOURCE_COMMIT
        self.release_author = "leeroo-coder"
        self.repository = repository
        self.repository_node_id = repository_node_id
        self.artifact_kind = artifact_kind
        self.tag = tag
        self.head_observer = head_observer

    def _record(self, event):
        self.events.append(event)
        if self.fail_event == event:
            raise InjectedFailure(event)

    def api_json(self, method, endpoint, body=None):
        if endpoint.endswith("/git/ref/heads/main"):
            return {"object": {"sha": self.head}}
        if endpoint.endswith("/git/blobs"):
            self._record("blob")
            content = base64.b64decode(body["content"])
            self.blob_contents.append(content)
            return {"sha": self._git_sha("blob", content)}
        if endpoint.endswith("/git/trees"):
            self._record("tree")
            if "base_tree" in body:
                return {"sha": "e" * 40}
            if all("content" in entry for entry in body["tree"]):
                self.source_tree_contents = {
                    entry["path"]: entry["content"].encode("utf-8")
                    for entry in body["tree"]
                }
                files = {
                    entry["path"]: (
                        self._git_sha("blob", entry["content"].encode("utf-8")),
                        entry["mode"],
                    )
                    for entry in body["tree"]
                }
                tree_sha = git_tree_shas(files)[""]
                self.source_tree_sha = tree_sha
                return {"sha": tree_sha}
            payload = b"".join(
                entry["mode"].lstrip("0").encode("ascii")
                + b" "
                + entry["path"].encode("utf-8")
                + b"\0"
                + bytes.fromhex(entry["sha"])
                for entry in sorted(body["tree"], key=lambda value: value["path"])
            )
            tree_sha = self._git_sha("tree", payload)
            return {"sha": tree_sha}
        if endpoint.endswith("/git/commits") and method == "POST":
            message = body["message"]
            if message.startswith("Publish "):
                event = "source_commit"
                commit_sha = SOURCE_COMMIT
            elif message.startswith("Claim "):
                event = "intent_commit"
                commit_sha = INTENT_COMMIT
            elif message.startswith("Record "):
                event = "identity_commit"
                commit_sha = IDENTITY_COMMIT
            else:
                event = "pointer_commit"
                commit_sha = POINTER_COMMIT
            self._record(event)
            return {"sha": commit_sha}
        if endpoint.endswith(f"/git/commits/{SOURCE_COMMIT}"):
            return {
                "tree": {"sha": self.source_tree_sha},
                "parents": [{"sha": EXPECTED_PARENT}],
            }
        if "/git/commits/" in endpoint and method == "GET":
            return {
                "tree": {"sha": "0" * 40},
                "parents": [{"sha": "0" * 40}],
            }
        if endpoint.endswith("/releases") and method == "POST":
            self._record("draft")
            return {
                "id": 7,
                "draft": True,
                "tag_name": self.tag,
                "target_commitish": SOURCE_COMMIT,
                "author": {"login": self.release_author},
                "assets": [],
            }
        if endpoint.endswith("/releases/7") and method == "GET":
            return self._release(
                draft=not self.release_published,
                immutable=self.release_published,
            )
        if endpoint.endswith("/releases/7") and method == "PATCH":
            self._record("publish")
            self.release_published = True
            return self._release(draft=False, immutable=True)
        if endpoint.endswith(f"/git/ref/tags/{self.tag}"):
            return {
                "ref": f"refs/tags/{self.tag}",
                "object": {"type": "commit", "sha": SOURCE_COMMIT},
            }
        raise AssertionError((method, endpoint, body))

    def _release(self, draft, immutable):
        assets = []
        if self.starter_asset:
            assets.append(
                {
                    "id": 11,
                    "name": self.starter_name,
                    "content_type": self.asset.media_type,
                    "size": 0,
                    "digest": None,
                    "state": "starter",
                }
            )
        elif self.uploaded:
            assets.append(
                {
                    "id": 11,
                    "name": self.asset.name,
                    "content_type": self.asset.media_type,
                    "size": self.asset.size,
                    "digest": self.asset.sha256,
                    "state": "uploaded",
                }
            )
        return {
            "id": 7,
            "draft": draft,
            "immutable": immutable,
            "tag_name": self.tag,
            "target_commitish": SOURCE_COMMIT,
            "published_at": "2026-07-20T15:00:00Z",
            "author": {"login": self.release_author},
            "assets": assets,
        }

    def upload_release_asset(
        self, repository, release_id, path, asset_name, media_type, asset_size
    ):
        assert repository == self.repository
        assert release_id == 7
        assert path == self.asset.path
        assert asset_name == self.asset.name
        assert media_type == self.asset.media_type
        assert asset_size == self.asset.size
        if self.fail_event == "upload":
            self.starter_asset = True
        self._record("upload")
        self.uploaded = True

    def delete_release_asset(self, repository, asset_id):
        assert repository == self.repository
        assert asset_id == 11
        assert self.starter_asset
        self._record("delete_starter")
        self.starter_asset = False

    def verify_release(self, repository, tag, commit_sha, asset_digests):
        assert repository == self.repository
        assert tag == self.tag
        assert self.release_published
        assert commit_sha == SOURCE_COMMIT
        assert asset_digests == {self.asset.name: self.asset.sha256}
        self._record("attestation")
        return release_attestation(repository, tag, commit_sha, asset_digests)

    def create_ref_if_absent(self, repository, qualified_ref, commit_sha):
        assert repository == self.repository
        if qualified_ref == f"refs/tags/{self.tag}":
            assert commit_sha == SOURCE_COMMIT
            if self.tag_target != commit_sha:
                raise GitHubCompareAndSwapError("tag targets another commit")
            self._record("tag_ref")
        elif qualified_ref.startswith("refs/kapso-publication-intents/"):
            assert commit_sha == INTENT_COMMIT
            self.intent_payload = self.blob_contents[-1]
            self._record("intent_ref")
        else:
            assert qualified_ref.startswith(
                f"refs/kapso-artifacts/{self.artifact_kind.value}/"
            )
            assert commit_sha == IDENTITY_COMMIT
            self.identity_payload = self.blob_contents[-1]
            self._record("identity_ref")
        return {"ref": qualified_ref, "object": {"sha": commit_sha}}

    def update_ref_compare_and_swap(
        self, repository, repository_node_id, branch, expected_sha, commit_sha
    ):
        assert repository == self.repository
        assert repository_node_id == self.repository_node_id
        if self.head != expected_sha:
            raise GitHubCompareAndSwapError("stale")
        event = "source_ref" if commit_sha == SOURCE_COMMIT else "pointer_ref"
        self.head = commit_sha
        if self.head_observer is not None:
            self.head_observer(commit_sha)
        self._record(event)
        return {"object": {"sha": commit_sha}}

    def _git_sha(self, object_kind, payload):
        header = f"{object_kind} {len(payload)}\0".encode("ascii")
        return hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()


def build_publisher(client, resolver, tmp_path, settings=None):
    github = settings or cross_run_settings().github
    materializer = GitHubArtifactMaterializer(client, github, tmp_path / "state")
    return AutonomousGitHubPublisher(client, resolver, materializer, github)


def test_publisher_runs_draft_verify_publish_attest_then_pointer_transaction(tmp_path):
    envelope, _ = build_envelope(tmp_path)
    client = FakePublisherClient(envelope.assets[0])
    resolver = FakeResolver()
    publisher = build_publisher(client, resolver, tmp_path)

    telemetry = publisher.publish(envelope)

    assert telemetry.publication_record.artifact_id == envelope.artifact_id
    assert telemetry.publication_record.repository_full_name == REPOSITORY
    assert telemetry.source_commit_sha == SOURCE_COMMIT
    assert telemetry.pointer_commit_sha == POINTER_COMMIT
    assert not telemetry.idempotent_replay
    assert "blob" not in client.events[: client.events.index("source_commit")]
    assert client.events[: client.events.index("source_commit")] == ["tree"]
    assert client.events.index("source_ref") < client.events.index("draft")
    assert client.events.index("intent_ref") < client.events.index("draft")
    assert client.events.index("upload") < client.events.index("publish")
    assert client.events.index("attestation") < client.events.index("pointer_commit")
    assert client.events[-1] == "pointer_ref"


@pytest.mark.parametrize(
    "failure_event",
    [
        "source_ref",
        "intent_commit",
        "intent_ref",
        "tag_ref",
        "draft",
        "upload",
        "publish",
        "attestation",
        "identity_commit",
        "identity_ref",
        "pointer_commit",
        "pointer_ref",
    ],
)
def test_publication_failure_never_activates_current_early(tmp_path, failure_event):
    envelope, _ = build_envelope(tmp_path)
    client = FakePublisherClient(envelope.assets[0], fail_event=failure_event)
    publisher = build_publisher(client, FakeResolver(), tmp_path)

    with pytest.raises(InjectedFailure):
        publisher.publish(envelope)

    if failure_event != "pointer_ref":
        assert "pointer_ref" not in client.events
    assert client.events.index(failure_event) == len(client.events) - 1


def test_retry_resumes_existing_immutable_release_without_duplicate_upload(tmp_path):
    envelope, _ = build_envelope(tmp_path)
    client = FakePublisherClient(envelope.assets[0], fail_event="attestation")
    with pytest.raises(InjectedFailure):
        build_publisher(client, FakeResolver(), tmp_path).publish(envelope)
    client.fail_event = None
    client.events.clear()
    resolver = FakeResolver(
        intent=publication_intent(envelope, client.source_tree_sha), release_id=7
    )

    telemetry = build_publisher(client, resolver, tmp_path).publish(envelope)

    assert telemetry.pointer_commit_sha == POINTER_COMMIT
    assert "source_ref" not in client.events
    assert "draft" not in client.events
    assert "upload" not in client.events
    assert "publish" not in client.events
    assert resolver.verified_source_intents == [resolver.intent]


def test_retry_resumes_partially_uploaded_draft_without_duplicate_asset(tmp_path):
    envelope, _ = build_envelope(tmp_path)
    client = FakePublisherClient(envelope.assets[0], fail_event="publish")
    with pytest.raises(InjectedFailure):
        build_publisher(client, FakeResolver(), tmp_path).publish(envelope)
    client.fail_event = None
    client.events.clear()
    resolver = FakeResolver(
        intent=publication_intent(envelope, client.source_tree_sha), release_id=7
    )

    telemetry = build_publisher(client, resolver, tmp_path).publish(envelope)

    assert telemetry.pointer_commit_sha == POINTER_COMMIT
    assert "source_ref" not in client.events
    assert "draft" not in client.events
    assert "upload" not in client.events
    assert "publish" in client.events
    assert resolver.verified_source_intents == [resolver.intent]


def test_retry_reclaims_github_failed_upload_starter_asset(tmp_path):
    envelope, _ = build_envelope(tmp_path)
    client = FakePublisherClient(envelope.assets[0], fail_event="upload")
    with pytest.raises(InjectedFailure):
        build_publisher(client, FakeResolver(), tmp_path).publish(envelope)
    assert client.starter_asset
    client.fail_event = None
    client.events.clear()
    resolver = FakeResolver(
        intent=publication_intent(envelope, client.source_tree_sha), release_id=7
    )

    telemetry = build_publisher(client, resolver, tmp_path).publish(envelope)

    assert telemetry.pointer_commit_sha == POINTER_COMMIT
    assert client.events.index("delete_starter") < client.events.index("upload")
    assert client.events.index("upload") < client.events.index("publish")
    assert not client.starter_asset


def test_retry_never_deletes_unexpected_starter_asset(tmp_path):
    envelope, _ = build_envelope(tmp_path)
    client = FakePublisherClient(envelope.assets[0], fail_event="upload")
    with pytest.raises(InjectedFailure):
        build_publisher(client, FakeResolver(), tmp_path).publish(envelope)
    client.fail_event = None
    client.starter_name = "foreign.bin"
    client.events.clear()
    resolver = FakeResolver(
        intent=publication_intent(envelope, client.source_tree_sha), release_id=7
    )

    with pytest.raises(GitHubPublicationError, match="reclaimable"):
        build_publisher(client, resolver, tmp_path).publish(envelope)

    assert "delete_starter" not in client.events
    assert "upload" not in client.events


def test_pre_release_intent_survives_release_crash_and_intervening_head(tmp_path):
    envelope, _ = build_envelope(tmp_path)
    client = FakePublisherClient(envelope.assets[0], fail_event="attestation")

    with pytest.raises(InjectedFailure):
        build_publisher(client, FakeResolver(), tmp_path).publish(envelope)

    assert client.release_published
    assert client.intent_payload is not None
    intent = ArtifactPublicationIntent.from_json_bytes(client.intent_payload)
    assert client.identity_payload is None
    client.fail_event = None
    client.events.clear()
    client.head = "f" * 40

    with pytest.raises(GitHubCompareAndSwapError):
        build_publisher(
            client,
            FakeResolver(intent=intent, release_id=7),
            tmp_path,
        ).publish(envelope)

    assert client.identity_payload is not None
    identity = CurrentArtifactPointer.from_json_bytes(client.identity_payload)
    assert intent.binds(identity)
    assert "identity_ref" in client.events

    conflicting_client = FakePublisherClient(envelope.assets[0])
    with pytest.raises(GitHubPublicationError, match="publication intent"):
        build_publisher(
            conflicting_client,
            FakeResolver(identity=identity, intent=intent),
            tmp_path,
        ).publish(replace(envelope, tag="knowledge/S000001-conflict"))
    assert conflicting_client.events == []

    with pytest.raises(GitHubCompareAndSwapError, match="not the active CURRENT"):
        build_publisher(
            client,
            FakeResolver(identity=identity, intent=intent),
            tmp_path,
        ).publish(envelope)


def test_source_commit_preserves_prior_current_pointer_during_release_failure(
    tmp_path,
):
    envelope, _ = build_envelope(tmp_path)
    seed_client = FakePublisherClient(envelope.assets[0])
    seed = build_publisher(seed_client, FakeResolver(), tmp_path).publish(envelope)
    prior_record = type(seed.publication_record).mint(
        **{
            key: value
            for key, value in seed.publication_record.to_dict().items()
            if key not in {"publication_id", "artifact_id"}
        },
        artifact_id=content_id("fixture", {"prior": 1}),
    )
    prior_pointer = CurrentArtifactPointer(
        scope_id="ml_ai",
        publication_record=prior_record,
        publication_intent_digest=tree_or_blob_digest(b"prior-intent"),
        source_tree_digest=tree_or_blob_digest(b"prior-tree"),
        source_git_tree_sha=seed_client.source_tree_sha,
        materialized_tree_digest=tree_or_blob_digest(b"prior-package"),
        manifest_relative_path="snapshot.json",
        manifest_digest=tree_or_blob_digest(b"prior-manifest"),
        validation_closure_ids=(content_id("fixture", {"prior-review": 1}),),
    )
    client = FakePublisherClient(envelope.assets[0], fail_event="draft")

    with pytest.raises(InjectedFailure):
        build_publisher(
            client,
            FakeResolver(existing=prior_pointer),
            tmp_path,
        ).publish(envelope)

    assert client.source_tree_contents["CURRENT.json"] == prior_pointer.to_json_bytes()
    assert client.head == SOURCE_COMMIT


def test_prior_current_is_in_complete_source_bound_before_remote_write(tmp_path):
    envelope, _ = build_envelope(tmp_path)
    seed_client = FakePublisherClient(envelope.assets[0])
    seed = build_publisher(seed_client, FakeResolver(), tmp_path).publish(envelope)
    prior_record = type(seed.publication_record).mint(
        **{
            key: value
            for key, value in seed.publication_record.to_dict().items()
            if key not in {"publication_id", "artifact_id"}
        },
        artifact_id=content_id("fixture", {"prior-bound": 1}),
    )
    prior_pointer = CurrentArtifactPointer(
        scope_id="ml_ai",
        publication_record=prior_record,
        publication_intent_digest=tree_or_blob_digest(b"prior-intent"),
        source_tree_digest=tree_or_blob_digest(b"prior-tree"),
        source_git_tree_sha=seed_client.source_tree_sha,
        materialized_tree_digest=tree_or_blob_digest(b"prior-package"),
        manifest_relative_path="snapshot.json",
        manifest_digest=tree_or_blob_digest(b"prior-manifest"),
        validation_closure_ids=tuple(
            sorted(
                content_id("fixture", {"review": position}) for position in range(400)
            )
        ),
    )
    candidate_source_bytes = sum(
        path.stat().st_size
        for path in envelope.source_tree.rglob("*")
        if path.is_file()
    )
    source_bound = max(candidate_source_bytes, envelope.assets[0].size)
    assert candidate_source_bytes + len(prior_pointer.to_json_bytes()) > source_bound
    settings = replace(
        cross_run_settings().github,
        source_tree_size_bytes=source_bound,
    )
    client = FakePublisherClient(envelope.assets[0])

    with pytest.raises(GitHubPublicationError, match="complete publication source"):
        build_publisher(
            client,
            FakeResolver(existing=prior_pointer),
            tmp_path,
            settings,
        ).publish(envelope)

    assert client.events == []


def test_identical_replay_is_idempotent_but_conflicting_bytes_fail(tmp_path):
    envelope, _ = build_envelope(tmp_path)
    first_client = FakePublisherClient(envelope.assets[0])
    first = build_publisher(first_client, FakeResolver(), tmp_path).publish(envelope)
    intent = publication_intent(envelope, first_client.source_tree_sha)
    existing = CurrentArtifactPointer(
        scope_id=envelope.scope_id,
        publication_record=first.publication_record,
        publication_intent_digest=intent.digest,
        source_tree_digest=first.source_tree_digest,
        source_git_tree_sha=first_client.source_tree_sha,
        materialized_tree_digest=first.source_tree_digest,
        manifest_relative_path=envelope.manifest_relative_path,
        manifest_digest=tree_or_blob_digest(
            (envelope.source_tree / envelope.manifest_relative_path).read_bytes()
        ),
        validation_closure_ids=envelope.validation_closure_ids,
    )
    replay_resolver = FakeResolver(
        existing=existing,
        identity=existing,
        intent=intent,
        current_head=POINTER_COMMIT,
    )
    replay_client = FakePublisherClient(envelope.assets[0])
    replay_client.head = POINTER_COMMIT
    replay_client.source_tree_sha = first_client.source_tree_sha

    replay = build_publisher(replay_client, replay_resolver, tmp_path).publish(envelope)

    assert replay.idempotent_replay
    assert replay.pointer_commit_sha == POINTER_COMMIT
    assert replay_client.events == []
    assert replay_resolver.verified == [existing]

    conflicting_current = replace(
        existing,
        manifest_digest=tree_or_blob_digest(b"conflicting active pointer"),
    )
    conflict_client = FakePublisherClient(envelope.assets[0])
    with pytest.raises(GitHubPublicationError, match="write-once"):
        build_publisher(
            conflict_client,
            FakeResolver(
                existing=conflicting_current,
                identity=existing,
                intent=intent,
            ),
            tmp_path,
        ).publish(envelope)
    assert conflict_client.events == []

    with pytest.raises(GitHubPublicationError, match="publication intent"):
        build_publisher(
            FakePublisherClient(envelope.assets[0]),
            FakeResolver(identity=existing, intent=intent),
            tmp_path,
        ).publish(replace(envelope, expected_parent_sha="f" * 40))

    with tarfile.open(envelope.assets[0].path, "w") as package:
        for name in ("data.txt", "snapshot.json"):
            payload = (envelope.source_tree / name).read_bytes()
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            member.mtime = 1
            package.addfile(member, io.BytesIO(payload))
    conflicting_payload = envelope.assets[0].path.read_bytes()
    conflicting_asset = replace(
        envelope.assets[0],
        size=len(conflicting_payload),
        sha256=tree_or_blob_digest(conflicting_payload),
    )
    with pytest.raises(GitHubPublicationError):
        build_publisher(
            FakePublisherClient(conflicting_asset),
            FakeResolver(identity=existing, intent=intent),
            tmp_path,
        ).publish(replace(envelope, assets=(conflicting_asset,)))


def test_replay_uses_one_pinned_current_state_and_rejects_superseded_identity(
    tmp_path,
):
    envelope, _ = build_envelope(tmp_path)
    first_client = FakePublisherClient(envelope.assets[0])
    first = build_publisher(first_client, FakeResolver(), tmp_path).publish(envelope)
    intent = publication_intent(envelope, first_client.source_tree_sha)
    identity = CurrentArtifactPointer(
        scope_id=envelope.scope_id,
        publication_record=first.publication_record,
        publication_intent_digest=intent.digest,
        source_tree_digest=first.source_tree_digest,
        source_git_tree_sha=first_client.source_tree_sha,
        materialized_tree_digest=first.source_tree_digest,
        manifest_relative_path=envelope.manifest_relative_path,
        manifest_digest=tree_or_blob_digest(
            (envelope.source_tree / envelope.manifest_relative_path).read_bytes()
        ),
        validation_closure_ids=envelope.validation_closure_ids,
    )
    successor_record = type(first.publication_record).mint(
        **{
            key: value
            for key, value in first.publication_record.to_dict().items()
            if key not in {"publication_id", "artifact_id"}
        },
        artifact_id=content_id("fixture", {"successor": 1}),
    )
    successor = replace(identity, publication_record=successor_record)
    replay_client = FakePublisherClient(envelope.assets[0])
    replay_client.source_tree_sha = first_client.source_tree_sha

    with pytest.raises(GitHubCompareAndSwapError, match="not the active CURRENT"):
        build_publisher(
            replay_client,
            FakeResolver(
                existing=successor,
                identity=identity,
                intent=intent,
                current_head="f" * 40,
            ),
            tmp_path,
        ).publish(envelope)

    assert replay_client.events == []


def test_foreign_draft_is_rejected_before_upload_or_immutable_publication(tmp_path):
    envelope, _ = build_envelope(tmp_path)
    client = FakePublisherClient(envelope.assets[0])
    client.release_author = "foreign-writer"

    with pytest.raises(GitHubPublicationError, match="release author mismatch"):
        build_publisher(client, FakeResolver(), tmp_path).publish(envelope)

    assert not client.release_published
    assert "upload" not in client.events
    assert "publish" not in client.events


def test_stale_parent_and_unsafe_local_tree_fail_before_remote_write(tmp_path):
    envelope, _ = build_envelope(tmp_path)
    stale_client = FakePublisherClient(envelope.assets[0])
    stale_client.head = "f" * 40
    with pytest.raises(GitHubCompareAndSwapError):
        build_publisher(stale_client, FakeResolver(), tmp_path).publish(envelope)

    (envelope.source_tree / ".gitmodules").write_text(
        '[submodule "forbidden"]', encoding="utf-8"
    )
    safe_client = FakePublisherClient(envelope.assets[0])
    with pytest.raises(GitHubPublicationError):
        build_publisher(safe_client, FakeResolver(), tmp_path).publish(envelope)
    assert safe_client.events == []


def test_new_publication_rejects_observed_source_commit_before_branch_write(tmp_path):
    envelope, _ = build_envelope(tmp_path)
    client = FakePublisherClient(envelope.assets[0])
    original_api_json = client.api_json

    def malformed_source_commit(method, endpoint, body=None):
        if method == "GET" and endpoint.endswith(f"/git/commits/{SOURCE_COMMIT}"):
            return {
                "tree": {"sha": "f" * 40},
                "parents": [{"sha": "8" * 40}],
            }
        return original_api_json(method, endpoint, body)

    client.api_json = malformed_source_commit

    with pytest.raises(GitHubPublicationError, match="source commit mismatch"):
        build_publisher(client, FakeResolver(), tmp_path).publish(envelope)

    assert client.head == EXPECTED_PARENT
    assert "source_ref" not in client.events
    assert "draft" not in client.events
    assert not client.release_published


def test_two_publishers_from_one_head_produce_typed_compare_and_swap_conflict(
    tmp_path,
):
    envelope, _ = build_envelope(tmp_path)
    client = FakePublisherClient(envelope.assets[0])
    original_update = client.update_ref_compare_and_swap
    raced = False

    def race_on_source_ref(
        repository, repository_node_id, branch, expected_sha, commit_sha
    ):
        nonlocal raced
        if not raced and commit_sha == SOURCE_COMMIT:
            raced = True
            client.head = "f" * 40
        return original_update(
            repository, repository_node_id, branch, expected_sha, commit_sha
        )

    client.update_ref_compare_and_swap = race_on_source_ref

    with pytest.raises(GitHubCompareAndSwapError):
        build_publisher(client, FakeResolver(), tmp_path).publish(envelope)

    assert "draft" not in client.events
    assert "pointer_ref" not in client.events


def test_publication_rejects_oversize_and_symlink(tmp_path):
    envelope, _ = build_envelope(tmp_path)
    publisher = build_publisher(
        FakePublisherClient(envelope.assets[0]), FakeResolver(), tmp_path
    )

    small_settings = replace(
        cross_run_settings().github,
        release_asset_size_bytes=1,
    )
    with pytest.raises(GitHubPublicationError):
        build_publisher(
            FakePublisherClient(envelope.assets[0]),
            FakeResolver(),
            tmp_path,
            small_settings,
        ).publish(envelope)

    second_asset = replace(envelope.assets[0], name="zz-second-snapshot.tar")
    count_constrained = replace(
        cross_run_settings().github,
        release_asset_count_limit=1,
    )
    with pytest.raises(GitHubPublicationError, match="count"):
        build_publisher(
            FakePublisherClient(envelope.assets[0]),
            FakeResolver(),
            tmp_path,
            count_constrained,
        ).publish(replace(envelope, assets=(*envelope.assets, second_asset)))

    symlink = envelope.source_tree / "linked"
    symlink.symlink_to(envelope.source_tree / "data.txt")
    with pytest.raises(GitHubPublicationError):
        publisher.publish(envelope)


def test_publication_source_entry_bound_fails_before_remote_write(tmp_path):
    envelope, _ = build_envelope(tmp_path)
    settings = replace(cross_run_settings().github, source_entry_limit=1)
    client = FakePublisherClient(envelope.assets[0])

    with pytest.raises(GitHubPublicationError, match="entry limit"):
        build_publisher(client, FakeResolver(), tmp_path, settings).publish(envelope)

    assert client.events == []


def test_git_tree_request_bound_fails_before_remote_write(tmp_path):
    envelope, _ = build_envelope(tmp_path)
    settings = replace(
        cross_run_settings().github,
        git_tree_request_size_bytes=1,
    )
    client = FakePublisherClient(envelope.assets[0])

    with pytest.raises(GitHubPublicationError, match="tree request"):
        build_publisher(client, FakeResolver(), tmp_path, settings).publish(envelope)

    assert client.events == []


def test_package_preflight_rejects_unusable_release_before_remote_write(tmp_path):
    envelope, _ = build_envelope(tmp_path)
    manifest = (envelope.source_tree / envelope.manifest_relative_path).read_bytes()
    unsafe_payload = io.BytesIO()
    with tarfile.open(fileobj=unsafe_payload, mode="w") as package:
        member = tarfile.TarInfo("snapshot.json")
        member.size = len(manifest)
        package.addfile(member, io.BytesIO(manifest))
        member = tarfile.TarInfo("linked")
        member.type = tarfile.SYMTYPE
        member.linkname = "snapshot.json"
        package.addfile(member)
    envelope.assets[0].path.write_bytes(unsafe_payload.getvalue())
    asset = replace(
        envelope.assets[0],
        size=len(unsafe_payload.getvalue()),
        sha256=tree_or_blob_digest(unsafe_payload.getvalue()),
    )
    client = FakePublisherClient(asset)

    with pytest.raises(MaterializationError):
        build_publisher(client, FakeResolver(), tmp_path).publish(
            replace(envelope, assets=(asset,))
        )

    assert client.events == []


def test_package_preflight_rejects_content_outside_declared_closure(tmp_path):
    envelope, _ = build_envelope(tmp_path)
    archive = envelope.assets[0].path
    with tarfile.open(archive, "w") as package:
        for name in ("data.txt", "snapshot.json"):
            payload = (envelope.source_tree / name).read_bytes()
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            package.addfile(member, io.BytesIO(payload))
        payload = b"must not enter the release"
        member = tarfile.TarInfo("credentials.txt")
        member.size = len(payload)
        package.addfile(member, io.BytesIO(payload))
    archive_payload = archive.read_bytes()
    asset = replace(
        envelope.assets[0],
        size=len(archive_payload),
        sha256=tree_or_blob_digest(archive_payload),
    )
    client = FakePublisherClient(asset)

    with pytest.raises(MaterializationError, match="not closed"):
        build_publisher(client, FakeResolver(), tmp_path).publish(
            replace(envelope, assets=(asset,))
        )

    assert client.events == []


def test_package_preflight_rejects_opaque_asset_outside_manifest_closure(tmp_path):
    envelope, _ = build_envelope(tmp_path)
    opaque_path = tmp_path / "credentials.bin"
    opaque_path.write_bytes(b"must not enter an unbound release asset")
    opaque_asset = replace(
        envelope.assets[0],
        path=opaque_path,
        name=opaque_path.name,
        media_type="application/octet-stream",
        size=opaque_path.stat().st_size,
        sha256=tree_or_blob_digest(opaque_path.read_bytes()),
    )
    assets = tuple(
        sorted((*envelope.assets, opaque_asset), key=lambda asset: asset.name)
    )
    client = FakePublisherClient(envelope.assets[0])

    with pytest.raises(MaterializationError, match="outside manifest closure"):
        build_publisher(client, FakeResolver(), tmp_path).publish(
            replace(envelope, assets=assets)
        )

    assert client.events == []


def test_invalid_git_tag_fails_before_remote_write(tmp_path):
    envelope, _ = build_envelope(tmp_path)
    client = FakePublisherClient(envelope.assets[0])

    with pytest.raises(GitHubPublicationError, match="Git ref"):
        build_publisher(client, FakeResolver(), tmp_path).publish(
            replace(envelope, tag="knowledge/invalid..tag")
        )

    assert client.events == []


def test_preexisting_conflicting_tag_fails_before_release_creation(tmp_path):
    envelope, _ = build_envelope(tmp_path)
    client = FakePublisherClient(envelope.assets[0])
    client.tag_target = "f" * 40

    with pytest.raises(GitHubCompareAndSwapError, match="another commit"):
        build_publisher(client, FakeResolver(), tmp_path).publish(envelope)

    assert "intent_ref" in client.events
    assert "draft" not in client.events
    assert "upload" not in client.events
    assert not client.release_published


def test_package_preflight_accepts_manifest_bound_asset_only_content(
    tmp_path,
):
    envelope, _ = build_envelope(tmp_path)
    extra_payload = b"not part of the validated Git source tree"
    manifest = snapshot_manifest(
        (envelope.source_tree / "data.txt").read_bytes(),
        {"extra.txt": tree_or_blob_digest(extra_payload)},
    )
    (envelope.source_tree / "snapshot.json").write_bytes(manifest.to_json_bytes())
    archive = envelope.assets[0].path
    with tarfile.open(archive, "w") as package:
        for name, payload in (
            ("data.txt", (envelope.source_tree / "data.txt").read_bytes()),
            ("extra.txt", extra_payload),
            ("snapshot.json", manifest.to_json_bytes()),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            package.addfile(member, io.BytesIO(payload))
    archive_payload = archive.read_bytes()
    asset = replace(
        envelope.assets[0],
        size=len(archive_payload),
        sha256=tree_or_blob_digest(archive_payload),
    )
    updated = replace(
        envelope,
        artifact_id=manifest.snapshot_id,
        assets=(asset,),
    )
    client = FakePublisherClient(asset)

    telemetry = build_publisher(client, FakeResolver(), tmp_path).publish(updated)

    assert telemetry.publication_record.artifact_id == manifest.snapshot_id
    assert client.events[-1] == "pointer_ref"


def test_expert_publication_requires_sealed_domain_authorization_before_writes(
    tmp_path,
):
    envelope = build_expert_envelope(tmp_path)
    client = FakePublisherClient(
        envelope.assets[0],
        repository=repositories().expert_repository,
        artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
        tag=envelope.tag,
    )
    publisher = AutonomousGitHubPublisher(
        client,
        FakeResolver(
            artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
            repository=repositories().expert_repository,
        ),
        object(),
        cross_run_settings().github,
    )

    with pytest.raises(GitHubPublicationError, match="sealed authorization"):
        publisher.publish(envelope)

    assert client.events == []


@pytest.mark.parametrize("closure_change", ("missing", "extra"))
def test_expert_publication_requires_its_exact_dependency_closure(
    tmp_path,
    closure_change,
):
    envelope = build_expert_envelope(tmp_path)
    if closure_change == "missing":
        validation_closure_ids = envelope.validation_closure_ids[1:]
    else:
        validation_closure_ids = tuple(
            sorted(
                {
                    *envelope.validation_closure_ids,
                    content_id("fixture", {"unexpected": "dependency"}),
                }
            )
        )
    envelope = replace(
        envelope,
        validation_closure_ids=validation_closure_ids,
    )
    client = FakePublisherClient(
        envelope.assets[0],
        repository=repositories().expert_repository,
        artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
        tag=envelope.tag,
    )
    publisher = AutonomousGitHubPublisher(
        client,
        FakeResolver(
            artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
            repository=repositories().expert_repository,
        ),
        object(),
        cross_run_settings().github,
    )

    with pytest.raises(GitHubPublicationError, match="closure is not exact"):
        publisher.publish(envelope)

    assert client.events == []
    assert not tuple(tmp_path.rglob(".validation-*"))


def test_package_preflight_rejects_source_file_omitted_from_manifest_checksums(
    tmp_path,
):
    envelope, _ = build_envelope(tmp_path)
    unchecksummed_payload = b"present in source and package but absent from checksums"
    (envelope.source_tree / "unchecksummed.txt").write_bytes(unchecksummed_payload)
    archive = envelope.assets[0].path
    with tarfile.open(archive, "w") as package:
        for name, payload in (
            ("data.txt", (envelope.source_tree / "data.txt").read_bytes()),
            ("snapshot.json", (envelope.source_tree / "snapshot.json").read_bytes()),
            ("unchecksummed.txt", unchecksummed_payload),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            package.addfile(member, io.BytesIO(payload))
    archive_payload = archive.read_bytes()
    asset = replace(
        envelope.assets[0],
        size=len(archive_payload),
        sha256=tree_or_blob_digest(archive_payload),
    )
    client = FakePublisherClient(asset)

    with pytest.raises(MaterializationError, match="source closure"):
        build_publisher(client, FakeResolver(), tmp_path).publish(
            replace(envelope, assets=(asset,))
        )

    assert client.events == []


def test_oversize_control_manifest_fails_before_remote_write(tmp_path):
    envelope, _ = build_envelope(tmp_path)
    manifest_size = (envelope.source_tree / "snapshot.json").stat().st_size
    constrained = replace(
        cross_run_settings().github,
        control_blob_size_bytes=manifest_size - 1,
    )
    client = FakePublisherClient(envelope.assets[0])

    with pytest.raises(GitHubPublicationError, match="control bound"):
        build_publisher(
            client,
            FakeResolver(),
            tmp_path,
            constrained,
        ).publish(envelope)

    assert client.events == []
