import copy
import hashlib
from dataclasses import replace

import pytest

from kapso.core.config import load_config
from kapso.cross_run.canonical import (
    canonical_json_bytes,
    content_id,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    GitHubPublicationRecord,
    GitHubReleaseAsset,
    PublicationArtifactKind,
    ScopeRepositorySettings,
)
from kapso.cross_run.git_refs import git_object_sha, git_tree_shas
from kapso.cross_run.github.command import BoundedJsonResponse
from kapso.cross_run.github.resolver import (
    ArtifactPublicationIntent,
    CurrentArtifactPointer,
    GitHubArtifactResolver,
    GitHubArtifactActivationWitness,
    GitHubResolutionError,
    PublicationAssetIntent,
    PublicationSourceFile,
    release_attestation_reference,
)
from kapso.cross_run.settings import CrossRunSettings
from kapso.cross_run.settings import CrossRunConfigurationError
from tests.cross_run_github_fixtures import release_attestation

CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"
REPOSITORY = "Leeroo-AI/kapso-knowledge"
COMMIT_SHA = "a" * 40
POINTER_SHA = "b" * 40
ACTIVATION_SHA = "c" * 40
IDENTITY_SHA = "d" * 40
INTENT_SHA = "e" * 40
PARENT_SHA = "9" * 40
SOURCE_PAYLOAD = b"manifest"
SOURCE_BLOB_SHA = hashlib.sha1(
    f"blob {len(SOURCE_PAYLOAD)}\0".encode("ascii") + SOURCE_PAYLOAD,
    usedforsecurity=False,
).hexdigest()
SOURCE_FILES = (
    PublicationSourceFile(
        relative_path="snapshot.json",
        mode="100644",
        size=len(SOURCE_PAYLOAD),
        sha256=tree_or_blob_digest(SOURCE_PAYLOAD),
        git_blob_sha=SOURCE_BLOB_SHA,
    ),
)
TREE_SHA = git_tree_shas(
    {
        source.relative_path: (source.git_blob_sha, source.mode)
        for source in SOURCE_FILES
    }
)[""]


def github_settings():
    return CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    ).github


def scope_registry():
    return CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    ).scopes


def repositories():
    return ScopeRepositorySettings(
        scope_id="ml_ai",
        expert_repository="Leeroo-AI/kapso-expert",
        knowledge_repository=REPOSITORY,
        security_repository="Leeroo-AI/kapso-security",
    )


def text_blob(text, *, truncated=False):
    return {
        "byteSize": len(text.encode("utf-8")),
        "isBinary": False,
        "isTruncated": truncated,
        "text": text,
    }


def publication_fixture():
    asset = GitHubReleaseAsset(
        asset_id="11",
        name="snapshot.tar.zst",
        media_type="application/zstd",
        size=7,
        sha256=tree_or_blob_digest(b"archive"),
    )
    attestation = release_attestation(
        REPOSITORY,
        "knowledge/S000001",
        COMMIT_SHA,
        {asset.name: asset.sha256},
    )
    record = GitHubPublicationRecord.mint(
        artifact_kind=PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
        artifact_id=content_id("fixture", {"snapshot": 1}),
        repository_node_id="repository-node",
        repository_full_name=REPOSITORY,
        commit_sha=COMMIT_SHA,
        immutable_release_id="7",
        tag="knowledge/S000001",
        assets=(asset,),
        release_attestation_ref=release_attestation_reference(attestation),
        published_at="2026-07-20T15:00:00Z",
        publisher_identity="leeroo-coder",
    )
    source_digest = source_tree_digest(
        {
            source.relative_path: (source.sha256, source.mode, source.size)
            for source in SOURCE_FILES
        }
    )
    intent = ArtifactPublicationIntent(
        scope_id="ml_ai",
        artifact_kind=record.artifact_kind,
        artifact_id=record.artifact_id,
        repository_node_id=record.repository_node_id,
        repository_full_name=record.repository_full_name,
        expected_parent_sha=PARENT_SHA,
        source_commit_sha=record.commit_sha,
        source_tree_digest=source_digest,
        source_git_tree_sha=TREE_SHA,
        source_files=SOURCE_FILES,
        preserved_current=None,
        materialized_tree_digest=tree_or_blob_digest(b"package"),
        manifest_relative_path="snapshot.json",
        manifest_digest=tree_or_blob_digest(SOURCE_PAYLOAD),
        tag=record.tag,
        assets=tuple(
            PublicationAssetIntent(
                name=value.name,
                media_type=value.media_type,
                size=value.size,
                sha256=value.sha256,
            )
            for value in record.assets
        ),
        validation_closure_ids=(content_id("fixture", {"review": 1}),),
        publisher_identity=record.publisher_identity,
        committed_at=record.published_at,
    )
    pointer = CurrentArtifactPointer(
        scope_id="ml_ai",
        publication_record=record,
        publication_intent_digest=intent.digest,
        source_tree_digest=source_digest,
        source_git_tree_sha=TREE_SHA,
        materialized_tree_digest=intent.materialized_tree_digest,
        manifest_relative_path=intent.manifest_relative_path,
        manifest_digest=intent.manifest_digest,
        validation_closure_ids=intent.validation_closure_ids,
    )
    release = {
        "id": 7,
        "draft": False,
        "immutable": True,
        "tag_name": record.tag,
        "published_at": record.published_at,
        "author": {"login": record.publisher_identity},
        "assets": [
            {
                "id": 11,
                "name": asset.name,
                "content_type": asset.media_type,
                "size": asset.size,
                "digest": asset.sha256,
                "state": "uploaded",
            }
        ],
    }
    return pointer, release, attestation


def publication_intent(pointer):
    record = pointer.publication_record
    return ArtifactPublicationIntent(
        scope_id=pointer.scope_id,
        artifact_kind=record.artifact_kind,
        artifact_id=record.artifact_id,
        repository_node_id=record.repository_node_id,
        repository_full_name=record.repository_full_name,
        expected_parent_sha=PARENT_SHA,
        source_commit_sha=record.commit_sha,
        source_tree_digest=pointer.source_tree_digest,
        source_git_tree_sha=pointer.source_git_tree_sha,
        source_files=SOURCE_FILES,
        preserved_current=None,
        materialized_tree_digest=pointer.materialized_tree_digest,
        manifest_relative_path=pointer.manifest_relative_path,
        manifest_digest=pointer.manifest_digest,
        tag=record.tag,
        assets=tuple(
            PublicationAssetIntent(
                name=asset.name,
                media_type=asset.media_type,
                size=asset.size,
                sha256=asset.sha256,
            )
            for asset in record.assets
        ),
        validation_closure_ids=pointer.validation_closure_ids,
        publisher_identity=record.publisher_identity,
        committed_at=record.published_at,
    )


class FakeResolverClient:
    def __init__(self, pointer, release, attestation):
        self.pointer = pointer
        self.identity_pointer = pointer
        self.publication_intent = (
            publication_intent(pointer) if pointer is not None else None
        )
        self.release = release
        self.attestation = attestation
        self.noncanonical_control = None
        self.activation_preparation_ref_sha = ACTIVATION_SHA
        self.activation_witness_ref_sha = ACTIVATION_SHA
        self.activation_commit_sha = ACTIVATION_SHA
        self.activation_tree_sha = None
        self.activation_parent_sha = None

    def _control_text(self, control, payload):
        prefix = " " if self.noncanonical_control == control else ""
        return prefix + payload.to_json_bytes().decode("utf-8")

    def api_json(self, method, endpoint, body=None):
        assert body is None
        if endpoint == f"repos/{REPOSITORY}":
            return {
                "full_name": REPOSITORY,
                "node_id": "repository-node",
                "private": True,
                "default_branch": "main",
                "permissions": {"push": True},
            }
        if endpoint == f"repos/{REPOSITORY}/immutable-releases":
            return {"enabled": True, "enforced_by_owner": False}
        if endpoint == "user":
            return {"login": "leeroo-coder"}
        if endpoint == f"repos/{REPOSITORY}/releases/7":
            return self.release
        if endpoint == f"repos/{REPOSITORY}/git/ref/tags/knowledge/S000001":
            return {
                "ref": "refs/tags/knowledge/S000001",
                "object": {"type": "commit", "sha": COMMIT_SHA},
            }
        if endpoint == f"repos/{REPOSITORY}/git/commits/{COMMIT_SHA}":
            return {
                "sha": COMMIT_SHA,
                "tree": {"sha": TREE_SHA},
                "parents": [{"sha": PARENT_SHA}],
            }
        if endpoint == f"repos/{REPOSITORY}/git/commits/{POINTER_SHA}":
            return {
                "sha": POINTER_SHA,
                "tree": {"sha": TREE_SHA},
                "parents": [{"sha": self.first_parent_by_commit[POINTER_SHA]}],
            }
        if endpoint == (
            f"repos/{REPOSITORY}/git/commits/{self.activation_preparation_ref_sha}"
        ):
            pointer = (
                self.pointer if self.pointer is not None else self.identity_pointer
            )
            pointer_blob_sha = git_object_sha("blob", pointer.to_json_bytes())
            activation_tree_sha = git_tree_shas(
                {
                    **{
                        source.relative_path: (
                            source.git_blob_sha,
                            source.mode,
                        )
                        for source in self.publication_intent.source_files
                    },
                    "CURRENT.json": (pointer_blob_sha, "100644"),
                }
            )[""]
            return {
                "sha": self.activation_commit_sha,
                "tree": {
                    "sha": (
                        activation_tree_sha
                        if self.activation_tree_sha is None
                        else self.activation_tree_sha
                    )
                },
                "parents": [
                    {
                        "sha": (
                            self.publication_intent.source_commit_sha
                            if self.activation_parent_sha is None
                            else self.activation_parent_sha
                        )
                    }
                ],
            }
        if endpoint == f"repos/{REPOSITORY}/git/trees/{TREE_SHA}":
            return {
                "sha": TREE_SHA,
                "truncated": False,
                "tree": [
                    {
                        "path": "snapshot.json",
                        "mode": "100644",
                        "type": "blob",
                        "sha": SOURCE_BLOB_SHA,
                        "size": len(SOURCE_PAYLOAD),
                    }
                ],
            }
        raise AssertionError((method, endpoint))

    def read_ref_commit(self, repository, qualified_ref, *, allow_missing):
        if "kapso-activation-preparations" in qualified_ref:
            value = self.activation_preparation_ref_sha
        elif "kapso-activations" in qualified_ref:
            value = self.activation_witness_ref_sha
        elif "publication-intents" in qualified_ref:
            value = None if self.publication_intent is None else INTENT_SHA
        else:
            value = None if self.identity_pointer is None else IDENTITY_SHA
        return value

    def graphql(self, query, variables):
        if "defaultBranchRef" in query:
            return {
                "data": {
                    "repository": {
                        "defaultBranchRef": {
                            "name": "main",
                            "target": {"oid": POINTER_SHA},
                        }
                    }
                }
            }
        expression = variables["expression"]
        if expression == f"{COMMIT_SHA}:snapshot.json":
            return {"data": {"repository": {"object": text_blob("manifest")}}}
        if expression == f"{IDENTITY_SHA}:PUBLICATION.json":
            return {
                "data": {
                    "repository": {
                        "object": text_blob(
                            self._control_text("identity", self.identity_pointer)
                        )
                    }
                }
            }
        if expression == f"{INTENT_SHA}:PUBLICATION_INTENT.json":
            return {
                "data": {
                    "repository": {
                        "object": text_blob(
                            self._control_text("intent", self.publication_intent)
                        )
                    }
                }
            }
        assert expression == f"{POINTER_SHA}:CURRENT.json"
        blob = (
            None
            if self.pointer is None
            else text_blob(self._control_text("current", self.pointer))
        )
        return {
            "data": {
                "repository": {
                    "object": blob,
                }
            }
        }

    def api_json_bounded(self, method, endpoint, maximum_bytes):
        response = self.api_json(method, endpoint)
        payload_size = len(canonical_json_bytes(response))
        assert payload_size <= maximum_bytes
        return BoundedJsonResponse(response, payload_size)

    def verify_release(self, repository, tag, commit_sha, asset_digests):
        assert repository == REPOSITORY
        assert tag == "knowledge/S000001"
        assert commit_sha == COMMIT_SHA
        assert asset_digests == {
            asset["name"]: asset["digest"] for asset in self.release["assets"]
        }
        return self.attestation

    def read_git_blob(self, repository, blob_sha, maximum_bytes):
        assert repository == REPOSITORY
        assert blob_sha == SOURCE_BLOB_SHA
        assert len(SOURCE_PAYLOAD) <= maximum_bytes
        return SOURCE_PAYLOAD


def test_resolver_pins_and_verifies_complete_immutable_release():
    pointer, release, attestation = publication_fixture()
    client = FakeResolverClient(pointer, release, attestation)
    resolver = GitHubArtifactResolver(client, github_settings(), scope_registry())

    resolved = resolver.resolve_current(
        "ml_ai", PublicationArtifactKind.KNOWLEDGE_SNAPSHOT
    )

    assert resolved.pointer == pointer
    assert resolved.pointer_commit_sha == POINTER_SHA
    assert resolved.policy.repository_node_id == "repository-node"
    assert resolved.policy.authenticated_actor == "leeroo-coder"
    assert resolved.policy.immutable_releases


@pytest.mark.parametrize("control", ("current", "identity", "intent"))
def test_resolver_rejects_noncanonical_control_objects(control):
    pointer, release, attestation = publication_fixture()
    client = FakeResolverClient(pointer, release, attestation)
    client.noncanonical_control = control
    resolver = GitHubArtifactResolver(client, github_settings(), scope_registry())

    with pytest.raises(GitHubResolutionError, match="not canonical"):
        if control == "current":
            resolver.resolve_current(
                "ml_ai",
                PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
            )
        elif control == "identity":
            resolver.read_artifact_pointer(
                "ml_ai",
                PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
                pointer.publication_record.artifact_id,
            )
        else:
            resolver.read_artifact_intent(
                "ml_ai",
                PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
                pointer.publication_record.artifact_id,
            )


def test_attestation_reference_ignores_version_dependent_verifier_metadata():
    _, _, attestation = publication_fixture()
    changed_wrapper = copy.deepcopy(attestation)
    changed_wrapper["verificationResult"]["policy"] = {"cli_version": "future-version"}

    assert release_attestation_reference(changed_wrapper) == (
        release_attestation_reference(attestation)
    )


def test_resolver_reads_global_artifact_identity_after_current_is_superseded():
    pointer, release, attestation = publication_fixture()
    client = FakeResolverClient(None, release, attestation)
    client.identity_pointer = pointer
    resolver = GitHubArtifactResolver(client, github_settings(), scope_registry())

    resolved = resolver.read_artifact_pointer(
        "ml_ai",
        PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
        pointer.publication_record.artifact_id,
    )

    assert resolved == pointer


def test_resolver_verifies_immutable_identity_without_current_activation():
    pointer, release, attestation = publication_fixture()
    client = FakeResolverClient(None, release, attestation)
    client.identity_pointer = pointer
    client.publication_intent = publication_intent(pointer)
    resolver = GitHubArtifactResolver(client, github_settings(), scope_registry())

    resolved = resolver.resolve_artifact(
        "ml_ai",
        PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
        pointer.publication_record.artifact_id,
    )

    assert resolved.pointer == pointer
    assert resolved.pointer_commit_sha == IDENTITY_SHA


def test_resolver_requires_exact_write_once_intent_and_identity():
    pointer, release, attestation = publication_fixture()
    intent = publication_intent(pointer)
    client = FakeResolverClient(pointer, release, attestation)
    resolver = GitHubArtifactResolver(client, github_settings(), scope_registry())

    resolver.require_artifact_intent(
        "ml_ai",
        PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
        pointer.publication_record.artifact_id,
        intent,
    )
    resolver.require_artifact_pointer(
        "ml_ai",
        PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
        pointer.publication_record.artifact_id,
        pointer,
    )

    client.publication_intent = None
    with pytest.raises(GitHubResolutionError, match="intent ref differs"):
        resolver.require_artifact_intent(
            "ml_ai",
            PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
            pointer.publication_record.artifact_id,
            intent,
        )
    client.publication_intent = intent
    client.identity_pointer = None
    with pytest.raises(GitHubResolutionError, match="identity ref differs"):
        resolver.require_artifact_pointer(
            "ml_ai",
            PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
            pointer.publication_record.artifact_id,
            pointer,
        )


def test_resolver_verifies_exact_prepared_activation_commit():
    pointer, release, attestation = publication_fixture()
    client = FakeResolverClient(pointer, release, attestation)
    resolver = GitHubArtifactResolver(client, github_settings(), scope_registry())

    activation_commit_sha = resolver.resolve_artifact_activation_preparation(
        "ml_ai",
        PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
        pointer.publication_record.artifact_id,
        client.publication_intent,
        pointer,
    )

    assert activation_commit_sha == ACTIVATION_SHA


def test_resolver_rejects_missing_prepared_activation_ref():
    pointer, release, attestation = publication_fixture()
    client = FakeResolverClient(pointer, release, attestation)
    client.activation_preparation_ref_sha = None
    resolver = GitHubArtifactResolver(client, github_settings(), scope_registry())

    with pytest.raises(GitHubResolutionError, match="preparation is missing"):
        resolver.resolve_artifact_activation_preparation(
            "ml_ai",
            PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
            pointer.publication_record.artifact_id,
            client.publication_intent,
            pointer,
        )
    assert (
        resolver.resolve_artifact_activation_preparation(
            "ml_ai",
            PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
            pointer.publication_record.artifact_id,
            client.publication_intent,
            pointer,
            allow_missing=True,
        )
        is None
    )


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("ref", "activation commit mismatch"),
        ("tree", "activation commit mismatch"),
        ("parent", "activation parent mismatch"),
    ),
)
def test_resolver_rejects_mismatched_prepared_activation(
    corruption,
    message,
):
    pointer, release, attestation = publication_fixture()
    client = FakeResolverClient(pointer, release, attestation)
    if corruption == "ref":
        client.activation_commit_sha = "f" * 40
    elif corruption == "tree":
        client.activation_tree_sha = "f" * 40
    else:
        client.activation_parent_sha = "f" * 40
    resolver = GitHubArtifactResolver(client, github_settings(), scope_registry())

    with pytest.raises(GitHubResolutionError, match=message):
        resolver.resolve_artifact_activation_preparation(
            "ml_ai",
            PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
            pointer.publication_record.artifact_id,
            client.publication_intent,
            pointer,
        )


def test_resolver_authenticates_post_cas_activation_witness():
    pointer, release, attestation = publication_fixture()
    client = FakeResolverClient(pointer, release, attestation)
    resolver = GitHubArtifactResolver(client, github_settings(), scope_registry())

    witness = resolver.resolve_artifact_activation_witness(
        "ml_ai",
        PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
        pointer.publication_record.artifact_id,
        client.publication_intent,
        pointer,
    )

    assert witness == GitHubArtifactActivationWitness.mint(
        scope_id="ml_ai",
        scope_repository_binding_hash=repositories().binding_fingerprint,
        artifact_kind=PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
        artifact_id=pointer.publication_record.artifact_id,
        repository_full_name=REPOSITORY,
        activation_commit_sha=ACTIVATION_SHA,
        publication_intent_digest=client.publication_intent.digest,
        current_pointer_digest=tree_or_blob_digest(pointer.to_json_bytes()),
    )


def test_resolver_distinguishes_missing_and_conflicting_activation_witness():
    pointer, release, attestation = publication_fixture()
    client = FakeResolverClient(pointer, release, attestation)
    resolver = GitHubArtifactResolver(client, github_settings(), scope_registry())
    client.activation_witness_ref_sha = None

    assert (
        resolver.resolve_artifact_activation_witness(
            "ml_ai",
            PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
            pointer.publication_record.artifact_id,
            client.publication_intent,
            pointer,
            allow_missing=True,
        )
        is None
    )
    with pytest.raises(GitHubResolutionError, match="witness is missing"):
        resolver.resolve_artifact_activation_witness(
            "ml_ai",
            PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
            pointer.publication_record.artifact_id,
            client.publication_intent,
            pointer,
        )

    client.activation_witness_ref_sha = "f" * 40
    with pytest.raises(GitHubResolutionError, match="differs from its preparation"):
        resolver.resolve_artifact_activation_witness(
            "ml_ai",
            PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
            pointer.publication_record.artifact_id,
            client.publication_intent,
            pointer,
        )


@pytest.mark.parametrize("identity_state", ["missing", "mismatch"])
def test_resolver_rejects_current_without_matching_write_once_identity(
    identity_state,
):
    pointer, release, attestation = publication_fixture()
    client = FakeResolverClient(pointer, release, attestation)
    client.identity_pointer = (
        None
        if identity_state == "missing"
        else CurrentArtifactPointer(
            **{
                **pointer.to_dict(),
                "manifest_digest": tree_or_blob_digest(b"different manifest"),
            }
        )
    )

    with pytest.raises(GitHubResolutionError, match="write-once"):
        GitHubArtifactResolver(
            client, github_settings(), scope_registry()
        ).resolve_current("ml_ai", PublicationArtifactKind.KNOWLEDGE_SNAPSHOT)


def test_resolver_rejects_current_without_matching_pre_release_intent():
    pointer, release, attestation = publication_fixture()
    missing = FakeResolverClient(pointer, release, attestation)
    missing.publication_intent = None
    with pytest.raises(GitHubResolutionError, match="publication intent"):
        GitHubArtifactResolver(
            missing, github_settings(), scope_registry()
        ).resolve_current("ml_ai", PublicationArtifactKind.KNOWLEDGE_SNAPSHOT)

    conflicting = FakeResolverClient(pointer, release, attestation)
    conflicting.publication_intent = replace(
        conflicting.publication_intent,
        materialized_tree_digest=tree_or_blob_digest(b"another package"),
    )
    with pytest.raises(GitHubResolutionError, match="publication intent"):
        GitHubArtifactResolver(
            conflicting, github_settings(), scope_registry()
        ).resolve_current("ml_ai", PublicationArtifactKind.KNOWLEDGE_SNAPSHOT)


def test_resolver_rejects_self_consistent_false_source_descriptor():
    pointer, release, attestation = publication_fixture()
    client = FakeResolverClient(pointer, release, attestation)
    forged_source = replace(
        SOURCE_FILES[0],
        sha256=tree_or_blob_digest(b"forged source bytes"),
    )
    forged_digest = source_tree_digest(
        {
            forged_source.relative_path: (
                forged_source.sha256,
                forged_source.mode,
                forged_source.size,
            )
        }
    )
    forged_intent = replace(
        client.publication_intent,
        source_files=(forged_source,),
        source_tree_digest=forged_digest,
    )
    forged_pointer = replace(
        pointer,
        publication_intent_digest=forged_intent.digest,
        source_tree_digest=forged_digest,
    )
    client.pointer = forged_pointer
    client.identity_pointer = forged_pointer
    client.publication_intent = forged_intent

    with pytest.raises(GitHubResolutionError, match="blob digest"):
        GitHubArtifactResolver(
            client, github_settings(), scope_registry()
        ).resolve_current("ml_ai", PublicationArtifactKind.KNOWLEDGE_SNAPSHOT)


def test_resolver_enforces_the_dedicated_source_tree_size_bound():
    pointer, release, attestation = publication_fixture()
    constrained = replace(
        github_settings(),
        source_tree_size_bytes=len(SOURCE_PAYLOAD) - 1,
    )

    with pytest.raises(GitHubResolutionError, match="source commit exceeds"):
        GitHubArtifactResolver(
            FakeResolverClient(pointer, release, attestation),
            constrained,
            scope_registry(),
        ).resolve_current("ml_ai", PublicationArtifactKind.KNOWLEDGE_SNAPSHOT)


@pytest.mark.parametrize("forgery", ["blob", "tree"])
def test_resolver_recomputes_remote_git_object_identities(forgery):
    pointer, release, attestation = publication_fixture()
    client = FakeResolverClient(pointer, release, attestation)
    forged_blob_sha = "7" * 40 if forgery == "blob" else SOURCE_BLOB_SHA
    forged_tree_sha = "8" * 40
    forged_source = replace(SOURCE_FILES[0], git_blob_sha=forged_blob_sha)
    forged_intent = replace(
        client.publication_intent,
        source_files=(forged_source,),
        source_git_tree_sha=forged_tree_sha,
    )
    forged_pointer = replace(
        pointer,
        publication_intent_digest=forged_intent.digest,
        source_git_tree_sha=forged_tree_sha,
    )
    client.pointer = forged_pointer
    client.identity_pointer = forged_pointer
    client.publication_intent = forged_intent
    original_api_json = client.api_json

    def forged_git_objects(method, endpoint, body=None):
        if endpoint == f"repos/{REPOSITORY}/git/commits/{COMMIT_SHA}":
            return {
                "tree": {"sha": forged_tree_sha},
                "parents": [{"sha": PARENT_SHA}],
            }
        if endpoint == f"repos/{REPOSITORY}/git/trees/{forged_tree_sha}":
            return {
                "sha": forged_tree_sha,
                "truncated": False,
                "tree": [
                    {
                        "path": "snapshot.json",
                        "mode": "100644",
                        "type": "blob",
                        "sha": forged_blob_sha,
                        "size": len(SOURCE_PAYLOAD),
                    }
                ],
            }
        return original_api_json(method, endpoint, body)

    client.api_json = forged_git_objects
    client.read_git_blob = lambda repository, blob_sha, maximum_bytes: SOURCE_PAYLOAD

    with pytest.raises(GitHubResolutionError, match="Git (blob|tree) identity"):
        GitHubArtifactResolver(
            client, github_settings(), scope_registry()
        ).resolve_current("ml_ai", PublicationArtifactKind.KNOWLEDGE_SNAPSHOT)


def test_resolver_rejects_non_utf8_remote_source_blob():
    pointer, release, attestation = publication_fixture()
    client = FakeResolverClient(pointer, release, attestation)
    invalid_payload = b"\xff"
    invalid_blob_sha = git_object_sha("blob", invalid_payload)
    invalid_source = PublicationSourceFile(
        relative_path="snapshot.json",
        mode="100644",
        size=len(invalid_payload),
        sha256=tree_or_blob_digest(invalid_payload),
        git_blob_sha=invalid_blob_sha,
    )
    invalid_tree_sha = git_tree_shas(
        {invalid_source.relative_path: (invalid_blob_sha, invalid_source.mode)}
    )[""]
    invalid_intent = replace(
        client.publication_intent,
        source_tree_digest=source_tree_digest(
            {
                invalid_source.relative_path: (
                    invalid_source.sha256,
                    invalid_source.mode,
                    invalid_source.size,
                )
            }
        ),
        source_git_tree_sha=invalid_tree_sha,
        source_files=(invalid_source,),
    )
    original_api_json = client.api_json

    def invalid_source_objects(method, endpoint, body=None):
        if endpoint == f"repos/{REPOSITORY}/git/commits/{COMMIT_SHA}":
            return {
                "tree": {"sha": invalid_tree_sha},
                "parents": [{"sha": PARENT_SHA}],
            }
        if endpoint == f"repos/{REPOSITORY}/git/trees/{invalid_tree_sha}":
            return {
                "sha": invalid_tree_sha,
                "truncated": False,
                "tree": [
                    {
                        "path": invalid_source.relative_path,
                        "mode": invalid_source.mode,
                        "type": "blob",
                        "sha": invalid_blob_sha,
                        "size": invalid_source.size,
                    }
                ],
            }
        return original_api_json(method, endpoint, body)

    client.api_json = invalid_source_objects
    client.read_git_blob = lambda repository, blob_sha, maximum_bytes: invalid_payload

    with pytest.raises(UnicodeDecodeError):
        GitHubArtifactResolver(
            client, github_settings(), scope_registry()
        ).verify_publication_intent_source(REPOSITORY, invalid_intent)


@pytest.mark.parametrize(
    ("source_entry_limit", "message"),
    [(1, "entry limit"), (2, "directory closure")],
)
def test_resolver_bounds_and_closes_remote_directory_entries(
    source_entry_limit,
    message,
):
    pointer, release, attestation = publication_fixture()
    client = FakeResolverClient(pointer, release, attestation)
    original_api_json = client.api_json

    def extra_empty_tree(method, endpoint, body=None):
        if endpoint == f"repos/{REPOSITORY}/git/trees/{'f' * 40}":
            return {
                "sha": "f" * 40,
                "truncated": False,
                "tree": [],
            }
        response = original_api_json(method, endpoint, body)
        if endpoint == f"repos/{REPOSITORY}/git/trees/{TREE_SHA}":
            return {
                **response,
                "tree": [
                    {
                        "path": "unused",
                        "mode": "040000",
                        "type": "tree",
                        "sha": "f" * 40,
                    },
                    *response["tree"],
                ],
            }
        return response

    client.api_json = extra_empty_tree
    constrained = replace(github_settings(), source_entry_limit=source_entry_limit)

    with pytest.raises(GitHubResolutionError, match=message):
        GitHubArtifactResolver(client, constrained, scope_registry()).resolve_current(
            "ml_ai", PublicationArtifactKind.KNOWLEDGE_SNAPSHOT
        )


def test_source_tree_walk_is_non_recursive_and_globally_metadata_bounded():
    root_sha = "1" * 40
    child_sha = "2" * 40
    blob_sha = "3" * 40
    responses = {
        f"repos/{REPOSITORY}/git/trees/{root_sha}": {
            "sha": root_sha,
            "truncated": False,
            "tree": [
                {
                    "path": "nested",
                    "mode": "040000",
                    "type": "tree",
                    "sha": child_sha,
                }
            ],
        },
        f"repos/{REPOSITORY}/git/trees/{child_sha}": {
            "sha": child_sha,
            "truncated": False,
            "tree": [
                {
                    "path": "payload.bin",
                    "mode": "100644",
                    "type": "blob",
                    "sha": blob_sha,
                    "size": 7,
                }
            ],
        },
    }

    class NonRecursiveTreeClient:
        def __init__(self):
            self.calls = []

        def api_json_bounded(self, method, endpoint, maximum_bytes):
            assert method == "GET"
            assert "recursive" not in endpoint
            self.calls.append(endpoint)
            response = responses[endpoint]
            size = len(canonical_json_bytes(response))
            assert size <= maximum_bytes
            return BoundedJsonResponse(response, size)

    client = NonRecursiveTreeClient()
    resolver = GitHubArtifactResolver(client, github_settings(), scope_registry())

    blobs, directories = resolver._read_source_tree(REPOSITORY, root_sha)

    assert set(blobs) == {"nested/payload.bin"}
    assert set(directories) == {"nested"}
    assert client.calls == list(responses)

    root_size = len(canonical_json_bytes(responses[next(iter(responses))]))
    constrained = replace(github_settings(), git_tree_metadata_size_bytes=root_size)
    bounded_client = NonRecursiveTreeClient()
    with pytest.raises(GitHubResolutionError, match="metadata"):
        GitHubArtifactResolver(
            bounded_client,
            constrained,
            scope_registry(),
        )._read_source_tree(REPOSITORY, root_sha)
    assert bounded_client.calls == [f"repos/{REPOSITORY}/git/trees/{root_sha}"]

    responses[f"repos/{REPOSITORY}/git/trees/{root_sha}"]["truncated"] = True
    with pytest.raises(GitHubResolutionError, match="incomplete"):
        GitHubArtifactResolver(
            NonRecursiveTreeClient(),
            github_settings(),
            scope_registry(),
        )._read_source_tree(REPOSITORY, root_sha)


def test_resolver_rejects_annotated_tag_instead_of_exact_commit_ref():
    pointer, release, attestation = publication_fixture()
    client = FakeResolverClient(pointer, release, attestation)
    original_api_json = client.api_json

    def annotated_tag(method, endpoint, body=None):
        if endpoint == f"repos/{REPOSITORY}/git/ref/tags/knowledge/S000001":
            return {
                "ref": "refs/tags/knowledge/S000001",
                "object": {"type": "tag", "sha": COMMIT_SHA},
            }
        return original_api_json(method, endpoint, body)

    client.api_json = annotated_tag

    with pytest.raises(GitHubResolutionError, match="directly target"):
        GitHubArtifactResolver(
            client, github_settings(), scope_registry()
        ).resolve_current("ml_ai", PublicationArtifactKind.KNOWLEDGE_SNAPSHOT)


def test_resolver_rejects_forged_intent_parent_even_when_pointer_binds_it():
    pointer, release, attestation = publication_fixture()
    client = FakeResolverClient(pointer, release, attestation)
    forged_intent = replace(client.publication_intent, expected_parent_sha="8" * 40)
    forged_pointer = replace(
        pointer,
        publication_intent_digest=forged_intent.digest,
    )
    client.pointer = forged_pointer
    client.identity_pointer = forged_pointer
    client.publication_intent = forged_intent

    with pytest.raises(GitHubResolutionError, match="parent mismatch"):
        GitHubArtifactResolver(
            client, github_settings(), scope_registry()
        ).resolve_current("ml_ai", PublicationArtifactKind.KNOWLEDGE_SNAPSHOT)


@pytest.mark.parametrize("blob_kind", ["current", "identity"])
def test_resolver_rejects_truncated_pointer_blobs(blob_kind):
    pointer, release, attestation = publication_fixture()
    client = FakeResolverClient(pointer, release, attestation)
    original_graphql = client.graphql

    def truncated_pointer(query, variables):
        response = original_graphql(query, variables)
        expression = variables.get("expression")
        wanted = (
            f"{POINTER_SHA}:CURRENT.json"
            if blob_kind == "current"
            else f"{IDENTITY_SHA}:PUBLICATION.json"
        )
        if expression == wanted:
            response["data"]["repository"]["object"]["isTruncated"] = True
        return response

    client.graphql = truncated_pointer

    with pytest.raises(GitHubResolutionError, match="truncated"):
        GitHubArtifactResolver(
            client, github_settings(), scope_registry()
        ).resolve_current("ml_ai", PublicationArtifactKind.KNOWLEDGE_SNAPSHOT)


@pytest.mark.parametrize(
    "corruption",
    [
        "draft",
        "mutable",
        "asset_digest",
        "tag_commit",
        "source_tree",
        "source_manifest",
        "truncated_manifest",
        "attestation",
    ],
)
def test_resolver_rejects_incomplete_or_replaced_release(corruption):
    pointer, release, attestation = publication_fixture()
    release = copy.deepcopy(release)
    if corruption == "draft":
        release["draft"] = True
    elif corruption == "mutable":
        release["immutable"] = False
    elif corruption == "asset_digest":
        release["assets"][0]["digest"] = tree_or_blob_digest(b"different")
    client = FakeResolverClient(pointer, release, attestation)
    if corruption == "tag_commit":
        original = client.api_json

        def wrong_tag(method, endpoint, body=None):
            if endpoint == f"repos/{REPOSITORY}/git/ref/tags/knowledge/S000001":
                return {
                    "ref": "refs/tags/knowledge/S000001",
                    "object": {"type": "commit", "sha": "c" * 40},
                }
            return original(method, endpoint, body)

        client.api_json = wrong_tag
    if corruption == "source_tree":
        original = client.api_json

        def wrong_tree(method, endpoint, body=None):
            if endpoint == f"repos/{REPOSITORY}/git/commits/{COMMIT_SHA}":
                return {"tree": {"sha": "d" * 40}}
            return original(method, endpoint, body)

        client.api_json = wrong_tree
    if corruption == "source_manifest":
        original_graphql = client.graphql

        def wrong_manifest(query, variables):
            if variables.get("expression") == f"{COMMIT_SHA}:snapshot.json":
                return {"data": {"repository": {"object": text_blob("different")}}}
            return original_graphql(query, variables)

        client.graphql = wrong_manifest
    if corruption == "truncated_manifest":
        original_graphql = client.graphql

        def truncated_manifest(query, variables):
            if variables.get("expression") == f"{COMMIT_SHA}:snapshot.json":
                return {
                    "data": {
                        "repository": {"object": text_blob("manifest", truncated=True)}
                    }
                }
            return original_graphql(query, variables)

        client.graphql = truncated_manifest
    if corruption == "attestation":
        client.attestation = {}

    with pytest.raises(GitHubResolutionError):
        GitHubArtifactResolver(
            client, github_settings(), scope_registry()
        ).resolve_current("ml_ai", PublicationArtifactKind.KNOWLEDGE_SNAPSHOT)


def test_resolver_rejects_wrong_scope_and_missing_current_pointer():
    pointer, release, attestation = publication_fixture()
    wrong_scope = CurrentArtifactPointer(
        **{**pointer.to_dict(), "scope_id": "other_scope"}
    )
    with pytest.raises(GitHubResolutionError):
        GitHubArtifactResolver(
            FakeResolverClient(wrong_scope, release, attestation),
            github_settings(),
            scope_registry(),
        ).resolve_current("ml_ai", PublicationArtifactKind.KNOWLEDGE_SNAPSHOT)

    client = FakeResolverClient(None, release, attestation)
    resolver = GitHubArtifactResolver(client, github_settings(), scope_registry())
    assert (
        resolver.read_current_pointer(
            "ml_ai",
            PublicationArtifactKind.KNOWLEDGE_SNAPSHOT,
            allow_missing=True,
        )
        is None
    )
    with pytest.raises(GitHubResolutionError):
        resolver.resolve_current("ml_ai", PublicationArtifactKind.KNOWLEDGE_SNAPSHOT)


def test_repository_policy_rejects_swapped_or_public_repository_identity():
    pointer, release, attestation = publication_fixture()
    client = FakeResolverClient(pointer, release, attestation)
    original = client.api_json

    def public_repository(method, endpoint, body=None):
        response = original(method, endpoint, body)
        if endpoint == f"repos/{REPOSITORY}":
            return {**response, "private": False}
        return response

    client.api_json = public_repository
    with pytest.raises(GitHubResolutionError):
        GitHubArtifactResolver(
            client, github_settings(), scope_registry()
        ).diagnose_repository("ml_ai", PublicationArtifactKind.KNOWLEDGE_SNAPSHOT)

    swapped_client = FakeResolverClient(pointer, release, attestation)
    swapped_original = swapped_client.api_json

    def swapped_repository(method, endpoint, body=None):
        if endpoint == "repos/Leeroo-AI/kapso-expert":
            return {
                "full_name": "Leeroo-AI/kapso-expert",
                "node_id": "expert-repository-node",
                "private": True,
                "default_branch": "main",
                "permissions": {"push": True},
            }
        if endpoint == "repos/Leeroo-AI/kapso-expert/immutable-releases":
            return {"enabled": True, "enforced_by_owner": False}
        return swapped_original(method, endpoint, body)

    swapped_client.api_json = swapped_repository
    with pytest.raises(GitHubResolutionError):
        GitHubArtifactResolver(
            swapped_client, github_settings(), scope_registry()
        ).resolve_current("ml_ai", PublicationArtifactKind.EXPERT_BASE_RELEASE)


def test_scope_registry_is_the_only_repository_routing_authority():
    pointer, release, attestation = publication_fixture()
    resolver = GitHubArtifactResolver(
        FakeResolverClient(pointer, release, attestation),
        github_settings(),
        scope_registry(),
    )

    with pytest.raises(CrossRunConfigurationError):
        resolver.resolve_current(
            "unregistered_scope", PublicationArtifactKind.KNOWLEDGE_SNAPSHOT
        )


def test_graphql_partial_data_with_errors_is_rejected():
    pointer, release, attestation = publication_fixture()
    client = FakeResolverClient(pointer, release, attestation)
    original = client.graphql

    def partial_failure(query, variables):
        response = original(query, variables)
        if variables.get("expression") == f"{POINTER_SHA}:CURRENT.json":
            return {**response, "errors": [{"message": "authorization changed"}]}
        return response

    client.graphql = partial_failure

    with pytest.raises(GitHubResolutionError):
        GitHubArtifactResolver(
            client, github_settings(), scope_registry()
        ).resolve_current("ml_ai", PublicationArtifactKind.KNOWLEDGE_SNAPSHOT)
