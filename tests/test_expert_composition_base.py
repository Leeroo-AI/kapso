from __future__ import annotations

from dataclasses import fields, replace
from types import SimpleNamespace

import pytest

from kapso.cross_run.canonical import (
    content_id,
    source_tree_digest,
    tree_or_blob_digest,
)
from kapso.cross_run.contracts import (
    ExpertBaseReleaseManifest,
    PublicationArtifactKind,
    SourceFileDescriptor,
)
from kapso.cross_run.expert.book import (
    EXPERT_BOOK_PATH,
    EXPERT_REPOSITORY_MAP_PATH,
    compile_expert_semantic_book,
    expert_module_contract_path,
    expert_semantic_book_digest,
)
from kapso.cross_run.expert.composition_base import (
    ExpertCompositionBaseClosure,
    ExpertCompositionBaseError,
    build_expert_composition_base_closure,
)
from kapso.cross_run.expert.triggers import ExpertParentTreeReceipt
from kapso.cross_run.github.materializer import (
    SOURCE_ARCHIVE_EXTRACTOR_VERSION,
    CacheVerificationReceipt,
    SourceArchiveExtractionReceipt,
)
from test_expert_triggers import expert_records


def _digest(label: str) -> str:
    return tree_or_blob_digest(label.encode("utf-8"))


def _remint(record, **changes):
    payload = {
        field.name: getattr(record, field.name)
        for field in fields(record)
        if field.name != record.IDENTITY_FIELD
    }
    payload.update(changes)
    return type(record).mint(**payload)


def _source_descriptors(source_contents):
    return tuple(
        SourceFileDescriptor(
            relative_path=path,
            digest=tree_or_blob_digest(payload),
            mode="100644",
            size=len(payload),
        )
        for path, payload in sorted(source_contents.items())
    )


def _source_tree_hash(descriptors):
    return source_tree_digest(
        {
            descriptor.relative_path: (
                descriptor.digest,
                descriptor.mode,
                descriptor.size,
            )
            for descriptor in descriptors
        }
    )


def _parent_receipt(
    release,
    repository_map,
    modules,
    source_contents,
    *,
    cache_label="canonical",
    extractor_version=SOURCE_ARCHIVE_EXTRACTOR_VERSION,
    materializer_version="kapso.expert_materializer.v1",
    descriptor_modes=None,
):
    modes = descriptor_modes or {}
    descriptors = tuple(
        replace(descriptor, mode=modes.get(descriptor.relative_path, descriptor.mode))
        for descriptor in _source_descriptors(source_contents)
    )
    tree_hash = _source_tree_hash(descriptors)
    archive_digest = release.checksums[release.source_archive_ref]
    cache_receipt = CacheVerificationReceipt(
        artifact_kind=PublicationArtifactKind.EXPERT_BASE_RELEASE,
        artifact_id=release.release_id,
        materialized_tree_digest=_digest(f"{cache_label} materialized package"),
        manifest_relative_path="expert-release.json",
        manifest_digest=tree_or_blob_digest(release.to_json_bytes()),
        cache_tree_digest=_digest(f"{cache_label} cache tree"),
        asset_digests={release.source_archive_ref: archive_digest},
    )
    extraction_receipt = SourceArchiveExtractionReceipt.mint(
        artifact_id=release.release_id,
        source_archive_ref=release.source_archive_ref,
        source_archive_digest=archive_digest,
        source_tree_hash=tree_hash,
        source_tree_files=descriptors,
        extractor_version=extractor_version,
    )
    return ExpertParentTreeReceipt.mint(
        release_id=release.release_id,
        cache_verification_receipt=cache_receipt,
        source_extraction_receipt=extraction_receipt,
        parent_tree_hash=tree_hash,
        repository_map_id=repository_map.repository_map_id,
        module_contract_ids=tuple(
            sorted(module.module_contract_id for module in modules)
        ),
        materializer_version=materializer_version,
    )


def _case():
    scope, module, repository_map, original_release = expert_records()
    modules = (module,)
    book = compile_expert_semantic_book(scope, repository_map, modules)
    source_contents = {
        EXPERT_BOOK_PATH: book,
        EXPERT_REPOSITORY_MAP_PATH: repository_map.to_json_bytes(),
        expert_module_contract_path(module.module_contract_id): module.to_json_bytes(),
        "src/reproducible_execution/__init__.py": b"def resume():\n    return True\n",
        "tests/replay_resume.py": b"def replay():\n    return True\n",
        "tests/test_resume.py": b"def test_resume():\n    assert True\n",
    }
    release = _remint(
        original_release,
        semantic_book_digest=expert_semantic_book_digest(book),
        checksums={
            **original_release.checksums,
            original_release.source_archive_ref: _digest("verified source archive"),
        },
    )
    parent_receipt = _parent_receipt(
        release,
        repository_map,
        modules,
        source_contents,
    )
    return SimpleNamespace(
        scope=scope,
        module=module,
        modules=modules,
        repository_map=repository_map,
        release=release,
        parent_receipt=parent_receipt,
        source_contents=source_contents,
    )


def _build(case, **changes):
    values = {
        "scope_contract": case.scope,
        "release_manifest": case.release,
        "parent_tree_receipt": case.parent_receipt,
        "repository_map": case.repository_map,
        "module_contracts": case.modules,
        "source_contents": case.source_contents,
    }
    values.update(changes)
    return build_expert_composition_base_closure(**values)


def _with_source_contents(case, source_contents):
    return SimpleNamespace(
        **{
            **vars(case),
            "source_contents": source_contents,
            "parent_receipt": _parent_receipt(
                case.release,
                case.repository_map,
                case.modules,
                source_contents,
            ),
        }
    )


def test_builds_immutable_process_local_verified_base():
    case = _case()
    mutable_contents = dict(case.source_contents)

    base = _build(case, source_contents=mutable_contents)
    mutable_contents["tests/test_resume.py"] = b"mutated after verification"

    assert base.reference.release_id == case.release.release_id
    assert base.reference.source_tree_hash == case.parent_receipt.parent_tree_hash
    assert base.reference.repository_map_id == case.repository_map.repository_map_id
    assert base.reference.module_contract_ids == (case.module.module_contract_id,)
    assert base.source_contents["tests/test_resume.py"] == (
        case.source_contents["tests/test_resume.py"]
    )
    assert base.source_files == (
        case.parent_receipt.source_extraction_receipt.source_tree_files
    )
    assert base.source_tree == case.parent_receipt.source_extraction_receipt
    assert not hasattr(base, "to_dict")
    with pytest.raises(TypeError):
        base.source_contents["new.py"] = b"forbidden"


def test_reference_excludes_publication_cache_and_extraction_metadata():
    case = _case()
    first = _build(case)
    republished_receipt = _parent_receipt(
        case.release,
        case.repository_map,
        case.modules,
        case.source_contents,
        cache_label="republished",
        extractor_version="kapso.source_archive_extractor.v2",
        materializer_version="kapso.expert_materializer.v2",
    )

    republished = _build(
        case,
        release_manifest=case.release,
        parent_tree_receipt=republished_receipt,
    )

    assert republished.reference == first.reference
    assert republished.parent_tree_receipt != first.parent_tree_receipt
    assert set(first.reference.to_dict()).isdisjoint(
        {
            "publisher_attestation",
            "publication_id",
            "current_pointer_digest",
            "default_branch_head_commit_sha",
            "cache_verification_receipt",
            "parent_tree_receipt_id",
            "source_extraction_receipt",
        }
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        (
            {
                "repository_map": lambda case: _remint(
                    case.repository_map,
                    scope_contract_id=_remint(
                        case.scope,
                        scope_id="another_scope",
                    ).scope_contract_id,
                )
            },
            "release, map, and scope",
        ),
        (
            {
                "parent_tree_receipt": lambda case: _remint(
                    case.parent_receipt,
                    repository_map_id=content_id(
                        "expert-repository-map",
                        {"foreign": True},
                    ),
                )
            },
            "parent receipt",
        ),
    ),
)
def test_rejects_scope_release_map_module_and_receipt_substitution(
    changes,
    message,
):
    case = _case()
    resolved = {name: value(case) for name, value in changes.items()}

    with pytest.raises(ExpertCompositionBaseError, match=message):
        _build(case, **resolved)


@pytest.mark.parametrize("substitution", ("repository_map", "module_version"))
def test_rejects_release_topology_substitution(substitution):
    case = _case()
    if substitution == "repository_map":
        foreign_map_id = content_id("expert-repository-map", {"foreign": True})
        release = _remint(
            case.release,
            repository_map_ref=foreign_map_id,
            consumed_dependency_ids=tuple(
                sorted(
                    {
                        *case.release.consumed_dependency_ids,
                        foreign_map_id,
                    }
                )
            ),
        )
    else:
        release = _remint(
            case.release,
            module_versions={case.module.module_id: "v2"},
        )
    parent_receipt = _parent_receipt(
        release,
        case.repository_map,
        case.modules,
        case.source_contents,
    )

    with pytest.raises(ExpertCompositionBaseError, match="exact topology"):
        _build(
            case,
            release_manifest=release,
            parent_tree_receipt=parent_receipt,
        )


def test_rejects_cache_manifest_and_archive_substitution():
    case = _case()
    cache = case.parent_receipt.cache_verification_receipt
    forged_cache = replace(cache, manifest_digest=_digest("foreign manifest"))
    forged_parent = _remint(
        case.parent_receipt,
        cache_verification_receipt=forged_cache,
    )

    with pytest.raises(ExpertCompositionBaseError, match="cache receipt"):
        _build(case, parent_tree_receipt=forged_parent)

    forged_cache = replace(
        cache,
        asset_digests={case.release.source_archive_ref: _digest("foreign archive")},
    )
    with pytest.raises(ValueError, match="verified release asset"):
        _remint(
            case.parent_receipt,
            cache_verification_receipt=forged_cache,
        )


@pytest.mark.parametrize("mutation", ("missing", "extra", "changed", "non_bytes"))
def test_rejects_inexact_source_byte_closure(mutation):
    case = _case()
    contents = dict(case.source_contents)
    path = "tests/test_resume.py"
    if mutation == "missing":
        del contents[path]
    elif mutation == "extra":
        contents["unrecorded.py"] = b"not in receipt"
    elif mutation == "changed":
        contents[path] = b"changed bytes"
    else:
        contents[path] = bytearray(contents[path])

    with pytest.raises(ExpertCompositionBaseError, match="source"):
        _build(case, source_contents=contents)


@pytest.mark.parametrize(
    ("path", "payload"),
    (
        (EXPERT_BOOK_PATH, b"# Forged book\n"),
        (EXPERT_REPOSITORY_MAP_PATH, b'{"forged":"map"}'),
    ),
)
def test_rejects_generated_book_and_map_bytes_even_with_matching_tree_receipt(
    path,
    payload,
):
    case = _case()
    contents = {**case.source_contents, path: payload}
    forged_case = _with_source_contents(case, contents)

    with pytest.raises(ExpertCompositionBaseError, match="generated controls"):
        _build(forged_case)


def test_rejects_generated_module_bytes_even_with_matching_tree_receipt():
    case = _case()
    module_path = expert_module_contract_path(case.module.module_contract_id)
    forged_case = _with_source_contents(
        case,
        {**case.source_contents, module_path: b'{"forged":"module"}'},
    )

    with pytest.raises(ExpertCompositionBaseError, match="generated controls"):
        _build(forged_case)


def test_rejects_executable_generated_control_mode():
    case = _case()
    parent_receipt = _parent_receipt(
        case.release,
        case.repository_map,
        case.modules,
        case.source_contents,
        descriptor_modes={EXPERT_REPOSITORY_MAP_PATH: "100755"},
    )

    with pytest.raises(ExpertCompositionBaseError, match="generated controls"):
        _build(case, parent_tree_receipt=parent_receipt)


@pytest.mark.parametrize(
    ("path", "message"),
    (
        (".kapso/expert/undeclared.json", "undeclared expert control"),
        (".kapso/task-adapter/injected.py", "external task adapter"),
        ("unowned.py", "exactly one owner"),
    ),
)
def test_rejects_extra_controls_adapter_leakage_and_unowned_source(path, message):
    case = _case()
    forged_case = _with_source_contents(
        case,
        {**case.source_contents, path: b"forged"},
    )

    with pytest.raises(ExpertCompositionBaseError, match=message):
        _build(forged_case)


def test_rejects_topology_module_bijection_mismatch():
    case = _case()
    changed_module = _remint(
        case.module,
        purpose="A substituted module contract with the same semantic identity.",
    )
    changed_modules = (changed_module,)
    contents = {
        path: payload
        for path, payload in case.source_contents.items()
        if path != expert_module_contract_path(case.module.module_contract_id)
    }
    contents[expert_module_contract_path(changed_module.module_contract_id)] = (
        changed_module.to_json_bytes()
    )
    parent_receipt = _parent_receipt(
        case.release,
        case.repository_map,
        changed_modules,
        contents,
    )

    with pytest.raises(ExpertCompositionBaseError, match="bijection"):
        _build(
            case,
            parent_tree_receipt=parent_receipt,
            module_contracts=changed_modules,
            source_contents=contents,
        )


def test_rejects_forged_stable_reference_on_direct_closure_construction():
    case = _case()
    base = _build(case)
    forged_reference = _remint(
        base.reference,
        source_tree_hash=_digest("foreign source tree"),
    )

    with pytest.raises(ExpertCompositionBaseError, match="reference differs"):
        ExpertCompositionBaseClosure(
            reference=forged_reference,
            scope_contract=base.scope_contract,
            release_manifest=base.release_manifest,
            parent_tree_receipt=base.parent_tree_receipt,
            repository_map=base.repository_map,
            module_contracts=base.module_contracts,
            source_contents=base.source_contents,
        )


def test_changed_verified_source_tree_changes_stable_base_identity():
    case = _case()
    changed_case = _with_source_contents(
        case,
        {
            **case.source_contents,
            "tests/test_resume.py": b"def test_resume():\n    assert 2 + 2 == 4\n",
        },
    )

    original = _build(case)
    changed = _build(changed_case)

    assert changed.reference.release_id == original.reference.release_id
    assert changed.reference.source_tree_hash != original.reference.source_tree_hash
    assert changed.reference != original.reference
