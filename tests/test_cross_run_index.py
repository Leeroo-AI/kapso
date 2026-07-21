from dataclasses import replace

import pytest

from kapso.core.embeddings import EmbeddingSpaceId
from kapso.cross_run.canonical import (
    CANONICALIZER_VERSION,
    canonical_json_bytes,
    tree_or_blob_digest,
)
from kapso.cross_run.knowledge.index import (
    EmbeddingSpace,
    EmbeddingVector,
    EmbeddingVectorSet,
    SnapshotIndexError,
    SnapshotSearchIndex,
)
from kapso.cross_run.knowledge.package import KnowledgeSnapshotPackageBuilder
from test_knowledge_snapshot_package import digest, finalize, populated_generation


def vector_set(prepared):
    space = EmbeddingSpace.mint(
        provider="openai",
        model="text-embedding-3-small",
        dimensions=2,
        canonicalizer_version=CANONICALIZER_VERSION,
    )
    vectors = tuple(
        EmbeddingVector(
            record_id=record_id,
            input_digest=tree_or_blob_digest(
                canonical_json_bytes(prepared.record_by_id(record_id))
            ),
            values=(1.0, 0.0),
        )
        for record_id in prepared.retrieval_root_ids
    )
    return EmbeddingVectorSet(space=space, vectors=vectors)


def prepared_snapshot():
    scope, idea, _, generation, objects = populated_generation()
    prepared = KnowledgeSnapshotPackageBuilder.prepare(
        scope,
        generation,
        objects.__getitem__,
    )
    return prepared, idea


def test_index_exposes_metadata_lexical_and_exact_cosine_primitives():
    prepared, idea = prepared_snapshot()
    vectors = vector_set(prepared)
    search_index = SnapshotSearchIndex.build(prepared, (vectors,))
    package = finalize(
        prepared,
        search_files=search_index.files,
        embedding_sidecars=search_index.embedding_sidecars,
    )

    search_index.verify(package.manifest)
    assert search_index.record_closure_digest == prepared.record_closure_digest
    assert search_index.metadata_by_id[idea.prior_idea_id]["outcome"] == "frontier"
    assert search_index.metadata_by_id[idea.prior_idea_id]["task_family_id"] == (
        "language_model_post_training"
    )
    assert tuple(search_index.lexical_scores("hypothesis")) == (idea.prior_idea_id,)
    assert (
        search_index.semantic_scores(
            (1.0, 0.0),
            vectors.space.embedding_space_id,
        )[idea.prior_idea_id]
        == 1.0
    )


def test_index_is_byte_deterministic_and_embedding_spaces_do_not_mix():
    prepared, _ = prepared_snapshot()
    vectors = vector_set(prepared)
    first = SnapshotSearchIndex.build(prepared, (vectors,))
    second = SnapshotSearchIndex.build(prepared, (vectors,))

    assert first.files == second.files
    assert first.manifest.index_manifest_id == second.manifest.index_manifest_id
    other_space = EmbeddingSpace.mint(
        provider="openai",
        model="text-embedding-3-large",
        dimensions=2,
        canonicalizer_version=CANONICALIZER_VERSION,
    )
    with pytest.raises(SnapshotIndexError, match="absent"):
        first.semantic_scores((1.0, 0.0), other_space.embedding_space_id)


def test_index_embedding_space_identity_matches_shared_provider_boundary():
    vectors = vector_set(prepared_snapshot()[0])

    assert (
        vectors.space.embedding_space_id
        == EmbeddingSpaceId(
            provider=vectors.space.provider,
            model=vectors.space.model,
            dimensions=vectors.space.dimensions,
            canonicalizer_version=vectors.space.canonicalizer_version,
        ).value
    )


def test_index_rejects_stale_inputs_and_corrupt_float_data():
    prepared, _ = prepared_snapshot()
    vectors = vector_set(prepared)
    stale_vector = replace(vectors.vectors[0], input_digest=digest("stale-input"))
    stale_set = EmbeddingVectorSet(space=vectors.space, vectors=(stale_vector,))

    with pytest.raises(SnapshotIndexError, match="input digest"):
        SnapshotSearchIndex.build(prepared, (stale_set,))

    valid = SnapshotSearchIndex.build(prepared, (vectors,))
    data_ref = valid.manifest.vector_sidecars[0].data_ref
    corrupt_files = dict(valid.files)
    corrupt_files[data_ref] = b"broken"
    with pytest.raises(SnapshotIndexError, match="checksum mismatch"):
        SnapshotSearchIndex.open(prepared, corrupt_files)


def test_metadata_and_lexical_indexes_work_without_embeddings():
    prepared, idea = prepared_snapshot()
    search_index = SnapshotSearchIndex.build(prepared)
    package = finalize(prepared, search_files=search_index.files)

    search_index.verify(package.manifest)
    assert search_index.embedding_sidecars == ()
    assert tuple(search_index.lexical_scores(idea.prior_idea_id)) == (
        idea.prior_idea_id,
    )
