# File: `libs/langchain_v1/tests/integration_tests/cache/fake_embeddings.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 91 |
| Classes | `FakeEmbeddings`, `ConsistentFakeEmbeddings`, `AngularTwoDimensionalEmbeddings` |
| Imports | langchain_core, math, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides fake embedding implementations for testing cache functionality without requiring real embedding models.

**Mechanism:** Implements three classes extending langchain_core.embeddings.Embeddings: (1) FakeEmbeddings returns simple index-based vectors, (2) ConsistentFakeEmbeddings maintains state to return consistent vectors for seen texts, (3) AngularTwoDimensionalEmbeddings converts numeric strings to 2D unit vectors using trigonometry.

**Significance:** Essential testing utility that enables deterministic, fast cache integration tests without API calls or model loading. The different implementations test various scenarios: simple embeddings, consistency requirements, and geometric relationships for similarity testing.
