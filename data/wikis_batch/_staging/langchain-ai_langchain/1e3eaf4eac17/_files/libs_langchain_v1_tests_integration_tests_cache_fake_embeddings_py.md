# File: `libs/langchain_v1/tests/integration_tests/cache/fake_embeddings.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 91 |
| Classes | `FakeEmbeddings`, `ConsistentFakeEmbeddings`, `AngularTwoDimensionalEmbeddings` |
| Imports | langchain_core, math, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides fake embedding implementations for testing cache and vector store functionality without external API calls.

**Mechanism:** Three implementations: FakeEmbeddings (simple sequential vectors), ConsistentFakeEmbeddings (remembers texts for consistency), and AngularTwoDimensionalEmbeddings (uses angles in pi units for geometric testing). Each generates deterministic vectors for predictable test outcomes.

**Significance:** Essential testing utility that eliminates external dependencies and costs when testing embedding-based features. Allows predictable distance calculations and cache behavior validation without real API calls.
