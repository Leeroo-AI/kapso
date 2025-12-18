# File: `libs/langchain_v1/tests/unit_tests/agents/test_system_message.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 1010 |
| Classes | `TestModelRequestSystemMessage`, `TestCreateAgentSystemMessage`, `TestSystemMessageUpdateViaMiddleware`, `TestMultipleMiddlewareChaining`, `TestCacheControlPreservation`, `TestMetadataMerging`, `TestDynamicSystemPromptMiddleware`, `TestSystemMessageMiddlewareIntegration`, `TestEdgeCasesAndErrorHandling`, `FakeRuntime` |
| Imports | langchain, langchain_core, langgraph, pytest, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive unit tests for system message handling in agents. Validates ModelRequest system_message field, system message updates via middleware, cache control preservation, metadata merging, and dynamic prompt capabilities.

**Mechanism:** Organized into 9 test classes covering different aspects: ModelRequest creation/override, create_agent variations, middleware chains, cache control, metadata merging, and edge cases. Tests use GenericFakeChatModel for deterministic responses and validate system message content, metadata (additional_kwargs, response_metadata), and complex content blocks with cache control.

**Significance:** Replicates langchainjs PR #9459 functionality in Python. Ensures robust system message handling across the agent lifecycle, including backward compatibility with string prompts while supporting rich SystemMessage objects with metadata and cache control. Critical for prompt engineering and agent customization.
