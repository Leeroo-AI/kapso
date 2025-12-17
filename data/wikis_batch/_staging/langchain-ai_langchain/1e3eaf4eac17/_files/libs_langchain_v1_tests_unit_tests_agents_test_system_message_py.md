# File: `libs/langchain_v1/tests/unit_tests/agents/test_system_message.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 1010 |
| Classes | `TestModelRequestSystemMessage`, `TestCreateAgentSystemMessage`, `TestSystemMessageUpdateViaMiddleware`, `TestMultipleMiddlewareChaining`, `TestCacheControlPreservation`, `TestMetadataMerging`, `TestDynamicSystemPromptMiddleware`, `TestSystemMessageMiddlewareIntegration`, `TestEdgeCasesAndErrorHandling`, `FakeRuntime` |
| Imports | langchain, langchain_core, langgraph, pytest, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive test suite for system message handling in agents, covering ModelRequest system_message field, middleware-based system message updates, cache control preservation, metadata merging, and backward compatibility with string system prompts.

**Mechanism:** Organized into nine test classes covering distinct aspects:
1. `TestModelRequestSystemMessage`: Tests ModelRequest with `system_message` field, override methods, system_prompt property, validation, and content blocks
2. `TestCreateAgentSystemMessage`: Tests create_agent accepts various system_prompt formats (None, string, SystemMessage, with metadata, with complex content)
3. `TestSystemMessageUpdateViaMiddleware`: Tests middleware can set/update system messages using SystemMessage objects
4. `TestMultipleMiddlewareChaining`: Tests sequential middleware modifications with metadata preservation
5. `TestCacheControlPreservation`: Tests cache_control metadata in content blocks is preserved across middleware
6. `TestMetadataMerging`: Tests additional_kwargs and response_metadata merge correctly
7. `TestDynamicSystemPromptMiddleware`: Tests middleware returning SystemMessage with dynamic content from runtime context
8. `TestSystemMessageMiddlewareIntegration`: Tests complete middleware chains with metadata preservation
9. `TestEdgeCasesAndErrorHandling`: Tests edge cases like empty content, multiple blocks, resetting to None

Replicates functionality from langchainjs PR #9459, ensuring Python implementation has feature parity with TypeScript.

**Significance:** Critical for validating system message flexibility and middleware capabilities. Ensures agents can handle complex system prompts with metadata, cache control directives, and dynamic content injection. Tests that middleware can modify system messages while preserving important metadata, enabling advanced features like prompt caching (cache_control) and dynamic prompt generation based on runtime context.
