# File: `libs/langchain_v1/tests/unit_tests/agents/middleware/core/test_wrap_model_call.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 1271 |
| Classes | `TestBasicWrapModelCall`, `TestRetryLogic`, `TestResponseRewriting`, `TestErrorHandling`, `TestShortCircuit`, `TestRequestModification`, `TestStateAndRuntime`, `TestMiddlewareComposition`, `TestWrapModelCallDecorator`, `TestAsyncWrapModelCall`, `TestSyncAsyncInterop`, `TestEdgeCases`, `PassthroughMiddleware`, `LoggingMiddleware`, `CountingMiddleware`, `FailOnceThenSucceed`, `RetryOnceMiddleware`, `AlwaysFailModel`, `MaxRetriesMiddleware`, `FailingModel`, `NoRetryMiddleware`, `AlwaysFailingModel`, `LimitedRetryMiddleware`, `UppercaseMiddleware`, `PrefixMiddleware`, `MultiTransformMiddleware`, `AlwaysFailModel`, `ErrorToSuccessMiddleware`, `SpecificErrorModel`, `SelectiveErrorMiddleware`, `ErrorRecoveryMiddleware`, `AlwaysFailModel`, `CachingMiddleware`, `TrackingModel`, `SystemPromptMiddleware`, `StateAwareMiddleware`, `StateTrackingRetryMiddleware`, `FailOnceThenSucceed`, `OuterMiddleware`, `InnerMiddleware`, `FirstMiddleware`, `SecondMiddleware`, `ThirdMiddleware`, `FailOnceThenSucceed`, `LoggingMiddleware`, `RetryMiddleware`, `PrefixMiddleware`, `SuffixMiddleware`, `FailOnceThenSucceed`, `RetryMiddleware`, `UppercaseMiddleware`, `OuterMiddleware`, `MiddleRetryMiddleware`, `InnerMiddleware`, `TrackingModel`, `FailOnceThenSucceed`, `AlwaysFailModel`, `CustomState`, `ClassMiddleware`, `UnreliableModel`, `LoggingMiddleware`, `AsyncFailOnceThenSucceed`, `RetryMiddleware`, `AsyncOnlyMiddleware`, `MixedMiddleware`, `RequestModifyingMiddleware`, `MultiModelRetryMiddleware`, `FailFirstSucceedSecond` |
| Imports | collections, langchain, langchain_core, pytest, tests |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive tests for wrap_model_call hook functionality

**Mechanism:** 12 test classes cover passthrough, logging, retry logic (with max attempts), response rewriting (uppercase, prefix, multi-stage transforms), error handling (conversion to success, selective catching), short-circuit patterns (caching), request modification (system prompts, messages), state/runtime access, middleware composition (2-3 layers including nested retries), decorator usage (with/without parameters), and async execution paths, validating both class-based and decorator-created middleware.

**Significance:** Most comprehensive test suite for the wrap_model_call hook which is the primary mechanism for intercepting and modifying model interactions, ensuring all common middleware patterns (retry, caching, transformations, error recovery) work correctly in isolation and composition.
