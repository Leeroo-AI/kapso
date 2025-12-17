# File: `libs/langchain_v1/tests/unit_tests/agents/middleware/core/test_wrap_model_call.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 1271 |
| Classes | `TestBasicWrapModelCall`, `TestRetryLogic`, `TestResponseRewriting`, `TestErrorHandling`, `TestShortCircuit`, `TestRequestModification`, `TestStateAndRuntime`, `TestMiddlewareComposition`, `TestWrapModelCallDecorator`, `TestAsyncWrapModelCall`, `TestSyncAsyncInterop`, `TestEdgeCases`, `PassthroughMiddleware`, `LoggingMiddleware`, `CountingMiddleware`, `FailOnceThenSucceed`, `RetryOnceMiddleware`, `AlwaysFailModel`, `MaxRetriesMiddleware`, `FailingModel`, `NoRetryMiddleware`, `AlwaysFailingModel`, `LimitedRetryMiddleware`, `UppercaseMiddleware`, `PrefixMiddleware`, `MultiTransformMiddleware`, `AlwaysFailModel`, `ErrorToSuccessMiddleware`, `SpecificErrorModel`, `SelectiveErrorMiddleware`, `ErrorRecoveryMiddleware`, `AlwaysFailModel`, `CachingMiddleware`, `TrackingModel`, `SystemPromptMiddleware`, `StateAwareMiddleware`, `StateTrackingRetryMiddleware`, `FailOnceThenSucceed`, `OuterMiddleware`, `InnerMiddleware`, `FirstMiddleware`, `SecondMiddleware`, `ThirdMiddleware`, `FailOnceThenSucceed`, `LoggingMiddleware`, `RetryMiddleware`, `PrefixMiddleware`, `SuffixMiddleware`, `FailOnceThenSucceed`, `RetryMiddleware`, `UppercaseMiddleware`, `OuterMiddleware`, `MiddleRetryMiddleware`, `InnerMiddleware`, `TrackingModel`, `FailOnceThenSucceed`, `AlwaysFailModel`, `CustomState`, `ClassMiddleware`, `UnreliableModel`, `LoggingMiddleware`, `AsyncFailOnceThenSucceed`, `RetryMiddleware`, `AsyncOnlyMiddleware`, `MixedMiddleware`, `RequestModifyingMiddleware`, `MultiModelRetryMiddleware`, `FailFirstSucceedSecond` |
| Imports | collections, langchain, langchain_core, pytest, tests |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive testing of wrap_model_call middleware hook in three forms: method, decorator, and async

**Mechanism:** Tests cover the full spectrum of model call wrapping scenarios:

**Basic Functionality:**
- Passthrough (no modification)
- Logging before/after model calls
- Call counting and metrics

**Retry Logic:**
- Simple retry on error
- Maximum retry limits (3, 10 attempts)
- Conditional retry strategies
- Error propagation when retries exhausted

**Response Transformation:**
- Uppercase/lowercase transformations
- Prefix/suffix additions
- Multi-stage transformations (chaining multiple operations)

**Error Handling:**
- Converting errors to success responses
- Selective error handling (specific exception types)
- Error recovery with fallback responses
- Success path vs error path differentiation

**Short-Circuit Patterns:**
- Cache-based short-circuiting
- Returning cached responses without calling model
- Cache hit/miss tracking

**Request Modification:**
- Adding/modifying system prompts
- Modifying messages before model call
- Model settings override
- State-based request modifications

**Middleware Composition:**
- Two middleware (outer-inner execution order)
- Three middleware (first-second-third nesting)
- Retry + logging combinations
- Transform + retry patterns
- Multiple transformations in sequence
- Middle middleware retrying (inner executes multiple times)

**Decorator API:**
- @wrap_model_call basic usage
- Custom naming
- State schema configuration
- Tools parameter
- Parentheses optional syntax
- Mixed with class-based middleware

**Async Support:**
- awrap_model_call for async model execution
- Async retry logic
- Async decorators
- Sync/async interoperability rules
- NotImplementedError for incompatible paths

**Significance:** The most comprehensive test suite for model call wrapping - this is the core middleware capability that enables retry logic, response transformation, caching, and error handling. Tests ensure all common patterns work correctly and compose properly, which is essential since model call middleware is the most frequently used middleware type.
