# File: `benchmarks/disagg_benchmarks/rate_limiter.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 45 |
| Classes | `RateLimiter` |
| Imports | asyncio, time |

## Understanding

**Status:** ✅ Explored

**Purpose:** Token bucket rate limiter

**Mechanism:** Async token bucket implementation with configurable requests/second limit. Refills tokens every second, blocks acquisition when depleted. Uses asyncio.Lock for thread-safety. Supports async context manager protocol for convenient usage.

**Significance:** Utility for controlled load generation in disaggregation benchmarks. Enables realistic traffic patterns with configurable QPS limits, essential for measuring system behavior under various load conditions.
