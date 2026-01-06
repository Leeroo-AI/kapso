# File: `benchmarks/disagg_benchmarks/rate_limiter.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 45 |
| Classes | `RateLimiter` |
| Imports | asyncio, time |

## Understanding

**Status:** ✅ Explored

**Purpose:** Token bucket rate limiter for controlling request throughput.

**Mechanism:** Implements async context manager with token bucket algorithm. Maintains num_available_tokens (initially equals rate_limit) that refills to rate_limit every second. acquire() method blocks until a token is available, consuming one token per call. Uses asyncio.Lock for thread-safe token management and time.monotonic() for precise timing. Calculates wait_time when tokens are exhausted and sleeps until next refill period.

**Significance:** Infrastructure component for disagg benchmarks that need controlled request rates. Token bucket allows bursts up to rate_limit while maintaining average throughput. Used by proxy servers to prevent overwhelming backend services during benchmarking. Ensures fair comparison between different serving configurations by controlling the offered load precisely.
