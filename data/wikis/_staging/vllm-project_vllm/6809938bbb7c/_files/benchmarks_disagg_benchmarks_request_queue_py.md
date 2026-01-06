# File: `benchmarks/disagg_benchmarks/request_queue.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 39 |
| Classes | `RequestQueue` |
| Imports | asyncio, collections |

## Understanding

**Status:** ✅ Explored

**Purpose:** Async request queue with concurrency control for proxy servers.

**Mechanism:** Wraps a deque with asyncio primitives to manage request queuing and concurrent processing. enqueue() adds tasks to the queue (returns False if max_queue_size exceeded). process() continuously dequeues and executes tasks using a semaphore to limit concurrent execution to max_concurrent. Uses asyncio.Lock to prevent race conditions when modifying the queue. Yields control to the event loop with short sleeps to avoid blocking.

**Significance:** Infrastructure component for load-balancing proxies and request management in disagg benchmarks. Prevents overwhelming backend services by limiting both queue depth and concurrent requests. Used by proxy servers to implement backpressure and ensure smooth request flow. Critical for realistic benchmark scenarios that need to model production queue behavior and concurrency limits.
