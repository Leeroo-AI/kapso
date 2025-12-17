# Heuristic: Validation Cache Size

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n Task Runner Python|https://github.com/n8n-io/n8n/tree/master/packages/@n8n/task-runner-python]]
|-
! Domains
| [[domain::Optimization]], [[domain::Caching]], [[domain::Security]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

LRU cache limited to 500 entries for security validation results, balancing memory usage against cache hit rate for repeated code submissions.

=== Description ===

The TaskAnalyzer uses an LRU (Least Recently Used) cache to store the results of AST-based security validation. Each cache entry maps a hash of the code and allowlists configuration to the list of security violations (or an empty list for valid code). The cache avoids redundant AST parsing and validation for frequently executed code snippets.

=== Usage ===

This heuristic automatically applies when using the **TaskAnalyzer.validate()** method. The cache is shared across all TaskAnalyzer instances (class-level cache) and requires no configuration. Apply this understanding when:

- Debugging validation performance issues
- Tuning memory usage for high-concurrency runners
- Understanding why repeated submissions skip validation

== The Insight (Rule of Thumb) ==

* **Action:** Cache security validation results keyed by (code_hash, allowlists_tuple)
* **Value:** `MAX_VALIDATION_CACHE_SIZE = 500` entries
* **Trade-off:** 500 entries provides good hit rate for typical workloads while limiting memory to ~1-5 MB depending on code size
* **Eviction Policy:** LRU with FIFO fallback when cache is full
* **Cache Key:** SHA-256 hash of code + tuple of sorted allowlists

== Reasoning ==

AST parsing and security validation is CPU-bound work that scales with code complexity. By caching validation results:

1. **Repeated Submissions:** Same code in retry scenarios skips validation entirely
2. **Workflow Reruns:** Users testing workflows benefit from cached results
3. **Bounded Memory:** Fixed 500-entry limit prevents unbounded growth

The SHA-256 hash ensures collision resistance while the allowlists tuple ensures cache invalidation when security settings change.

== Code Evidence ==

Cache size constant from `constants.py:37`:
<syntaxhighlight lang="python">
MAX_VALIDATION_CACHE_SIZE = 500  # cached validation results
</syntaxhighlight>

Cache key generation from `task_analyzer.py:203-205`:
<syntaxhighlight lang="python">
def _to_cache_key(self, code: str) -> CacheKey:
    code_hash = hashlib.sha256(code.encode()).hexdigest()
    return (code_hash, self._allowlists)
</syntaxhighlight>

LRU eviction from `task_analyzer.py:207-212`:
<syntaxhighlight lang="python">
def _set_in_cache(self, cache_key: CacheKey, violations: CachedViolations) -> None:
    if len(self._cache) >= MAX_VALIDATION_CACHE_SIZE:
        self._cache.popitem(last=False)  # FIFO

    self._cache[cache_key] = violations.copy()
    self._cache.move_to_end(cache_key)
</syntaxhighlight>

Cache hit logic from `task_analyzer.py:176-186`:
<syntaxhighlight lang="python">
cache_key = self._to_cache_key(code)
cached_violations = self._cache.get(cache_key)
cache_hit = cached_violations is not None

if cache_hit:
    self._cache.move_to_end(cache_key)

    if len(cached_violations) == 0:
        return  # Valid code, skip re-validation

    self._raise_security_error(cached_violations)
</syntaxhighlight>

== Related Pages ==

* [[uses_heuristic::Implementation:n8n-io_n8n_TaskAnalyzer_validate]]
* [[uses_heuristic::Implementation:n8n-io_n8n_TaskAnalyzer_cache]]
* [[uses_heuristic::Principle:n8n-io_n8n_Validation_Caching]]
