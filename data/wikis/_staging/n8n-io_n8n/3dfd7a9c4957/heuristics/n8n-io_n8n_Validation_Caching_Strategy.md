# Heuristic: Validation Caching Strategy

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n-io/n8n|https://github.com/n8n-io/n8n]]
* [[source::Code|task_analyzer.py|packages/@n8n/task-runner-python/src/task_analyzer.py]]
|-
! Domains
| [[domain::Performance]], [[domain::Security]], [[domain::Caching]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==
LRU cache with SHA256 hash keys and allowlist-aware cache invalidation to avoid redundant AST security validation.

=== Description ===
The Python Task Runner performs AST-based security validation on every piece of user code before execution. Since the same code snippets may be submitted multiple times (e.g., in workflows with loops or retries), caching validation results significantly improves performance. The cache key includes both the code hash AND the current allowlist configuration, ensuring that cache entries are invalidated when security policies change.

=== Usage ===
This heuristic is automatically applied by the `TaskAnalyzer` class. Use this pattern when:
* You have an expensive validation/analysis operation that may be called repeatedly with the same input
* The validation result depends on both the input AND configuration state
* You need bounded memory usage (LRU eviction)

== The Insight (Rule of Thumb) ==

* **Action:** Use `OrderedDict` with SHA256 hash keys for LRU caching of validation results
* **Value:** `MAX_VALIDATION_CACHE_SIZE = 500` entries (configurable in constants.py)
* **Trade-off:** Memory usage (~500 cache entries) vs. CPU time for repeated AST parsing
* **Key Design:** Include allowlist configuration in cache key to prevent stale results when security policies change

== Reasoning ==

AST parsing and traversal for security validation involves:
1. `ast.parse()` - Parsing Python source to AST (CPU-bound)
2. `SecurityValidator.visit()` - Walking the entire AST tree (CPU-bound)
3. Multiple import/attribute checks per node

For repeated code submissions (common in workflow loops), this overhead is wasteful. By caching results keyed on `(code_hash, allowlists_tuple)`, subsequent validations of the same code with the same security configuration return instantly.

The cache key structure ensures correctness:
```python
CacheKey = tuple[str, tuple]  # (code_hash, allowlists_tuple)
```

If the allowlist changes, all cached entries effectively become invalid (different key), preventing security bypasses.

== Code Evidence ==

From `task_analyzer.py:159-212`:

<syntaxhighlight lang="python">
class TaskAnalyzer:
    _cache: ValidationCache = OrderedDict()

    def __init__(self, security_config: SecurityConfig):
        self._security_config = security_config
        self._allowlists = (
            tuple(sorted(security_config.stdlib_allow)),
            tuple(sorted(security_config.external_allow)),
        )
        self._allow_all = (
            "*" in security_config.stdlib_allow
            and "*" in security_config.external_allow
        )

    def validate(self, code: str) -> None:
        if self._allow_all:
            return  # Skip validation entirely if all modules allowed

        cache_key = self._to_cache_key(code)
        cached_violations = self._cache.get(cache_key)
        cache_hit = cached_violations is not None

        if cache_hit:
            self._cache.move_to_end(cache_key)  # LRU: move to end on access

            if len(cached_violations) == 0:
                return  # Valid code, cached

            self._raise_security_error(cached_violations)

        # Cache miss: perform full validation
        tree = ast.parse(code)
        security_validator = SecurityValidator(self._security_config)
        security_validator.visit(tree)

        self._set_in_cache(cache_key, security_validator.violations)

        if security_validator.violations:
            self._raise_security_error(security_validator.violations)

    def _to_cache_key(self, code: str) -> CacheKey:
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        return (code_hash, self._allowlists)

    def _set_in_cache(self, cache_key: CacheKey, violations: CachedViolations) -> None:
        if len(self._cache) >= MAX_VALIDATION_CACHE_SIZE:
            self._cache.popitem(last=False)  # FIFO eviction of oldest

        self._cache[cache_key] = violations.copy()
        self._cache.move_to_end(cache_key)  # Mark as recently used
</syntaxhighlight>

== Related Pages ==

* [[uses_heuristic::Implementation:n8n-io_n8n_TaskAnalyzer_cache]]
* [[uses_heuristic::Implementation:n8n-io_n8n_TaskAnalyzer_validate]]
* [[uses_heuristic::Workflow:n8n-io_n8n_Security_Validation_Pipeline]]
