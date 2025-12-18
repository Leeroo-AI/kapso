# Principle: Validation Caching

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|n8n|https://github.com/n8n-io/n8n]]
|-
! Domains
| [[domain::Security]], [[domain::Performance]], [[domain::Caching]]
|-
! Last Updated
| [[last_updated::2024-12-18 12:00 GMT]]
|}

== Overview ==

Principle for caching security validation results to avoid repeated AST parsing and traversal for identical code with identical security policies.

=== Description ===

Validation Caching optimizes repeated validation of the same code:

1. **Cache Key Generation**: Combines SHA-256 hash of code with sorted allowlists tuple
2. **Cache Lookup**: Checks if this code+policy combination was previously validated
3. **LRU-like Eviction**: Moves accessed entries to end, evicts oldest when full (FIFO)
4. **Result Reuse**: Returns cached violations or success without re-parsing

Benefits:
- **Performance**: Avoids expensive AST parsing for repeated code
- **Policy-Aware**: Different policies create different cache entries
- **Memory Bounded**: Limited to MAX_VALIDATION_CACHE_SIZE entries (500)
- **Hash Collision Resistant**: SHA-256 provides strong uniqueness

=== Usage ===

Apply this principle when:
- Implementing expensive validation that may repeat
- Building systems where identical inputs occur frequently
- Optimizing security checks in hot paths
- Designing stateful validators with bounded memory

== Theoretical Basis ==

Validation caching uses a **Keyed Result Cache** pattern:

<syntaxhighlight lang="python">
# Pseudo-code for validation caching

class TaskAnalyzer:
    _cache: OrderedDict = OrderedDict()  # Class-level cache
    MAX_SIZE = 500

    def _to_cache_key(self, code: str) -> CacheKey:
        # 1. Hash the code
        code_hash = sha256(code.encode()).hexdigest()

        # 2. Include policy in key (sorted for stability)
        allowlists = (
            tuple(sorted(self.stdlib_allow)),
            tuple(sorted(self.external_allow)),
        )

        return (code_hash, allowlists)

    def validate(self, code: str) -> None:
        cache_key = self._to_cache_key(code)

        # 3. Check cache
        if cached := self._cache.get(cache_key):
            self._cache.move_to_end(cache_key)  # LRU touch
            if cached:  # Has violations
                raise SecurityViolationError(cached)
            return  # Clean code, cached success

        # 4. Perform validation
        violations = self._do_validation(code)

        # 5. Store in cache (FIFO eviction)
        if len(self._cache) >= MAX_SIZE:
            self._cache.popitem(last=False)  # Remove oldest
        self._cache[cache_key] = violations.copy()
</syntaxhighlight>

Cache key structure: `(code_sha256_hex, (stdlib_tuple, external_tuple))`

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_TaskAnalyzer_cache]]
