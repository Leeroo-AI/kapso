{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|n8n Python Task Runner|https://docs.n8n.io]]
|-
! Domains
| [[domain::Performance_Optimization]], [[domain::Security]]
|-
! Last Updated
| [[last_updated::2025-12-17 20:00 GMT]]
|}

== Overview ==

Caching results of expensive validation operations to avoid redundant computation for identical inputs.

=== Description ===

Validation caching is an optimization technique that stores the results of security validation analysis indexed by the code content and security configuration. When the same code is submitted for validation multiple times with the same security settings, the cached result is returned immediately without re-parsing or re-analyzing the code.

This principle solves the performance problem of repeated static analysis:
* AST parsing is computationally expensive
* Security validation requires full AST traversal
* Identical code produces identical validation results
* Workflow executions often reuse the same code blocks

The cache uses a Least Recently Used (LRU) eviction policy to maintain a bounded memory footprint while maximizing hit rates for frequently executed workflows.

=== Usage ===

Apply this principle when:
* Performing expensive deterministic computations
* Processing repeated requests with identical inputs
* Optimizing stateless validation or analysis operations
* Balancing memory usage with computational cost

== Theoretical Basis ==

Validation caching implements memoization with LRU eviction:

<pre>
Cache Key Construction:
  key = hash(code_text) + hash(security_config)

Memoization Pattern:
  function validate_with_cache(code, config):
    key = compute_key(code, config)

    if key in cache:
      return cache[key]  # Cache hit

    result = expensive_validation(code, config)
    cache[key] = result

    if cache.size > MAX_SIZE:
      evict_lru_entry()

    return result

LRU Eviction:
  - Track access time for each entry
  - When cache is full, remove least recently accessed
  - Move accessed entries to "most recent" position

Cache Invalidation:
  - Not needed (validation is deterministic)
  - Code change produces different hash
  - Config change produces different hash
</pre>

The correctness of caching relies on the deterministic property: identical inputs always produce identical validation results. The hash function must be collision-resistant to prevent false cache hits.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:n8n-io_n8n_TaskAnalyzer_cache]]
