# Phase 3: Enrichment Report

## Summary

- **Environment pages created:** 2
- **Heuristic pages created:** 4
- **Indexes updated:** 2 (_EnvironmentIndex.md, _HeuristicIndex.md)

---

## Environments Created

| Environment | Required By | Key Requirements |
|-------------|-------------|------------------|
| n8n-io_n8n_Python_Task_Runner | 16 implementations (Task Runner + Security) | Python 3.13+, Unix-only, websockets |
| n8n-io_n8n_Python_Workflow_Comparison | 8 implementations (Workflow Comparison) | Python 3.11+, NetworkX, NumPy, SciPy |

### Environment Details

#### Python Task Runner Environment
- **Python Version:** >= 3.13
- **OS Constraint:** Unix-like only (Linux, macOS) - Windows NOT supported
- **Core Dependencies:** websockets >= 15.0.1
- **Optional:** sentry-sdk >= 2.35.2
- **Key Environment Variables:**
  - `N8N_RUNNERS_GRANT_TOKEN` (required)
  - `N8N_RUNNERS_TASK_BROKER_URI`
  - `N8N_RUNNERS_MAX_CONCURRENCY`
  - `N8N_RUNNERS_STDLIB_ALLOW` / `N8N_RUNNERS_EXTERNAL_ALLOW`
- **Code Evidence:** Platform check in `main.py:67-70` explicitly rejects Windows

#### Python Workflow Comparison Environment
- **Python Version:** >= 3.11
- **OS Constraint:** Cross-platform (any)
- **Core Dependencies:**
  - networkx >= 3.2
  - numpy >= 2.3.4
  - pyyaml >= 6.0
  - scipy >= 1.16.3
- **No credentials required** (CLI tool for local files)

---

## Heuristics Created

| Heuristic | Applies To | Summary |
|-----------|------------|---------|
| n8n-io_n8n_Validation_Cache_Size | TaskAnalyzer_validate, TaskAnalyzer_cache | LRU cache limited to 500 entries |
| n8n-io_n8n_Offer_Validity_Window | TaskRunner_send_offers, Task_Acceptance | 5000ms + 500ms jitter offer expiry |
| n8n-io_n8n_Pipe_Reader_Timeout | TaskExecutor_execute_process, Result_Collection | Dynamic timeout: (payload * 0.1) / 100MB/s + 2s |
| n8n-io_n8n_Trigger_Priority_Multiplier | calculate_graph_edit_distance, GED_Calculation | 50x cost for trigger node mismatches |

### Heuristic Details

#### Validation Cache Size (500 entries)
- **Location:** `constants.py:37`
- **Purpose:** Balance memory vs. cache hit rate for security validation
- **Eviction:** LRU with FIFO fallback
- **Cache Key:** SHA-256 hash of code + sorted allowlists tuple

#### Offer Validity Window (5000ms + jitter)
- **Location:** `constants.py:33-36`
- **Purpose:** Prevent stale task assignments in distributed system
- **Components:**
  - Base: 5000ms
  - Jitter: 0-500ms random (prevents thundering herd)
  - Latency buffer: 100ms subtracted for network round-trip

#### Pipe Reader Timeout (dynamic)
- **Location:** `task_runner_config.py:105-109`
- **Formula:** `(max_payload * 0.1) / 100MB/s + 2.0s`
- **Default:** ~3 seconds for 1 GiB max payload
- **Purpose:** Prevent hung threads while accommodating large payloads

#### Trigger Priority Multiplier (50x)
- **Location:** `config_loader.py:109`
- **Purpose:** Heavily penalize wrong trigger types in workflow comparison
- **Rationale:** Trigger = workflow intent (time-based vs. event-based vs. manual)
- **Cost Hierarchy:** same_type (1.0) < similar (5.0) < different (15.0) < trigger (50.0)

---

## Links Added

- **Environment links added:** 24 (all implementations linked to their environments)
- **Heuristic links added:** 13 (implementations + principles + workflows)

### Connection Summary

| Page Type | Environment Links | Heuristic Links |
|-----------|-------------------|-----------------|
| Implementations | 24 | 6 |
| Principles | 0 | 5 |
| Workflows | 0 | 2 |

---

## Files Created

### Environment Pages (2 files)

```
environments/
├── n8n-io_n8n_Python_Task_Runner.md
└── n8n-io_n8n_Python_Workflow_Comparison.md
```

### Heuristic Pages (4 files)

```
heuristics/
├── n8n-io_n8n_Validation_Cache_Size.md
├── n8n-io_n8n_Offer_Validity_Window.md
├── n8n-io_n8n_Pipe_Reader_Timeout.md
└── n8n-io_n8n_Trigger_Priority_Multiplier.md
```

---

## Indexes Updated

| Index File | Status | Changes |
|------------|--------|---------|
| _EnvironmentIndex.md | ✅ Complete | 2 environment entries added |
| _HeuristicIndex.md | ✅ Complete | 4 heuristic entries added |

---

## Notes for Audit Phase

### Potential Review Items

1. **Environment pages reference all 24 implementations** - verify bidirectional links are correct
2. **Heuristic pages reference principles and implementations** - verify all linked pages exist
3. **No broken links expected** - all referenced pages were created in Phase 2

### Missing/Not Found

- **No TODO/NOTE/HACK comments** found in Python source code
- **No warnings.warn() calls** found (only logger.warning used)
- **No additional environment requirements** beyond Python version and package dependencies

### Observations

1. The codebase uses environment variables extensively for configuration rather than config files
2. Docker secrets support via `_FILE` suffix is well documented
3. Security configurations (allowlists, denylists) are critical tribal knowledge for operators
4. The 50x trigger multiplier is a design decision worth documenting for AI evaluation context

---

## Phase 3 Completion Status

**Phase 3 (Enrichment) is COMPLETE.**

All discovered environment requirements and heuristics have been:
1. ✅ Extracted into dedicated wiki pages
2. ✅ Linked to relevant implementation/principle pages
3. ✅ Indexed in tracking files
4. ✅ Documented with code evidence

Ready for Phase 4 (Audit) to verify:
- Bidirectional link integrity
- Page format compliance
- Coverage completeness
