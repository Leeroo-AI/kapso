# Heuristic Index: n8n-io_n8n

> Tracks Heuristic pages and which pages they apply to.
> **Update IMMEDIATELY** after creating or modifying a Heuristic page.

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| n8n-io_n8n_Validation_Cache_Size | [→](./heuristics/n8n-io_n8n_Validation_Cache_Size.md) | ✅Impl:n8n-io_n8n_TaskAnalyzer_validate, ✅Impl:n8n-io_n8n_TaskAnalyzer_cache, ✅Principle:n8n-io_n8n_Validation_Caching | LRU cache limited to 500 entries for security validation |
| n8n-io_n8n_Offer_Validity_Window | [→](./heuristics/n8n-io_n8n_Offer_Validity_Window.md) | ✅Impl:n8n-io_n8n_TaskRunner_send_offers, ✅Impl:n8n-io_n8n_TaskRunner_handle_task_offer_accept, ✅Principle:n8n-io_n8n_Offer_Based_Distribution, ✅Principle:n8n-io_n8n_Task_Acceptance | 5000ms + 500ms jitter offer expiry |
| n8n-io_n8n_Pipe_Reader_Timeout | [→](./heuristics/n8n-io_n8n_Pipe_Reader_Timeout.md) | ✅Impl:n8n-io_n8n_TaskExecutor_execute_process, ✅Principle:n8n-io_n8n_Result_Collection | Dynamic timeout based on max payload size |
| n8n-io_n8n_Trigger_Priority_Multiplier | [→](./heuristics/n8n-io_n8n_Trigger_Priority_Multiplier.md) | ✅Impl:n8n-io_n8n_calculate_graph_edit_distance, ✅Impl:n8n-io_n8n_load_config, ✅Principle:n8n-io_n8n_GED_Calculation, ✅Workflow:n8n-io_n8n_Workflow_Comparison | 50x cost multiplier for trigger node mismatches |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
