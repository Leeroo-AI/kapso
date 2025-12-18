# Heuristic Index: n8n-io_n8n

> Tracks Heuristic pages and which pages they apply to.
> **Update IMMEDIATELY** after creating or modifying a Heuristic page.

## Summary

| Heuristic | Applies To |
|-----------|------------|
| Validation_Caching_Strategy | Security validation |
| Pipe_Timeout_Scaling | Large payload handling |
| Print_Output_Truncation | User code output |
| Offer_Validity_Jitter | Distributed task offers |
| GED_Performance_Note | Graph comparison |
| Graceful_Shutdown_Timeout | Process lifecycle |

## Pages

| Page | File | Connections | Notes |
|------|------|-------------|-------|
| n8n-io_n8n_Validation_Caching_Strategy | [→](./heuristics/n8n-io_n8n_Validation_Caching_Strategy.md) | ✅Impl:n8n-io_n8n_TaskAnalyzer_cache, ✅Impl:n8n-io_n8n_TaskAnalyzer_validate, ✅Workflow:n8n-io_n8n_Security_Validation_Pipeline | LRU cache with SHA256 hash keys for AST validation |
| n8n-io_n8n_Pipe_Timeout_Scaling | [→](./heuristics/n8n-io_n8n_Pipe_Timeout_Scaling.md) | ✅Impl:n8n-io_n8n_TaskExecutor_put_result, ✅Impl:n8n-io_n8n_TaskRunner_execute_task, ✅Workflow:n8n-io_n8n_Python_Task_Execution | Dynamic timeout based on payload size |
| n8n-io_n8n_Print_Output_Truncation | [→](./heuristics/n8n-io_n8n_Print_Output_Truncation.md) | ✅Impl:n8n-io_n8n_TaskExecutor_put_result, ✅Impl:n8n-io_n8n_TaskExecutor_execute, ✅Workflow:n8n-io_n8n_Python_Task_Execution | Limit print() to 100 statements |
| n8n-io_n8n_Offer_Validity_Jitter | [→](./heuristics/n8n-io_n8n_Offer_Validity_Jitter.md) | ✅Impl:n8n-io_n8n_TaskRunner_send_offers, ✅Workflow:n8n-io_n8n_Python_Task_Execution, ✅Principle:n8n-io_n8n_Task_Offer_Negotiation | Random 0-500ms jitter prevents thundering herd |
| n8n-io_n8n_GED_Performance_Note | [→](./heuristics/n8n-io_n8n_GED_Performance_Note.md) | ✅Impl:n8n-io_n8n_calculate_graph_edit_distance, ✅Workflow:n8n-io_n8n_AI_Workflow_Comparison, ✅Principle:n8n-io_n8n_Graph_Edit_Distance | GED slow but OK for small workflow graphs |
| n8n-io_n8n_Graceful_Shutdown_Timeout | [→](./heuristics/n8n-io_n8n_Graceful_Shutdown_Timeout.md) | ✅Impl:n8n-io_n8n_TaskRunner_execute_task, ✅Workflow:n8n-io_n8n_Python_Task_Execution, ✅Principle:n8n-io_n8n_Result_Delivery | Two-phase shutdown: wait 10s then force kill |

---

**Legend:** `✅Type:Name` = page exists | `⬜Type:Name` = page needs creation
