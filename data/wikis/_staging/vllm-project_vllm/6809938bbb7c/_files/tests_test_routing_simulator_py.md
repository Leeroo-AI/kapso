# File: `tests/test_routing_simulator.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 199 |
| Classes | None (fixture device) |
| Functions | `device`, `test_basic_functionality`, `test_routing_strategy_integration`, `test_distribution_based_routing_with_custom_strategy`, `test_instance_compatibility` |
| Imports | pytest, tempfile, torch, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** MoE routing simulator tests

**Mechanism:** Tests the token-to-expert routing simulator including: basic functionality across different strategies (varied num_tokens, hidden_size, num_experts, top_k), integration with FusedMoE layer via VLLM_MOE_ROUTING_SIMULATION_STRATEGY environment variable, custom distribution-based routing strategies, and static method compatibility.

**Significance:** Validates the routing simulator used for testing and analyzing different MoE routing strategies without requiring full model execution. Important for MoE model development and optimization.
