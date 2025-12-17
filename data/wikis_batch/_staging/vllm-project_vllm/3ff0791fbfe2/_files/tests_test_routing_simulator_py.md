# File: `tests/test_routing_simulator.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 199 |
| Functions | `device`, `test_basic_functionality`, `test_routing_strategy_integration`, `test_distribution_based_routing_with_custom_strategy`, `test_instance_compatibility` |
| Imports | pytest, tempfile, torch, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** MoE routing simulation testing

**Mechanism:** Tests RoutingSimulator for token-to-expert routing with different strategies (uniform_random, distribution-based). Validates integration with FusedMoE layer, custom strategy registration, and output shape/range validation across various num_tokens, hidden_size, num_experts, and top_k configurations.

**Significance:** Enables testing and analysis of MoE routing strategies without running full model inference, critical for optimizing expert selection and load balancing in mixture-of-experts models.
