# File: `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_pii.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 638 |
| Classes | `TestEmailDetection`, `TestCreditCardDetection`, `TestIPDetection`, `TestMACAddressDetection`, `TestURLDetection`, `TestRedactStrategy`, `TestMaskStrategy`, `TestHashStrategy`, `TestBlockStrategy`, `TestPIIMiddlewareIntegration`, `TestCustomDetector`, `TestMultipleMiddleware` |
| Imports | langchain, langchain_core, pytest, tests |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests the PIIMiddleware that detects and redacts personally identifiable information from agent conversations.

**Mechanism:** Organized into test classes covering: (1) Detection functions - validate regex patterns for email, credit card (with Luhn validation), IP addresses, MAC addresses, and URLs against valid/invalid inputs, (2) Strategy tests - verify redact (replace with [REDACTED_TYPE]), mask (partial visibility like user@****.com), hash (deterministic hashing), and block (raise PIIDetectionError) strategies, (3) Integration tests - confirm apply_to_input/output/tool_results configuration, message type handling, and behavior with no PII found, (4) Custom detectors - test regex string or callable function detectors with proper error handling for unknown types, and (5) Multiple middleware - verify sequential application and composition patterns. Tests use FakeToolCallingModel for agent integration.

**Significance:** Critical data privacy and compliance validation ensuring the middleware correctly identifies and handles sensitive information across all conversation message types, supporting regulatory requirements like GDPR, HIPAA, and PCI-DSS.
