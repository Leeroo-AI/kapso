# File: `libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_pii.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 638 |
| Classes | `TestEmailDetection`, `TestCreditCardDetection`, `TestIPDetection`, `TestMACAddressDetection`, `TestURLDetection`, `TestRedactStrategy`, `TestMaskStrategy`, `TestHashStrategy`, `TestBlockStrategy`, `TestPIIMiddlewareIntegration`, `TestCustomDetector`, `TestMultipleMiddleware` |
| Imports | langchain, langchain_core, pytest, tests |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive unit tests for PIIMiddleware, validating detection and handling of personally identifiable information (PII) in agent messages. Tests cover built-in detectors (email, credit card, IP, MAC, URL), multiple protection strategies (redact, mask, hash, block), and custom detector integration.

**Mechanism:** Test suite organized into distinct validation areas:

**Detection Function Tests** (5 test classes):
- **Email**: Validates regex detection of valid email formats, rejects invalid formats
- **Credit Card**: Tests Luhn algorithm validation for card numbers with various separators (spaces, dashes)
- **IP Address**: Detects valid IPv4 addresses, rejects out-of-range octets
- **MAC Address**: Matches colon and dash-separated MAC addresses (upper/lowercase)
- **URL**: Detects HTTP/HTTPS URLs, www domains, and domains with paths

**Strategy Tests** (4 test classes):
- **Redact**: Replaces PII with `[REDACTED_TYPE]` placeholders
- **Mask**: Partially obscures PII (e.g., `user@****.com`, `*.*.*.100`, last 4 digits of cards)
- **Hash**: Deterministic hashing for consistent de-identification
- **Block**: Raises `PIIDetectionError` to prevent processing messages with PII

**Integration Tests**:
- **Direction control**: Tests `apply_to_input`, `apply_to_output`, `apply_to_tool_results` flags
- **Agent integration**: Validates middleware works with `create_agent` and processes message flows
- **Custom detectors**: Tests regex patterns and callable detector functions
- **Multiple middleware**: Validates sequential application of multiple PII types and composition patterns

**Significance:** Critical for production agent deployments handling sensitive data. Ensures agents comply with privacy regulations (GDPR, HIPAA, etc.) by detecting and protecting PII in conversations. The flexible strategy system allows different handling based on use case (analytics with hashing, compliance with blocking). Custom detector support enables domain-specific PII detection (API keys, employee IDs, etc.). Comprehensive test coverage ensures PII protection works correctly across all message types and agent execution paths.
