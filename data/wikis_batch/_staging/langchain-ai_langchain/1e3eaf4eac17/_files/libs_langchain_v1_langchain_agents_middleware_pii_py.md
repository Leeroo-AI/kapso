# File: `libs/langchain_v1/langchain/agents/middleware/pii.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 369 |
| Classes | `PIIMiddleware` |
| Imports | __future__, langchain, langchain_core, typing, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides middleware for detecting and handling Personally Identifiable Information (PII) in agent conversations, supporting multiple detection types (email, credit card, IP, MAC address, URL) and strategies (block, redact, mask, hash) to protect sensitive data.

**Mechanism:** The PIIMiddleware class intercepts messages in the agent lifecycle through `before_model` and `after_model` hooks. It uses built-in detectors or custom regex patterns to identify PII in user input, AI output, and tool results. When PII is detected, it applies configurable strategies: blocking execution with PIIDetectionError, redacting with placeholders like `[REDACTED_EMAIL]`, masking to show partial content (e.g., `****-1234`), or hashing for pseudonymous tracking. The middleware processes message content, creates modified copies with sanitized text, and returns updated state to the agent graph.

**Significance:** This file is a critical security and compliance component for production agent systems. It enables organizations to meet privacy regulations (GDPR, CCPA, HIPAA) by automatically detecting and sanitizing PII before it reaches language models or gets logged. The flexible strategy system allows different handling for different contexts (development vs production, different PII types), while the extensible detector framework supports custom organizational requirements beyond the built-in types.
