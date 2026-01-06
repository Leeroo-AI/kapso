# File: `libs/langchain_v1/langchain/agents/middleware/pii.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 369 |
| Classes | `PIIMiddleware` |
| Imports | __future__, langchain, langchain_core, typing, typing_extensions |

## Understanding

**Status:** ✅ Explored

**Purpose:** Detects and handles Personally Identifiable Information (PII) in agent conversations using configurable strategies.

**Mechanism:** Implements PIIMiddleware that hooks into before_model (checks user/tool messages) and after_model (checks AI messages). Uses detector functions from _redaction module to identify common PII types (email, credit card, IP, MAC, URL) via regex patterns. Applies configurable strategies: block (raise exception), redact (replace with [REDACTED_TYPE]), mask (partial masking), or hash (deterministic hashing). Processes message content and returns updated messages with sanitized content.

**Significance:** Critical middleware for data compliance and privacy protection in agent systems. Provides production-grade PII detection/handling capabilities to prevent sensitive data leakage in agent conversations. Supports both built-in PII types and custom detectors via regex or callables.
