# File: `libs/langchain_v1/langchain/agents/middleware/_redaction.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 364 |
| Classes | `PIIMatch`, `PIIDetectionError`, `RedactionRule`, `ResolvedRedactionRule` |
| Functions | `detect_email`, `detect_credit_card`, `detect_ip`, `detect_mac_address`, `detect_url`, `apply_strategy`, `resolve_detector` |
| Imports | __future__, collections, dataclasses, hashlib, ipaddress, re, typing, typing_extensions, urllib |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides shared PII detection and redaction utilities for middleware components, identifying sensitive data patterns (emails, credit cards, IPs, MAC addresses, URLs) and applying configurable redaction strategies.

**Mechanism:** Implements five built-in regex-based detectors (detect_email, detect_credit_card with Luhn validation, detect_ip with ipaddress validation, detect_mac_address, detect_url with urlparse validation) that return PIIMatch tuples with type/value/start/end positions. RedactionRule and ResolvedRedactionRule classes combine detectors with four strategies: 'block' raises PIIDetectionError, 'redact' replaces with [REDACTED_TYPE], 'mask' shows last 4 chars (e.g., ****1234), 'hash' computes SHA256[:8] fingerprint. Supports custom detectors via regex strings or callables.

**Significance:** Reusable PII handling foundation shared across multiple middleware implementations (PIIMiddleware, ShellToolMiddleware) - centralizes detection logic and strategy application to ensure consistent privacy protection patterns throughout the agent system.
