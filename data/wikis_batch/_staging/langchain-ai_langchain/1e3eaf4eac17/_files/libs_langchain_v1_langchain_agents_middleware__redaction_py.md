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

**Purpose:** Provides PII (Personally Identifiable Information) detection and redaction utilities for agent middleware, protecting sensitive data in agent inputs and outputs through configurable strategies.

**Mechanism:** The module implements a registry of built-in detectors (`BUILTIN_DETECTORS`) that use regular expressions and validation logic to identify sensitive data types: emails (RFC-compliant regex), credit cards (Luhn checksum validation), IP addresses (IPv4 with `ipaddress` validation), MAC addresses (colon/hyphen-separated hex), and URLs (with/without schemes using `urlparse`). Each detector returns `PIIMatch` objects containing type, value, and position. Four redaction strategies handle detected PII: `redact` replaces with `[REDACTED_TYPE]` tags, `mask` preserves partial data (last 4 digits of cards, final IP octet), `hash` uses SHA-256 digests for deterministic anonymization, and `block` raises `PIIDetectionError` to prevent processing. The `RedactionRule` dataclass combines a PII type, strategy, and optional custom detector (callable or regex string).

**Significance:** This is a foundational privacy and security module used by PIIMiddleware to prevent accidental leakage of sensitive information through agent interactions. It enables compliance with data protection regulations (GDPR, CCPA) by automatically sanitizing logs, prompts, and responses. The flexible strategy system supports different use cases: masking for user display, hashing for analytics, redaction for logging, and blocking for high-security environments. Custom detectors allow domain-specific patterns beyond the built-in types.
