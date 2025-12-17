{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai/langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::Agent Middleware]], [[domain::Privacy]], [[domain::PII Detection]], [[domain::Data Protection]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==
Shared utilities for detecting and redacting personally identifiable information (PII) in agent text content using configurable detection patterns and redaction strategies.

=== Description ===
The PII redaction module provides a comprehensive framework for identifying and handling sensitive data in text streams. It includes built-in detectors for common PII types (email addresses, credit cards, IP addresses, MAC addresses, URLs) and supports custom detection patterns via regex or callable functions.

The module defines four redaction strategies: "block" (raise exception on detection), "redact" (replace with type labels like [REDACTED_EMAIL]), "mask" (partial masking preserving some characters), and "hash" (replace with SHA-256 hash prefixes). Each detector returns PIIMatch objects containing the matched value, type, and position information.

RedactionRule and ResolvedRedactionRule provide immutable configuration objects that combine a PII type with a detection method and redaction strategy. The apply_strategy function executes the configured strategy on matched content, while resolve_detector handles the conversion from configuration (string regex or callable) to executable detector functions.

=== Usage ===
Use this module when building agent middleware that needs to protect sensitive information in messages, tool outputs, or state data. Configure RedactionRule objects for each PII type you want to detect, choosing appropriate strategies based on whether you need strict blocking, full redaction, partial visibility, or deterministic hashing. The built-in detectors cover common cases, but custom detectors can be provided for domain-specific PII patterns.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai/langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain_v1/langchain/agents/middleware/_redaction.py libs/langchain_v1/langchain/agents/middleware/_redaction.py]

=== Signature ===
<syntaxhighlight lang="python">
RedactionStrategy = Literal["block", "redact", "mask", "hash"]
Detector = Callable[[str], list[PIIMatch]]


class PIIMatch(TypedDict):
    type: str
    value: str
    start: int
    end: int


class PIIDetectionError(Exception):
    def __init__(self, pii_type: str, matches: Sequence[PIIMatch]) -> None:
        self.pii_type = pii_type
        self.matches = list(matches)


@dataclass(frozen=True)
class RedactionRule:
    pii_type: str
    strategy: RedactionStrategy = "redact"
    detector: Detector | str | None = None

    def resolve(self) -> ResolvedRedactionRule: ...


@dataclass(frozen=True)
class ResolvedRedactionRule:
    pii_type: str
    strategy: RedactionStrategy
    detector: Detector

    def apply(self, content: str) -> tuple[str, list[PIIMatch]]: ...


# Built-in detectors
def detect_email(content: str) -> list[PIIMatch]: ...
def detect_credit_card(content: str) -> list[PIIMatch]: ...
def detect_ip(content: str) -> list[PIIMatch]: ...
def detect_mac_address(content: str) -> list[PIIMatch]: ...
def detect_url(content: str) -> list[PIIMatch]: ...

BUILTIN_DETECTORS: dict[str, Detector]
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain.agents.middleware._redaction import (
    RedactionRule,
    ResolvedRedactionRule,
    PIIMatch,
    PIIDetectionError,
    detect_email,
    detect_credit_card,
    detect_ip,
    detect_mac_address,
    detect_url,
    apply_strategy,
)
</syntaxhighlight>

== I/O Contract ==

=== RedactionRule Parameters ===
{| class="wikitable"
|-
! Parameter
! Type
! Default
! Description
|-
| pii_type
| str
| Required
| Name of the PII type (e.g., "email", "credit_card")
|-
| strategy
| RedactionStrategy
| "redact"
| How to handle detected PII: "block", "redact", "mask", "hash"
|-
| detector
| Detector, str, or None
| None
| Custom detector function or regex pattern; None uses built-in
|}

=== PIIMatch Fields ===
{| class="wikitable"
|-
! Field
! Type
! Description
|-
| type
| str
| The PII type identifier
|-
| value
| str
| The matched sensitive value
|-
| start
| int
| Starting position in original text
|-
| end
| int
| Ending position in original text
|}

=== Detector Functions ===
{| class="wikitable"
|-
! Function
! Returns
! Description
|-
| detect_email(content: str)
| list[PIIMatch]
| Find email addresses using regex pattern
|-
| detect_credit_card(content: str)
| list[PIIMatch]
| Find credit card numbers with Luhn validation
|-
| detect_ip(content: str)
| list[PIIMatch]
| Find IPv4 addresses with validation
|-
| detect_mac_address(content: str)
| list[PIIMatch]
| Find MAC addresses in colon or hyphen format
|-
| detect_url(content: str)
| list[PIIMatch]
| Find URLs with or without scheme
|}

=== Utility Functions ===
{| class="wikitable"
|-
! Function
! Returns
! Description
|-
| apply_strategy(content, matches, strategy)
| str
| Apply redaction strategy to content with matches
|-
| resolve_detector(pii_type, detector)
| Detector
| Convert configuration to callable detector
|}

== Usage Examples ==

=== Basic Email Redaction ===
<syntaxhighlight lang="python">
from langchain.agents.middleware._redaction import (
    RedactionRule,
    detect_email
)

# Create a redaction rule for emails
rule = RedactionRule(pii_type="email", strategy="redact")
resolved = rule.resolve()

# Apply to content
text = "Contact us at support@example.com or sales@example.com"
redacted_text, matches = resolved.apply(text)

print(redacted_text)
# Output: "Contact us at [REDACTED_EMAIL] or [REDACTED_EMAIL]"
print(f"Found {len(matches)} email addresses")
</syntaxhighlight>

=== Credit Card Masking ===
<syntaxhighlight lang="python">
from langchain.agents.middleware._redaction import RedactionRule

# Mask credit cards, showing last 4 digits
rule = RedactionRule(pii_type="credit_card", strategy="mask")
resolved = rule.resolve()

text = "Please charge card 4532-1488-0343-6467 for payment"
redacted_text, matches = resolved.apply(text)

print(redacted_text)
# Output: "Please charge card ****-****-****-6467 for payment"
</syntaxhighlight>

=== Blocking on Sensitive Data ===
<syntaxhighlight lang="python">
from langchain.agents.middleware._redaction import (
    RedactionRule,
    PIIDetectionError
)

# Block execution if credit card detected
rule = RedactionRule(pii_type="credit_card", strategy="block")
resolved = rule.resolve()

text = "My card number is 4532-1488-0343-6467"

try:
    redacted_text, matches = resolved.apply(text)
except PIIDetectionError as e:
    print(f"Blocked: Found {len(e.matches)} instances of {e.pii_type}")
    for match in e.matches:
        print(f"  - {match['value']} at position {match['start']}")
</syntaxhighlight>

=== Hashing for Audit Logs ===
<syntaxhighlight lang="python">
from langchain.agents.middleware._redaction import RedactionRule

# Hash emails for audit trail
rule = RedactionRule(pii_type="email", strategy="hash")
resolved = rule.resolve()

text = "User email: user@example.com"
redacted_text, matches = resolved.apply(text)

print(redacted_text)
# Output: "User email: <email_hash:a1b2c3d4>"
# Hash is deterministic: same email -> same hash
</syntaxhighlight>

=== Custom Regex Detector ===
<syntaxhighlight lang="python">
from langchain.agents.middleware._redaction import RedactionRule

# Detect Social Security Numbers with custom pattern
ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"
rule = RedactionRule(
    pii_type="ssn",
    strategy="mask",
    detector=ssn_pattern
)
resolved = rule.resolve()

text = "SSN: 123-45-6789"
redacted_text, matches = resolved.apply(text)

print(redacted_text)
# Output: "SSN: ****6789"
</syntaxhighlight>

=== Custom Callable Detector ===
<syntaxhighlight lang="python">
from langchain.agents.middleware._redaction import (
    RedactionRule,
    PIIMatch
)

# Custom detector for API keys
def detect_api_key(content: str) -> list[PIIMatch]:
    import re
    pattern = r"\b[A-Za-z0-9]{32}\b"
    matches = []
    for match in re.finditer(pattern, content):
        # Additional validation logic
        value = match.group()
        if value.startswith(('sk_', 'pk_', 'api_')):
            matches.append(PIIMatch(
                type="api_key",
                value=value,
                start=match.start(),
                end=match.end()
            ))
    return matches


rule = RedactionRule(
    pii_type="api_key",
    strategy="redact",
    detector=detect_api_key
)
resolved = rule.resolve()

text = "Use API key: sk_1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p"
redacted_text, matches = resolved.apply(text)

print(redacted_text)
# Output: "Use API key: [REDACTED_API_KEY]"
</syntaxhighlight>

=== Multiple PII Types ===
<syntaxhighlight lang="python">
from langchain.agents.middleware._redaction import RedactionRule

# Configure multiple redaction rules
rules = [
    RedactionRule(pii_type="email", strategy="redact"),
    RedactionRule(pii_type="credit_card", strategy="mask"),
    RedactionRule(pii_type="ip", strategy="hash"),
]

# Resolve all rules
resolved_rules = [rule.resolve() for rule in rules]

# Apply sequentially
text = """
Contact: admin@example.com
Card: 4532-1488-0343-6467
Server: 192.168.1.100
"""

for resolved in resolved_rules:
    text, matches = resolved.apply(text)
    if matches:
        print(f"Redacted {len(matches)} {resolved.pii_type} instances")

print("\nFinal text:")
print(text)
</syntaxhighlight>

=== IP Address Detection with Validation ===
<syntaxhighlight lang="python">
from langchain.agents.middleware._redaction import (
    RedactionRule,
    detect_ip
)

# Built-in IP detector validates IP format
text = "Server at 192.168.1.1 and backup at 10.0.0.256"  # Second is invalid
matches = detect_ip(text)

print(f"Found {len(matches)} valid IP addresses")
# Only finds 192.168.1.1 (256 is invalid for octet)

rule = RedactionRule(pii_type="ip", strategy="mask")
resolved = rule.resolve()
redacted_text, _ = resolved.apply(text)

print(redacted_text)
# Output: "Server at *.*.*.1 and backup at 10.0.0.256"
</syntaxhighlight>

=== URL Redaction ===
<syntaxhighlight lang="python">
from langchain.agents.middleware._redaction import RedactionRule

# Redact URLs to prevent data leakage
rule = RedactionRule(pii_type="url", strategy="mask")
resolved = rule.resolve()

text = """
Visit https://example.com/api/v1/users/123
Or go to www.example.com/profile?token=abc
"""

redacted_text, matches = resolved.apply(text)

print(redacted_text)
# Output shows masked URLs
print(f"Found {len(matches)} URLs")
</syntaxhighlight>

=== Combining with Exception Handling ===
<syntaxhighlight lang="python">
from langchain.agents.middleware._redaction import (
    RedactionRule,
    PIIDetectionError
)

def safe_process_text(text: str) -> str:
    # Block on credit cards, redact everything else
    rules = [
        RedactionRule(pii_type="credit_card", strategy="block"),
        RedactionRule(pii_type="email", strategy="redact"),
        RedactionRule(pii_type="ip", strategy="redact"),
    ]

    for rule in rules:
        resolved = rule.resolve()
        try:
            text, matches = resolved.apply(text)
        except PIIDetectionError as e:
            # Log and reject the request
            print(f"Rejected: {e.pii_type} detected")
            raise ValueError("Content contains sensitive payment information")

    return text


# This will succeed
safe_text = safe_process_text("Email me at user@example.com")
print(safe_text)
# Output: "Email me at [REDACTED_EMAIL]"

# This will raise ValueError
try:
    safe_process_text("Card: 4532-1488-0343-6467")
except ValueError as e:
    print(f"Error: {e}")
</syntaxhighlight>

=== MAC Address Detection ===
<syntaxhighlight lang="python">
from langchain.agents.middleware._redaction import RedactionRule

# Detect MAC addresses in logs
rule = RedactionRule(pii_type="mac_address", strategy="mask")
resolved = rule.resolve()

text = "Device MAC: 00:1B:44:11:3A:B7 and 00-1B-44-11-3A-B8"
redacted_text, matches = resolved.apply(text)

print(redacted_text)
# Output: "Device MAC: **:**:**:**:**:B7 and **-**-**-**-**-B8"
print(f"Found {len(matches)} MAC addresses")
</syntaxhighlight>

== Related Pages ==
* [[langchain-ai_langchain_PIIRedactionMiddleware]] - Middleware that uses these utilities
* [[principle::Privacy by Design]]
* [[principle::Data Minimization]]
* [[environment::GDPR Compliance]]
* [[environment::Healthcare Applications (HIPAA)]]
* [[environment::Financial Services (PCI-DSS)]]
