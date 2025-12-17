{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai/langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::Agent Middleware]], [[domain::Privacy]], [[domain::PII Detection]], [[domain::Data Security]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==
PIIMiddleware detects and handles Personally Identifiable Information (PII) in agent conversations by applying configurable strategies to redact, mask, hash, or block sensitive data in user input, AI output, and tool results.

=== Description ===
This middleware provides comprehensive PII detection and handling for LangChain agents. It monitors conversations for common PII types (emails, credit cards, IP addresses, MAC addresses, URLs) and applies configurable strategies to sanitize or block detected sensitive information before it reaches the model or is returned to users.

The middleware operates at three configurable interception points:
* User input messages (before model invocation)
* AI output messages (after model invocation)
* Tool result messages (before model invocation)

Each PII type can be handled independently using different strategies, and custom detectors can be added using regex patterns or callable functions.

Built-in PII detectors include validation logic (e.g., Luhn algorithm for credit cards, stdlib validation for IP addresses) to minimize false positives.

=== Usage ===
Use this middleware when you need to:
* Prevent sensitive information from being sent to language models
* Comply with data protection regulations (GDPR, HIPAA, etc.)
* Sanitize logs and conversation histories
* Maintain pseudonymous analytics with deterministic hashing
* Block agent execution when specific PII types are detected

== Code Reference ==
'''Source location:''' `/tmp/praxium_repo_wjjl6pl8/libs/langchain_v1/langchain/agents/middleware/pii.py`

'''Signature:'''
<syntaxhighlight lang="python">
class PIIMiddleware(AgentMiddleware):
    def __init__(
        self,
        pii_type: Literal["email", "credit_card", "ip", "mac_address", "url"] | str,
        *,
        strategy: Literal["block", "redact", "mask", "hash"] = "redact",
        detector: Callable[[str], list[PIIMatch]] | str | None = None,
        apply_to_input: bool = True,
        apply_to_output: bool = False,
        apply_to_tool_results: bool = False,
    ) -> None
</syntaxhighlight>

'''Import statement:'''
<syntaxhighlight lang="python">
from langchain.agents.middleware import PIIMiddleware
from langchain.agents.middleware.pii import PIIDetectionError, detect_email, detect_credit_card
</syntaxhighlight>

== I/O Contract ==

=== Initialization Parameters ===
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| pii_type || str/Literal || (required) || Type of PII to detect. Built-in: "email", "credit_card", "ip", "mac_address", "url", or custom name
|-
| strategy || Literal || "redact" || How to handle detected PII: "block" (raise exception), "redact" (replace with placeholder), "mask" (partial masking), "hash" (deterministic hash)
|-
| detector || Callable/str/None || None || Custom detector function or regex pattern. If None, uses built-in detector for pii_type
|-
| apply_to_input || bool || True || Whether to check user messages before model call
|-
| apply_to_output || bool || False || Whether to check AI messages after model call
|-
| apply_to_tool_results || bool || False || Whether to check tool result messages
|}

=== Hook Methods ===
{| class="wikitable"
! Method !! Execution Point !! Input !! Output
|-
| before_model() || Before model invocation || AgentState, Runtime || dict[str, Any] or None (updated state with sanitized messages)
|-
| after_model() || After model invocation || AgentState, Runtime || dict[str, Any] or None (updated state with sanitized AI message)
|}

=== Strategy Behavior ===
{| class="wikitable"
! Strategy !! Preserves Identity !! Output Format !! Best For
|-
| block || N/A || Raises PIIDetectionError || Avoiding PII completely, strict compliance
|-
| redact || No || [REDACTED_TYPE] || General compliance, log sanitization
|-
| mask || No || ****-****-****-1234 || Human readability, customer service UIs
|-
| hash || Yes (pseudonymous) || <type_hash:a1b2c3d4> || Analytics, debugging, tracking across sessions
|}

== Usage Examples ==

=== Example 1: Basic Email Redaction ===
<syntaxhighlight lang="python">
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware

# Redact all emails in user input
agent = create_agent(
    "openai:gpt-4o",
    middleware=[
        PIIMiddleware("email", strategy="redact"),
    ],
)

# User message: "Contact me at john@example.com"
# Model receives: "Contact me at [REDACTED_email]"
</syntaxhighlight>

=== Example 2: Multiple PII Types with Different Strategies ===
<syntaxhighlight lang="python">
from langchain.agents.middleware import PIIMiddleware

# Apply different handling to different PII types
agent = create_agent(
    "openai:gpt-4o",
    middleware=[
        PIIMiddleware("credit_card", strategy="mask"),      # Show last 4 digits
        PIIMiddleware("url", strategy="redact"),            # Full redaction
        PIIMiddleware("ip", strategy="hash"),               # Pseudonymous tracking
        PIIMiddleware("email", strategy="block"),           # Raise error
    ],
)

# Credit card "4532015112830366" -> "****-****-****-0366"
# URL "https://internal.corp/docs" -> "[REDACTED_url]"
# IP "192.168.1.1" -> "<ip_hash:a7f3d9e2>"
# Email detected -> raises PIIDetectionError
</syntaxhighlight>

=== Example 3: Custom PII Detector with Regex ===
<syntaxhighlight lang="python">
# Detect and block API keys
agent = create_agent(
    "openai:gpt-4o",
    middleware=[
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block"
        ),
    ],
)

# User message containing "sk-abc123..." will raise PIIDetectionError
</syntaxhighlight>

=== Example 4: Custom Detector Function ===
<syntaxhighlight lang="python">
from langchain.agents.middleware.pii import PIIMatch

def detect_employee_ids(content: str) -> list[PIIMatch]:
    """Detect employee IDs in format EMP-12345."""
    import re
    matches = []
    for match in re.finditer(r'\bEMP-\d{5}\b', content):
        matches.append(PIIMatch(
            start=match.start(),
            end=match.end(),
            value=match.group()
        ))
    return matches

agent = create_agent(
    "openai:gpt-4o",
    middleware=[
        PIIMiddleware(
            "employee_id",
            detector=detect_employee_ids,
            strategy="hash"
        ),
    ],
)
</syntaxhighlight>

=== Example 5: Protecting AI Output and Tool Results ===
<syntaxhighlight lang="python">
# Redact PII in AI responses and tool results
agent = create_agent(
    "openai:gpt-4o",
    tools=[database_search_tool],
    middleware=[
        PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=True,          # Redact user input
            apply_to_output=True,         # Redact AI responses
            apply_to_tool_results=True    # Redact database results
        ),
    ],
)

# Protects PII at all conversation stages:
# - User input sanitized before reaching model
# - Model output sanitized before returning to user
# - Tool results sanitized before feeding back to model
</syntaxhighlight>

=== Example 6: Handling PIIDetectionError ===
<syntaxhighlight lang="python">
from langchain.agents.middleware.pii import PIIDetectionError

agent = create_agent(
    "openai:gpt-4o",
    middleware=[
        PIIMiddleware("credit_card", strategy="block"),
    ],
)

try:
    result = agent.invoke({
        "messages": [{"role": "user", "content": "My card is 4532015112830366"}]
    })
except PIIDetectionError as e:
    print(f"PII detected: {e.pii_type}")
    print(f"Matches found: {len(e.matches)}")
    # Handle appropriately (log, notify user, etc.)
</syntaxhighlight>

=== Example 7: Built-in Detector Functions ===
<syntaxhighlight lang="python">
from langchain.agents.middleware.pii import (
    detect_email,
    detect_credit_card,
    detect_ip,
    detect_mac_address,
    detect_url
)

# Test detectors independently
content = "Contact admin@example.com or visit https://example.com"
email_matches = detect_email(content)
url_matches = detect_url(content)

print(f"Found {len(email_matches)} emails")
print(f"Found {len(url_matches)} URLs")
</syntaxhighlight>

== Implementation Details ==

=== Detection Process ===
The middleware processes content through these steps:
# Extract string content from message
# Apply detector function (built-in or custom)
# If matches found, apply configured strategy
# Return modified content or raise exception

=== Strategy Application ===
'''Redact:''' Replaces matched PII with `[REDACTED_{pii_type}]`

'''Mask:''' Shows last few characters (implementation varies by PII type)

'''Hash:''' Generates deterministic SHA-256 hash, formatted as `<{pii_type}_hash:{digest[:8]}>`

'''Block:''' Raises `PIIDetectionError` with matches and PII type

=== Message Processing Order ===
The `before_model` hook processes in order:
# Last HumanMessage (if `apply_to_input=True`)
# All ToolMessages after last AIMessage (if `apply_to_tool_results=True`)

The `after_model` hook processes:
# Last AIMessage (if `apply_to_output=True`)

=== Validation and False Positives ===
Built-in detectors include validation:
* Credit cards: Luhn algorithm validation
* IP addresses: Python stdlib validation
* Others: Pattern matching with reasonable constraints

=== Performance Considerations ===
* Detectors run on every applicable message
* Regex patterns are compiled once during initialization
* Processing is synchronous (async methods delegate to sync implementation)
* Message modification creates new message instances (immutable)

== Related Pages ==
* [[langchain-ai_langchain_AgentMiddleware|AgentMiddleware]] - Base middleware class
* [[langchain-ai_langchain_ShellToolMiddleware|ShellToolMiddleware]] - Uses redaction rules for command output
* [[langchain-ai_langchain_RedactionRule|RedactionRule]] - Underlying redaction configuration
* [[Agent Security Patterns]] - Best practices for agent safety
* [[Privacy Compliance Guide]] - Using middleware for regulatory compliance
