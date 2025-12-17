---
title: OpenAIModerationChain
type: implementation
project: langchain-ai/langchain
file: libs/langchain/langchain_classic/chains/moderation.py
category: content_safety
---

= OpenAIModerationChain Implementation =

== Overview ==

'''OpenAIModerationChain''' passes input text through OpenAI's content moderation endpoint to detect potentially harmful content. It can either flag problematic content or raise an error, making it useful for building safe AI applications that comply with content policies.

This chain integrates with OpenAI's Moderation API to identify content that violates OpenAI's usage policies, including categories like hate speech, violence, self-harm, and sexual content.

== Code Reference ==

=== Source Location ===
<syntaxhighlight lang="text">
File: /tmp/praxium_repo_wjjl6pl8/libs/langchain/langchain_classic/chains/moderation.py
Lines: 16-129
Package: langchain-classic
</syntaxhighlight>

=== Class Signature ===
<syntaxhighlight lang="python">
class OpenAIModerationChain(Chain):
    """Pass input through a moderation endpoint."""
</syntaxhighlight>

=== Import Statement ===
<syntaxhighlight lang="python">
from langchain_classic.chains import OpenAIModerationChain
</syntaxhighlight>

== Class Attributes ==

{| class="wikitable"
|+ Configuration Attributes
! Attribute !! Type !! Default !! Description
|-
| client || <code>Any</code> || None || OpenAI client instance (auto-initialized)
|-
| async_client || <code>Any</code> || None || Async OpenAI client (auto-initialized for openai>=1.0)
|-
| model_name || <code>str \| None</code> || None || Moderation model name to use
|-
| error || <code>bool</code> || False || Whether to raise error if bad content found
|-
| input_key || <code>str</code> || "input" || Key for input text in inputs dict
|-
| output_key || <code>str</code> || "output" || Key for output in results dict
|-
| openai_api_key || <code>str \| None</code> || None || OpenAI API key (or from OPENAI_API_KEY env var)
|-
| openai_organization || <code>str \| None</code> || None || OpenAI organization ID
|-
| openai_pre_1_0 || <code>bool</code> || False || Internal flag for OpenAI SDK version
|}

== Input/Output Contract ==

{| class="wikitable"
|+ Input Schema
! Key !! Type !! Description
|-
| <code>input_key</code> (default: "input") || <code>str</code> || Text content to moderate
|}

{| class="wikitable"
|+ Output Schema
! Key !! Type !! Description
|-
| <code>output_key</code> (default: "output") || <code>str</code> || Input text (if safe) or error message (if flagged)
|}

{| class="wikitable"
|+ Exception Behavior
! Condition !! error=False !! error=True
|-
| Content flagged || Returns error message string || Raises <code>ValueError</code>
|-
| Content safe || Returns original input text || Returns original input text
|}

== Environment Setup ==

=== Required Environment Variables ===

<syntaxhighlight lang="bash">
# Required
export OPENAI_API_KEY="sk-..."

# Optional
export OPENAI_ORGANIZATION="org-..."
</syntaxhighlight>

=== Required Package ===

<syntaxhighlight lang="bash">
pip install openai
</syntaxhighlight>

== Usage Examples ==

=== Basic Content Moderation ===

<syntaxhighlight lang="python">
from langchain_classic.chains import OpenAIModerationChain

# Create moderation chain (warning mode)
moderation = OpenAIModerationChain()

# Check safe content
result = moderation.invoke({"input": "Hello, how are you?"})
print(result["output"])  # "Hello, how are you?"

# Check problematic content
result = moderation.invoke({"input": "Some harmful text..."})
print(result["output"])  # "Text was found that violates OpenAI's content policy."
</syntaxhighlight>

=== Error Mode ===

<syntaxhighlight lang="python">
from langchain_classic.chains import OpenAIModerationChain

# Create moderation chain (error mode)
moderation = OpenAIModerationChain(error=True)

# Safe content passes through
result = moderation.invoke({"input": "This is fine."})
print(result["output"])  # "This is fine."

# Problematic content raises exception
try:
    result = moderation.invoke({"input": "Harmful content..."})
except ValueError as e:
    print(f"Content blocked: {e}")
    # Raises: ValueError("Text was found that violates OpenAI's content policy.")
</syntaxhighlight>

=== Custom Keys ===

<syntaxhighlight lang="python">
from langchain_classic.chains import OpenAIModerationChain

# Custom input/output keys
moderation = OpenAIModerationChain(
    input_key="user_message",
    output_key="moderated_message"
)

result = moderation.invoke({
    "user_message": "Hello world!"
})

print(result["moderated_message"])  # "Hello world!"
</syntaxhighlight>

=== In a Chain Pipeline ===

<syntaxhighlight lang="python">
from langchain_classic.chains import OpenAIModerationChain, LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Create moderation chain with error mode
moderation = OpenAIModerationChain(error=True)

# Create LLM chain
llm_chain = LLMChain(
    llm=ChatOpenAI(),
    prompt=PromptTemplate(
        template="Respond to: {input}",
        input_variables=["input"]
    )
)

# Moderate input before processing
user_input = "Tell me about AI safety"

try:
    # Step 1: Moderate input
    moderated = moderation.invoke({"input": user_input})

    # Step 2: Process with LLM if safe
    response = llm_chain.invoke({"input": moderated["output"]})

    print(response["text"])

except ValueError as e:
    print(f"Input blocked by moderation: {e}")
</syntaxhighlight>

=== Async Usage ===

<syntaxhighlight lang="python">
import asyncio
from langchain_classic.chains import OpenAIModerationChain

async def moderate_content(text: str) -> str:
    """Async content moderation."""
    moderation = OpenAIModerationChain()

    result = await moderation.ainvoke({"input": text})
    return result["output"]

# Run async
async def main():
    texts = [
        "Hello world",
        "Nice to meet you",
        "How are you today?"
    ]

    # Moderate multiple texts
    results = await asyncio.gather(*[
        moderate_content(text) for text in texts
    ])

    for text, result in zip(texts, results):
        print(f"{text} -> {result}")

asyncio.run(main())
</syntaxhighlight>

=== With Custom Model ===

<syntaxhighlight lang="python">
from langchain_classic.chains import OpenAIModerationChain

# Specify moderation model (if OpenAI offers different versions)
moderation = OpenAIModerationChain(
    model_name="text-moderation-stable",  # or "text-moderation-latest"
    error=False
)

result = moderation.invoke({"input": "Test content"})
print(result["output"])
</syntaxhighlight>

== Implementation Details ==

=== Environment Validation ===

The class uses a Pydantic <code>model_validator</code> to set up OpenAI client:

<syntaxhighlight lang="python">
@model_validator(mode="before")
@classmethod
def validate_environment(cls, values: dict) -> Any:
    """Validate that api key and python package exists in environment."""
    # Get API key from dict or environment
    openai_api_key = get_from_dict_or_env(
        values, "openai_api_key", "OPENAI_API_KEY"
    )

    # Import and configure openai package
    import openai
    openai.api_key = openai_api_key

    # Detect OpenAI SDK version
    try:
        check_package_version("openai", gte_version="1.0")
        values["openai_pre_1_0"] = False
        values["client"] = openai.OpenAI(api_key=openai_api_key)
        values["async_client"] = openai.AsyncOpenAI(api_key=openai_api_key)
    except ValueError:
        values["openai_pre_1_0"] = True
        values["client"] = openai.Moderation

    return values
</syntaxhighlight>

=== Moderation Logic ===

<syntaxhighlight lang="python">
def _moderate(self, text: str, results: Any) -> str:
    """Check if content is flagged and handle accordingly."""
    # Extract flagged status (different for old vs new SDK)
    condition = results["flagged"] if self.openai_pre_1_0 else results.flagged

    if condition:
        error_str = "Text was found that violates OpenAI's content policy."
        if self.error:
            raise ValueError(error_str)
        return error_str

    return text  # Content is safe
</syntaxhighlight>

=== Synchronous Call ===

<syntaxhighlight lang="python">
def _call(
    self,
    inputs: dict[str, Any],
    run_manager: CallbackManagerForChainRun | None = None,
) -> dict[str, Any]:
    text = inputs[self.input_key]

    if self.openai_pre_1_0:
        # Old SDK (openai < 1.0)
        results = self.client.create(text)
        output = self._moderate(text, results["results"][0])
    else:
        # New SDK (openai >= 1.0)
        results = self.client.moderations.create(input=text)
        output = self._moderate(text, results.results[0])

    return {self.output_key: output}
</syntaxhighlight>

=== Asynchronous Call ===

<syntaxhighlight lang="python">
async def _acall(
    self,
    inputs: dict[str, Any],
    run_manager: AsyncCallbackManagerForChainRun | None = None,
) -> dict[str, Any]:
    # Old SDK doesn't support async, falls back to sync
    if self.openai_pre_1_0:
        return await super()._acall(inputs, run_manager=run_manager)

    # New SDK supports async
    text = inputs[self.input_key]
    results = await self.async_client.moderations.create(input=text)
    output = self._moderate(text, results.results[0])
    return {self.output_key: output}
</syntaxhighlight>

== OpenAI Moderation Categories ==

OpenAI's moderation API checks for these content categories:

{| class="wikitable"
|+ Content Categories
! Category !! Description
|-
| hate || Content promoting hate based on identity
|-
| hate/threatening || Hateful content with violence or threats
|-
| self-harm || Content promoting self-harm
|-
| sexual || Sexual content
|-
| sexual/minors || Sexual content involving minors
|-
| violence || Content depicting violence
|-
| violence/graphic || Graphic violent content
|}

== Use Cases ==

=== Content Filtering ===
* Filter user-generated content before display
* Moderate chat messages in real-time
* Screen submissions in content platforms

=== Compliance ===
* Ensure AI applications comply with content policies
* Prevent generation of harmful content
* Audit content for safety violations

=== Input Validation ===
* Validate prompts before sending to LLM
* Protect against prompt injection with harmful content
* Screen user inputs in conversational AI

=== Workflow Protection ===
* Add safety gates in multi-stage pipelines
* Prevent unsafe content from reaching production systems
* Log and monitor policy violations

== Performance Considerations ==

{| class="wikitable"
|+ Performance Characteristics
! Aspect !! Impact !! Notes
|-
| Latency || +100-300ms || Additional API call per moderation
|-
| Cost || Per-request || OpenAI charges for moderation API calls
|-
| Rate Limits || API-dependent || Subject to OpenAI rate limits
|-
| Async Support || Available || Full async support in openai>=1.0
|}

== Error Handling ==

<syntaxhighlight lang="python">
from langchain_classic.chains import OpenAIModerationChain

moderation = OpenAIModerationChain(error=True)

try:
    result = moderation.invoke({"input": user_input})
    # Process safe content
    process_content(result["output"])

except ValueError as e:
    # Content policy violation
    if "violates OpenAI's content policy" in str(e):
        log_violation(user_input)
        notify_user("Your message was flagged for policy violation")

except ImportError as e:
    # OpenAI package not installed
    print("Please install openai: pip install openai")

except Exception as e:
    # API errors, network issues, etc.
    log_error(f"Moderation error: {e}")
    # Fail safe or allow content?
</syntaxhighlight>

== Best Practices ==

=== Safety First ===
* Use <code>error=True</code> for critical applications
* Always moderate user inputs before processing
* Log all moderation failures for review

=== Performance ===
* Moderate asynchronously when possible
* Batch moderate if processing multiple inputs
* Cache results for repeated content

=== User Experience ===
* Provide clear feedback when content is flagged
* Allow users to rephrase flagged content
* Document content policy for users

=== Monitoring ===
* Track moderation failure rates
* Monitor for false positives
* Review flagged content regularly

== Limitations ==

=== API Dependency ===
* Requires OpenAI API access
* Subject to API availability and rate limits
* Incurs cost per moderation check

=== Language Support ===
* Primarily optimized for English
* May have reduced accuracy for other languages

=== False Positives/Negatives ===
* May flag benign content (false positive)
* May miss subtle policy violations (false negative)
* Requires human review for edge cases

== Configuration Example ==

<syntaxhighlight lang="python">
from langchain_classic.chains import OpenAIModerationChain

# Production configuration
moderation = OpenAIModerationChain(
    error=True,  # Strict mode: raise errors
    input_key="user_message",
    output_key="safe_message",
    openai_api_key="sk-...",  # Or from env var
    openai_organization="org-...",  # Optional
)

# Development configuration
moderation_dev = OpenAIModerationChain(
    error=False,  # Warning mode: return error message
    input_key="input",
    output_key="output",
)
</syntaxhighlight>

== Related Components ==

* '''Chain''' (langchain-classic) - Base chain class
* '''OpenAI Moderation API''' - Underlying moderation service
* '''CallbackManager''' (langchain-core) - Chain execution callbacks
* '''LLMChain''' (langchain-classic) - Can be combined with moderation

== See Also ==

* [[langchain-ai_langchain_Chain|Chain]] - Base chain class
* [[langchain-ai_langchain_LLMChain|LLMChain]] - LLM chain for text generation
* OpenAI Moderation API documentation
* Content safety best practices for AI applications
