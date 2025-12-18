{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|OpenAI Moderation|https://platform.openai.com/docs/guides/moderation]]
|-
! Domains
| [[domain::Safety]], [[domain::Content_Moderation]], [[domain::Chains]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Chain class `OpenAIModerationChain` that validates text through OpenAI's content moderation API to detect policy violations.

=== Description ===

The `OpenAIModerationChain` passes text through OpenAI's moderation endpoint to check for content that violates OpenAI's usage policies (hate speech, violence, self-harm, etc.). It can either return an error message when violations are detected or raise an exception, making it useful as a safety filter in LLM pipelines.

=== Usage ===

Use this chain as a preprocessing or postprocessing step to filter unsafe content before sending to users or as input validation for user-submitted text.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/chains/moderation.py libs/langchain/langchain_classic/chains/moderation.py]
* '''Lines:''' 1-129

=== Signature ===
<syntaxhighlight lang="python">
class OpenAIModerationChain(Chain):
    """Pass input through OpenAI's moderation endpoint.

    Attributes:
        client: OpenAI client instance.
        async_client: Async OpenAI client instance.
        model_name: Moderation model to use (optional).
        error: Whether to raise exception on policy violation.
        input_key: Key for input text (default: "input").
        output_key: Key for output (default: "output").
        openai_api_key: API key for OpenAI.
    """

    client: Any = None
    async_client: Any = None
    model_name: str | None = None
    error: bool = False
    input_key: str = "input"
    output_key: str = "output"
    openai_api_key: str | None = None
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain_classic.chains import OpenAIModerationChain
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| input || str || Yes || Text to check for policy violations
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| output || str || Original text if safe, error message if flagged (when error=False)
|}

== Usage Examples ==

=== Basic Content Moderation ===
<syntaxhighlight lang="python">
from langchain_classic.chains import OpenAIModerationChain

# Create moderation chain (uses OPENAI_API_KEY env var)
moderation_chain = OpenAIModerationChain()

# Check safe content
result = moderation_chain.invoke({"input": "Hello, how are you today?"})
print(result["output"])  # "Hello, how are you today?"

# Check flagged content
result = moderation_chain.invoke({"input": "harmful content here..."})
print(result["output"])  # "Text was found that violates OpenAI's content policy."
</syntaxhighlight>

=== Raising Exceptions on Violations ===
<syntaxhighlight lang="python">
# Configure to raise exception instead of returning message
moderation_chain = OpenAIModerationChain(error=True)

try:
    result = moderation_chain.invoke({"input": "potentially harmful text"})
except ValueError as e:
    print(f"Content blocked: {e}")
    # Handle the violation appropriately
</syntaxhighlight>

=== Composing with Other Chains ===
<syntaxhighlight lang="python">
from langchain_classic.chains import SequentialChain, LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

# Moderate user input before processing
moderation = OpenAIModerationChain(error=True)

# Process with LLM
prompt = PromptTemplate.from_template("Respond helpfully to: {output}")
llm_chain = LLMChain(llm=OpenAI(), prompt=prompt)

# Chain: moderation -> LLM
# Modern approach: use LCEL
safe_chain = moderation | (lambda x: {"text": x["output"]}) | llm_chain
</syntaxhighlight>

=== Async Moderation ===
<syntaxhighlight lang="python">
# Async support for high-throughput applications
moderation_chain = OpenAIModerationChain()

async def check_content(text: str) -> str:
    result = await moderation_chain.ainvoke({"input": text})
    return result["output"]

# Batch check with asyncio
import asyncio
texts = ["text1", "text2", "text3"]
results = await asyncio.gather(*[check_content(t) for t in texts])
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]

