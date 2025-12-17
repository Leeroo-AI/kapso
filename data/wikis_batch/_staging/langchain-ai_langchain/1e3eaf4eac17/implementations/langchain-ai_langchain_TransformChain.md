= TransformChain Implementation =

== Metadata ==
{| class="wikitable"
|-
! Field !! Value
|-
| Source File || <code>/tmp/praxium_repo_wjjl6pl8/libs/langchain/langchain_classic/chains/transform.py</code>
|-
| Class Name || <code>TransformChain</code>
|-
| Package || <code>langchain_classic.chains</code>
|-
| Extends || <code>Chain</code>
|-
| Lines of Code || 79
|-
| Status || Approved
|}

== Overview ==
<code>TransformChain</code> is a simple chain wrapper that executes an arbitrary Python function to transform input data into output data. It allows users to inject custom data transformation logic into a chain pipeline without involving language models or external services. The chain supports both synchronous and asynchronous execution modes.

This implementation is particularly useful for:
* Preprocessing data before passing it to another chain
* Postprocessing data after receiving it from another chain
* Applying custom business logic transformations
* Extracting or reformatting data structures

== Code Reference ==

=== Source Location ===
<syntaxhighlight lang="text">
File: /tmp/praxium_repo_wjjl6pl8/libs/langchain/langchain_classic/chains/transform.py
Lines: 20-79
</syntaxhighlight>

=== Class Signature ===
<syntaxhighlight lang="python">
class TransformChain(Chain):
    """Chain that transforms the chain output.

    Example:
        ```python
        from langchain_classic.chains import TransformChain
        transform_chain = TransformChain(input_variables=["text"],
         output_variables["entities"], transform=func())

        ```
    """

    input_variables: list[str]
    """The keys expected by the transform's input dictionary."""

    output_variables: list[str]
    """The keys returned by the transform's output dictionary."""

    transform_cb: Callable[[dict[str, str]], dict[str, str]] = Field(alias="transform")
    """The transform function."""

    atransform_cb: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None = Field(
        None, alias="atransform"
    )
    """The async coroutine transform function."""
</syntaxhighlight>

=== Import Statement ===
<syntaxhighlight lang="python">
from langchain_classic.chains import TransformChain
</syntaxhighlight>

== I/O Contract ==

=== Constructor Parameters ===
{| class="wikitable"
|-
! Parameter !! Type !! Required !! Default !! Description
|-
| <code>input_variables</code> || <code>list[str]</code> || Yes || N/A || Keys expected by the transform's input dictionary
|-
| <code>output_variables</code> || <code>list[str]</code> || Yes || N/A || Keys returned by the transform's output dictionary
|-
| <code>transform</code> || <code>Callable[[dict[str, str]], dict[str, str]]</code> || Yes || N/A || The synchronous transform function (aliased to <code>transform_cb</code>)
|-
| <code>atransform</code> || <code>Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]</code> || No || <code>None</code> || The async coroutine transform function (aliased to <code>atransform_cb</code>)
|}

=== Key Methods ===
{| class="wikitable"
|-
! Method !! Parameters !! Returns !! Description
|-
| <code>_call</code> || <code>inputs: dict[str, str]</code><br/><code>run_manager: CallbackManagerForChainRun &#124; None</code> || <code>dict[str, str]</code> || Executes the synchronous transform function
|-
| <code>_acall</code> || <code>inputs: dict[str, Any]</code><br/><code>run_manager: AsyncCallbackManagerForChainRun &#124; None</code> || <code>dict[str, Any]</code> || Executes the async transform function, falls back to sync if not provided
|-
| <code>input_keys</code> || N/A (property) || <code>list[str]</code> || Returns the expected input keys
|-
| <code>output_keys</code> || N/A (property) || <code>list[str]</code> || Returns the output keys
|}

=== Input Format ===
{| class="wikitable"
|-
! Field !! Type !! Description
|-
| <code>inputs</code> || <code>dict[str, str]</code> or <code>dict[str, Any]</code> || Dictionary containing the keys specified in <code>input_variables</code>
|}

=== Output Format ===
{| class="wikitable"
|-
! Field !! Type !! Description
|-
| (return) || <code>dict[str, str]</code> or <code>dict[str, Any]</code> || Dictionary containing the keys specified in <code>output_variables</code>
|}

== Implementation Details ==

=== Core Logic ===
The <code>TransformChain</code> implementation is straightforward:

1. '''Synchronous Execution''' (<code>_call</code>): Directly invokes the <code>transform_cb</code> function with the input dictionary
2. '''Asynchronous Execution''' (<code>_acall</code>):
   * If <code>atransform_cb</code> is provided, awaits its execution
   * Otherwise, logs a warning (once, via LRU cache) and falls back to the synchronous <code>transform_cb</code>

=== Key Features ===
* '''Field Aliases''': The <code>transform</code> and <code>atransform</code> constructor parameters map to <code>transform_cb</code> and <code>atransform_cb</code> internal fields
* '''Logging Once''': Uses <code>@functools.lru_cache</code> on <code>_log_once</code> to ensure warnings about missing async transforms are only logged once
* '''Flexible Async Support''': Gracefully falls back to synchronous execution if async transform is not provided

== Usage Examples ==

=== Basic Text Transformation ===
<syntaxhighlight lang="python">
from langchain_classic.chains import TransformChain

def extract_words(inputs: dict[str, str]) -> dict[str, str]:
    """Extract words from text."""
    text = inputs["text"]
    words = text.split()
    return {"words": " ".join(words), "word_count": str(len(words))}

transform_chain = TransformChain(
    input_variables=["text"],
    output_variables=["words", "word_count"],
    transform=extract_words
)

result = transform_chain.run("Hello world from LangChain")
# result: {"words": "Hello world from LangChain", "word_count": "4"}
</syntaxhighlight>

=== JSON Parsing Example ===
<syntaxhighlight lang="python">
import json
from langchain_classic.chains import TransformChain

def parse_json_output(inputs: dict[str, str]) -> dict[str, str]:
    """Parse JSON string and extract fields."""
    json_str = inputs["json_string"]
    data = json.loads(json_str)
    return {
        "name": data.get("name", ""),
        "email": data.get("email", "")
    }

chain = TransformChain(
    input_variables=["json_string"],
    output_variables=["name", "email"],
    transform=parse_json_output
)

result = chain.run('{"name": "Alice", "email": "alice@example.com"}')
# result: {"name": "Alice", "email": "alice@example.com"}
</syntaxhighlight>

=== Async Transformation ===
<syntaxhighlight lang="python">
import asyncio
from langchain_classic.chains import TransformChain

def sync_transform(inputs: dict[str, str]) -> dict[str, str]:
    text = inputs["text"]
    return {"upper": text.upper()}

async def async_transform(inputs: dict[str, Any]) -> dict[str, Any]:
    """Async transformation with delay."""
    await asyncio.sleep(0.1)
    text = inputs["text"]
    return {"upper": text.upper()}

# Chain with both sync and async transforms
chain = TransformChain(
    input_variables=["text"],
    output_variables=["upper"],
    transform=sync_transform,
    atransform=async_transform
)

# Async execution will use async_transform
result = await chain.acall({"text": "hello"})
# result: {"upper": "HELLO"}
</syntaxhighlight>

=== Chaining with Other Chains ===
<syntaxhighlight lang="python">
from langchain_classic.chains import TransformChain, SimpleSequentialChain, LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

# First: Transform input
def prepare_input(inputs: dict[str, str]) -> dict[str, str]:
    return {"formatted": f"Question: {inputs['question']}"}

transform_chain = TransformChain(
    input_variables=["question"],
    output_variables=["formatted"],
    transform=prepare_input
)

# Second: LLM processing
llm_chain = LLMChain(
    llm=OpenAI(),
    prompt=PromptTemplate(
        input_variables=["formatted"],
        template="{formatted}\nAnswer:"
    )
)

# Combine chains
overall_chain = SimpleSequentialChain(
    chains=[transform_chain, llm_chain]
)
</syntaxhighlight>

== Related Pages ==
* [[langchain-ai_langchain_Chain|Chain Base Class]] - Parent class providing chain abstraction
* [[langchain-ai_langchain_LLMChain|LLMChain]] - Chain for LLM invocations
* [[langchain-ai_langchain_SequentialChain|SequentialChain]] - Chain for sequential execution
* [[langchain-ai_langchain_load_chain|load_chain]] - Loading chains from configuration

== Notes ==
* The transform function should be pure and deterministic for consistent results
* For async execution, provide an <code>atransform</code> function to avoid blocking the event loop
* Input and output dictionaries must contain exactly the keys specified in <code>input_variables</code> and <code>output_variables</code>
* This chain does not invoke any LLMs or external services - it's purely for data transformation
* Consider using this chain for preprocessing/postprocessing in larger chain pipelines

== See Also ==
* [https://docs.langchain.com/docs/modules/chains/foundational/transform LangChain Transform Chain Documentation]
* [https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain_classic/chains/transform.py Source Code on GitHub]
