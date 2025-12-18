{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain Docs|https://docs.langchain.com]]
|-
! Domains
| [[domain::Chains]], [[domain::LLM_Orchestration]], [[domain::Deprecated]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Deprecated `LLMChain` class that formats prompts and calls language models, now replaced by LangChain Expression Language (LCEL) with the `prompt | llm` pattern.

=== Description ===

The `LLMChain` class was the original way to combine a prompt template with a language model in LangChain. It handles prompt formatting, LLM invocation, output parsing, and batch processing. The class is deprecated since version 0.1.17 in favor of LCEL's composable `RunnableSequence` pattern which offers better flexibility, streaming, and async support.

=== Usage ===

This class should only be used for maintaining legacy code. For new code, use LCEL: `chain = prompt | llm | output_parser` instead of `LLMChain(llm=llm, prompt=prompt)`.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/chains/llm.py libs/langchain/langchain_classic/chains/llm.py]
* '''Lines:''' 1-432

=== Signature ===
<syntaxhighlight lang="python">
@deprecated(
    since="0.1.17",
    alternative="RunnableSequence, e.g., `prompt | llm`",
    removal="1.0",
)
class LLMChain(Chain):
    """Chain to run queries against LLMs.

    Attributes:
        prompt: Prompt template to use.
        llm: Language model to call.
        output_key: Key for the output in returned dict.
        output_parser: Parser for LLM output.
        return_final_only: Whether to return only parsed result.
        llm_kwargs: Additional kwargs for LLM calls.
    """

    prompt: BasePromptTemplate
    llm: Runnable[LanguageModelInput, str] | Runnable[LanguageModelInput, BaseMessage]
    output_key: str = "text"
    output_parser: BaseLLMOutputParser = Field(default_factory=StrOutputParser)
    return_final_only: bool = True
    llm_kwargs: dict = Field(default_factory=dict)

    def generate(
        self,
        input_list: list[dict[str, Any]],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> LLMResult:
        """Generate LLM result from inputs."""

    def predict(self, callbacks: Callbacks = None, **kwargs: Any) -> str:
        """Format prompt with kwargs and pass to LLM."""

    @classmethod
    def from_string(cls, llm: BaseLanguageModel, template: str) -> LLMChain:
        """Create LLMChain from LLM and template string."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Legacy (deprecated)
from langchain_classic.chains import LLMChain

# Modern replacement - no import needed, use LCEL
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| inputs || dict[str, Any] || Yes || Dictionary with keys matching prompt.input_variables
|-
| stop || list[str] || No || Stop sequences for LLM generation
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| text || str || Parsed LLM output (default key)
|-
| full_generation || list[Generation] || Raw generation (if return_final_only=False)
|}

== Usage Examples ==

=== Legacy LLMChain Usage (Deprecated) ===
<syntaxhighlight lang="python">
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

# Old way - deprecated
prompt = PromptTemplate(
    input_variables=["adjective"],
    template="Tell me a {adjective} joke"
)
chain = LLMChain(llm=OpenAI(), prompt=prompt)

# Call the chain
result = chain.invoke({"adjective": "funny"})
# {"adjective": "funny", "text": "Why did the..."}

# Or use predict for direct string output
joke = chain.predict(adjective="funny")
</syntaxhighlight>

=== Modern LCEL Replacement ===
<syntaxhighlight lang="python">
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

# Modern way - use LCEL
prompt = PromptTemplate.from_template("Tell me a {adjective} joke")
model = OpenAI()
chain = prompt | model | StrOutputParser()

# Invoke the chain
result = chain.invoke({"adjective": "funny"})
# "Why did the chicken..."

# Streaming works naturally
for chunk in chain.stream({"adjective": "funny"}):
    print(chunk, end="")

# Async support
result = await chain.ainvoke({"adjective": "funny"})
</syntaxhighlight>

=== Batch Processing ===
<syntaxhighlight lang="python">
# Legacy batch
results = chain.apply([
    {"adjective": "funny"},
    {"adjective": "sad"},
    {"adjective": "scary"},
])

# Modern batch with LCEL
results = chain.batch([
    {"adjective": "funny"},
    {"adjective": "sad"},
    {"adjective": "scary"},
])
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
* [[uses_heuristic::Heuristic:langchain-ai_langchain_Warning_Deprecated_langchain_classic]]

