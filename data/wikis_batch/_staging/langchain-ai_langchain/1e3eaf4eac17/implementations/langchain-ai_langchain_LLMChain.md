{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai/langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::LangChain]], [[domain::LLM]], [[domain::Prompts]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==
Deprecated chain implementation that formats a prompt template and sends it to a language model for completion.

=== Description ===
LLMChain is a legacy Chain implementation that combines a prompt template with a language model. It formats input variables into the prompt, sends the formatted prompt to the LLM, and optionally parses the output. While deprecated since version 0.1.17 in favor of the more modern RunnableSequence pattern (prompt | llm), LLMChain remains in the codebase for backward compatibility and is still widely used in existing code.

The class supports both traditional BaseLanguageModel instances and modern Runnable instances, handles batch generation for multiple inputs simultaneously, supports output parsing, and integrates with the Chain callback system for observability. It can work with both text completion LLMs and chat models, automatically detecting the model type and handling responses appropriately.

=== Usage ===
Use LLMChain when maintaining legacy code that relies on the Chain interface. For new code, prefer the modern pattern: `prompt | llm | output_parser`. LLMChain is particularly useful when you need backward compatibility, are working with existing Chain-based applications, or need the convenience methods like predict() and apply().

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai/langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/chains/llm.py libs/langchain/langchain_classic/chains/llm.py]
* '''Lines:''' 1-433

=== Signature ===
<syntaxhighlight lang="python">
@deprecated(
    since="0.1.17",
    alternative="RunnableSequence, e.g., `prompt | llm`",
    removal="1.0",
)
class LLMChain(Chain):
    prompt: BasePromptTemplate
    llm: Runnable[LanguageModelInput, str] | Runnable[LanguageModelInput, BaseMessage]
    output_key: str = "text"
    output_parser: BaseLLMOutputParser = Field(default_factory=StrOutputParser)
    return_final_only: bool = True
    llm_kwargs: dict = Field(default_factory=dict)
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Deprecated import
from langchain_classic.chains import LLMChain

# Modern alternative
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
prompt = PromptTemplate(...)
chain = prompt | llm | StrOutputParser()
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| prompt || BasePromptTemplate || Yes || Template for formatting inputs into prompts
|-
| llm || Runnable || Yes || Language model or chat model to query
|-
| output_key || str || No || Key name for output in result dict (default: "text")
|-
| output_parser || BaseLLMOutputParser || No || Parser for LLM output (default: StrOutputParser)
|-
| return_final_only || bool || No || If True, only return parsed output; if False, include full generation info
|-
| llm_kwargs || dict || No || Additional keyword arguments to pass to LLM
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| output_key || str || Parsed LLM response (default key: "text")
|-
| full_generation || LLMResult || Complete generation info (only if return_final_only=False)
|}

== Properties ==

=== input_keys ===
<syntaxhighlight lang="python">
@property
def input_keys(self) -> list[str]:
    """Will be whatever keys the prompt expects."""
    return self.prompt.input_variables
</syntaxhighlight>

Dynamically determined from the prompt template's input variables.

=== output_keys ===
<syntaxhighlight lang="python">
@property
def output_keys(self) -> list[str]:
    """Will always return text key."""
    if self.return_final_only:
        return [self.output_key]
    return [self.output_key, "full_generation"]
</syntaxhighlight>

Returns either just the output_key or both output_key and full_generation.

== Core Methods ==

=== _call ===
<syntaxhighlight lang="python">
def _call(
    self,
    inputs: dict[str, Any],
    run_manager: CallbackManagerForChainRun | None = None,
) -> dict[str, str]:
    response = self.generate([inputs], run_manager=run_manager)
    return self.create_outputs(response)[0]
</syntaxhighlight>

Implements the Chain._call abstract method. Generates a response for a single input and returns formatted outputs.

=== generate ===
<syntaxhighlight lang="python">
def generate(
    self,
    input_list: list[dict[str, Any]],
    run_manager: CallbackManagerForChainRun | None = None,
) -> LLMResult:
    """Generate LLM result from inputs."""
    prompts, stop = self.prep_prompts(input_list, run_manager=run_manager)
    callbacks = run_manager.get_child() if run_manager else None

    if isinstance(self.llm, BaseLanguageModel):
        return self.llm.generate_prompt(
            prompts,
            stop,
            callbacks=callbacks,
            **self.llm_kwargs,
        )

    results = self.llm.bind(stop=stop, **self.llm_kwargs).batch(
        cast("list", prompts),
        {"callbacks": callbacks},
    )
    generations: list[list[Generation]] = []
    for res in results:
        if isinstance(res, BaseMessage):
            generations.append([ChatGeneration(message=res)])
        else:
            generations.append([Generation(text=res)])
    return LLMResult(generations=generations)
</syntaxhighlight>

Handles both legacy BaseLanguageModel and modern Runnable LLMs. Batches multiple inputs for efficiency.

=== prep_prompts ===
<syntaxhighlight lang="python">
def prep_prompts(
    self,
    input_list: list[dict[str, Any]],
    run_manager: CallbackManagerForChainRun | None = None,
) -> tuple[list[PromptValue], list[str] | None]:
    """Prepare prompts from inputs."""
    stop = None
    if len(input_list) == 0:
        return [], stop
    if "stop" in input_list[0]:
        stop = input_list[0]["stop"]

    prompts = []
    for inputs in input_list:
        selected_inputs = {k: inputs[k] for k in self.prompt.input_variables}
        prompt = self.prompt.format_prompt(**selected_inputs)
        _colored_text = get_colored_text(prompt.to_string(), "green")
        _text = "Prompt after formatting:\n" + _colored_text
        if run_manager:
            run_manager.on_text(_text, end="\n", verbose=self.verbose)
        if "stop" in inputs and inputs["stop"] != stop:
            raise ValueError("If `stop` is present in any inputs, should be present in all.")
        prompts.append(prompt)
    return prompts, stop
</syntaxhighlight>

Formats prompt templates with input variables and extracts stop sequences. Logs formatted prompts to callbacks.

=== create_outputs ===
<syntaxhighlight lang="python">
def create_outputs(self, llm_result: LLMResult) -> list[dict[str, Any]]:
    """Create outputs from response."""
    result = [
        {
            self.output_key: self.output_parser.parse_result(generation),
            "full_generation": generation,
        }
        for generation in llm_result.generations
    ]
    if self.return_final_only:
        result = [{self.output_key: r[self.output_key]} for r in result]
    return result
</syntaxhighlight>

Parses LLM results through the output_parser and formats into output dictionaries.

== Convenience Methods ==

=== predict ===
<syntaxhighlight lang="python">
def predict(self, callbacks: Callbacks = None, **kwargs: Any) -> str:
    """Format prompt with kwargs and pass to LLM.

    Args:
        callbacks: Callbacks to pass to LLMChain
        **kwargs: Keys to pass to prompt template.

    Returns:
        Completion from LLM.

    Example:
        ```python
        completion = llm.predict(adjective="funny")
        ```
    """
    return self(kwargs, callbacks=callbacks)[self.output_key]
</syntaxhighlight>

Convenient method for single predictions with keyword arguments.

=== apply ===
<syntaxhighlight lang="python">
def apply(
    self,
    input_list: list[dict[str, Any]],
    callbacks: Callbacks = None,
) -> list[dict[str, str]]:
    """Utilize the LLM generate method for speed gains."""
    callback_manager = CallbackManager.configure(
        callbacks,
        self.callbacks,
        self.verbose,
    )
    run_manager = callback_manager.on_chain_start(
        None,
        {"input_list": input_list},
        name=self.get_name(),
    )
    try:
        response = self.generate(input_list, run_manager=run_manager)
    except BaseException as e:
        run_manager.on_chain_error(e)
        raise
    outputs = self.create_outputs(response)
    run_manager.on_chain_end({"outputs": outputs})
    return outputs
</syntaxhighlight>

Processes multiple inputs in a single batch for efficiency.

=== predict_and_parse (Deprecated) ===
<syntaxhighlight lang="python">
def predict_and_parse(
    self,
    callbacks: Callbacks = None,
    **kwargs: Any,
) -> str | list[str] | dict[str, Any]:
    """Call predict and then parse the results."""
    warnings.warn(
        "The predict_and_parse method is deprecated, "
        "instead pass an output parser directly to LLMChain.",
        stacklevel=2,
    )
    result = self.predict(callbacks=callbacks, **kwargs)
    if self.prompt.output_parser is not None:
        return self.prompt.output_parser.parse(result)
    return result
</syntaxhighlight>

Deprecated method for parsing. Use output_parser parameter instead.

== Async Support ==

=== agenerate ===
<syntaxhighlight lang="python">
async def agenerate(
    self,
    input_list: list[dict[str, Any]],
    run_manager: AsyncCallbackManagerForChainRun | None = None,
) -> LLMResult:
    """Generate LLM result from inputs."""
    prompts, stop = await self.aprep_prompts(input_list, run_manager=run_manager)
    callbacks = run_manager.get_child() if run_manager else None

    if isinstance(self.llm, BaseLanguageModel):
        return await self.llm.agenerate_prompt(
            prompts,
            stop,
            callbacks=callbacks,
            **self.llm_kwargs,
        )

    results = await self.llm.bind(stop=stop, **self.llm_kwargs).abatch(
        cast("list", prompts),
        {"callbacks": callbacks},
    )
    generations: list[list[Generation]] = []
    for res in results:
        if isinstance(res, BaseMessage):
            generations.append([ChatGeneration(message=res)])
        else:
            generations.append([Generation(text=res)])
    return LLMResult(generations=generations)
</syntaxhighlight>

Async version of generate with full async/await support.

=== apredict ===
<syntaxhighlight lang="python">
async def apredict(self, callbacks: Callbacks = None, **kwargs: Any) -> str:
    """Format prompt with kwargs and pass to LLM.

    Example:
        ```python
        completion = await llm.apredict(adjective="funny")
        ```
    """
    return (await self.acall(kwargs, callbacks=callbacks))[self.output_key]
</syntaxhighlight>

Async version of predict.

== Helper Functions ==

=== _get_language_model ===
<syntaxhighlight lang="python">
def _get_language_model(llm_like: Runnable) -> BaseLanguageModel:
    """Extract BaseLanguageModel from various Runnable wrappers."""
    if isinstance(llm_like, BaseLanguageModel):
        return llm_like
    if isinstance(llm_like, RunnableBinding):
        return _get_language_model(llm_like.bound)
    if isinstance(llm_like, RunnableWithFallbacks):
        return _get_language_model(llm_like.runnable)
    if isinstance(llm_like, (RunnableBranch, DynamicRunnable)):
        return _get_language_model(llm_like.default)
    raise ValueError(
        f"Unable to extract BaseLanguageModel from llm_like object of type "
        f"{type(llm_like)}"
    )
</syntaxhighlight>

Unwraps Runnable wrappers to find the underlying BaseLanguageModel.

== Factory Methods ==

=== from_string ===
<syntaxhighlight lang="python">
@classmethod
def from_string(cls, llm: BaseLanguageModel, template: str) -> LLMChain:
    """Create LLMChain from LLM and template."""
    prompt_template = PromptTemplate.from_template(template)
    return cls(llm=llm, prompt=prompt_template)
</syntaxhighlight>

Convenience factory for creating chains from template strings.

== Usage Examples ==

=== Basic Usage (Deprecated Pattern) ===
<syntaxhighlight lang="python">
from langchain_classic.chains import LLMChain
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

# Create prompt template
prompt = PromptTemplate(
    input_variables=["adjective"],
    template="Tell me a {adjective} joke"
)

# Create LLM
llm = OpenAI()

# Create chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run chain
result = chain.invoke({"adjective": "funny"})
print(result["text"])
</syntaxhighlight>

=== Modern Alternative (Recommended) ===
<syntaxhighlight lang="python">
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAI

# Create prompt template
prompt = PromptTemplate(
    input_variables=["adjective"],
    template="Tell me a {adjective} joke"
)

# Create chain using pipe operator
chain = prompt | OpenAI() | StrOutputParser()

# Run chain
result = chain.invoke({"adjective": "funny"})
print(result)
</syntaxhighlight>

=== Batch Processing ===
<syntaxhighlight lang="python">
from langchain_classic.chains import LLMChain
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a haiku about {topic}"
)

chain = LLMChain(llm=OpenAI(), prompt=prompt)

# Process multiple inputs at once
inputs = [
    {"topic": "nature"},
    {"topic": "technology"},
    {"topic": "ocean"}
]
results = chain.apply(inputs)

for result in results:
    print(result["text"])
</syntaxhighlight>

=== With Output Parser ===
<syntaxhighlight lang="python">
from langchain_classic.chains import LLMChain
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser

prompt = PromptTemplate(
    input_variables=["count"],
    template="List {count} random colors, separated by commas"
)

parser = CommaSeparatedListOutputParser()

chain = LLMChain(
    llm=OpenAI(),
    prompt=prompt,
    output_parser=parser
)

result = chain.invoke({"count": "5"})
print(result["text"])  # ['red', 'blue', 'green', 'yellow', 'purple']
</syntaxhighlight>

=== Async Usage ===
<syntaxhighlight lang="python">
import asyncio
from langchain_classic.chains import LLMChain
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

async def main():
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Explain {topic} in one sentence"
    )

    chain = LLMChain(llm=OpenAI(), prompt=prompt)

    # Async single prediction
    result = await chain.ainvoke({"topic": "quantum computing"})
    print(result["text"])

    # Async batch processing
    inputs = [
        {"topic": "AI"},
        {"topic": "blockchain"},
        {"topic": "cloud computing"}
    ]
    results = await chain.aapply(inputs)
    for result in results:
        print(result["text"])

asyncio.run(main())
</syntaxhighlight>

=== With Stop Sequences ===
<syntaxhighlight lang="python">
from langchain_classic.chains import LLMChain
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["question"],
    template="Q: {question}\nA:"
)

chain = LLMChain(llm=OpenAI(), prompt=prompt)

# Stop generation at newline
result = chain.invoke({"question": "What is the capital of France?", "stop": ["\n"]})
print(result["text"])
</syntaxhighlight>

=== Predict Convenience Method ===
<syntaxhighlight lang="python">
from langchain_classic.chains import LLMChain
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["product", "feature"],
    template="Write a tagline for {product} highlighting its {feature}"
)

chain = LLMChain(llm=OpenAI(), prompt=prompt)

# Use predict with kwargs instead of dict
tagline = chain.predict(product="smartphone", feature="camera quality")
print(tagline)
</syntaxhighlight>

=== From String Factory ===
<syntaxhighlight lang="python">
from langchain_classic.chains import LLMChain
from langchain_openai import OpenAI

# Create chain directly from template string
chain = LLMChain.from_string(
    llm=OpenAI(),
    template="Translate '{text}' to {language}"
)

result = chain.invoke({"text": "hello", "language": "Spanish"})
print(result["text"])
</syntaxhighlight>

== Related Pages ==
* [[extends::Implementation:langchain-ai_langchain_Chain]]
* [[uses::Concept:PromptTemplate]]
* [[uses::Concept:OutputParser]]
* [[deprecated_in_favor_of::Concept:RunnableSequence]]
