---
title: SequentialChain and SimpleSequentialChain
type: implementation
project: langchain-ai/langchain
file: libs/langchain/langchain_classic/chains/sequential.py
category: chain_composition
---

= SequentialChain and SimpleSequentialChain Implementation =

== Overview ==

'''SequentialChain''' and '''SimpleSequentialChain''' enable pipeline-style composition where the outputs of one chain feed directly into the next. These classes are part of the legacy chain system and provide a way to build multi-stage processing workflows.

* '''SequentialChain''': Flexible chain supporting multiple inputs/outputs per stage
* '''SimpleSequentialChain''': Simplified version where each chain has single input/output

== Code Reference ==

=== Source Location ===
<syntaxhighlight lang="text">
File: /tmp/praxium_repo_wjjl6pl8/libs/langchain/langchain_classic/chains/sequential.py
Lines: 1-209
Package: langchain-classic
</syntaxhighlight>

=== Class Signatures ===
<syntaxhighlight lang="python">
class SequentialChain(Chain):
    """Chain where the outputs of one chain feed directly into next."""

class SimpleSequentialChain(Chain):
    """Simple chain where the outputs of one step feed directly into next."""
</syntaxhighlight>

=== Import Statement ===
<syntaxhighlight lang="python">
from langchain_classic.chains.sequential import SequentialChain, SimpleSequentialChain
</syntaxhighlight>

== SequentialChain ==

=== Class Attributes ===

{| class="wikitable"
|+ Required Attributes
! Attribute !! Type !! Description
|-
| chains || <code>list[Chain]</code> || List of chains to execute sequentially
|-
| input_variables || <code>list[str]</code> || Initial input keys required by the chain
|-
| output_variables || <code>list[str]</code> || Output keys to return from the chain
|}

{| class="wikitable"
|+ Optional Attributes
! Attribute !! Type !! Default !! Description
|-
| return_all || <code>bool</code> || False || Return all intermediate outputs or just output_variables
|}

=== Input/Output Contract ===

{| class="wikitable"
|+ Input Schema
! Key !! Type !! Description
|-
| <code>input_variables</code> || Various || Keys specified in input_variables list
|-
| Memory keys || Various || Additional keys from memory if configured
|}

{| class="wikitable"
|+ Output Schema
! Key !! Type !! Description
|-
| <code>output_variables</code> || Various || Keys specified in output_variables list
|-
| Intermediate outputs || Various || If return_all=True, all generated keys
|}

=== Usage Examples ===

==== Basic Sequential Processing ====

<syntaxhighlight lang="python">
from langchain_classic.chains import LLMChain, SequentialChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

llm = ChatOpenAI()

# Chain 1: Generate a topic
topic_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template="Given the category '{category}', suggest a specific topic.",
        input_variables=["category"]
    ),
    output_key="topic"
)

# Chain 2: Write an outline
outline_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template="Create an outline for a blog post about '{topic}'.",
        input_variables=["topic"]
    ),
    output_key="outline"
)

# Chain 3: Write introduction
intro_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template="Write an introduction based on this outline:\n{outline}",
        input_variables=["outline"]
    ),
    output_key="introduction"
)

# Compose sequential chain
overall_chain = SequentialChain(
    chains=[topic_chain, outline_chain, intro_chain],
    input_variables=["category"],
    output_variables=["topic", "outline", "introduction"]
)

# Execute
result = overall_chain.invoke({"category": "machine learning"})

print("Topic:", result["topic"])
print("Outline:", result["outline"])
print("Introduction:", result["introduction"])
</syntaxhighlight>

==== With Multiple Initial Inputs ====

<syntaxhighlight lang="python">
from langchain_classic.chains import LLMChain, SequentialChain

# Chain 1: Uses both inputs
analysis_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template="Analyze {product} in the {market} market.",
        input_variables=["product", "market"]
    ),
    output_key="analysis"
)

# Chain 2: Uses analysis and original product
recommendations_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template="Given the analysis: {analysis}\nProvide recommendations for {product}.",
        input_variables=["analysis", "product"]
    ),
    output_key="recommendations"
)

# Compose with multiple inputs
chain = SequentialChain(
    chains=[analysis_chain, recommendations_chain],
    input_variables=["product", "market"],
    output_variables=["recommendations"]
)

result = chain.invoke({
    "product": "smartphone",
    "market": "enterprise"
})

print(result["recommendations"])
</syntaxhighlight>

==== With return_all ====

<syntaxhighlight lang="python">
from langchain_classic.chains import SequentialChain

# Return all intermediate outputs
chain = SequentialChain(
    chains=[chain1, chain2, chain3],
    input_variables=["initial_input"],
    return_all=True  # Returns all outputs, not just final ones
)

result = chain.invoke({"initial_input": "value"})

# Result contains all intermediate outputs
print(result.keys())  # Shows all output keys from all chains
</syntaxhighlight>

==== With Memory ====

<syntaxhighlight lang="python">
from langchain_classic.chains import SequentialChain
from langchain_classic.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="history"
)

chain = SequentialChain(
    chains=[chain1, chain2],
    input_variables=["user_input"],
    output_variables=["final_output"],
    memory=memory
)

# Memory variables automatically available to chains
result = chain.invoke({"user_input": "Hello"})
</syntaxhighlight>

=== Validation ===

SequentialChain performs extensive validation:

<syntaxhighlight lang="python">
@model_validator(mode="before")
@classmethod
def validate_chains(cls, values: dict) -> Any:
    """Validate that the correct inputs exist for all chains."""
    chains = values["chains"]
    input_variables = values["input_variables"]

    # Track known variables
    known_variables = set(input_variables)

    # Check each chain
    for chain in chains:
        # Verify all input keys are available
        missing_vars = set(chain.input_keys).difference(known_variables)
        if missing_vars:
            raise ValueError(f"Missing required input keys: {missing_vars}")

        # Check for overlapping keys
        overlapping_keys = known_variables.intersection(chain.output_keys)
        if overlapping_keys:
            raise ValueError(f"Chain returned keys that already exist: {overlapping_keys}")

        # Add new output keys to known variables
        known_variables |= set(chain.output_keys)

    # Validate output_variables exist
    if "output_variables" in values:
        missing_vars = set(values["output_variables"]).difference(known_variables)
        if missing_vars:
            raise ValueError(f"Expected output variables not found: {missing_vars}")

    return values
</syntaxhighlight>

{| class="wikitable"
|+ Validation Rules
! Rule !! Description
|-
| Input availability || Each chain's inputs must be available from previous chains or initial inputs
|-
| No key conflicts || Chain outputs cannot overwrite existing keys
|-
| Output existence || output_variables must be produced by chains
|-
| Memory separation || Memory keys cannot overlap with input_variables
|}

=== Execution Flow ===

<syntaxhighlight lang="python">
def _call(
    self,
    inputs: dict[str, str],
    run_manager: CallbackManagerForChainRun | None = None,
) -> dict[str, str]:
    known_values = inputs.copy()

    # Execute each chain sequentially
    for chain in self.chains:
        outputs = chain(known_values, return_only_outputs=True)
        known_values.update(outputs)  # Accumulate outputs

    # Return only requested output variables
    return {k: known_values[k] for k in self.output_variables}
</syntaxhighlight>

== SimpleSequentialChain ==

=== Class Attributes ===

{| class="wikitable"
|+ Required Attributes
! Attribute !! Type !! Description
|-
| chains || <code>list[Chain]</code> || List of chains with single input/output each
|}

{| class="wikitable"
|+ Optional Attributes
! Attribute !! Type !! Default !! Description
|-
| strip_outputs || <code>bool</code> || False || Strip whitespace from outputs before passing to next chain
|-
| input_key || <code>str</code> || "input" || Key for initial input
|-
| output_key || <code>str</code> || "output" || Key for final output
|}

=== Usage Examples ===

==== Basic Linear Pipeline ====

<syntaxhighlight lang="python">
from langchain_classic.chains import LLMChain, SimpleSequentialChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

llm = ChatOpenAI()

# Each chain has single input/output
chain1 = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template="Write a one-sentence synopsis of: {input}",
        input_variables=["input"]
    )
)

chain2 = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template="Translate to Spanish: {input}",
        input_variables=["input"]
    )
)

chain3 = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template="Make this more formal: {input}",
        input_variables=["input"]
    )
)

# Compose simple pipeline
pipeline = SimpleSequentialChain(
    chains=[chain1, chain2, chain3]
)

# Execute - output of each feeds into next
result = pipeline.invoke({
    "input": "A story about a brave knight"
})

print(result["output"])
</syntaxhighlight>

==== With Output Stripping ====

<syntaxhighlight lang="python">
from langchain_classic.chains import SimpleSequentialChain

# Strip whitespace between chains
pipeline = SimpleSequentialChain(
    chains=[chain1, chain2, chain3],
    strip_outputs=True  # Removes leading/trailing whitespace
)

result = pipeline.invoke({"input": "Initial text"})
</syntaxhighlight>

==== Text Processing Pipeline ====

<syntaxhighlight lang="python">
from langchain_classic.chains import SimpleSequentialChain, TransformChain, LLMChain

# Simple text transformation pipeline
clean_chain = TransformChain(
    input_variables=["input"],
    output_variables=["output"],
    transform=lambda x: {"output": x["input"].strip().lower()}
)

summarize_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template="Summarize: {input}",
        input_variables=["input"]
    )
)

sentiment_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template="What is the sentiment of: {input}",
        input_variables=["input"]
    )
)

pipeline = SimpleSequentialChain(
    chains=[clean_chain, summarize_chain, sentiment_chain],
    strip_outputs=True
)

result = pipeline.invoke({
    "input": "  Raw user input with extra spaces  "
})

print(result["output"])  # Cleaned -> Summarized -> Sentiment
</syntaxhighlight>

=== Validation ===

<syntaxhighlight lang="python">
@model_validator(mode="after")
def validate_chains(self) -> Self:
    """Validate that chains are all single input/output."""
    for chain in self.chains:
        if len(chain.input_keys) != 1:
            raise ValueError(
                f"Chains used in SimplePipeline should all have one input, got "
                f"{chain} with {len(chain.input_keys)} inputs."
            )
        if len(chain.output_keys) != 1:
            raise ValueError(
                f"Chains used in SimplePipeline should all have one output, got "
                f"{chain} with {len(chain.output_keys)} outputs."
            )
    return self
</syntaxhighlight>

=== Execution Flow ===

<syntaxhighlight lang="python">
def _call(
    self,
    inputs: dict[str, str],
    run_manager: CallbackManagerForChainRun | None = None,
) -> dict[str, str]:
    _input = inputs[self.input_key]

    # Execute chains sequentially
    for i, chain in enumerate(self.chains):
        _input = chain.run(_input)

        # Strip whitespace if configured
        if self.strip_outputs:
            _input = _input.strip()

        # Log intermediate output for observability
        _run_manager.on_text(_input, color=color_mapping[str(i)], end="\n")

    return {self.output_key: _input}
</syntaxhighlight>

== Comparison ==

{| class="wikitable"
|+ SequentialChain vs SimpleSequentialChain
! Feature !! SequentialChain !! SimpleSequentialChain
|-
| Input/Output || Multiple keys per chain || Single key per chain
|-
| Complexity || Flexible, complex workflows || Simple linear pipelines
|-
| Validation || Extensive input/output checking || Simple single I/O check
|-
| Use Case || Multi-variable workflows || Linear transformations
|-
| Configuration || More verbose || Simpler setup
|-
| Observability || Less detailed || Color-coded intermediate outputs
|}

== Common Use Cases ==

=== SequentialChain ===
* Multi-stage document processing
* Complex data transformation pipelines
* Workflows with branching logic (through input/output keys)
* Business process automation

=== SimpleSequentialChain ===
* Text refinement pipelines (edit, translate, format)
* Sequential LLM calls (expand, then summarize, then extract)
* Data cleaning and preprocessing
* Simple multi-step transformations

== Configuration ==

<syntaxhighlight lang="python">
model_config = ConfigDict(
    arbitrary_types_allowed=True,
    extra="forbid",
)
</syntaxhighlight>

Both chains forbid extra attributes and allow arbitrary types (for Chain objects).

== Async Support ==

Both classes support async execution:

<syntaxhighlight lang="python">
import asyncio
from langchain_classic.chains import SequentialChain

async def run_pipeline():
    chain = SequentialChain(
        chains=[chain1, chain2, chain3],
        input_variables=["input"],
        output_variables=["output"]
    )

    result = await chain.ainvoke({"input": "Test"})
    return result

# Run async
result = asyncio.run(run_pipeline())
</syntaxhighlight>

== Performance Considerations ==

{| class="wikitable"
|+ Performance Characteristics
! Aspect !! Impact !! Notes
|-
| Execution || Sequential, not parallel || Each chain waits for previous to complete
|-
| Memory || Linear in chain count || Intermediate results stored in dictionary
|-
| Latency || Sum of all chain latencies || No optimization possible
|-
| Error propagation || Fails on first error || No built-in retry or recovery
|}

== Limitations ==

=== Sequential Execution ===
* No parallel execution of independent chains
* Cannot optimize by reordering chains
* Total latency is sum of all chains

=== Error Handling ===
* If any chain fails, entire pipeline fails
* No automatic retry logic
* Limited error recovery options

=== Modern Alternatives ===
* LCEL provides more flexible composition
* LangGraph enables parallel execution
* Better suited for simple use cases

== Migration to LCEL ==

Modern equivalent using LCEL:

<syntaxhighlight lang="python">
# Old: SequentialChain
from langchain_classic.chains import SequentialChain

seq_chain = SequentialChain(
    chains=[chain1, chain2, chain3],
    input_variables=["input"],
    output_variables=["output"]
)

# New: LCEL
from langchain_core.runnables import RunnablePassthrough

# Simple sequential
lcel_chain = chain1 | chain2 | chain3

# With passthrough (preserving intermediate values)
lcel_chain = (
    RunnablePassthrough.assign(step1=chain1)
    .assign(step2=chain2)
    .assign(step3=chain3)
)
</syntaxhighlight>

== Related Components ==

* '''Chain''' (langchain-classic) - Base class
* '''LLMChain''' (langchain-classic) - Common component in pipelines
* '''TransformChain''' (langchain-classic) - Non-LLM transformations
* '''RunnablePassthrough''' (langchain-core) - LCEL alternative
* '''CallbackManager''' (langchain-core) - Execution callbacks

== See Also ==

* [[langchain-ai_langchain_Chain|Chain]] - Base chain class
* [[langchain-ai_langchain_LLMChain|LLMChain]] - LLM chain component
* [[langchain-ai_langchain_RunnablePassthrough|RunnablePassthrough]] - LCEL primitive
* LangChain documentation on chain composition
* LCEL guide for modern composition patterns
