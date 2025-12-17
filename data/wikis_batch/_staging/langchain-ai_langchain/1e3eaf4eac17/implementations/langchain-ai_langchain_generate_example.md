---
title: generate_example
type: implementation
project: langchain-ai/langchain
file: libs/langchain/langchain_classic/chains/example_generator.py
category: utility_function
---

= generate_example Implementation =

== Overview ==

'''generate_example''' is a utility function that uses an LLM to generate additional examples based on a provided list of existing examples. It leverages few-shot prompting to create new examples that follow the same pattern as the input examples.

This function is particularly useful for:
* Expanding training datasets
* Creating test cases
* Generating synthetic examples for prompt engineering
* Augmenting few-shot learning examples

== Code Reference ==

=== Source Location ===
<syntaxhighlight lang="text">
File: /tmp/praxium_repo_wjjl6pl8/libs/langchain/langchain_classic/chains/example_generator.py
Lines: 9-22
Package: langchain-classic
</syntaxhighlight>

=== Function Signature ===
<syntaxhighlight lang="python">
def generate_example(
    examples: list[dict],
    llm: BaseLanguageModel,
    prompt_template: PromptTemplate,
) -> str:
    """Return another example given a list of examples for a prompt."""
</syntaxhighlight>

=== Import Statement ===
<syntaxhighlight lang="python">
from langchain_classic.chains.example_generator import generate_example
</syntaxhighlight>

== Parameters ==

{| class="wikitable"
|+ Input Parameters
! Parameter !! Type !! Required !! Description
|-
| examples || <code>list[dict]</code> || Yes || List of example dictionaries to use as few-shot context
|-
| llm || <code>BaseLanguageModel</code> || Yes || Language model to generate the new example
|-
| prompt_template || <code>PromptTemplate</code> || Yes || Template defining the format for each example
|}

== Return Value ==

{| class="wikitable"
|+ Output
! Type !! Description
|-
| <code>str</code> || Generated example as a string following the pattern of input examples
|}

== Implementation Details ==

The function creates a simple chain:

<syntaxhighlight lang="python">
# 1. Create few-shot prompt with examples
prompt = FewShotPromptTemplate(
    examples=examples,
    suffix=TEST_GEN_TEMPLATE_SUFFIX,  # "Add another example."
    input_variables=[],
    example_prompt=prompt_template,
)

# 2. Build chain: prompt -> LLM -> parse string output
chain = prompt | llm | StrOutputParser()

# 3. Invoke with empty input
return chain.invoke({})
</syntaxhighlight>

=== Template Suffix ===
<syntaxhighlight lang="python">
TEST_GEN_TEMPLATE_SUFFIX = "Add another example."
</syntaxhighlight>

This suffix is appended after all examples to prompt the LLM to generate a new one.

== Usage Examples ==

=== Basic Example Generation ===

<syntaxhighlight lang="python">
from langchain_classic.chains.example_generator import generate_example
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Define example format
prompt_template = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}"
)

# Provide existing examples
examples = [
    {"input": "cat", "output": "animal"},
    {"input": "rose", "output": "flower"},
    {"input": "oak", "output": "tree"},
]

# Generate new example
llm = ChatOpenAI(temperature=0.7)
new_example = generate_example(examples, llm, prompt_template)

# Result might be something like:
# "Input: eagle\nOutput: bird"
print(new_example)
</syntaxhighlight>

=== Sentiment Classification Example ===

<syntaxhighlight lang="python">
from langchain_classic.chains.example_generator import generate_example
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Template for sentiment examples
template = PromptTemplate(
    input_variables=["text", "sentiment"],
    template="Text: {text}\nSentiment: {sentiment}"
)

# Existing labeled examples
examples = [
    {"text": "I love this product!", "sentiment": "positive"},
    {"text": "This is terrible.", "sentiment": "negative"},
    {"text": "It's okay, nothing special.", "sentiment": "neutral"},
]

llm = ChatOpenAI(temperature=0.8)
new_example = generate_example(examples, llm, template)

# Generates new text-sentiment pair following the pattern
print(new_example)
</syntaxhighlight>

=== Code Example Generation ===

<syntaxhighlight lang="python">
from langchain_classic.chains.example_generator import generate_example
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Template for code examples
template = PromptTemplate(
    input_variables=["description", "code"],
    template="Description: {description}\nCode: {code}"
)

# Programming examples
examples = [
    {
        "description": "Print hello world",
        "code": "print('Hello, World!')"
    },
    {
        "description": "Create a list",
        "code": "my_list = [1, 2, 3]"
    },
    {
        "description": "Define a function",
        "code": "def greet(name):\n    return f'Hello, {name}'"
    },
]

llm = ChatOpenAI(temperature=0.7)
new_example = generate_example(examples, llm, template)

# Generates a new description-code pair
print(new_example)
</syntaxhighlight>

== Chain Composition ==

The function uses LCEL (LangChain Expression Language) to compose the chain:

{| class="wikitable"
|+ Chain Steps
! Step !! Component !! Purpose
|-
| 1 || <code>FewShotPromptTemplate</code> || Formats all examples and adds generation prompt
|-
| 2 || <code>BaseLanguageModel</code> || Generates new example based on pattern
|-
| 3 || <code>StrOutputParser</code> || Extracts string output from LLM response
|}

== Configuration Options ==

=== Temperature ===
Control randomness in generated examples:
* '''Low (0.0-0.3):''' More consistent, conservative examples
* '''Medium (0.4-0.7):''' Balanced creativity and consistency
* '''High (0.8-1.0):''' More creative, diverse examples

=== Model Selection ===
Choose appropriate model based on task:
* '''Simple patterns:''' Faster, smaller models (GPT-3.5, Claude Haiku)
* '''Complex patterns:''' More capable models (GPT-4, Claude Sonnet/Opus)

== Limitations and Considerations ==

=== Quality Control ===
* Generated examples may not always match desired format
* No built-in validation of generated content
* Quality depends on input example quality and diversity

=== Best Practices ===
* Provide 3-5 diverse examples for best results
* Use clear, consistent formatting in example_prompt
* Validate generated examples before use
* Consider post-processing to ensure format compliance

=== Input Requirements ===
* Examples list must not be empty
* All examples should follow the same schema
* prompt_template should match example structure

== Related Components ==

* '''FewShotPromptTemplate''' (langchain-core) - Formats examples for few-shot learning
* '''PromptTemplate''' (langchain-core) - Defines individual example format
* '''StrOutputParser''' (langchain-core) - Parses LLM string output
* '''BaseLanguageModel''' (langchain-core) - LLM interface for generation

== Design Pattern ==

This function implements the '''Few-Shot Learning''' pattern:

1. Provide examples demonstrating the desired pattern
2. Add instruction to generate similar example
3. Let LLM infer pattern and create new instance
4. Parse and return generated output

== Performance Considerations ==

* '''Latency:''' Single LLM call per invocation
* '''Cost:''' Proportional to number of examples (affects prompt tokens)
* '''Caching:''' Consider caching if generating multiple examples with same pattern

== Error Handling ==

The function has minimal error handling. Consider wrapping in try-catch:

<syntaxhighlight lang="python">
try:
    new_example = generate_example(examples, llm, template)
except Exception as e:
    print(f"Failed to generate example: {e}")
    # Handle error appropriately
</syntaxhighlight>

== See Also ==

* [[langchain-ai_langchain_FewShotPromptTemplate|FewShotPromptTemplate]] - Core component for few-shot prompting
* [[langchain-ai_langchain_PromptTemplate|PromptTemplate]] - Template system for prompts
* [[langchain-ai_langchain_StrOutputParser|StrOutputParser]] - Output parsing utilities
* LangChain documentation on few-shot learning
