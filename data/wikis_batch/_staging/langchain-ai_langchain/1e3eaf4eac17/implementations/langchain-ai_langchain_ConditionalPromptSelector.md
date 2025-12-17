---
title: ConditionalPromptSelector
type: implementation
project: langchain-ai/langchain
file: libs/langchain/langchain_classic/chains/prompt_selector.py
category: prompt_management
---

= ConditionalPromptSelector Implementation =

== Overview ==

'''ConditionalPromptSelector''' is a utility class that selects the appropriate prompt template based on the type of language model being used. It enables developers to maintain different prompt variations for different model types (e.g., chat models vs. completion models) and automatically select the right one at runtime.

This is particularly useful when building model-agnostic applications that need to work with both traditional LLMs (like GPT-3) and chat models (like GPT-4 or Claude).

== Code Reference ==

=== Source Location ===
<syntaxhighlight lang="text">
File: /tmp/praxium_repo_wjjl6pl8/libs/langchain/langchain_classic/chains/prompt_selector.py
Lines: 1-66
Package: langchain-classic
</syntaxhighlight>

=== Class Hierarchy ===

<syntaxhighlight lang="python">
# Abstract base class
class BasePromptSelector(BaseModel, ABC):
    """Base class for prompt selectors."""

    @abstractmethod
    def get_prompt(self, llm: BaseLanguageModel) -> BasePromptTemplate:
        """Get default prompt for a language model."""

# Concrete implementation
class ConditionalPromptSelector(BasePromptSelector):
    """Prompt collection that goes through conditionals."""
</syntaxhighlight>

=== Import Statement ===
<syntaxhighlight lang="python">
from langchain_classic.chains.prompt_selector import ConditionalPromptSelector
from langchain_classic.chains.prompt_selector import is_llm, is_chat_model
</syntaxhighlight>

== Class Attributes ==

{| class="wikitable"
|+ ConditionalPromptSelector Attributes
! Attribute !! Type !! Required !! Description
|-
| default_prompt || <code>BasePromptTemplate</code> || Yes || Fallback prompt if no conditionals match
|-
| conditionals || <code>list[tuple[Callable, BasePromptTemplate]]</code> || No || List of (condition_function, prompt) pairs
|}

== Method Reference ==

=== get_prompt ===

'''Method Signature:'''
<syntaxhighlight lang="python">
def get_prompt(self, llm: BaseLanguageModel) -> BasePromptTemplate:
    """Get default prompt for a language model."""
</syntaxhighlight>

{| class="wikitable"
|+ Parameters
! Parameter !! Type !! Description
|-
| llm || <code>BaseLanguageModel</code> || Language model to select prompt for
|}

{| class="wikitable"
|+ Return Value
! Type !! Description
|-
| <code>BasePromptTemplate</code> || Selected prompt template (from conditionals or default)
|}

'''Logic:''' Iterates through conditionals in order, returning the first prompt whose condition returns True. If no conditions match, returns <code>default_prompt</code>.

== Helper Functions ==

=== is_llm ===

<syntaxhighlight lang="python">
def is_llm(llm: BaseLanguageModel) -> bool:
    """Check if the language model is a LLM.

    Returns:
        `True` if the language model is a BaseLLM model, `False` otherwise.
    """
    return isinstance(llm, BaseLLM)
</syntaxhighlight>

Checks if model is a completion-style LLM (e.g., text-davinci-003).

=== is_chat_model ===

<syntaxhighlight lang="python">
def is_chat_model(llm: BaseLanguageModel) -> bool:
    """Check if the language model is a chat model.

    Returns:
        `True` if the language model is a BaseChatModel model, `False` otherwise.
    """
    return isinstance(llm, BaseChatModel)
</syntaxhighlight>

Checks if model is a chat-style model (e.g., GPT-4, Claude).

== Usage Examples ==

=== Basic Model-Aware Prompt Selection ===

<syntaxhighlight lang="python">
from langchain_classic.chains.prompt_selector import (
    ConditionalPromptSelector,
    is_chat_model,
    is_llm
)
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAI

# Define prompts for different model types
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{question}")
])

llm_prompt = PromptTemplate(
    template="Answer the following question: {question}\n\nAnswer:",
    input_variables=["question"]
)

default_prompt = PromptTemplate(
    template="Q: {question}\nA:",
    input_variables=["question"]
)

# Create selector
prompt_selector = ConditionalPromptSelector(
    default_prompt=default_prompt,
    conditionals=[
        (is_chat_model, chat_prompt),
        (is_llm, llm_prompt),
    ]
)

# Use with chat model
chat_model = ChatOpenAI()
selected_prompt = prompt_selector.get_prompt(chat_model)
print(type(selected_prompt))  # ChatPromptTemplate

# Use with completion model
completion_model = OpenAI()
selected_prompt = prompt_selector.get_prompt(completion_model)
print(type(selected_prompt))  # PromptTemplate
</syntaxhighlight>

=== Custom Conditional Logic ===

<syntaxhighlight lang="python">
from langchain_classic.chains.prompt_selector import ConditionalPromptSelector
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel

# Custom condition: check model name
def is_gpt4(llm: BaseLanguageModel) -> bool:
    """Check if model is GPT-4."""
    return hasattr(llm, 'model_name') and 'gpt-4' in llm.model_name.lower()

def is_claude(llm: BaseLanguageModel) -> bool:
    """Check if model is Claude."""
    return hasattr(llm, 'model_name') and 'claude' in llm.model_name.lower()

# Define model-specific prompts
gpt4_prompt = PromptTemplate(
    template="[GPT-4 Optimized] {task}",
    input_variables=["task"]
)

claude_prompt = PromptTemplate(
    template="[Claude Optimized] {task}",
    input_variables=["task"]
)

generic_prompt = PromptTemplate(
    template="[Generic] {task}",
    input_variables=["task"]
)

# Create selector with custom conditions
selector = ConditionalPromptSelector(
    default_prompt=generic_prompt,
    conditionals=[
        (is_gpt4, gpt4_prompt),
        (is_claude, claude_prompt),
    ]
)

# Select based on model
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4")
prompt = selector.get_prompt(model)
</syntaxhighlight>

=== Building a Model-Agnostic Chain ===

<syntaxhighlight lang="python">
from langchain_classic.chains import LLMChain
from langchain_classic.chains.prompt_selector import (
    ConditionalPromptSelector,
    is_chat_model,
    is_llm
)
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# Define prompts
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert summarizer."),
    ("user", "Summarize: {text}")
])

llm_prompt = PromptTemplate(
    template="Summarize the following text:\n\n{text}\n\nSummary:",
    input_variables=["text"]
)

# Create selector
selector = ConditionalPromptSelector(
    default_prompt=llm_prompt,
    conditionals=[(is_chat_model, chat_prompt)]
)

def create_summary_chain(llm):
    """Create a summary chain that works with any model type."""
    prompt = selector.get_prompt(llm)
    return LLMChain(llm=llm, prompt=prompt)

# Use with different models
from langchain_openai import ChatOpenAI, OpenAI

# Chat model version
chat_chain = create_summary_chain(ChatOpenAI())
result = chat_chain.invoke({"text": "Long document..."})

# Completion model version
completion_chain = create_summary_chain(OpenAI())
result = completion_chain.invoke({"text": "Long document..."})
</syntaxhighlight>

=== Multiple Conditions with Priority ===

<syntaxhighlight lang="python">
from langchain_classic.chains.prompt_selector import ConditionalPromptSelector
from langchain_core.prompts import PromptTemplate

# Conditions are checked in order - first match wins
def is_high_capability(llm) -> bool:
    """Check if model is high-capability."""
    high_cap_models = ['gpt-4', 'claude-3-opus', 'claude-3-sonnet']
    return hasattr(llm, 'model_name') and any(
        model in llm.model_name.lower() for model in high_cap_models
    )

def is_fast_model(llm) -> bool:
    """Check if model is optimized for speed."""
    fast_models = ['gpt-3.5', 'claude-3-haiku']
    return hasattr(llm, 'model_name') and any(
        model in llm.model_name.lower() for model in fast_models
    )

# Create prompts with different complexity
complex_prompt = PromptTemplate(
    template="Analyze in detail with step-by-step reasoning: {task}",
    input_variables=["task"]
)

simple_prompt = PromptTemplate(
    template="Briefly answer: {task}",
    input_variables=["task"]
)

basic_prompt = PromptTemplate(
    template="{task}",
    input_variables=["task"]
)

# Priority order: high-capability first, then fast, then default
selector = ConditionalPromptSelector(
    default_prompt=basic_prompt,
    conditionals=[
        (is_high_capability, complex_prompt),  # Checked first
        (is_fast_model, simple_prompt),        # Checked second
    ]
)
</syntaxhighlight>

=== Integration with LCEL ===

<syntaxhighlight lang="python">
from langchain_classic.chains.prompt_selector import ConditionalPromptSelector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Define prompts
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    ("user", "{input}")
])

# Create selector
selector = ConditionalPromptSelector(
    default_prompt=chat_prompt,
    conditionals=[]
)

# Use in LCEL chain
def create_chain(llm):
    prompt = selector.get_prompt(llm)
    return prompt | llm | StrOutputParser()

# Build chain
from langchain_openai import ChatOpenAI
chain = create_chain(ChatOpenAI())

result = chain.invoke({"input": "Hello!"})
print(result)
</syntaxhighlight>

== Design Patterns ==

=== Conditional Selection Pattern ===

The class implements a simple conditional selection pattern:

{| class="wikitable"
|+ Selection Algorithm
! Step !! Action
|-
| 1 || Iterate through conditionals list in order
|-
| 2 || For each (condition_fn, prompt) pair, call condition_fn(llm)
|-
| 3 || If condition returns True, return associated prompt
|-
| 4 || If no conditions match, return default_prompt
|}

=== Order Matters ===

Conditionals are evaluated in order, and the '''first match wins'''. Structure your conditionals from most specific to most general:

<syntaxhighlight lang="python">
conditionals=[
    (is_gpt4_turbo, gpt4_turbo_prompt),     # Most specific
    (is_gpt4, gpt4_prompt),                  # Less specific
    (is_chat_model, generic_chat_prompt),    # Most general
]
</syntaxhighlight>

== Use Cases ==

=== Cross-Model Compatibility ===
* Build applications that work with multiple LLM providers
* Switch between chat and completion models seamlessly
* Maintain model-specific optimizations

=== Prompt Optimization ===
* Use different prompt styles for different model architectures
* Optimize token usage based on model capabilities
* Adjust prompt complexity based on model sophistication

=== Testing and Development ===
* Test with cheaper models in development
* Use premium models in production
* Easy model swapping without code changes

=== Multi-Provider Support ===
* Support OpenAI, Anthropic, and other providers
* Maintain provider-specific prompt variations
* Abstract provider differences from application logic

== Best Practices ==

=== Clear Conditions ===
* Write descriptive condition functions with clear names
* Document what each condition checks
* Test conditions with different model types

=== Sensible Defaults ===
* Always provide a reasonable default_prompt
* Default should work with any model type
* Avoid model-specific syntax in default

=== Order Considerations ===
* Place most specific conditions first
* Place most general conditions last
* Document the evaluation order

=== Testing ===
* Test with all expected model types
* Verify correct prompt selected for each model
* Test fallback to default prompt

== Limitations ==

=== Static Conditions ===
* Conditions are evaluated once at selection time
* Cannot dynamically adapt based on runtime context
* No support for weighted or probabilistic selection

=== No Composition ===
* Cannot combine multiple conditions with AND/OR logic
* Must write custom condition functions for complex logic
* No built-in condition combinators

=== Limited Introspection ===
* Relies on isinstance checks and attribute inspection
* May not work with all model wrappers
* Requires models to follow expected interfaces

== Performance Considerations ==

=== Evaluation Cost ===
* Conditions evaluated sequentially until match
* Short-circuit evaluation (stops at first match)
* Negligible overhead for typical use cases

=== Caching ===
Consider caching the selected prompt if using same model repeatedly:

<syntaxhighlight lang="python">
from functools import lru_cache

@lru_cache(maxsize=128)
def get_cached_prompt(llm_class_name: str, selector: ConditionalPromptSelector):
    """Cache prompt selection by model class."""
    # Note: This is simplified; actual implementation needs hashable keys
    pass
</syntaxhighlight>

== Related Components ==

* '''BasePromptTemplate''' (langchain-core) - Base class for all prompts
* '''ChatPromptTemplate''' (langchain-core) - Chat-style prompts
* '''PromptTemplate''' (langchain-core) - Completion-style prompts
* '''BaseLLM''' (langchain-core) - Completion model interface
* '''BaseChatModel''' (langchain-core) - Chat model interface
* '''BaseLanguageModel''' (langchain-core) - Common model interface

== See Also ==

* [[langchain-ai_langchain_PromptTemplate|PromptTemplate]] - Basic prompt templates
* [[langchain-ai_langchain_ChatPromptTemplate|ChatPromptTemplate]] - Chat-specific prompts
* [[langchain-ai_langchain_BaseLLM|BaseLLM]] - Completion model interface
* [[langchain-ai_langchain_BaseChatModel|BaseChatModel]] - Chat model interface
* LangChain documentation on prompt templates
* Model-agnostic application design patterns
