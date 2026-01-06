{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::Prompts]], [[domain::Model_Selection]], [[domain::Chains]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Classes `BasePromptSelector` and `ConditionalPromptSelector` that dynamically select prompts based on the language model type.

=== Description ===

The prompt selector pattern allows different prompts to be used depending on whether the model is a chat model or completion model. `ConditionalPromptSelector` evaluates a list of conditions against the provided model and returns the first matching prompt, or a default if none match. Helper functions `is_llm` and `is_chat_model` are provided for common model type checks.

=== Usage ===

Use prompt selectors when building chains that need to work with different model types (chat vs completion) that require different prompt formats. This is common in libraries that want to support both legacy LLMs and modern chat models.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/chains/prompt_selector.py libs/langchain/langchain_classic/chains/prompt_selector.py]
* '''Lines:''' 1-65

=== Signature ===
<syntaxhighlight lang="python">
class BasePromptSelector(BaseModel, ABC):
    """Base class for prompt selectors."""

    @abstractmethod
    def get_prompt(self, llm: BaseLanguageModel) -> BasePromptTemplate:
        """Get default prompt for a language model."""


class ConditionalPromptSelector(BasePromptSelector):
    """Prompt collection that goes through conditionals.

    Attributes:
        default_prompt: Default prompt if no conditionals match.
        conditionals: List of (condition_func, prompt) tuples.
    """

    default_prompt: BasePromptTemplate
    conditionals: list[tuple[Callable[[BaseLanguageModel], bool], BasePromptTemplate]]

    def get_prompt(self, llm: BaseLanguageModel) -> BasePromptTemplate:
        """Get prompt matching the first true condition, or default."""


def is_llm(llm: BaseLanguageModel) -> bool:
    """Check if the language model is a BaseLLM (completion model)."""


def is_chat_model(llm: BaseLanguageModel) -> bool:
    """Check if the language model is a BaseChatModel."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain_classic.chains.prompt_selector import (
    BasePromptSelector,
    ConditionalPromptSelector,
    is_llm,
    is_chat_model,
)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| llm || BaseLanguageModel || Yes || Language model to select prompt for
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| return || BasePromptTemplate || Selected prompt template
|}

== Usage Examples ==

=== Basic Prompt Selection ===
<syntaxhighlight lang="python">
from langchain_classic.chains.prompt_selector import (
    ConditionalPromptSelector,
    is_chat_model,
)
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import OpenAI, ChatOpenAI

# Create different prompts for different model types
completion_prompt = PromptTemplate.from_template(
    "Answer the following question:\n\nQuestion: {question}\n\nAnswer:"
)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}"),
])

# Create selector
prompt_selector = ConditionalPromptSelector(
    default_prompt=completion_prompt,
    conditionals=[
        (is_chat_model, chat_prompt),
    ],
)

# Select prompt based on model type
chat_model = ChatOpenAI()
prompt = prompt_selector.get_prompt(chat_model)
# Returns chat_prompt

completion_model = OpenAI()
prompt = prompt_selector.get_prompt(completion_model)
# Returns completion_prompt (default)
</syntaxhighlight>

=== Custom Conditions ===
<syntaxhighlight lang="python">
# Create custom condition for specific models
def is_gpt4(llm: BaseLanguageModel) -> bool:
    if hasattr(llm, "model_name"):
        return "gpt-4" in llm.model_name
    return False

gpt4_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are GPT-4, an advanced AI assistant."),
    ("human", "{question}"),
])

prompt_selector = ConditionalPromptSelector(
    default_prompt=completion_prompt,
    conditionals=[
        (is_gpt4, gpt4_prompt),
        (is_chat_model, chat_prompt),
    ],
)

# Conditions are evaluated in order
gpt4_model = ChatOpenAI(model="gpt-4")
prompt = prompt_selector.get_prompt(gpt4_model)  # Returns gpt4_prompt
</syntaxhighlight>

=== Using in Chains ===
<syntaxhighlight lang="python">
from langchain_classic.chains import LLMChain

def create_qa_chain(llm: BaseLanguageModel):
    """Create a QA chain with model-appropriate prompt."""
    prompt = prompt_selector.get_prompt(llm)
    return LLMChain(llm=llm, prompt=prompt)

# Works with any model type
chain = create_qa_chain(ChatOpenAI())
# or
chain = create_qa_chain(OpenAI())
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]

