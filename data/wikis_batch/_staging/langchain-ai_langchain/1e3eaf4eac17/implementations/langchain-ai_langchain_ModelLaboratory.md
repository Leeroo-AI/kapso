= ModelLaboratory Implementation =

== Metadata ==
{| class="wikitable"
|-
! Field !! Value
|-
| Source File || <code>/tmp/praxium_repo_wjjl6pl8/libs/langchain/langchain_classic/model_laboratory.py</code>
|-
| Class Name || <code>ModelLaboratory</code>
|-
| Package || <code>langchain_classic</code>
|-
| Lines of Code || 98
|-
| Status || Approved
|}

== Overview ==
<code>ModelLaboratory</code> is a utility class designed for experimenting with and comparing the performance of different language models side-by-side. It allows developers to run the same input through multiple chains or LLMs simultaneously and view their outputs in a color-coded format for easy comparison.

Key use cases include:
* '''Model Evaluation''': Compare outputs from different models (e.g., GPT-4 vs Claude vs Llama)
* '''Prompt Testing''': Evaluate how different models respond to the same prompt
* '''Quality Assessment''': Identify which model produces better results for specific tasks
* '''Performance Analysis''': Observe differences in response style, accuracy, and completeness

== Code Reference ==

=== Source Location ===
<syntaxhighlight lang="text">
File: /tmp/praxium_repo_wjjl6pl8/libs/langchain/langchain_classic/model_laboratory.py
Lines: 15-98
</syntaxhighlight>

=== Class Signature ===
<syntaxhighlight lang="python">
class ModelLaboratory:
    """A utility to experiment with and compare the performance of different models."""

    def __init__(self, chains: Sequence[Chain], names: list[str] | None = None):
        """Initialize the ModelLaboratory with chains to experiment with.

        Args:
            chains: A sequence of chains to experiment with.
                Each chain must have exactly one input and one output variable.
            names: Optional list of names corresponding to each chain.
                If provided, its length must match the number of chains.

        Raises:
            ValueError: If any chain is not an instance of `Chain`.
            ValueError: If a chain does not have exactly one input variable.
            ValueError: If a chain does not have exactly one output variable.
            ValueError: If the length of `names` does not match the number of chains.
        """

    @classmethod
    def from_llms(
        cls,
        llms: list[BaseLLM],
        prompt: PromptTemplate | None = None,
    ) -> ModelLaboratory:
        """Initialize the ModelLaboratory with LLMs and an optional prompt.

        Args:
            llms: A list of LLMs to experiment with.
            prompt: An optional prompt to use with the LLMs.
                If provided, the prompt must contain exactly one input variable.

        Returns:
            An instance of `ModelLaboratory` initialized with LLMs.
        """

    def compare(self, text: str) -> None:
        """Compare model outputs on an input text.

        If a prompt was provided with starting the laboratory, then this text will be
        fed into the prompt. If no prompt was provided, then the input text is the
        entire prompt.

        Args:
            text: input text to run all models on.
        """
</syntaxhighlight>

=== Import Statement ===
<syntaxhighlight lang="python">
from langchain_classic import ModelLaboratory
</syntaxhighlight>

== I/O Contract ==

=== Constructor Parameters ===
{| class="wikitable"
|-
! Parameter !! Type !! Required !! Default !! Description
|-
| <code>chains</code> || <code>Sequence[Chain]</code> || Yes || N/A || Sequence of chains to experiment with (each must have 1 input and 1 output)
|-
| <code>names</code> || <code>list[str] &#124; None</code> || No || <code>None</code> || Optional names for each chain (defaults to string representation of chain)
|}

=== Class Method: <code>from_llms</code> ===
{| class="wikitable"
|-
! Parameter !! Type !! Required !! Default !! Description
|-
| <code>llms</code> || <code>list[BaseLLM]</code> || Yes || N/A || List of LLMs to experiment with
|-
| <code>prompt</code> || <code>PromptTemplate &#124; None</code> || No || <code>None</code> || Optional prompt template (defaults to passthrough template)
|}

'''Returns:''' <code>ModelLaboratory</code> instance

=== Instance Method: <code>compare</code> ===
{| class="wikitable"
|-
! Parameter !! Type !! Required !! Description
|-
| <code>text</code> || <code>str</code> || Yes || Input text to run all models on
|}

'''Returns:''' <code>None</code> (prints results to stdout)

=== Instance Attributes ===
{| class="wikitable"
|-
! Attribute !! Type !! Description
|-
| <code>chains</code> || <code>Sequence[Chain]</code> || The chains being compared
|-
| <code>names</code> || <code>list[str] &#124; None</code> || Optional names for each chain
|-
| <code>chain_colors</code> || <code>dict[str, str]</code> || Color mapping for chain outputs
|}

== Implementation Details ==

=== Validation Logic ===
The constructor performs strict validation:

1. '''Type Check''': Ensures all items in <code>chains</code> are instances of <code>Chain</code>
   * If not, raises <code>ValueError</code> with a migration hint to use <code>from_llms</code>
2. '''Input Keys Check''': Each chain must have exactly one input variable
   * Raises <code>ValueError</code> if <code>len(chain.input_keys) != 1</code>
3. '''Output Keys Check''': Each chain must have exactly one output variable
   * Raises <code>ValueError</code> if <code>len(chain.output_keys) != 1</code>
4. '''Names Length Check''': If <code>names</code> is provided, its length must match the number of chains
   * Raises <code>ValueError</code> if lengths don't match

=== Color Mapping ===
* Uses <code>get_color_mapping</code> from <code>langchain_core.utils.input</code>
* Assigns a unique color to each chain based on its index
* Colors are used to visually distinguish outputs in the terminal

=== Default Prompt Template ===
When using <code>from_llms</code> without a prompt:
* Creates a passthrough template: <code>PromptTemplate(input_variables=["_input"], template="{_input}")</code>
* This allows direct text input without additional formatting

=== Output Formatting ===
The <code>compare</code> method:
1. Prints the input text with bold formatting
2. For each chain:
   * Prints the chain name (or string representation)
   * Runs the chain with the input text
   * Prints the output in the assigned color
   * Adds spacing between outputs

== Usage Examples ==

=== Basic LLM Comparison ===
<syntaxhighlight lang="python">
from langchain_classic import ModelLaboratory
from langchain_openai import OpenAI
from langchain_anthropic import ChatAnthropic

# Create multiple LLMs
llms = [
    OpenAI(temperature=0.9),
    OpenAI(temperature=0.5),
    ChatAnthropic(model="claude-3-opus-20240229")
]

# Create laboratory
lab = ModelLaboratory.from_llms(llms=llms)

# Compare outputs
lab.compare("Write a short poem about artificial intelligence.")
</syntaxhighlight>

Output:
<syntaxhighlight lang="text">
Input:
Write a short poem about artificial intelligence.

OpenAI(temperature=0.9)
In circuits deep and code so bright,
AI learns throughout the night...

OpenAI(temperature=0.5)
Silicon minds that think and learn,
Processing data at every turn...

ChatAnthropic(model='claude-3-opus-20240229')
Algorithms weaving thought from code,
Intelligence in electric mode...
</syntaxhighlight>

=== Comparison with Custom Prompt ===
<syntaxhighlight lang="python">
from langchain_classic import ModelLaboratory
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Define a prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms suitable for a 10-year-old."
)

# Create LLMs
llms = [
    OpenAI(model_name="gpt-3.5-turbo-instruct"),
    ChatOpenAI(model_name="gpt-4")
]

# Create laboratory with prompt
lab = ModelLaboratory.from_llms(llms=llms, prompt=prompt)

# Compare
lab.compare("quantum computing")
</syntaxhighlight>

=== Comparing Custom Chains ===
<syntaxhighlight lang="python">
from langchain_classic import ModelLaboratory
from langchain_classic.chains import LLMChain
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

# Create custom chains with different configurations
prompt1 = PromptTemplate(
    input_variables=["question"],
    template="Q: {question}\nA: Let's think step by step."
)

prompt2 = PromptTemplate(
    input_variables=["question"],
    template="Question: {question}\nProvide a concise answer:"
)

chain1 = LLMChain(llm=OpenAI(temperature=0), prompt=prompt1)
chain2 = LLMChain(llm=OpenAI(temperature=0.7), prompt=prompt2)

# Create laboratory with custom names
lab = ModelLaboratory(
    chains=[chain1, chain2],
    names=["Step-by-step reasoning", "Concise answer"]
)

# Compare
lab.compare("What is the capital of France?")
</syntaxhighlight>

=== Multiple Model Providers ===
<syntaxhighlight lang="python">
from langchain_classic import ModelLaboratory
from langchain_openai import OpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Create prompt
prompt = PromptTemplate(
    input_variables=["task"],
    template="Task: {task}\n\nProvide a solution:"
)

# Mix different providers
llms = [
    OpenAI(model_name="gpt-3.5-turbo-instruct"),
    ChatAnthropic(model="claude-3-sonnet-20240229"),
    ChatGoogleGenerativeAI(model="gemini-pro")
]

lab = ModelLaboratory.from_llms(llms=llms, prompt=prompt)

lab.compare("Create a Python function to calculate Fibonacci numbers")
</syntaxhighlight>

=== Temperature Experimentation ===
<syntaxhighlight lang="python">
from langchain_classic import ModelLaboratory
from langchain_openai import OpenAI

# Compare same model with different temperatures
llms = [
    OpenAI(temperature=0.0),
    OpenAI(temperature=0.5),
    OpenAI(temperature=1.0)
]

lab = ModelLaboratory.from_llms(llms=llms)

# See how temperature affects creativity
lab.compare("Generate a creative company name for a coffee shop")
</syntaxhighlight>

=== Chain with Custom Logic ===
<syntaxhighlight lang="python">
from langchain_classic import ModelLaboratory
from langchain_classic.chains import LLMChain, TransformChain, SimpleSequentialChain
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

# Chain 1: Direct LLM
chain1 = LLMChain(
    llm=OpenAI(),
    prompt=PromptTemplate(
        input_variables=["text"],
        template="{text}"
    )
)

# Chain 2: Preprocessed input
def preprocess(inputs):
    return {"processed": f"CONTEXT: {inputs['text']}\nPlease elaborate:"}

transform = TransformChain(
    input_variables=["text"],
    output_variables=["processed"],
    transform=preprocess
)

chain2_llm = LLMChain(
    llm=OpenAI(),
    prompt=PromptTemplate(
        input_variables=["processed"],
        template="{processed}"
    )
)

# Note: This would require adapting for single input/output
# This is just to show the concept
lab = ModelLaboratory(
    chains=[chain1],
    names=["Direct", "Preprocessed"]
)

lab.compare("Artificial intelligence")
</syntaxhighlight>

== Related Pages ==
* [[langchain-ai_langchain_Chain|Chain]] - Base chain class
* [[langchain-ai_langchain_LLMChain|LLMChain]] - Chain for LLM invocations
* [[langchain-ai_langchain_BaseLLM|BaseLLM]] - Base LLM interface
* [[langchain-ai_langchain_PromptTemplate|PromptTemplate]] - Template for prompts

== Notes ==
* '''Output Display''': Results are printed to stdout with color coding for visual distinction
* '''Single I/O Constraint''': Each chain must have exactly one input and one output variable
* '''Synchronous Execution''': Chains are executed sequentially, not in parallel
* '''Terminal Colors''': Colors are assigned automatically from a predefined palette
* '''Name Display''': If no names are provided, uses the string representation of each chain
* '''Error Handling''': Validation errors are raised during initialization, not during comparison
* '''Use Case''': Best suited for interactive exploration and manual evaluation, not automated testing
* '''Migration Note''': Previously accepted LLMs directly; now requires using <code>from_llms</code> class method

== Limitations ==
* Only supports chains with exactly one input and one output
* No built-in metrics or scoring - purely visual comparison
* Sequential execution may be slow for many models
* No result persistence - output is ephemeral (printed to stdout)
* No statistical comparison or aggregation across multiple inputs

== See Also ==
* [https://docs.langchain.com/docs/modules/model_io/model_comparison LangChain Model Comparison Documentation]
* [https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain_classic/model_laboratory.py Source Code on GitHub]
* [[langchain-ai_langchain_evaluation|LangChain Evaluation Tools]] - For programmatic model evaluation
