= TokensPrompt for Speculative Decoding =

== Metadata ==
{| class="wikitable"
|-
! Field !! Value
|-
| Knowledge Sources || vLLM Input API Documentation, Speculative Decoding Examples
|-
| Domains || Input Processing, Token Management, API Design
|-
| Last Updated || 2025-12-17
|}

== Overview ==

<code>TokensPrompt</code> is an input class that allows direct specification of tokenized prompts for vLLM inference. When used with speculative decoding, it provides fine-grained control over token sequences and enables optimization of speculation-friendly inputs.

== Code Reference ==

=== Source Location ===
<syntaxhighlight lang="text">
File: vllm/inputs/data.py
Class: TokensPrompt
</syntaxhighlight>

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class TokensPrompt:
    """Prompt represented by pre-tokenized token IDs."""
    prompt_token_ids: list[int]
    multi_modal_data: MultiModalDataDict | None = None
    mm_processor_kwargs: dict[str, Any] | None = None
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm.inputs import TokensPrompt
from vllm import LLM, SamplingParams
</syntaxhighlight>

== Description ==

<code>TokensPrompt</code> implements the [[implements::Principle:vllm-project_vllm_speculative_prompt_prep]] principle by providing a direct token-level interface for speculative decoding. This is particularly useful when:

* Optimizing token sequences for ngram pattern matching
* Controlling exact tokenization for reproducibility
* Bypassing tokenization overhead in high-throughput scenarios
* Working with custom preprocessing pipelines

=== Key Attributes ===

* '''prompt_token_ids''': List of integer token IDs representing the prompt
* '''multi_modal_data''': Optional dict containing images, audio, or other modalities
* '''mm_processor_kwargs''': Optional processing arguments for multi-modal data

== Input/Output Contract ==

=== Input: TokensPrompt Construction ===

{| class="wikitable"
! Parameter !! Type !! Required !! Description
|-
| prompt_token_ids || list[int] || Yes || Pre-tokenized sequence of token IDs
|-
| multi_modal_data || dict || No || Multi-modal inputs (images, audio, etc.)
|-
| mm_processor_kwargs || dict || No || Multi-modal processor configuration
|}

=== Output ===

* Can be passed directly to <code>LLM.generate()</code>
* Works seamlessly with all speculative decoding methods
* Integrates with <code>SamplingParams</code> for generation control

== Usage Examples ==

=== Example 1: Basic TokensPrompt with N-gram ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from transformers import AutoTokenizer

# Initialize tokenizer and LLM with ngram speculation
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,
    }
)

# Pre-tokenize prompts
prompts_text = [
    "The capital of France is Paris. The capital of Germany is",
    "To sort a list in Python: list.sort(). To reverse a list in Python:"
]
prompt_ids = [
    tokenizer.encode(p, add_special_tokens=False) for p in prompts_text
]

# Create TokensPrompt objects
tokens_prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids]

# Generate with speculation
sampling_params = SamplingParams(temperature=0.0, max_tokens=50)
outputs = llm.generate(tokens_prompts, sampling_params)

for output in outputs:
    print(f"Generated: {output.outputs[0].text}")
</syntaxhighlight>

=== Example 2: Optimized Ngram Patterns ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 5,
    }
)

# Create prompt with repetitive pattern (good for ngram)
prompt_text = """
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
"""

prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
tokens_prompt = TokensPrompt(prompt_token_ids=prompt_ids)

outputs = llm.generate(
    [tokens_prompt],
    SamplingParams(temperature=0.0, max_tokens=100)
)

print(outputs[0].outputs[0].text)
</syntaxhighlight>

=== Example 3: TokensPrompt with EAGLE ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "eagle",
        "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        "num_speculative_tokens": 3,
    }
)

# Pre-tokenize for EAGLE
prompt_text = "Explain the theory of relativity in simple terms."
prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)

tokens_prompt = TokensPrompt(prompt_token_ids=prompt_ids)

outputs = llm.generate(
    [tokens_prompt],
    SamplingParams(temperature=0.0, max_tokens=200)
)

print(f"Output: {outputs[0].outputs[0].text}")
</syntaxhighlight>

=== Example 4: Batch Processing with TokensPrompt ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,
    }
)

# Batch of prompts
prompts_text = [
    "The first president of the United States was",
    "The largest planet in our solar system is",
    "The chemical symbol for gold is",
    "The speed of light is approximately",
]

# Tokenize all prompts
prompt_ids_list = [
    tokenizer.encode(p, add_special_tokens=True) for p in prompts_text
]

# Create TokensPrompt for each
tokens_prompts = [
    TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids_list
]

# Batch generate
sampling_params = SamplingParams(temperature=0.0, max_tokens=50)
outputs = llm.generate(tokens_prompts, sampling_params)

for i, output in enumerate(outputs):
    print(f"Prompt {i}: {prompts_text[i]}")
    print(f"Output {i}: {output.outputs[0].text}")
    print("-" * 50)
</syntaxhighlight>

=== Example 5: Custom Token Manipulation ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,
    }
)

# Custom preprocessing: inject special pattern tokens
base_text = "Translate to French: Hello"
base_ids = tokenizer.encode(base_text, add_special_tokens=False)

# Add custom instruction tokens (example)
special_token_id = tokenizer.eos_token_id
modified_ids = [tokenizer.bos_token_id] + base_ids

tokens_prompt = TokensPrompt(prompt_token_ids=modified_ids)

outputs = llm.generate(
    [tokens_prompt],
    SamplingParams(temperature=0.3, max_tokens=50)
)

print(outputs[0].outputs[0].text)
</syntaxhighlight>

=== Example 6: Extracting Token IDs from Outputs ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,
    }
)

# Generate with TokensPrompt
prompt_ids = tokenizer.encode("Count: one, two, three,", add_special_tokens=False)
tokens_prompt = TokensPrompt(prompt_token_ids=prompt_ids)

outputs = llm.generate(
    [tokens_prompt],
    SamplingParams(temperature=0.0, max_tokens=20)
)

# Access generated token IDs
output = outputs[0]
generated_ids = output.outputs[0].token_ids
generated_text = output.outputs[0].text

print(f"Generated token IDs: {generated_ids}")
print(f"Generated text: {generated_text}")

# Use generated IDs for next iteration (chaining)
next_prompt_ids = prompt_ids + generated_ids
next_tokens_prompt = TokensPrompt(prompt_token_ids=next_prompt_ids)
</syntaxhighlight>

== Design Details ==

=== Token ID Requirements ===

* Token IDs must be valid for the model's vocabulary
* Must include appropriate special tokens (BOS, EOS) as needed
* Should match tokenizer's encoding scheme exactly
* Range typically [0, vocab_size)

=== Integration with Speculative Methods ===

==== N-gram ====
* Token IDs stored in circular buffer for pattern matching
* Longer sequences provide more matching opportunities
* No special token handling required

==== EAGLE ====
* Token IDs converted to embeddings via target model
* Hidden states extracted for draft model input
* Works with standard tokenization

==== MLP Speculator ====
* Token IDs processed through embedding layer
* Context vectors derived from token sequence
* Standard preprocessing applies

=== Memory Considerations ===

* <code>TokensPrompt</code> stores only token IDs (int list)
* More memory efficient than storing full text
* Suitable for caching and reuse across requests
* No tokenization overhead during generation

=== Performance Benefits ===

* '''Skip Tokenization''': Eliminates tokenizer overhead
* '''Reproducibility''': Exact token control ensures consistent results
* '''Optimization''': Can craft sequences for better speculation
* '''Batching''': Pre-tokenization enables efficient batching

== Common Use Cases ==

=== Code Generation ===
Pre-tokenize code templates with repetitive patterns for ngram:
<syntaxhighlight lang="python">
code_template = "def func1():\n    pass\n\ndef func2():\n    pass\n\ndef func3():"
prompt_ids = tokenizer.encode(code_template, add_special_tokens=False)
tokens_prompt = TokensPrompt(prompt_token_ids=prompt_ids)
</syntaxhighlight>

=== Question Answering ===
Format QA pairs for better pattern matching:
<syntaxhighlight lang="python">
qa_prompt = "Q: Capital of France? A: Paris. Q: Capital of Germany? A:"
prompt_ids = tokenizer.encode(qa_prompt, add_special_tokens=False)
tokens_prompt = TokensPrompt(prompt_token_ids=prompt_ids)
</syntaxhighlight>

=== Chain-of-Thought ===
Build reasoning chains with consistent structure:
<syntaxhighlight lang="python">
cot = "Step 1: Identify the problem.\nStep 2: Break it down.\nStep 3:"
prompt_ids = tokenizer.encode(cot, add_special_tokens=False)
tokens_prompt = TokensPrompt(prompt_token_ids=prompt_ids)
</syntaxhighlight>

== Related Pages ==

* [[implements::Principle:vllm-project_vllm_speculative_prompt_prep]]
* [[implemented_by::vllm-project_vllm_LLM_generate_spec]]
* TextPrompt Documentation
* Multi-Modal Input Handling
* Tokenization Best Practices
