= SpeculativeConfig =

== Metadata ==
{| class="wikitable"
|-
! Field !! Value
|-
| Knowledge Sources || [https://arxiv.org/abs/2401.15077 EAGLE Paper], [https://arxiv.org/abs/2404.19124 MLP Speculator Paper], [https://arxiv.org/abs/2411.04975 Suffix Decoding Paper], vLLM Source Code
|-
| Domains || Configuration Management, Speculative Decoding, Model Configuration
|-
| Last Updated || 2025-12-17
|}

== Overview ==

<code>SpeculativeConfig</code> is a configuration dataclass that defines all parameters for enabling and controlling speculative decoding in vLLM. It manages method selection, draft model configuration, and method-specific parameters for various speculative decoding approaches.

== Code Reference ==

=== Source Location ===
<syntaxhighlight lang="text">
File: vllm/config/speculative.py
Class: SpeculativeConfig
Lines: 52-645
</syntaxhighlight>

=== Signature ===
<syntaxhighlight lang="python">
@config
@dataclass
class SpeculativeConfig:
    enforce_eager: bool | None = None
    num_speculative_tokens: int = Field(default=None, gt=0)
    model: str | None = None
    method: SpeculativeMethod | None = None
    draft_tensor_parallel_size: int | None = Field(default=None, ge=1)
    quantization: me_quant.QuantizationMethods | None = None
    max_model_len: int | None = Field(default=None, ge=1)
    revision: str | None = None
    code_revision: str | None = None
    disable_by_batch_size: int | None = Field(default=None, ge=2)
    disable_padded_drafter_batch: bool = False
    prompt_lookup_max: int | None = Field(default=None, ge=1)
    prompt_lookup_min: int | None = Field(default=None, ge=1)
    speculative_token_tree: str | None = None
    # ... additional fields
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm.config import SpeculativeConfig
# Or when initializing LLM
from vllm import LLM
</syntaxhighlight>

== Description ==

<code>SpeculativeConfig</code> implements the [[implements::Principle:vllm-project_vllm_spec_method_selection]] principle by providing a unified configuration interface for all speculative decoding methods. The configuration is typically passed as a dictionary to the <code>LLM</code> constructor via the <code>speculative_config</code> parameter.

=== Key Features ===

* '''Method Detection''': Automatically detects speculative method from model type when possible
* '''Validation''': Comprehensive validation of parameters and compatibility checks
* '''Draft Model Management''': Creates and configures draft model configurations internally
* '''Flexibility''': Supports both simple (ngram) and complex (EAGLE, MTP) methods

== Input/Output Contract ==

=== Configuration Parameters ===

{| class="wikitable"
! Parameter !! Type !! Required !! Description
|-
| method || SpeculativeMethod || Conditional || Method name: "ngram", "eagle", "eagle3", "mlp_speculator", "draft_model", "suffix", "mtp"
|-
| num_speculative_tokens || int || Yes || Number of tokens to speculate (must be > 0)
|-
| model || str || Conditional || Path to draft model, EAGLE head, or speculator weights
|-
| prompt_lookup_max || int || For ngram || Maximum n-gram window size for prompt lookup
|-
| prompt_lookup_min || int || For ngram || Minimum n-gram window size for prompt lookup
|-
| draft_tensor_parallel_size || int || Optional || TP size for draft model (1 or same as target)
|-
| quantization || str || Optional || Quantization method for draft model
|-
| disable_by_batch_size || int || Optional || Disable speculation when batch size exceeds threshold
|}

=== Output ===

Returns a validated <code>SpeculativeConfig</code> instance with:
* Populated <code>draft_model_config</code>
* Populated <code>draft_parallel_config</code>
* Validated and normalized parameters
* Auto-detected method when not explicitly specified

== Usage Examples ==

=== Example 1: N-gram Speculative Decoding ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

# Simple ngram speculation with no additional model
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,
        "prompt_lookup_min": 2,
    }
)

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, max_tokens=100)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Generated: {output.outputs[0].text}")
</syntaxhighlight>

=== Example 2: EAGLE Speculative Decoding ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

# EAGLE with feature-level autoregression
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=4,
    speculative_config={
        "method": "eagle",
        "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
        "num_speculative_tokens": 3,
        "draft_tensor_parallel_size": 1,  # EAGLE must run with TP=1
    }
)

outputs = llm.generate(["Write a story"], SamplingParams(max_tokens=256))
</syntaxhighlight>

=== Example 3: EAGLE3 for Better Acceptance ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

# EAGLE3 provides improved acceptance rates
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "eagle3",
        "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
        "num_speculative_tokens": 3,
    }
)

outputs = llm.generate(
    ["Explain quantum computing"],
    SamplingParams(temperature=0.0, max_tokens=200)
)
</syntaxhighlight>

=== Example 4: MLP Speculator ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

# Lightweight MLP-based speculation
llm = LLM(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    speculative_config={
        "model": "ibm-ai-platform/llama3-70b-accelerator",
        "draft_tensor_parallel_size": 1,
    }
)

outputs = llm.generate(["Hello world"], SamplingParams(max_tokens=100))
</syntaxhighlight>

=== Example 5: Suffix Decoding ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

# Suffix decoding for repetitive patterns
# Requires: pip install arctic-inference
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "suffix",
        "num_speculative_tokens": 32,  # Maximum, dynamically adjusted
        "suffix_decoding_max_spec_factor": 1.0,
        "suffix_decoding_min_token_prob": 0.1,
    }
)

outputs = llm.generate(
    ["Refactor this code: def foo(): pass"],
    SamplingParams(temperature=0.0, max_tokens=200)
)
</syntaxhighlight>

=== Example 6: MTP (Multi-Token Prediction) ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

# For models with native MTP layers like DeepSeek-V3
llm = LLM(
    model="deepseek-ai/DeepSeek-V3",
    speculative_config={
        "method": "mtp",
        "num_speculative_tokens": 4,
    }
)

outputs = llm.generate(["Hello"], SamplingParams(max_tokens=100))
</syntaxhighlight>

== Design Details ==

=== Method Auto-Detection ===

The configuration automatically detects the speculative method when <code>model</code> is provided but <code>method</code> is not:

* If model name contains "eagle-" → detects as "eagle"
* If model name contains "eagle3" → detects as "eagle3"
* If model type is "medusa" → detects as "medusa"
* If model type is "mlp_speculator" → detects as "mlp_speculator"
* If model type is MTP variant → detects as "mtp"
* Otherwise → defaults to "draft_model" (not yet implemented)

=== Validation Logic ===

The <code>__post_init__</code> and <code>_verify_args</code> methods perform:

* Parameter completeness checks
* Method compatibility validation
* Draft model configuration creation
* Tensor parallelism compatibility checks
* N-gram parameter normalization

=== Integration Points ===

* '''LLM Initialization''': Passed via <code>speculative_config</code> parameter
* '''Engine Creation''': Used by <code>EngineArgs</code> to configure speculation
* '''Scheduler''': Influences token allocation and batch management
* '''Model Runner''': Configures draft model loading and execution

== Related Pages ==

* [[implements::Principle:vllm-project_vllm_spec_method_selection]]
* [[implemented_by::vllm-project_vllm_LLM_speculative]]
* vLLM Speculative Decoding Documentation
* EngineArgs Configuration
