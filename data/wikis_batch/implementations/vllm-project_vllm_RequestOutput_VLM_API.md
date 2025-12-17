= RequestOutput VLM API =

{{Metadata
| Knowledge Sources = vllm/outputs.py, output data structures
| Domains = API Reference, Data Structures, Result Processing
| Last Updated = 2025-12-17
}}

== Overview ==

The '''RequestOutput VLM API''' implements the VLM Output Processing Principle by providing structured data classes that encapsulate generation results from vision-language models. This API defines the format and structure of outputs returned by the `generate()` method.

== Code Reference ==

=== Source Location ===

<syntaxhighlight lang="python">
# Output data structures
vllm/outputs.py: class RequestOutput
vllm/outputs.py: class CompletionOutput

# Logprob types
vllm/logprobs.py: SampleLogprobs, PromptLogprobs
</syntaxhighlight>

=== Data Structure ===

<syntaxhighlight lang="python">
@dataclass
class CompletionOutput:
    """Output data for one completion."""
    index: int                              # Output index
    text: str                               # Generated text
    token_ids: Sequence[int]                # Generated token IDs
    cumulative_logprob: Optional[float]     # Total log probability
    logprobs: Optional[SampleLogprobs]      # Per-token logprobs
    finish_reason: Optional[str]            # Why generation stopped
    stop_reason: Union[int, str, None]      # Stop token/string
    lora_request: Optional[LoRARequest]     # LoRA used if any

class RequestOutput:
    """Output data for a complete request."""
    request_id: str                         # Unique request ID
    prompt: Optional[str]                   # Original prompt text
    prompt_token_ids: Optional[List[int]]   # Prompt tokens
    prompt_logprobs: Optional[PromptLogprobs] # Prompt token logprobs
    outputs: List[CompletionOutput]         # Generated completions
    finished: bool                          # Is generation complete
    metrics: Optional[RequestMetrics]       # Performance metrics
    multi_modal_placeholders: MultiModalPlaceholderDict  # Placeholder info
    num_cached_tokens: Optional[int]        # Prefix cache hits
</syntaxhighlight>

=== Import ===

<syntaxhighlight lang="python">
from vllm.outputs import RequestOutput, CompletionOutput
</syntaxhighlight>

== Description ==

The output API provides a two-level structure:

* '''RequestOutput''': Top-level container for the entire request
  * Contains original prompt and all metadata
  * Holds list of completion outputs
  * Includes performance metrics and cache statistics

* '''CompletionOutput''': Individual completion within a request
  * Contains generated text and tokens
  * Provides finish reason and stop information
  * Includes log probabilities if requested

This structure supports multiple completions per request (via beam search or n > 1 sampling).

== I/O Contract ==

=== RequestOutput Fields ===

{| class="wikitable"
! Field !! Type !! Description
|-
| request_id || str || Unique identifier for the request
|-
| prompt || str || Original prompt text (may be None)
|-
| prompt_token_ids || List[int] || Tokenized prompt
|-
| outputs || List[CompletionOutput] || Generated completions
|-
| finished || bool || Whether generation completed
|-
| metrics || RequestMetrics || Timing and performance data
|-
| num_cached_tokens || int || Tokens from prefix cache
|}

=== CompletionOutput Fields ===

{| class="wikitable"
! Field !! Type !! Description
|-
| index || int || Completion index (for multiple outputs)
|-
| text || str || Generated text string
|-
| token_ids || Sequence[int] || Generated token IDs
|-
| cumulative_logprob || float || Sum of log probabilities
|-
| logprobs || SampleLogprobs || Per-token log probabilities
|-
| finish_reason || str || "stop", "length", "eos_token", etc.
|-
| stop_reason || int, str || Specific stop token/string
|}

== Usage Examples ==

=== Basic Output Access ===

<syntaxhighlight lang="python">
from PIL import Image
from vllm import LLM, SamplingParams

llm = LLM(model="llava-hf/llava-1.5-7b-hf")
image = Image.open("image.jpg")

prompt = {
    "prompt": "USER: <image>\nWhat is in this image?\nASSISTANT:",
    "multi_modal_data": {"image": image}
}

outputs = llm.generate([prompt], SamplingParams(max_tokens=100))

# Access first output
output = outputs[0]
print(f"Request ID: {output.request_id}")
print(f"Generated text: {output.outputs[0].text}")
print(f"Finished: {output.finished}")
</syntaxhighlight>

=== Extracting Token Information ===

<syntaxhighlight lang="python">
from PIL import Image
from vllm import LLM, SamplingParams

llm = LLM(model="llava-hf/llava-1.5-7b-hf")
image = Image.open("image.jpg")

prompt = {
    "prompt": "USER: <image>\nDescribe this image.\nASSISTANT:",
    "multi_modal_data": {"image": image}
}

outputs = llm.generate([prompt], SamplingParams(max_tokens=100))

completion = outputs[0].outputs[0]
print(f"Generated text: {completion.text}")
print(f"Token IDs: {completion.token_ids}")
print(f"Number of tokens: {len(completion.token_ids)}")
print(f"Finish reason: {completion.finish_reason}")
</syntaxhighlight>

=== Accessing Log Probabilities ===

<syntaxhighlight lang="python">
from PIL import Image
from vllm import LLM, SamplingParams

llm = LLM(model="llava-hf/llava-1.5-7b-hf")
image = Image.open("image.jpg")

prompt = {
    "prompt": "USER: <image>\nWhat is this?\nASSISTANT:",
    "multi_modal_data": {"image": image}
}

# Request logprobs for top 5 tokens
sampling_params = SamplingParams(max_tokens=50, logprobs=5)
outputs = llm.generate([prompt], sampling_params)

completion = outputs[0].outputs[0]

# Access logprobs for each generated token
if completion.logprobs:
    for i, token_logprobs in enumerate(completion.logprobs):
        token_id = completion.token_ids[i]
        print(f"Token {i} (ID {token_id}):")
        for tid, logprob_obj in token_logprobs.items():
            decoded = llm.get_tokenizer().decode([tid])
            print(f"  {decoded}: {logprob_obj.logprob:.4f}")
</syntaxhighlight>

=== Processing Batch Outputs ===

<syntaxhighlight lang="python">
from PIL import Image
from vllm import LLM, SamplingParams

llm = LLM(model="llava-hf/llava-1.5-7b-hf")

# Create batch of prompts
images = [Image.open(f"image{i}.jpg") for i in range(5)]
prompts = [
    {
        "prompt": f"USER: <image>\nDescribe image {i}.\nASSISTANT:",
        "multi_modal_data": {"image": img}
    }
    for i, img in enumerate(images)
]

outputs = llm.generate(prompts, SamplingParams(max_tokens=100))

# Process each output
for i, output in enumerate(outputs):
    print(f"\n=== Image {i} ===")
    print(f"Request ID: {output.request_id}")
    print(f"Prompt tokens: {len(output.prompt_token_ids)}")
    print(f"Generated: {output.outputs[0].text}")
    print(f"Output tokens: {len(output.outputs[0].token_ids)}")
    print(f"Finish reason: {output.outputs[0].finish_reason}")
</syntaxhighlight>

=== Checking Finish Reasons ===

<syntaxhighlight lang="python">
from PIL import Image
from vllm import LLM, SamplingParams

llm = LLM(model="llava-hf/llava-1.5-7b-hf")
image = Image.open("image.jpg")

prompt = {
    "prompt": "USER: <image>\nDescribe this image.\nASSISTANT:",
    "multi_modal_data": {"image": image}
}

# Generate with short max_tokens
sampling_params = SamplingParams(max_tokens=20)
outputs = llm.generate([prompt], sampling_params)

completion = outputs[0].outputs[0]

# Check why generation stopped
if completion.finish_reason == "length":
    print("Generation stopped due to max_tokens limit")
    print(f"Generated: {completion.text}")
    print("(may be truncated)")
elif completion.finish_reason == "stop":
    print("Generation stopped at stop sequence")
    print(f"Stop reason: {completion.stop_reason}")
elif completion.finish_reason == "eos_token":
    print("Generation completed naturally (EOS token)")
</syntaxhighlight>

=== Accessing Performance Metrics ===

<syntaxhighlight lang="python">
from PIL import Image
from vllm import LLM, SamplingParams

llm = LLM(model="llava-hf/llava-1.5-7b-hf")
image = Image.open("image.jpg")

prompt = {
    "prompt": "USER: <image>\nDescribe this image.\nASSISTANT:",
    "multi_modal_data": {"image": image}
}

outputs = llm.generate([prompt], SamplingParams(max_tokens=100))

output = outputs[0]

# Access metrics if available
if output.metrics:
    print(f"Metrics: {output.metrics}")

# Check prefix cache usage
if output.num_cached_tokens:
    print(f"Cached tokens: {output.num_cached_tokens}")
    print(f"Total prompt tokens: {len(output.prompt_token_ids)}")
    cache_ratio = output.num_cached_tokens / len(output.prompt_token_ids)
    print(f"Cache hit ratio: {cache_ratio:.2%}")
</syntaxhighlight>

=== Handling Multiple Completions ===

<syntaxhighlight lang="python">
from PIL import Image
from vllm import LLM, SamplingParams

llm = LLM(model="llava-hf/llava-1.5-7b-hf")
image = Image.open("image.jpg")

prompt = {
    "prompt": "USER: <image>\nWhat is in this image?\nASSISTANT:",
    "multi_modal_data": {"image": image}
}

# Generate multiple outputs with sampling
sampling_params = SamplingParams(
    n=3,  # Generate 3 different outputs
    temperature=0.8,
    max_tokens=100
)

outputs = llm.generate([prompt], sampling_params)

# Process all completions
request_output = outputs[0]
print(f"Generated {len(request_output.outputs)} completions:\n")

for i, completion in enumerate(request_output.outputs):
    print(f"Completion {i + 1}:")
    print(f"  Text: {completion.text}")
    print(f"  Cumulative logprob: {completion.cumulative_logprob:.4f}")
    print()
</syntaxhighlight>

=== Converting Output to Dictionary ===

<syntaxhighlight lang="python">
from PIL import Image
from vllm import LLM, SamplingParams

llm = LLM(model="llava-hf/llava-1.5-7b-hf")
image = Image.open("image.jpg")

prompt = {
    "prompt": "USER: <image>\nDescribe this image.\nASSISTANT:",
    "multi_modal_data": {"image": image}
}

outputs = llm.generate([prompt], SamplingParams(max_tokens=100))

output = outputs[0]
completion = output.outputs[0]

# Create custom output dictionary
result = {
    "request_id": output.request_id,
    "generated_text": completion.text,
    "prompt_length": len(output.prompt_token_ids),
    "output_length": len(completion.token_ids),
    "finish_reason": completion.finish_reason,
    "is_complete": output.finished
}

print(result)
</syntaxhighlight>

=== Streaming-style Processing ===

<syntaxhighlight lang="python">
from PIL import Image
from vllm import LLM, SamplingParams

llm = LLM(model="llava-hf/llava-1.5-7b-hf")

# Process images one at a time
image_files = ["image1.jpg", "image2.jpg", "image3.jpg"]

for img_file in image_files:
    image = Image.open(img_file)
    prompt = {
        "prompt": f"USER: <image>\nWhat's in this image?\nASSISTANT:",
        "multi_modal_data": {"image": image}
    }

    outputs = llm.generate([prompt], SamplingParams(max_tokens=100))

    output = outputs[0]
    print(f"\n{img_file}:")
    print(f"  {output.outputs[0].text}")
    print(f"  Tokens: {len(output.outputs[0].token_ids)}")
    print(f"  Status: {output.outputs[0].finish_reason}")
</syntaxhighlight>

=== Error Handling with Outputs ===

<syntaxhighlight lang="python">
from PIL import Image
from vllm import LLM, SamplingParams

llm = LLM(model="llava-hf/llava-1.5-7b-hf")
image = Image.open("image.jpg")

prompt = {
    "prompt": "USER: <image>\nDescribe this image.\nASSISTANT:",
    "multi_modal_data": {"image": image}
}

try:
    outputs = llm.generate([prompt], SamplingParams(max_tokens=100))

    for output in outputs:
        if not output.finished:
            print(f"Warning: Request {output.request_id} did not finish")
            continue

        completion = output.outputs[0]
        if completion.finish_reason == "length":
            print(f"Warning: Output may be truncated")

        print(f"Generated: {completion.text}")

except Exception as e:
    print(f"Generation error: {e}")
</syntaxhighlight>

== Related Pages ==

* [[implements::Principle:vllm-project_vllm_VLM_Output_Processing_Principle]]
* [[completes::Vision_Language_Multimodal_Inference_Workflow]]
* [[provides::Generation_Result_Structures]]
