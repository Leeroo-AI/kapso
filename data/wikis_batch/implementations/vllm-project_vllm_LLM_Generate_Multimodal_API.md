= LLM Generate Multimodal API =

{{Metadata
| Knowledge Sources = vllm/entrypoints/llm.py generate() method, vllm/outputs.py
| Domains = API Reference, Text Generation, Multimodal Inference
| Last Updated = 2025-12-17
}}

== Overview ==

The '''LLM Generate Multimodal API''' implements the Multimodal Generation Principle by providing the `generate()` method that processes vision-language prompts and produces text outputs. This API is the primary interface for performing inference with vision-language models in vLLM.

== Code Reference ==

=== Source Location ===

<syntaxhighlight lang="python">
# Main generation method
vllm/entrypoints/llm.py: class LLM.generate

# Output types
vllm/outputs.py: class RequestOutput, class CompletionOutput

# Sampling configuration
vllm/sampling_params.py: class SamplingParams
</syntaxhighlight>

=== Signature ===

<syntaxhighlight lang="python">
class LLM:
    def generate(
        self,
        prompts: Union[str, List[str], Dict, List[Dict]],
        sampling_params: Optional[Union[SamplingParams, List[SamplingParams]]] = None,
        *,
        use_tqdm: bool = True,
        lora_request: Optional[Union[LoRARequest, List[LoRARequest]]] = None,
        priority: Optional[List[int]] = None
    ) -> List[RequestOutput]:
        """Generate text outputs for the given prompts."""
</syntaxhighlight>

=== Import ===

<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
</syntaxhighlight>

== Description ==

The `generate()` method processes multimodal prompts through the following pipeline:

1. '''Input Validation''': Validates prompt structure and multimodal data
2. '''Preprocessing''': Processes images through vision encoder and prepares embeddings
3. '''Request Creation''': Creates inference requests with combined text and visual features
4. '''Batching''': Groups requests for efficient parallel processing
5. '''Model Execution''': Runs autoregressive generation with multimodal context
6. '''Decoding''': Converts output token IDs to text
7. '''Output Assembly''': Packages results into `RequestOutput` objects

The method supports both single and batch inference, automatically handling scheduling and memory management.

== I/O Contract ==

=== Input Parameters ===

{| class="wikitable"
! Parameter !! Type !! Description
|-
| prompts || str, List[str], Dict, List[Dict] || Text prompt or multimodal prompt dictionary
|-
| sampling_params || SamplingParams, List[SamplingParams] || Generation configuration
|-
| use_tqdm || bool || Show progress bar (default: True)
|-
| lora_request || LoRARequest, List[LoRARequest] || Optional LoRA adapters
|-
| priority || List[int] || Request priorities (if scheduler supports)
|}

=== Multimodal Prompt Structure ===

<syntaxhighlight lang="python">
{
    "prompt": str,                          # Text prompt with placeholders
    "multi_modal_data": {                   # Multimodal inputs
        "image": PIL.Image | List[PIL.Image],
        "video": np.ndarray | List[np.ndarray]
    },
    "multi_modal_uuids": {                  # Optional UUIDs for caching
        "image": str | List[str],
        "video": str | List[str]
    },
    "mm_processor_kwargs": Dict[str, Any]   # Optional processor overrides
}
</syntaxhighlight>

=== Output ===

Returns `List[RequestOutput]` where each `RequestOutput` contains:

* `request_id`: Unique request identifier
* `prompt`: Original prompt text
* `prompt_token_ids`: Tokenized prompt
* `outputs`: List of `CompletionOutput` objects with generated text
* `finished`: Whether generation completed
* `metrics`: Performance metrics

== Usage Examples ==

=== Basic Multimodal Generation ===

<syntaxhighlight lang="python">
from PIL import Image
from vllm import LLM, SamplingParams

# Initialize model
llm = LLM(model="llava-hf/llava-1.5-7b-hf")

# Load image
image = Image.open("photo.jpg")

# Create prompt
prompt = {
    "prompt": "USER: <image>\nWhat is in this image?\nASSISTANT:",
    "multi_modal_data": {"image": image}
}

# Generate with sampling params
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=100,
    top_p=0.9
)

outputs = llm.generate([prompt], sampling_params)

# Extract generated text
generated_text = outputs[0].outputs[0].text
print(generated_text)
</syntaxhighlight>

=== Batch Multimodal Generation ===

<syntaxhighlight lang="python">
from PIL import Image
from vllm import LLM, SamplingParams

llm = LLM(model="llava-hf/llava-1.5-7b-hf")

# Prepare multiple prompts with different images
images = [Image.open(f"image{i}.jpg") for i in range(1, 4)]
questions = [
    "Describe this image.",
    "What objects do you see?",
    "What is the main subject?"
]

prompts = [
    {
        "prompt": f"USER: <image>\n{question}\nASSISTANT:",
        "multi_modal_data": {"image": img}
    }
    for img, question in zip(images, questions)
]

# Batch generation
sampling_params = SamplingParams(max_tokens=150)
outputs = llm.generate(prompts, sampling_params)

# Process results
for i, output in enumerate(outputs):
    print(f"Image {i+1}: {output.outputs[0].text}")
    print("-" * 50)
</syntaxhighlight>

=== Generation with Different Sampling Strategies ===

<syntaxhighlight lang="python">
from PIL import Image
from vllm import LLM, SamplingParams

llm = LLM(model="llava-hf/llava-1.5-7b-hf")
image = Image.open("image.jpg")

prompt = {
    "prompt": "USER: <image>\nDescribe this image.\nASSISTANT:",
    "multi_modal_data": {"image": image}
}

# Greedy decoding (temperature=0)
greedy_params = SamplingParams(temperature=0, max_tokens=100)
greedy_output = llm.generate([prompt], greedy_params)[0]
print("Greedy:", greedy_output.outputs[0].text)

# Sampling with high temperature
creative_params = SamplingParams(temperature=1.0, max_tokens=100)
creative_output = llm.generate([prompt], creative_params)[0]
print("Creative:", creative_output.outputs[0].text)

# Top-p nucleus sampling
nucleus_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=100)
nucleus_output = llm.generate([prompt], nucleus_params)[0]
print("Nucleus:", nucleus_output.outputs[0].text)
</syntaxhighlight>

=== Generation with Stop Sequences ===

<syntaxhighlight lang="python">
from PIL import Image
from vllm import LLM, SamplingParams

llm = LLM(model="llava-hf/llava-1.5-7b-hf")
image = Image.open("image.jpg")

prompt = {
    "prompt": "USER: <image>\nList the objects you see:\n1.",
    "multi_modal_data": {"image": image}
}

# Stop at newline to get first item only
sampling_params = SamplingParams(
    max_tokens=50,
    stop=["\n", "2."],  # Stop at newline or next number
    temperature=0.7
)

output = llm.generate([prompt], sampling_params)[0]
print("First item:", output.outputs[0].text)
</syntaxhighlight>

=== Multi-Image Generation ===

<syntaxhighlight lang="python">
from PIL import Image
from vllm import LLM, SamplingParams

# Use model that supports multiple images
llm = LLM(
    model="OpenGVLab/InternVL3-2B",
    trust_remote_code=True,
    limit_mm_per_prompt={"image": 2}
)

image1 = Image.open("image1.jpg")
image2 = Image.open("image2.jpg")

prompt = {
    "prompt": "<image><image>\nCompare these two images. What are the similarities and differences?",
    "multi_modal_data": {"image": [image1, image2]}
}

sampling_params = SamplingParams(max_tokens=200, temperature=0.7)
output = llm.generate([prompt], sampling_params)[0]
print(output.outputs[0].text)
</syntaxhighlight>

=== Video Generation ===

<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.assets.video import VideoAsset

llm = LLM(model="llava-hf/LLaVA-NeXT-Video-7B-hf")

# Load video as numpy array
video = VideoAsset(name="baby_reading", num_frames=16).np_ndarrays

prompt = {
    "prompt": "USER: <video>\nDescribe what is happening in this video. ASSISTANT:",
    "multi_modal_data": {"video": video}
}

sampling_params = SamplingParams(max_tokens=150, temperature=0.7)
output = llm.generate([prompt], sampling_params)[0]
print(output.outputs[0].text)
</syntaxhighlight>

=== Accessing Output Metadata ===

<syntaxhighlight lang="python">
from PIL import Image
from vllm import LLM, SamplingParams

llm = LLM(model="llava-hf/llava-1.5-7b-hf")
image = Image.open("image.jpg")

prompt = {
    "prompt": "USER: <image>\nWhat is in this image?\nASSISTANT:",
    "multi_modal_data": {"image": image}
}

sampling_params = SamplingParams(max_tokens=100, logprobs=5)
output = llm.generate([prompt], sampling_params)[0]

# Access metadata
print(f"Request ID: {output.request_id}")
print(f"Prompt tokens: {len(output.prompt_token_ids)}")
print(f"Generated text: {output.outputs[0].text}")
print(f"Generated tokens: {output.outputs[0].token_ids}")
print(f"Finish reason: {output.outputs[0].finish_reason}")

# Access log probabilities if requested
if output.outputs[0].logprobs:
    print(f"Top token logprobs: {output.outputs[0].logprobs[0]}")
</syntaxhighlight>

=== Generation with Progress Tracking ===

<syntaxhighlight lang="python">
from PIL import Image
from vllm import LLM, SamplingParams
from tqdm import tqdm

llm = LLM(model="llava-hf/llava-1.5-7b-hf")

# Create many prompts for batch processing
images = [Image.open(f"image{i}.jpg") for i in range(10)]
prompts = [
    {
        "prompt": f"USER: <image>\nDescribe image {i}.\nASSISTANT:",
        "multi_modal_data": {"image": img}
    }
    for i, img in enumerate(images)
]

sampling_params = SamplingParams(max_tokens=100)

# Generate with progress bar
outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

# Process results
for i, output in enumerate(outputs):
    print(f"Output {i}: {output.outputs[0].text[:50]}...")
</syntaxhighlight>

=== Handling Generation Errors ===

<syntaxhighlight lang="python">
from PIL import Image
from vllm import LLM, SamplingParams

llm = LLM(model="llava-hf/llava-1.5-7b-hf")
image = Image.open("image.jpg")

prompt = {
    "prompt": "USER: <image>\nDescribe this image.\nASSISTANT:",
    "multi_modal_data": {"image": image}
}

sampling_params = SamplingParams(max_tokens=100)

try:
    outputs = llm.generate([prompt], sampling_params)

    for output in outputs:
        if output.finished:
            print("Generated:", output.outputs[0].text)
        else:
            print("Generation incomplete:", output.outputs[0].finish_reason)

except Exception as e:
    print(f"Generation failed: {e}")
</syntaxhighlight>

== Related Pages ==

* [[implements::Principle:vllm-project_vllm_Multimodal_Generation_Principle]]
* [[next_step::vllm-project_vllm_RequestOutput_VLM_API]]
* [[uses::vllm-project_vllm_SamplingParams]]
