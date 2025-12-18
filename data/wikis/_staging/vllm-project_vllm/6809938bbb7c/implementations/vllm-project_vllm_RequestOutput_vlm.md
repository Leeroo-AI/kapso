{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::Vision]], [[domain::NLP]], [[domain::Output_Processing]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

Concrete tool for extracting generated text and multimodal metadata from vision-language model inference results.

=== Description ===

`RequestOutput` from VLM inference includes the standard text output plus multimodal-specific metadata:
- **multi_modal_placeholders:** Positions where images were inserted
- Generated text describing or analyzing the images
- Standard completion metadata (finish_reason, token counts, etc.)

=== Usage ===

Process VLM outputs when:
- Extracting image captions or descriptions
- Analyzing VQA responses
- Debugging image-text alignment
- Building multimodal applications

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project/vllm]
* '''File:''' vllm/outputs.py
* '''Lines:''' L84-191

=== Interface ===
<syntaxhighlight lang="python">
class RequestOutput:
    """Output from VLM generation."""

    request_id: str
    prompt: str | None
    outputs: list[CompletionOutput]
    finished: bool

    multi_modal_placeholders: MultiModalPlaceholderDict | None
    """Positions where multimodal data was inserted."""

    # ... other fields

class CompletionOutput:
    text: str           # Generated text (caption/answer)
    token_ids: list[int]
    finish_reason: str
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| RequestOutput || RequestOutput || Yes || Output from VLM generate()
|}

=== Outputs (Accessible Fields) ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| outputs[0].text || str || Generated caption/answer
|-
| multi_modal_placeholders || dict || Image insertion positions (optional)
|}

== Usage Examples ==

=== Basic VLM Output Processing ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from PIL import Image

llm = LLM(model="llava-hf/llava-1.5-7b-hf", trust_remote_code=True)

image = Image.open("photo.jpg")

outputs = llm.generate(
    [{
        "prompt": "USER: <image>\nDescribe this image.\nASSISTANT:",
        "multi_modal_data": {"image": image},
    }],
    SamplingParams(max_tokens=200),
)

output = outputs[0]

# Extract generated caption
caption = output.outputs[0].text
print(f"Caption: {caption}")

# Check completion status
print(f"Finish reason: {output.outputs[0].finish_reason}")
</syntaxhighlight>

=== Batch Captioning with Results Collection ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from PIL import Image
from pathlib import Path

llm = LLM(model="llava-hf/llava-1.5-7b-hf", trust_remote_code=True)

# Process multiple images
image_dir = Path("images/")
results = []

prompts = []
file_names = []
for img_path in image_dir.glob("*.jpg"):
    image = Image.open(img_path)
    prompts.append({
        "prompt": "USER: <image>\nCaption this image.\nASSISTANT:",
        "multi_modal_data": {"image": image},
    })
    file_names.append(img_path.name)

outputs = llm.generate(prompts, SamplingParams(max_tokens=100))

# Collect results
for file_name, output in zip(file_names, outputs):
    results.append({
        "file": file_name,
        "caption": output.outputs[0].text.strip(),
        "tokens": len(output.outputs[0].token_ids),
    })

# Print results
for r in results:
    print(f"{r['file']}: {r['caption'][:50]}... ({r['tokens']} tokens)")
</syntaxhighlight>

=== VQA Response Analysis ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from PIL import Image

llm = LLM(model="llava-hf/llava-1.5-7b-hf", trust_remote_code=True)

image = Image.open("chart.png")

questions = [
    "What type of visualization is this?",
    "What is the main trend shown?",
    "What is the approximate maximum value?",
]

prompts = [
    {
        "prompt": f"USER: <image>\n{q}\nASSISTANT:",
        "multi_modal_data": {"image": image},
    }
    for q in questions
]

outputs = llm.generate(prompts, SamplingParams(max_tokens=100, temperature=0))

# Analyze responses
qa_pairs = []
for q, output in zip(questions, outputs):
    answer = output.outputs[0].text.strip()

    # Check for complete answers
    is_complete = output.outputs[0].finish_reason == "stop"

    qa_pairs.append({
        "question": q,
        "answer": answer,
        "complete": is_complete,
        "confidence": "high" if is_complete else "may be truncated",
    })

for qa in qa_pairs:
    print(f"Q: {qa['question']}")
    print(f"A: {qa['answer']} [{qa['confidence']}]\n")
</syntaxhighlight>

=== Handling Incomplete Generations ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from PIL import Image

llm = LLM(model="llava-hf/llava-1.5-7b-hf", trust_remote_code=True)

image = Image.open("complex_scene.jpg")

outputs = llm.generate(
    [{
        "prompt": "USER: <image>\nDescribe every object in this image.\nASSISTANT:",
        "multi_modal_data": {"image": image},
    }],
    SamplingParams(max_tokens=50),  # Low limit for demo
)

output = outputs[0]
completion = output.outputs[0]

if completion.finish_reason == "length":
    print("WARNING: Response was truncated")
    print(f"Generated text: {completion.text}...")

    # Could regenerate with higher max_tokens
    # or continue generation with the output as context

elif completion.finish_reason == "stop":
    print("Complete response:")
    print(completion.text)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_VLM_Output_Processing]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
