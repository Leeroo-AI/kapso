{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::NLP]], [[domain::Structured_Output]], [[domain::Sampling]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

Concrete tool for configuring SamplingParams with structured output constraints for constrained generation.

=== Description ===

`SamplingParams` with `structured_outputs` enables constraint-aware generation:
- Logit masking ensures only valid tokens are sampled
- Constraint state tracks generation progress
- Compatible with all other sampling parameters
- Transparent integration with standard generate() API

=== Usage ===

Configure structured SamplingParams when:
- Extracting structured data from text
- Building classification systems
- Implementing tool calling
- Enforcing output format compliance

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project/vllm]
* '''File:''' vllm/sampling_params.py
* '''Lines:''' L111-241

=== Signature ===
<syntaxhighlight lang="python">
SamplingParams(
    n: int = 1,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    max_tokens: int | None = 16,
    stop: str | list[str] | None = None,
    structured_outputs: StructuredOutputsParams | None = None,  # Key parameter
    ...
) -> SamplingParams
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| structured_outputs || StructuredOutputsParams || No || Constraint configuration
|-
| max_tokens || int || No || Maximum tokens to generate
|-
| temperature || float || No || Sampling temperature
|-
| top_p || float || No || Nucleus sampling threshold
|-
| stop || list[str] || No || Stop sequences
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| params || SamplingParams || Configured sampling with constraints
|}

== Usage Examples ==

=== JSON Schema Generation ===
<syntaxhighlight lang="python">
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

class Product(BaseModel):
    name: str
    price: float
    in_stock: bool

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

sampling_params = SamplingParams(
    structured_outputs=StructuredOutputsParams(
        json=Product.model_json_schema(),
    ),
    max_tokens=100,
    temperature=0.3,
)

outputs = llm.generate(
    ["Extract product info: The new iPhone 15 costs $999 and is available now."],
    sampling_params,
)

import json
product = json.loads(outputs[0].outputs[0].text)
# {"name": "iPhone 15", "price": 999.0, "in_stock": true}
</syntaxhighlight>

=== Classification with Choices ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

# Sentiment classification
sampling_params = SamplingParams(
    structured_outputs=StructuredOutputsParams(
        choice=["positive", "negative", "neutral"],
    ),
    max_tokens=10,
    temperature=0,  # Deterministic
)

texts = [
    "I absolutely love this product!",
    "This is terrible, don't buy it.",
    "It's okay, nothing special.",
]

prompts = [f"Classify sentiment: '{t}' Answer:" for t in texts]
outputs = llm.generate(prompts, sampling_params)

for text, output in zip(texts, outputs):
    print(f"{text[:30]}... -> {output.outputs[0].text}")
</syntaxhighlight>

=== Regex Pattern Matching ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

# Phone number extraction
sampling_params = SamplingParams(
    structured_outputs=StructuredOutputsParams(
        regex=r"\(\d{3}\) \d{3}-\d{4}",
    ),
    max_tokens=20,
    temperature=0,
)

outputs = llm.generate(
    ["Extract phone number: Call us at (555) 123-4567 for support."],
    sampling_params,
)
# Output: (555) 123-4567
</syntaxhighlight>

=== Complex Nested Schema ===
<syntaxhighlight lang="python">
from pydantic import BaseModel
from typing import Optional
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

class Address(BaseModel):
    street: str
    city: str
    zip_code: str

class Contact(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    address: Optional[Address] = None

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

sampling_params = SamplingParams(
    structured_outputs=StructuredOutputsParams(
        json=Contact.model_json_schema(),
    ),
    max_tokens=300,  # Generous for nested structure
    temperature=0.5,
)

outputs = llm.generate(
    ["Extract contact: John Smith (john@email.com) at 123 Main St, NYC 10001"],
    sampling_params,
)
</syntaxhighlight>

=== Batch Structured Generation ===
<syntaxhighlight lang="python">
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

class Entity(BaseModel):
    type: str
    name: str
    description: str

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

sampling_params = SamplingParams(
    structured_outputs=StructuredOutputsParams(
        json=Entity.model_json_schema(),
    ),
    max_tokens=150,
    temperature=0.3,
)

prompts = [
    "Extract entity: Apple Inc. is a technology company founded by Steve Jobs.",
    "Extract entity: The Eiffel Tower is a famous landmark in Paris, France.",
    "Extract entity: Python is a programming language created by Guido van Rossum.",
]

outputs = llm.generate(prompts, sampling_params)

import json
for prompt, output in zip(prompts, outputs):
    entity = json.loads(output.outputs[0].text)
    print(f"Type: {entity['type']}, Name: {entity['name']}")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_Structured_SamplingParams]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
