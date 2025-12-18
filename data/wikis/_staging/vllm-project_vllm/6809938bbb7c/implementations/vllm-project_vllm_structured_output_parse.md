{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::NLP]], [[domain::Structured_Output]], [[domain::Output_Processing]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

Concrete patterns for parsing and validating structured generation outputs into typed Python objects.

=== Description ===

Structured output parsing involves:
- **JSON Parsing:** `json.loads()` for JSON schema outputs
- **Pydantic Validation:** Re-validate against original model
- **Direct Use:** Choice and regex outputs used as strings
- **Error Handling:** Graceful handling of malformed outputs

=== Usage ===

Parse structured outputs when:
- Processing generation results
- Building data pipelines
- Validating extraction quality
- Converting to typed objects

== Code Reference ==

=== Source Location ===
* '''Pattern:''' Python stdlib `json` module
* '''Example:''' `examples/offline_inference/structured_outputs.py:L91-93`

=== Core Pattern ===
<syntaxhighlight lang="python">
import json

# Parse JSON output
text = output.outputs[0].text
data = json.loads(text)
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
import json
from pydantic import BaseModel, ValidationError
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| text || str || Yes || Generated output text
|-
| model || BaseModel || No || Pydantic model for validation
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| data || dict | BaseModel || Parsed structured data
|}

== Usage Examples ==

=== Basic JSON Parsing ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from pydantic import BaseModel
import json

class Product(BaseModel):
    name: str
    price: float
    category: str

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

sampling_params = SamplingParams(
    structured_outputs=StructuredOutputsParams(
        json=Product.model_json_schema(),
    ),
    max_tokens=100,
)

outputs = llm.generate(
    ["Extract: iPhone 15 Pro, $1199, Electronics"],
    sampling_params,
)

# Parse the output
text = outputs[0].outputs[0].text
product_dict = json.loads(text)

print(f"Name: {product_dict['name']}")
print(f"Price: ${product_dict['price']}")
print(f"Category: {product_dict['category']}")
</syntaxhighlight>

=== Pydantic Model Validation ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from pydantic import BaseModel, ValidationError
import json

class Person(BaseModel):
    name: str
    age: int
    email: str

def parse_with_validation(text: str, model: type[BaseModel]) -> BaseModel:
    """Parse JSON and validate against Pydantic model."""
    data = json.loads(text)
    return model.model_validate(data)

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

sampling_params = SamplingParams(
    structured_outputs=StructuredOutputsParams(
        json=Person.model_json_schema(),
    ),
    max_tokens=150,
)

outputs = llm.generate(
    ["Extract: John Doe, 30 years old, john@example.com"],
    sampling_params,
)

# Parse and validate
try:
    person = parse_with_validation(
        outputs[0].outputs[0].text,
        Person,
    )
    print(f"Valid person: {person.name}, {person.age}")
except ValidationError as e:
    print(f"Validation error: {e}")
</syntaxhighlight>

=== Choice Output Handling ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

sampling_params = SamplingParams(
    structured_outputs=StructuredOutputsParams(
        choice=["high", "medium", "low"],
    ),
    max_tokens=10,
)

outputs = llm.generate(
    ["Rate urgency of: Server is down! Answer:"],
    sampling_params,
)

# Direct use - no parsing needed
priority = outputs[0].outputs[0].text.strip()
print(f"Priority: {priority}")

# Use in logic
if priority == "high":
    print("Alerting on-call team...")
</syntaxhighlight>

=== Regex Output Processing ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from datetime import datetime

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

sampling_params = SamplingParams(
    structured_outputs=StructuredOutputsParams(
        regex=r"\d{4}-\d{2}-\d{2}",
    ),
    max_tokens=15,
)

outputs = llm.generate(
    ["Convert to ISO date: March 15, 2024"],
    sampling_params,
)

# Parse as date
date_str = outputs[0].outputs[0].text.strip()
date_obj = datetime.strptime(date_str, "%Y-%m-%d")
print(f"Parsed date: {date_obj}")
</syntaxhighlight>

=== Batch Parsing Pipeline ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from pydantic import BaseModel
import json

class Entity(BaseModel):
    type: str
    name: str
    description: str

def parse_batch(outputs, model: type[BaseModel]) -> list[dict]:
    """Parse batch of structured outputs."""
    results = []
    for output in outputs:
        text = output.outputs[0].text
        try:
            data = json.loads(text)
            validated = model.model_validate(data)
            results.append({
                "success": True,
                "data": validated.model_dump(),
            })
        except Exception as e:
            results.append({
                "success": False,
                "error": str(e),
                "raw": text,
            })
    return results

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

sampling_params = SamplingParams(
    structured_outputs=StructuredOutputsParams(
        json=Entity.model_json_schema(),
    ),
    max_tokens=150,
)

prompts = [
    "Extract entity: Apple Inc is a tech company",
    "Extract entity: The Mona Lisa is a famous painting",
    "Extract entity: Python is a programming language",
]

outputs = llm.generate(prompts, sampling_params)
results = parse_batch(outputs, Entity)

for prompt, result in zip(prompts, results):
    if result["success"]:
        print(f"✓ {result['data']['name']}: {result['data']['type']}")
    else:
        print(f"✗ Parse error: {result['error']}")
</syntaxhighlight>

=== Error Recovery ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
import json

def safe_parse_json(text: str, default: dict = None) -> dict:
    """Safely parse JSON with fallback."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to fix common issues
        text = text.strip()

        # Handle truncated JSON
        if not text.endswith("}"):
            text += "}"

        try:
            return json.loads(text)
        except:
            return default or {}

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

sampling_params = SamplingParams(
    structured_outputs=StructuredOutputsParams(
        json={"type": "object", "properties": {"value": {"type": "string"}}},
    ),
    max_tokens=50,
)

outputs = llm.generate(["Generate data"], sampling_params)

# Safe parsing with fallback
result = safe_parse_json(
    outputs[0].outputs[0].text,
    default={"value": "unknown"},
)
print(f"Result: {result}")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_Structured_Output_Parsing]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
