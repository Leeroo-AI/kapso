{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::NLP]], [[domain::Structured_Output]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

Concrete tool for running constrained generation using LLM.generate() with structured output parameters.

=== Description ===

`LLM.generate()` with `structured_outputs` in `SamplingParams` enables:
- Transparent constraint enforcement via logit masking
- Same API as standard generation
- Batch support for multiple constrained requests
- Automatic backend selection (outlines/lm-format-enforcer)

=== Usage ===

Use constrained generate() when:
- Extracting structured data
- Building classification systems
- Implementing tool calling
- Enforcing output formats

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project/vllm]
* '''File:''' vllm/entrypoints/llm.py
* '''Lines:''' L365-434
* '''Backend:''' vllm/v1/sample/logits_processor.py

=== Signature ===
<syntaxhighlight lang="python">
def generate(
    self,
    prompts: PromptType | Sequence[PromptType],
    sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
    *,
    use_tqdm: bool | Callable = True,
    lora_request: LoRARequest | None = None,
    priority: list[int] | None = None,
) -> list[RequestOutput]:
    """Generate with optional structured output constraints."""
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
| prompts || PromptType | Sequence || Yes || Input prompts for generation
|-
| sampling_params || SamplingParams || No || Params with structured_outputs set
|-
| use_tqdm || bool || No || Show progress bar
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| outputs || list[RequestOutput] || Generated outputs conforming to constraints
|-
| text || str || Constraint-conforming generated text
|-
| finish_reason || str || "stop" when constraint completed
|}

== Usage Examples ==

=== Basic JSON Extraction ===
<syntaxhighlight lang="python">
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
import json

class MovieReview(BaseModel):
    title: str
    rating: int
    summary: str

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

sampling_params = SamplingParams(
    structured_outputs=StructuredOutputsParams(
        json=MovieReview.model_json_schema(),
    ),
    max_tokens=200,
    temperature=0.3,
)

outputs = llm.generate(
    ["Extract review: The Dark Knight (2008) is a masterpiece. 10/10. "
     "Heath Ledger's Joker is unforgettable."],
    sampling_params,
)

review = json.loads(outputs[0].outputs[0].text)
print(f"Movie: {review['title']}")
print(f"Rating: {review['rating']}/10")
print(f"Summary: {review['summary']}")
</syntaxhighlight>

=== Batch Classification ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

sampling_params = SamplingParams(
    structured_outputs=StructuredOutputsParams(
        choice=["spam", "not_spam"],
    ),
    max_tokens=10,
    temperature=0,
)

emails = [
    "Congratulations! You've won $1,000,000! Click here now!",
    "Hi John, please find the quarterly report attached.",
    "FREE PILLS!!! Act now before it's too late!!!",
    "Meeting reminder: 3pm tomorrow in Conference Room A.",
]

prompts = [f"Classify as spam or not_spam: '{e}' Answer:" for e in emails]
outputs = llm.generate(prompts, sampling_params)

for email, output in zip(emails, outputs):
    classification = output.outputs[0].text.strip()
    print(f"{email[:40]}... -> {classification}")
</syntaxhighlight>

=== Data Extraction Pipeline ===
<syntaxhighlight lang="python">
from pydantic import BaseModel
from typing import Optional
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
import json

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str

class Person(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[Address] = None

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

sampling_params = SamplingParams(
    structured_outputs=StructuredOutputsParams(
        json=Person.model_json_schema(),
    ),
    max_tokens=300,
    temperature=0.2,
)

documents = [
    "Contact: Jane Doe, jane@example.com, 555-1234, "
    "123 Main St, Boston, MA 02101",
    "John Smith can be reached at john.smith@corp.com",
]

prompts = [f"Extract person info from: {doc}" for doc in documents]
outputs = llm.generate(prompts, sampling_params)

for doc, output in zip(documents, outputs):
    person = json.loads(output.outputs[0].text)
    print(f"Name: {person['name']}, Email: {person.get('email', 'N/A')}")
</syntaxhighlight>

=== Multi-turn Tool Calling ===
<syntaxhighlight lang="python">
from pydantic import BaseModel
from typing import Literal
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
import json

class ToolCall(BaseModel):
    tool: Literal["search", "calculator", "weather"]
    query: str

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

sampling_params = SamplingParams(
    structured_outputs=StructuredOutputsParams(
        json=ToolCall.model_json_schema(),
    ),
    max_tokens=100,
    temperature=0.1,
)

user_queries = [
    "What's the weather in New York?",
    "Calculate 15% tip on $85",
    "Search for Python documentation",
]

prompts = [
    f"Select tool and query for: '{q}' "
    "Available tools: search, calculator, weather"
    for q in user_queries
]

outputs = llm.generate(prompts, sampling_params)

for query, output in zip(user_queries, outputs):
    tool_call = json.loads(output.outputs[0].text)
    print(f"Query: {query}")
    print(f"  Tool: {tool_call['tool']}, Args: {tool_call['query']}\n")
</syntaxhighlight>

=== Regex Structured Extraction ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

# Extract dates in ISO format
sampling_params = SamplingParams(
    structured_outputs=StructuredOutputsParams(
        regex=r"\d{4}-\d{2}-\d{2}",
    ),
    max_tokens=15,
    temperature=0,
)

texts = [
    "The meeting is scheduled for January 15, 2024.",
    "Project deadline: March 1st, 2024",
    "Event date: December 25, 2023",
]

prompts = [f"Convert to YYYY-MM-DD: {t}" for t in texts]
outputs = llm.generate(prompts, sampling_params)

for text, output in zip(texts, outputs):
    date = output.outputs[0].text.strip()
    print(f"{text} -> {date}")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_Constrained_Generation]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
