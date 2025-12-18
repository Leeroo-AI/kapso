{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Embeddings]], [[domain::Classification]], [[domain::Pooling]], [[domain::Reranking]], [[domain::Semantic Search]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
PoolingParams defines API parameters for embedding, classification, scoring, and reranking tasks in vLLM pooling models.

=== Description ===
PoolingParams is a configuration class for controlling pooling model behavior across different tasks. Key features include:

* '''Multi-task support:''' embed, classify, score, token_embed, token_classify
* '''Matryoshka embeddings:''' Support for dimension reduction in compatible models
* '''Normalization:''' L2 normalization of embedding outputs
* '''Activation functions:''' Configurable softmax/sigmoid for classification tasks
* '''Truncation control:''' Flexible prompt token truncation strategies
* '''Task validation:''' Ensures only valid parameters are used per task
* '''Plugin extensibility:''' Custom pooling tasks via plugin system

The class uses msgspec for efficient serialization and provides task-specific parameter validation based on model capabilities.

=== Usage ===
Use this class when you need to:
* Generate sentence or document embeddings
* Perform text classification or scoring
* Rerank documents for search
* Control embedding dimensions for Matryoshka models
* Configure token-level embeddings or classification
* Implement custom pooling tasks via plugins

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/vllm/pooling_params.py vllm/pooling_params.py]

=== Signature ===
<syntaxhighlight lang="python">
class PoolingParams(msgspec.Struct):
    # Common parameters
    truncate_prompt_tokens: int | None = None  # -1 for model default, k for last k tokens, None to disable

    # Embedding parameters
    dimensions: int | None = None
    normalize: bool | None = None

    # Classification/scoring parameters
    use_activation: bool | None = None

    # Step pooling parameters
    step_tag_id: int | None = None
    returned_token_ids: list[int] | None = None

    # Internal
    task: PoolingTask | None = None
    requires_token_ids: bool = False
    skip_reading_prefix_cache: bool | None = None
    extra_kwargs: dict[str, Any] | None = None
    output_kind: RequestOutputKind = RequestOutputKind.FINAL_ONLY

    def verify(self, task: PoolingTask, model_config: ModelConfig | None) -> None
    def clone(self) -> "PoolingParams"
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm.pooling_params import PoolingParams
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| truncate_prompt_tokens || int &#124; None || -1: model default, k: keep last k tokens, None: no truncation
|-
| dimensions || int &#124; None || Output dimensions for Matryoshka embeddings
|-
| normalize || bool &#124; None || Whether to L2-normalize embeddings
|-
| use_activation || bool &#124; None || Apply softmax/sigmoid to classification outputs
|-
| task || PoolingTask &#124; None || Task type: embed, classify, score, token_embed, token_classify, plugin
|-
| step_tag_id || int &#124; None || Step tag ID for STEP pooling type
|-
| returned_token_ids || list[int] &#124; None || Specific token IDs to return for token-level tasks
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| PoolingParams || PoolingParams || Validated configuration object ready for inference
|}

== Usage Examples ==

=== Embedding Generation ===
<syntaxhighlight lang="python">
from vllm import LLM
from vllm.pooling_params import PoolingParams

# Basic embedding
params = PoolingParams(
    normalize=True,
    dimensions=None  # Full dimensions
)

llm = LLM(model="BAAI/bge-base-en-v1.5", task="embed")
outputs = llm.encode("Hello world", pooling_params=params)

# Matryoshka embedding (reduced dimensions)
params = PoolingParams(
    normalize=True,
    dimensions=256  # Reduce from 768 to 256
)

llm = LLM(model="nomic-ai/nomic-embed-text-v1.5", task="embed")
outputs = llm.encode("Hello world", pooling_params=params)
</syntaxhighlight>

=== Text Classification ===
<syntaxhighlight lang="python">
from vllm import LLM
from vllm.pooling_params import PoolingParams

# Binary classification with sigmoid
params = PoolingParams(
    use_activation=True  # Apply sigmoid
)

llm = LLM(model="your-classification-model", task="classify")
outputs = llm.classify("This movie is great!", pooling_params=params)

# Multi-class without activation (raw logits)
params = PoolingParams(
    use_activation=False
)
outputs = llm.classify("Document text", pooling_params=params)
</syntaxhighlight>

=== Document Scoring/Reranking ===
<syntaxhighlight lang="python">
from vllm import LLM
from vllm.pooling_params import PoolingParams

params = PoolingParams(
    use_activation=True  # Apply softmax for relevance scores
)

llm = LLM(model="cross-encoder/ms-marco-MiniLM-L-6-v2", task="score")

# Score query-document pairs
queries = ["What is AI?", "What is AI?"]
documents = ["AI is artificial intelligence", "The sky is blue"]

outputs = llm.score(
    prompts=[f"{q} [SEP] {d}" for q, d in zip(queries, documents)],
    pooling_params=params
)

for output in outputs:
    print(f"Relevance score: {output.outputs.data}")
</syntaxhighlight>

=== Token-Level Embeddings ===
<syntaxhighlight lang="python">
from vllm import LLM
from vllm.pooling_params import PoolingParams

params = PoolingParams(
    normalize=True,
    dimensions=768
)

llm = LLM(model="bert-base-uncased", task="token_embed")
outputs = llm.encode("Hello world", pooling_params=params)

# Output contains embeddings for each token
# Shape: (num_tokens, dimensions)
</syntaxhighlight>

=== Truncation Strategies ===
<syntaxhighlight lang="python">
from vllm.pooling_params import PoolingParams

# Use model's default truncation
params = PoolingParams(
    truncate_prompt_tokens=-1,
    normalize=True
)

# Keep only last 512 tokens (left truncation)
params = PoolingParams(
    truncate_prompt_tokens=512,
    normalize=True
)

# Disable truncation (may fail if exceeds model limit)
params = PoolingParams(
    truncate_prompt_tokens=None,
    normalize=True
)
</syntaxhighlight>

=== Plugin Task ===
<syntaxhighlight lang="python">
from vllm import LLM
from vllm.pooling_params import PoolingParams

# Custom pooling task
params = PoolingParams(
    extra_kwargs={"custom_param": "value"}
)

llm = LLM(
    model="custom-plugin-model",
    task="plugin"
)

outputs = llm.encode(
    prompts={"prompt_token_ids": [1, 2, 3], "multi_modal_data": {...}},
    pooling_params=params
)
</syntaxhighlight>

=== Validation and Defaults ===
<syntaxhighlight lang="python">
from vllm.pooling_params import PoolingParams
from vllm.config import ModelConfig

params = PoolingParams(
    dimensions=384,
    normalize=None  # Will be set to True by default for embed tasks
)

# Verify against task and model config
model_config = ModelConfig(...)
params.verify(task="embed", model_config=model_config)

print(params.normalize)  # True (default for embed tasks)

# Clone for modification
params2 = params.clone()
params2.dimensions = 256
</syntaxhighlight>

=== Error Handling ===
<syntaxhighlight lang="python">
from vllm.pooling_params import PoolingParams

# Invalid: using classification parameter for embedding task
params = PoolingParams(
    normalize=True,
    use_activation=True  # Not valid for embed task
)

try:
    params.verify(task="embed", model_config=None)
except ValueError as e:
    print(e)  # "Task embed only supports ['dimensions', 'normalize'] parameters"

# Invalid: dimensions on non-Matryoshka model
params = PoolingParams(
    dimensions=256
)

try:
    # Model that doesn't support Matryoshka
    params.verify(task="embed", model_config=non_matryoshka_config)
except ValueError as e:
    print(e)  # "Model does not support matryoshka representation"
</syntaxhighlight>

== Valid Parameters by Task ==

{| class="wikitable"
|-
! Task !! Valid Parameters
|-
| embed || dimensions, normalize
|-
| classify || use_activation
|-
| score || use_activation
|-
| token_embed || dimensions, normalize
|-
| token_classify || use_activation
|-
| plugin || All (no validation)
|}

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
* [[LLM]]
* [[PoolingTask]]
* [[ModelConfig]]
* [[PoolerConfig]]
* [[Matryoshka Embeddings]]
