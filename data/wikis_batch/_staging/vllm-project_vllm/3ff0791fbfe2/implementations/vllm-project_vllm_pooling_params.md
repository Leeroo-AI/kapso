# Implementation: Pooling Parameters for Embedding and Classification Models

**File:** `/tmp/praxium_repo_583nq7ea/vllm/pooling_params.py` (230 lines)
**Repository:** [vllm-project/vllm](https://github.com/vllm-project/vllm)

## Overview

The `pooling_params.py` module defines the `PoolingParams` class, which serves as the user-facing API for configuring pooling/embedding models in vLLM. It is the non-generative counterpart to `SamplingParams`, providing parameters for tasks like embedding generation, text classification, scoring, and reranking.

**Key Components:**
- `PoolingParams` class with task-specific parameters
- Parameter validation and verification logic
- Support for multiple pooling tasks: embed, classify, score, token_embed, token_classify
- Integration with model configuration defaults
- Matryoshka embedding dimension support

## Implementation Details

### Core Class: PoolingParams

```python
class PoolingParams(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
):  # type: ignore[call-arg]
    """API parameters for pooling models.

    Attributes:
        truncate_prompt_tokens: Controls prompt truncation.
            Set to -1 to use the model's default truncation size.
            Set to k to keep only the last k tokens (left truncation).
            Set to None to disable truncation.
        dimensions: Reduce the dimensions of embeddings
            if model support matryoshka representation.
        normalize: Whether to normalize the embeddings outputs.
        softmax: softmax will be deprecated, please use use_activation instead.
        activation: activation will be deprecated, please use use_activation instead.
        use_activation: Whether to apply activation function to
            the classification outputs.
    """
```

**Design Choice: msgspec.Struct**
- High-performance serialization/deserialization
- Type validation at construction time
- `omit_defaults=True`: Only serialize non-default values
- `array_like=True`: Enables array unpacking behavior

### Parameter Categories

#### 1. Common Parameters

```python
# --8<-- [start:common-pooling-params]
truncate_prompt_tokens: Annotated[int, msgspec.Meta(ge=-1)] | None = None
# --8<-- [end:common-pooling-params]
```

**Truncation Modes:**
- `None`: No truncation (default)
- `-1`: Use model's default truncation size
- `k > 0`: Keep last k tokens (left truncation)

**Use Cases:**
- Long document embeddings that exceed model context
- Enforcing consistent input lengths
- Optimizing for relevant content (keeping end of text)

#### 2. Embedding Parameters

```python
## for embeddings models
# --8<-- [start:embedding-pooling-params]
dimensions: int | None = None
normalize: bool | None = None
# --8<-- [end:embedding-pooling-params]
```

**`dimensions`:**
- For Matryoshka Representation Learning models
- Allows reducing output dimensionality post-hoc
- Example: Model outputs 768 dims, request 128 dims

**`normalize`:**
- L2 normalization of output embeddings
- Common for similarity search (cosine similarity)
- Default: `True` for embed tasks

#### 3. Classification/Scoring Parameters

```python
## for classification, scoring and rerank
# --8<-- [start:classification-pooling-params]
softmax: bool | None = None      # Deprecated
activation: bool | None = None    # Deprecated
use_activation: bool | None = None
# --8<-- [end:classification-pooling-params]
```

**`use_activation`:**
- Apply activation function (softmax/sigmoid) to outputs
- For classification: softmax for probabilities
- For scoring: sigmoid for bounded scores
- Default: `True` for classify/score tasks

**Deprecation Handling:**
```python
# raise deprecated warning for softmax and activation
self.use_activation = get_use_activation(self)
```

#### 4. Step Pooling Parameters

```python
## for step pooling models
step_tag_id: int | None = None
returned_token_ids: list[int] | None = None
```

**Use Case:**
- Specialized pooling at specific token positions
- Multi-step reasoning models
- Attention-guided pooling

#### 5. Internal Parameters

```python
## Internal use only
task: PoolingTask | None = None
requires_token_ids: bool = False
skip_reading_prefix_cache: bool | None = None
extra_kwargs: dict[str, Any] | None = None
output_kind: RequestOutputKind = RequestOutputKind.FINAL_ONLY
```

**`task`:**
- Set during verification: "embed", "classify", "score", etc.
- Determines which parameters are valid

**`skip_reading_prefix_cache`:**
- Controls prefix caching behavior
- `True` for token-level tasks (token_embed, token_classify)
- `False` for sequence-level tasks

**`output_kind`:**
- Always `FINAL_ONLY` for pooling (no streaming)

### Task-Specific Valid Parameters

```python
@property
def valid_parameters(self):
    return {
        "embed": ["dimensions", "normalize"],
        "classify": ["use_activation"],
        "score": ["use_activation"],
        "token_embed": ["dimensions", "normalize"],
        "token_classify": ["use_activation"],
    }
```

**Design Rationale:**
- Prevents misuse (e.g., normalize on classification task)
- Clear documentation of supported parameters per task
- Enables meaningful validation errors

### Clone Method

```python
def clone(self) -> "PoolingParams":
    """Returns a deep copy of the PoolingParams instance."""
    return deepcopy(self)
```

**Use Cases:**
- Creating variations of parameters
- Preserving original params while modifying copy
- Thread-safe parameter handling

### Verification Pipeline

```python
def verify(
    self, task: PoolingTask, model_config: Optional["ModelConfig"] = None
) -> None:
    if self.task is None:
        self.task = task
    elif self.task != task:
        msg = f"You cannot overwrite {self.task=!r} with {task=!r}!"
        raise ValueError(msg)

    # raise deprecated warning for softmax and activation
    self.use_activation = get_use_activation(self)

    # plugin task uses io_processor.parse_request to verify inputs,
    # skipping PoolingParams verify
    if self.task == "plugin":
        if self.skip_reading_prefix_cache is None:
            self.skip_reading_prefix_cache = True
        return

    self._merge_default_parameters(model_config)
    self._set_default_parameters(model_config)
    self._verify_valid_parameters()
```

**Verification Steps:**
1. Set or validate task
2. Handle deprecated parameters
3. Skip validation for plugin tasks
4. Merge model config defaults
5. Set task-specific defaults
6. Verify parameter validity

### Merging Model Config Defaults

```python
def _merge_default_parameters(
    self, model_config: Optional["ModelConfig"] = None
) -> None:
    if model_config is None:
        return

    pooler_config = model_config.pooler_config
    if pooler_config is None:
        return

    assert self.task is not None, "task must be set"
    valid_parameters = self.valid_parameters[self.task]

    for k in valid_parameters:
        if getattr(pooler_config, k, None) is None:
            continue

        if getattr(self, k, None) is None:
            setattr(self, k, getattr(pooler_config, k))

    if self.skip_reading_prefix_cache is None:
        # If prefix caching is enabled,
        # the output of all pooling may less than n_prompt_tokens,
        # we need to skip reading cache at this request.
        if self.task in ["token_embed", "token_classify"]:
            self.skip_reading_prefix_cache = True
        else:
            self.skip_reading_prefix_cache = False

    self._verify_step_pooling(pooler_config, valid_parameters)
```

**Merge Strategy:**
- User-provided values take precedence
- Model config provides defaults for unset parameters
- Only merges valid parameters for the task
- Sets cache behavior based on task type

### Step Pooling Verification

```python
def _verify_step_pooling(
    self, pooler_config: "PoolerConfig", valid_parameters: list[str]
):
    step_pooling_parameters = ["step_tag_id", "returned_token_ids"]
    if pooler_config.pooling_type != "STEP":
        invalid_parameters = []
        for k in step_pooling_parameters:
            if getattr(self, k, None) is not None:
                invalid_parameters.append(k)

        if invalid_parameters:
            raise ValueError(
                f"Task {self.task} only supports {valid_parameters} "
                f"parameters, does not support "
                f"{invalid_parameters} parameters"
            )
    else:
        for k in step_pooling_parameters:
            if getattr(pooler_config, k, None) is None:
                continue

            if getattr(self, k, None) is None:
                setattr(self, k, getattr(pooler_config, k))
```

**Logic:**
- If not STEP pooling: Reject step-specific parameters
- If STEP pooling: Merge step-specific defaults

### Setting Default Parameters

```python
def _set_default_parameters(self, model_config: Optional["ModelConfig"]):
    if self.task in ["embed", "token_embed"]:
        if self.normalize is None:
            self.normalize = True

        if self.dimensions is not None and model_config is not None:
            if not model_config.is_matryoshka:
                raise ValueError(
                    f'Model "{model_config.served_model_name}" does not '
                    f"support matryoshka representation, "
                    f"changing output dimensions will lead to poor results."
                )

            mds = model_config.matryoshka_dimensions
            if mds is not None:
                if self.dimensions not in mds:
                    raise ValueError(
                        f'Model "{model_config.served_model_name}" '
                        f"only supports {str(mds)} matryoshka dimensions, "
                        f"use other output dimensions will "
                        f"lead to poor results."
                    )
                elif self.dimensions < 1:
                    raise ValueError("Dimensions must be greater than 0")

    elif self.task in ["classify", "score", "token_classify"]:
        if self.use_activation is None:
            self.use_activation = True
    else:
        raise ValueError(f"Unknown pooling task: {self.task}")
```

**Default Setting Logic:**

**For Embedding Tasks:**
- Set `normalize = True` by default
- Validate Matryoshka dimensions if specified:
  - Check model supports Matryoshka
  - Check requested dimension is in supported set
  - Ensure dimension > 0

**For Classification/Scoring Tasks:**
- Set `use_activation = True` by default

### Parameter Validation

```python
def _verify_valid_parameters(self):
    assert self.task is not None, "task must be set"
    valid_parameters = self.valid_parameters[self.task]
    invalid_parameters = []
    for k in self.all_parameters:
        if k in valid_parameters:
            continue

        if getattr(self, k, None) is not None:
            invalid_parameters.append(k)

    if invalid_parameters:
        raise ValueError(
            f"Task {self.task} only supports {valid_parameters} "
            f"parameters, does not support "
            f"{invalid_parameters} parameters"
        )
```

**Validation Logic:**
- Check each parameter against valid set for task
- Collect all invalid parameters (not just first)
- Provide comprehensive error message

**Example Error:**
```
ValueError: Task embed only supports ['dimensions', 'normalize']
parameters, does not support ['use_activation'] parameters
```

### String Representation

```python
def __repr__(self) -> str:
    return (
        f"PoolingParams("
        f"task={self.task}, "
        f"normalize={self.normalize}, "
        f"dimensions={self.dimensions}, "
        f"use_activation={self.use_activation}, "
        f"step_tag_id={self.step_tag_id}, "
        f"returned_token_ids={self.returned_token_ids}, "
        f"requires_token_ids={self.requires_token_ids}, "
        f"skip_reading_prefix_cache={self.skip_reading_prefix_cache}, "
        f"truncate_prompt_tokens={self.truncate_prompt_tokens}, "
        f"extra_kwargs={self.extra_kwargs})"
    )
```

### Post-Initialization Validation

```python
def __post_init__(self) -> None:
    assert self.output_kind == RequestOutputKind.FINAL_ONLY, (
        "For pooling output_kind has to be FINAL_ONLY"
    )
```

**Invariant Enforcement:**
- Pooling tasks don't support streaming
- Only final output is meaningful

## Usage Patterns

### Basic Embedding

```python
from vllm import PoolingParams

# Simple embedding with defaults
params = PoolingParams()
# After verification: normalize=True, dimensions=None

# Custom normalization
params = PoolingParams(normalize=False)
```

### Matryoshka Embeddings

```python
# Request reduced dimensionality
params = PoolingParams(dimensions=256)

# Model must support [128, 256, 512, 768]
# Requesting 300 would raise ValueError
```

### Text Classification

```python
# Classification with probability output
params = PoolingParams()  # use_activation=True by default

# Raw logits (no softmax)
params = PoolingParams(use_activation=False)
```

### Scoring/Reranking

```python
# Semantic similarity scoring
params = PoolingParams()  # use_activation=True by default

# Useful for cross-encoder rerankers
```

### Truncation

```python
# Truncate to last 512 tokens (for long documents)
params = PoolingParams(truncate_prompt_tokens=512)

# Use model's default truncation
params = PoolingParams(truncate_prompt_tokens=-1)

# No truncation (may fail if input exceeds max_model_len)
params = PoolingParams(truncate_prompt_tokens=None)
```

### Token-Level Tasks

```python
# Token embeddings (e.g., for token classification)
params = PoolingParams(normalize=True)
# After verification for "token_embed" task:
#   skip_reading_prefix_cache=True
```

### Complete Example

```python
from vllm import LLM, PoolingParams

# Load embedding model
llm = LLM(model="BAAI/bge-small-en-v1.5", task="embed")

# Create pooling params
pooling_params = PoolingParams(
    normalize=True,
    truncate_prompt_tokens=512
)

# Generate embeddings
outputs = llm.encode(
    ["Hello world", "Goodbye world"],
    pooling_params=pooling_params
)

for output in outputs:
    print(f"Embedding: {output.outputs.embedding}")
```

## Task-Specific Behaviors

### Task: "embed"

**Valid Parameters:**
- `dimensions`: Output dimensionality (Matryoshka)
- `normalize`: L2 normalization

**Defaults:**
- `normalize = True`

**Use Cases:**
- Semantic search
- Document similarity
- Clustering

### Task: "classify"

**Valid Parameters:**
- `use_activation`: Apply softmax

**Defaults:**
- `use_activation = True`

**Use Cases:**
- Sentiment analysis
- Topic classification
- Intent detection

### Task: "score"

**Valid Parameters:**
- `use_activation`: Apply sigmoid/softmax

**Defaults:**
- `use_activation = True`

**Use Cases:**
- Cross-encoder reranking
- Relevance scoring
- Pairwise comparison

### Task: "token_embed"

**Valid Parameters:**
- `dimensions`: Output dimensionality
- `normalize`: L2 normalization

**Defaults:**
- `normalize = True`
- `skip_reading_prefix_cache = True`

**Use Cases:**
- Named entity recognition embeddings
- Token-level similarity

### Task: "token_classify"

**Valid Parameters:**
- `use_activation`: Apply activation per token

**Defaults:**
- `use_activation = True`
- `skip_reading_prefix_cache = True`

**Use Cases:**
- Part-of-speech tagging
- Named entity recognition
- Token-level sentiment

## Integration Points

### Model Configuration

```python
# Model defines default pooling config
model_config = ModelConfig(
    pooler_config=PoolerConfig(
        normalize=True,
        dimensions=768,
        pooling_type="LAST"
    )
)

# User params merge with defaults
user_params = PoolingParams(dimensions=256)  # Override dimensions
# Result: normalize=True (from config), dimensions=256 (from user)
```

### Request Processing

```python
# In API endpoint
def handle_embed_request(request):
    pooling_params = PoolingParams(
        normalize=request.normalize,
        dimensions=request.dimensions
    )

    pooling_params.verify(task="embed", model_config=model.config)

    return llm.encode(request.texts, pooling_params=pooling_params)
```

### Prefix Caching Integration

```python
# Token-level tasks skip prefix cache
if pooling_params.skip_reading_prefix_cache:
    # Don't use cached results
    process_full_sequence()
else:
    # Can use prefix cache for sequence-level pooling
    use_cached_prefix_if_available()
```

## Validation Examples

### Valid Usage

```python
# Embed task with valid params
params = PoolingParams(normalize=True, dimensions=256)
params.verify(task="embed", model_config=config)
# ✓ Success

# Classify task with valid params
params = PoolingParams(use_activation=True)
params.verify(task="classify", model_config=config)
# ✓ Success
```

### Invalid Usage

```python
# Embed task with classification params
params = PoolingParams(use_activation=True)
params.verify(task="embed", model_config=config)
# ✗ ValueError: Task embed only supports ['dimensions', 'normalize']
#              parameters, does not support ['use_activation'] parameters

# Classify task with embedding params
params = PoolingParams(normalize=True)
params.verify(task="classify", model_config=config)
# ✗ ValueError: Task classify only supports ['use_activation']
#              parameters, does not support ['normalize'] parameters

# Invalid Matryoshka dimension
params = PoolingParams(dimensions=300)  # Model supports [128, 256, 512]
params.verify(task="embed", model_config=config)
# ✗ ValueError: Model "..." only supports [128, 256, 512] matryoshka
#              dimensions, use other output dimensions will lead to poor results

# Non-Matryoshka model with dimensions
params = PoolingParams(dimensions=256)
params.verify(task="embed", model_config=non_matryoshka_config)
# ✗ ValueError: Model "..." does not support matryoshka representation,
#              changing output dimensions will lead to poor results
```

## Performance Considerations

### Serialization Efficiency

msgspec.Struct provides:
- Fast serialization (faster than dataclasses)
- Compact representation (`omit_defaults=True`)
- Type validation at construction

### Parameter Merging

Lazy parameter merging:
- Only happens during `verify()`
- Not on every parameter access
- Cached in object state

### Memory Efficiency

```python
# Only non-default values stored (omit_defaults=True)
params1 = PoolingParams()  # Minimal memory
params2 = PoolingParams(
    normalize=True,
    dimensions=256,
    use_activation=False,
    # ... many parameters
)  # More memory
```

## Design Rationale

### Why Task-Specific Validation?

**Alternative:** Allow all parameters for all tasks
**Problem:** User confusion, silent bugs

**Chosen Approach:** Strict validation
**Benefits:**
- Clear error messages
- Prevents misuse
- Documents intended usage

### Why Merge Model Config Defaults?

**Alternative:** Require explicit parameter values
**Problem:** Verbose API, model-specific knowledge required

**Chosen Approach:** Merge defaults from model config
**Benefits:**
- Model authors define sensible defaults
- Users only override when needed
- Forward compatibility (new parameters get defaults)

### Why Deprecation Path for softmax/activation?

**Alternative:** Breaking change
**Problem:** Existing code breaks

**Chosen Approach:** Gradual deprecation with warnings
**Benefits:**
- Smooth migration
- Clear upgrade path
- Backward compatibility period

## Testing Considerations

### Test Cases

1. **Default Values:**
```python
def test_embed_defaults():
    params = PoolingParams()
    params.verify(task="embed")
    assert params.normalize == True
    assert params.dimensions is None
```

2. **Parameter Validation:**
```python
def test_invalid_param_for_task():
    params = PoolingParams(use_activation=True)
    with pytest.raises(ValueError, match="only supports"):
        params.verify(task="embed")
```

3. **Matryoshka Validation:**
```python
def test_invalid_matryoshka_dim():
    config = ModelConfig(matryoshka_dimensions=[128, 256])
    params = PoolingParams(dimensions=300)
    with pytest.raises(ValueError, match="only supports"):
        params.verify(task="embed", model_config=config)
```

4. **Config Merging:**
```python
def test_merge_config_defaults():
    config = ModelConfig(pooler_config=PoolerConfig(normalize=False))
    params = PoolingParams()  # No explicit normalize
    params.verify(task="embed", model_config=config)
    assert params.normalize == False  # From config
```

5. **Task Switching Prevention:**
```python
def test_cannot_change_task():
    params = PoolingParams()
    params.verify(task="embed")
    with pytest.raises(ValueError, match="cannot overwrite"):
        params.verify(task="classify")
```

## Related Components

- **vllm.config.PoolerConfig**: Model-level pooling configuration
- **vllm.tasks.PoolingTask**: Task type definitions
- **vllm.sampling_params.SamplingParams**: Analogous for generation tasks
- **vllm.outputs**: Output structures for pooling results
- **vllm.entrypoints.openai**: OpenAI-compatible embedding API

## Future Enhancements

1. **Additional Pooling Strategies:**
   - Mean pooling
   - Max pooling
   - Attention-weighted pooling

2. **Dynamic Dimension Reduction:**
   - PCA-based reduction for non-Matryoshka models
   - Learned projection layers

3. **Batch-Specific Parameters:**
   - Per-input dimensions/normalization
   - Heterogeneous batching support

4. **Advanced Truncation:**
   - Sliding window truncation
   - Importance-based truncation
   - Bidirectional truncation

5. **Output Formatting:**
   - Sparse embedding support
   - Quantized embeddings
   - Binary embeddings

## Summary

The `PoolingParams` class provides a robust, task-aware API for configuring pooling models in vLLM. Its validation system ensures correct parameter usage while its default merging strategy balances ease-of-use with flexibility. The support for Matryoshka embeddings and task-specific behaviors makes it a comprehensive solution for diverse non-generative NLP tasks, from semantic search to text classification.
