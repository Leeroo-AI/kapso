{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Model Loading]], [[domain::Serialization]], [[domain::Performance Optimization]], [[domain::Cloud Storage]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Tensorizer is a fast GPU model loading system that serializes and deserializes vLLM models directly to/from GPU memory over HTTP/S3/local storage with optional encryption.

=== Description ===
The tensorize_vllm_model.py script provides a complete solution for serializing vLLM models to tensorized format and loading them extremely quickly. Key features include:

* Direct GPU loading without CPU intermediate stage
* Support for distributed tensor-parallel models with per-shard serialization
* S3, HTTP/HTTPS, and local filesystem storage backends
* Optional encryption/decryption with libsodium
* LoRA adapter serialization and loading support
* Automatic handling of multi-GPU sharded models with template-based naming

The tensorized format significantly reduces model loading time by streaming tensors directly to GPU memory, making it ideal for serverless and dynamic scaling scenarios.

=== Usage ===
Use this script when you need to:
* Reduce model loading time for production deployments
* Load large models quickly from cloud storage (S3, GCS, etc.)
* Deploy models in serverless environments with cold start requirements
* Protect model weights with encryption
* Manage LoRA adapters alongside base models

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/others/tensorize_vllm_model.py examples/others/tensorize_vllm_model.py]

=== Signature ===
<syntaxhighlight lang="python">
# Main serialization function
def tensorize_vllm_model(
    engine_args: EngineArgs,
    tensorizer_config: TensorizerConfig
) -> None

# Main deserialization function
def deserialize(
    args,
    tensorizer_config: TensorizerConfig
) -> LLM

# Configuration class
class TensorizerConfig:
    tensorizer_uri: str  # Path to serialized tensors
    tensorizer_dir: str | None  # Directory with model artifacts
    encryption_keyfile: str | None  # Path to encryption key
    s3_access_key_id: str | None
    s3_secret_access_key: str | None
    s3_endpoint: str | None
    serialization_kwargs: dict
    deserialization_kwargs: dict
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm.model_executor.model_loader.tensorizer import (
    TensorizerArgs,
    TensorizerConfig,
    tensorize_lora_adapter,
    tensorize_vllm_model,
)
from vllm import LLM
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || str || HuggingFace model ID or path
|-
| serialized_directory || str || Target directory for serialized model (S3/local)
|-
| suffix || str || Version suffix for organizing serialized models
|-
| keyfile || str (optional) || Path to encryption key file
|-
| tensor_parallel_size || int || Number of GPUs for distributed loading
|-
| lora_path || str (optional) || HuggingFace LoRA adapter ID to serialize
|-
| s3_credentials || dict || S3 access credentials if using S3 storage
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model.tensors || file || Serialized model weights (or model-rank-%03d.tensors for sharded)
|-
| lora_artifacts || files || Serialized LoRA adapter files if --lora-path provided
|-
| LLM instance || LLM || Loaded model instance (deserialize mode)
|}

== Usage Examples ==

=== Serialize a Model to S3 ===
<syntaxhighlight lang="bash">
# Serialize single-GPU model
python examples/others/tensorize_vllm_model.py \
    --model facebook/opt-125m \
    serialize \
    --serialized-directory s3://my-bucket \
    --suffix v1

# Result: s3://my-bucket/vllm/facebook/opt-125m/v1/model.tensors

# Serialize multi-GPU model with encryption
python examples/others/tensorize_vllm_model.py \
    --model EleutherAI/gpt-j-6B \
    --tensor-parallel-size 2 \
    serialize \
    --serialized-directory s3://my-bucket \
    --suffix v1 \
    --keyfile encryption.key

# Result: s3://my-bucket/vllm/EleutherAI/gpt-j-6B/v1/model-rank-000.tensors
#         s3://my-bucket/vllm/EleutherAI/gpt-j-6B/v1/model-rank-001.tensors
</syntaxhighlight>

=== Deserialize and Load a Model ===
<syntaxhighlight lang="bash">
# Deserialize from S3
python examples/others/tensorize_vllm_model.py \
    --model EleutherAI/gpt-j-6B \
    --dtype float16 \
    deserialize \
    --path-to-tensors s3://my-bucket/vllm/EleutherAI/gpt-j-6B/v1/model.tensors

# Deserialize encrypted model
python examples/others/tensorize_vllm_model.py \
    --model EleutherAI/gpt-j-6B \
    deserialize \
    --serialized-directory s3://my-bucket/vllm/EleutherAI/gpt-j-6B/v1 \
    --keyfile encryption.key
</syntaxhighlight>

=== Use with LLM Class ===
<syntaxhighlight lang="python">
from vllm import LLM

# Load tensorized model directly
llm = LLM(
    "s3://my-bucket/vllm/facebook/opt-125m/v1",
    load_format="tensorizer",
)

# Use with vLLM server
# vllm serve s3://my-bucket/vllm/facebook/opt-125m/v1 \
#     --load-format tensorizer
</syntaxhighlight>

=== Serialize with LoRA Adapter ===
<syntaxhighlight lang="bash">
# Serialize base model with LoRA adapter
python examples/others/tensorize_vllm_model.py \
    --model meta-llama/Llama-2-7b-hf \
    --lora-path sql-lora-adapter \
    serialize \
    --serialized-directory s3://my-bucket \
    --suffix v1

# Use in server with LoRA enabled
# vllm serve s3://my-bucket/vllm/meta-llama/Llama-2-7b-hf/v1 \
#     --load-format tensorizer \
#     --enable-lora
</syntaxhighlight>

=== Configure S3 Credentials ===
<syntaxhighlight lang="bash">
# Via environment variables
export S3_ACCESS_KEY_ID=your_key_id
export S3_SECRET_ACCESS_KEY=your_secret_key
export S3_ENDPOINT_URL=https://s3.amazonaws.com

python examples/others/tensorize_vllm_model.py \
    --model facebook/opt-125m \
    serialize \
    --serialized-directory s3://my-bucket

# Or via command-line arguments
python examples/others/tensorize_vllm_model.py \
    --model facebook/opt-125m \
    --s3-access-key-id your_key_id \
    --s3-secret-access-key your_secret_key \
    --s3-endpoint https://s3.amazonaws.com \
    serialize \
    --serialized-directory s3://my-bucket
</syntaxhighlight>

== Performance Characteristics ==

* '''Loading Speed:''' 10-100x faster than standard model loading depending on network bandwidth
* '''Memory Efficiency:''' Direct GPU loading avoids CPU memory staging
* '''Storage:''' Tensorized models are approximately same size as original weights
* '''Encryption:''' Minimal performance overhead with libsodium

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
* [[LLM]]
* [[TensorizerConfig]]
* [[Model Loading Strategies]]
* [[LoRA Support]]
