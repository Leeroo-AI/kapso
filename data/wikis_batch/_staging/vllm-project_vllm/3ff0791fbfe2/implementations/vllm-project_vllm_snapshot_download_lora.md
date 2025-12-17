'''Metadata'''
{| class="wikitable"
|-
! Key !! Value
|-
| Knowledge Sources || Hugging Face Hub Library, vLLM Examples
|-
| Domains || Model Download, Repository Management
|-
| Last Updated || 2025-12-17
|}

== Overview ==

'''snapshot_download_lora''' is a wrapper implementation using the Hugging Face Hub library's snapshot_download function to retrieve complete LoRA adapter repositories from Hugging Face Hub. This function downloads all adapter files to local storage for use with vLLM LoRA inference.

== Description ==

The snapshot_download function from huggingface_hub provides the primary mechanism for acquiring LoRA adapters from Hugging Face repositories. This implementation downloads an entire repository snapshot, including adapter weights, configuration files, and metadata, to a local cache directory.

=== Core Functionality ===

snapshot_download handles:
* '''Repository Resolution''': Converting repo_id strings (e.g., "username/adapter-name") to Hub API endpoints
* '''Authentication''': Using HF tokens for private repositories or gated models
* '''Revision Control''': Supporting specific commits, branches, or tags via revision parameter
* '''Caching''': Utilizing Hugging Face's local cache (~/.cache/huggingface/hub) with symlink-based storage
* '''Partial Downloads''': Optionally filtering files with allow_patterns/ignore_patterns
* '''Progress Tracking''': Displaying download progress for large adapter files

The function returns the local path to the downloaded snapshot, which can be passed directly to LoRARequest for adapter serving.

=== Integration with vLLM ===

In vLLM workflows, snapshot_download typically executes before request submission:

1. Download adapter repository to local cache
2. Obtain local path from snapshot_download return value
3. Create LoRARequest with the local path
4. Submit requests with the LoRARequest to the engine

This separation allows offline adapter preparation and efficient reuse across engine restarts.

== Code Reference ==

=== Source Location ===
* '''Library''': huggingface_hub (external dependency)
* '''Module''': huggingface_hub
* '''Function''': snapshot_download
* '''vLLM Usage''': examples/offline_inference/multilora_inference.py

=== Signature ===

<syntaxhighlight lang="python">
def snapshot_download(
    repo_id: str,
    *,
    revision: str | None = None,
    repo_type: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    token: str | bool | None = None,
    allow_patterns: list[str] | str | None = None,
    ignore_patterns: list[str] | str | None = None,
    resume_download: bool = True,
    force_download: bool = False,
    proxies: dict | None = None,
    etag_timeout: float = 10,
    local_files_only: bool = False,
    max_workers: int = 8,
) -> str:
    """
    Download a complete repository snapshot from Hugging Face Hub.

    Returns:
        Local path to the downloaded repository
    """
</syntaxhighlight>

=== Import ===

<syntaxhighlight lang="python">
from huggingface_hub import snapshot_download
</syntaxhighlight>

== I/O Contract ==

=== Input Parameters ===

{| class="wikitable"
|-
! Parameter !! Type !! Default !! Description
|-
| repo_id || str || required || Hugging Face repository ID (e.g., "username/lora-adapter")
|-
| revision || str \| None || None || Git revision (commit hash, branch, tag)
|-
| cache_dir || str \| Path \| None || None || Custom cache directory (default: ~/.cache/huggingface/hub)
|-
| token || str \| bool \| None || None || HF authentication token (True = load from ~/.huggingface/token)
|-
| allow_patterns || list[str] \| str \| None || None || File patterns to download (e.g., ["*.safetensors", "*.json"])
|-
| ignore_patterns || list[str] \| str \| None || None || File patterns to skip
|-
| local_files_only || bool || False || Use only cached files, don't download
|-
| resume_download || bool || True || Resume interrupted downloads
|-
| force_download || bool || False || Re-download even if cached
|}

=== Output ===

{| class="wikitable"
|-
! Return Type !! Description
|-
| str || Local filesystem path to downloaded repository snapshot
|}

== Usage Examples ==

=== Example 1: Basic Adapter Download ===

<syntaxhighlight lang="python">
from huggingface_hub import snapshot_download
from vllm import LLMEngine
from vllm.lora.request import LoRARequest

# Download LoRA adapter from Hugging Face Hub
lora_path = snapshot_download(repo_id="jeeejeee/llama32-3b-text2sql-spider")

print(f"Adapter downloaded to: {lora_path}")
# Output: /home/user/.cache/huggingface/hub/models--jeeejeee--llama32-3b-text2sql-spider/snapshots/abc123...

# Use with vLLM
lora_request = LoRARequest("sql-lora", 1, lora_path)
</syntaxhighlight>

=== Example 2: Download with Authentication ===

<syntaxhighlight lang="python">
from huggingface_hub import snapshot_download
import os

# Download private adapter requiring authentication
token = os.environ.get("HF_TOKEN")

lora_path = snapshot_download(
    repo_id="myorg/private-lora-adapter",
    token=token
)

# Or use auto-token loading
lora_path = snapshot_download(
    repo_id="myorg/private-lora-adapter",
    token=True  # Loads token from ~/.huggingface/token
)
</syntaxhighlight>

=== Example 3: Specific Revision ===

<syntaxhighlight lang="python">
from huggingface_hub import snapshot_download

# Download specific commit for reproducibility
lora_path = snapshot_download(
    repo_id="username/lora-adapter",
    revision="v1.0.0"  # Tag, branch, or commit hash
)

# Production deployment with pinned version
lora_path = snapshot_download(
    repo_id="username/production-adapter",
    revision="3a7b9c1e2d4f5g6h"  # Specific commit hash
)
</syntaxhighlight>

=== Example 4: Selective File Download ===

<syntaxhighlight lang="python">
from huggingface_hub import snapshot_download

# Download only adapter weights and config
lora_path = snapshot_download(
    repo_id="username/lora-adapter",
    allow_patterns=["*.safetensors", "*.json"],
    ignore_patterns=["*.md", "*.txt"]  # Skip documentation
)

# Minimal download for faster cold start
lora_path = snapshot_download(
    repo_id="username/lora-adapter",
    allow_patterns=["adapter_model.safetensors", "adapter_config.json"]
)
</syntaxhighlight>

=== Example 5: Offline Mode ===

<syntaxhighlight lang="python">
from huggingface_hub import snapshot_download

# Use only cached adapters (no network access)
try:
    lora_path = snapshot_download(
        repo_id="username/lora-adapter",
        local_files_only=True
    )
    print("Using cached adapter")
except FileNotFoundError:
    print("Adapter not in cache, download required")
</syntaxhighlight>

=== Example 6: Multi-LoRA Download ===

<syntaxhighlight lang="python">
from huggingface_hub import snapshot_download
from vllm.lora.request import LoRARequest

# Download multiple adapters for multi-LoRA serving
adapter_repos = [
    "user1/math-lora",
    "user2/code-lora",
    "user3/sql-lora"
]

lora_paths = {}
for repo_id in adapter_repos:
    lora_paths[repo_id] = snapshot_download(repo_id=repo_id)
    print(f"Downloaded {repo_id} to {lora_paths[repo_id]}")

# Create LoRA requests
lora_requests = [
    LoRARequest("math-lora", 1, lora_paths["user1/math-lora"]),
    LoRARequest("code-lora", 2, lora_paths["user2/code-lora"]),
    LoRARequest("sql-lora", 3, lora_paths["user3/sql-lora"])
]
</syntaxhighlight>

=== Example 7: Custom Cache Location ===

<syntaxhighlight lang="python">
from huggingface_hub import snapshot_download
from pathlib import Path

# Use custom cache directory (e.g., fast NVMe storage)
custom_cache = Path("/mnt/nvme/hf_cache")
custom_cache.mkdir(parents=True, exist_ok=True)

lora_path = snapshot_download(
    repo_id="username/lora-adapter",
    cache_dir=str(custom_cache)
)

print(f"Adapter cached in: {lora_path}")
</syntaxhighlight>

== Performance Considerations ==

* '''First Download''': May take 10s-5min depending on adapter size and network speed
* '''Subsequent Access''': Instant when using cached version
* '''Parallel Downloads''': Uses 8 worker threads by default (configurable with max_workers)
* '''Disk Space''': Cache grows with number of unique adapters downloaded

== Error Handling ==

<syntaxhighlight lang="python">
from huggingface_hub import snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

try:
    lora_path = snapshot_download(repo_id="username/lora-adapter")
except RepositoryNotFoundError:
    print("Repository not found - check repo ID and permissions")
except RevisionNotFoundError:
    print("Specified revision does not exist")
except Exception as e:
    print(f"Download failed: {e}")
</syntaxhighlight>

== Related Pages ==

* [[implements::Principle:vllm-project_vllm_LoRA_Adapter_Loading]]
* [[related_to::vllm-project_vllm_LoRARequest]]
* [[related_to::vllm-project_vllm_LoRAModelManager]]
