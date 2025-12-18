{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::HTTP_Client]], [[domain::Networking]], [[domain::File_Download]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
HTTP client wrapper providing synchronous and asynchronous methods for downloading models, images, and other assets.

=== Description ===
The connections.py module is a 189-line HTTP client abstraction that provides a clean, unified interface for making HTTP requests throughout vLLM. It wraps the popular requests library for synchronous operations and aiohttp for asynchronous operations, providing consistent error handling, automatic retries, and proper resource management.

The HTTPConnection class supports both synchronous and asynchronous workflows with methods for common operations: (1) get_response/get_async_response for raw HTTP responses with streaming support; (2) get_bytes/async_get_bytes for downloading binary data; (3) get_text/async_get_text for retrieving text content; (4) get_json/async_get_json for fetching and parsing JSON; (5) download_file/async_download_file for saving files to disk with chunked transfers. All methods validate URLs to ensure http/https schemes and include vLLM version in User-Agent headers.

The module supports client reuse to leverage connection pooling for better performance, controlled via the reuse_client parameter. It provides a global singleton instance (global_http_connection) used throughout vLLM for downloading model files from Hugging Face Hub, fetching images/videos for multimodal models, and accessing remote resources. The implementation properly handles async context managers and ensures resources are cleaned up correctly.

=== Usage ===
Use the global_http_connection instance for standard HTTP operations, or create custom instances for specific requirements.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/vllm/connections.py vllm/connections.py]
* '''Lines:''' 1-189

=== Signature ===
<syntaxhighlight lang="python">
class HTTPConnection:
    def __init__(self, *, reuse_client: bool = True) -> None

    def get_sync_client(self) -> requests.Session
    async def get_async_client(self) -> aiohttp.ClientSession

    def get_response(
        self,
        url: str,
        *,
        stream: bool = False,
        timeout: float | None = None,
        extra_headers: Mapping[str, str] | None = None,
        allow_redirects: bool = True,
    ) -> requests.Response

    async def get_async_response(
        self,
        url: str,
        *,
        timeout: float | None = None,
        extra_headers: Mapping[str, str] | None = None,
        allow_redirects: bool = True,
    ) -> aiohttp.ClientResponse

    def get_bytes(
        self,
        url: str,
        *,
        timeout: float | None = None,
        allow_redirects: bool = True
    ) -> bytes

    async def async_get_bytes(
        self,
        url: str,
        *,
        timeout: float | None = None,
        allow_redirects: bool = True,
    ) -> bytes

    def get_text(
        self,
        url: str,
        *,
        timeout: float | None = None
    ) -> str

    async def async_get_text(
        self,
        url: str,
        *,
        timeout: float | None = None,
    ) -> str

    def get_json(
        self,
        url: str,
        *,
        timeout: float | None = None
    ) -> dict

    async def async_get_json(
        self,
        url: str,
        *,
        timeout: float | None = None,
    ) -> dict

    def download_file(
        self,
        url: str,
        save_path: Path,
        *,
        timeout: float | None = None,
        chunk_size: int = 128,
    ) -> Path

    async def async_download_file(
        self,
        url: str,
        save_path: Path,
        *,
        timeout: float | None = None,
        chunk_size: int = 128,
    ) -> Path

# Global instance
global_http_connection: HTTPConnection
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm.connections import global_http_connection, HTTPConnection

# Use global instance for standard operations
data = global_http_connection.get_bytes("https://example.com/data.bin")

# Or create custom instance
custom_conn = HTTPConnection(reuse_client=True)
</syntaxhighlight>

== I/O Contract ==

=== Key Exports ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| HTTPConnection || Class || HTTP client with sync/async methods
|-
| global_http_connection || HTTPConnection || Global singleton instance
|-
| get_response || Method || Get raw HTTP response (sync)
|-
| get_async_response || Method || Get raw HTTP response (async)
|-
| get_bytes || Method || Download binary data (sync)
|-
| async_get_bytes || Method || Download binary data (async)
|-
| get_text || Method || Get text content (sync)
|-
| async_get_text || Method || Get text content (async)
|-
| get_json || Method || Fetch and parse JSON (sync)
|-
| async_get_json || Method || Fetch and parse JSON (async)
|-
| download_file || Method || Save file to disk (sync)
|-
| async_download_file || Method || Save file to disk (async)
|}

== Usage Examples ==

<syntaxhighlight lang="python">
import asyncio
from pathlib import Path
from vllm.connections import global_http_connection, HTTPConnection

# Example 1: Download binary data (images, model files)
url = "https://example.com/image.png"
image_bytes = global_http_connection.get_bytes(url, timeout=30.0)
print(f"Downloaded {len(image_bytes)} bytes")

# Example 2: Fetch JSON configuration
config_url = "https://huggingface.co/meta-llama/Llama-3.1-8B/raw/main/config.json"
config = global_http_connection.get_json(config_url)
print(f"Model config: {config}")

# Example 3: Download file to disk
model_url = "https://example.com/model.safetensors"
save_path = Path("/tmp/model.safetensors")

downloaded_path = global_http_connection.download_file(
    model_url,
    save_path,
    timeout=300.0,  # 5 minutes
    chunk_size=1024,  # 1KB chunks
)
print(f"Downloaded to: {downloaded_path}")

# Example 4: Async operations
async def fetch_multiple_files():
    urls = [
        "https://example.com/file1.json",
        "https://example.com/file2.json",
        "https://example.com/file3.json",
    ]

    # Fetch all concurrently
    tasks = [
        global_http_connection.async_get_json(url)
        for url in urls
    ]
    results = await asyncio.gather(*tasks)

    return results

# Run async example
# results = asyncio.run(fetch_multiple_files())

# Example 5: Custom headers and streaming
response = global_http_connection.get_response(
    "https://example.com/large_file.bin",
    stream=True,
    timeout=60.0,
    extra_headers={"Authorization": "Bearer token123"},
)

# Process response in chunks
with response:
    for chunk in response.iter_content(chunk_size=8192):
        # Process chunk
        pass

# Example 6: Error handling
try:
    data = global_http_connection.get_bytes(
        "https://invalid-domain-that-does-not-exist.com/file",
        timeout=10.0
    )
except Exception as e:
    print(f"Download failed: {e}")

# Example 7: Custom HTTPConnection instance
# Useful when you need different settings
no_reuse_conn = HTTPConnection(reuse_client=False)

# Each request gets a fresh client (no connection pooling)
data1 = no_reuse_conn.get_text("https://example.com/1")
data2 = no_reuse_conn.get_text("https://example.com/2")

# Example 8: Async file download
async def download_async_example():
    url = "https://example.com/video.mp4"
    save_path = Path("/tmp/video.mp4")

    downloaded = await global_http_connection.async_download_file(
        url,
        save_path,
        timeout=600.0,  # 10 minutes
        chunk_size=4096,
    )

    print(f"Async download complete: {downloaded}")
    return downloaded

# asyncio.run(download_async_example())

# Example 9: Integration with vLLM multimodal
# Used internally by vLLM to fetch images/videos
async def fetch_multimodal_data(url: str) -> bytes:
    """Fetch image or video data for multimodal models"""
    import vllm.envs as envs

    timeout = envs.VLLM_IMAGE_FETCH_TIMEOUT
    allow_redirects = envs.VLLM_MEDIA_URL_ALLOW_REDIRECTS

    return await global_http_connection.async_get_bytes(
        url,
        timeout=timeout,
        allow_redirects=allow_redirects,
    )

# Example 10: URL validation
# HTTPConnection validates URL schemes
try:
    # This will raise ValueError
    data = global_http_connection.get_bytes("ftp://invalid.com/file")
except ValueError as e:
    print(f"Invalid URL: {e}")

# Only http and https are supported
data = global_http_connection.get_bytes("https://valid.com/file")
</syntaxhighlight>

== Related Pages ==
* [[used_by::Module:vllm-project_vllm_Model_Loader]]
* [[used_by::Module:vllm-project_vllm_Multimodal_Inputs]]
* [[wraps::Library:requests]]
* [[wraps::Library:aiohttp]]
* [[related::Module:vllm-project_vllm_Environment_Variables]]
