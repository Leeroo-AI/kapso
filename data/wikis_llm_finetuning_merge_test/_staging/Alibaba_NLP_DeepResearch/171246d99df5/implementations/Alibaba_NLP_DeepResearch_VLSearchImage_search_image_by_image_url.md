# Implementation: VLSearchImage_search_image_by_image_url

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::File|vl_search_image.py|WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/vl_search_image.py]]
* [[source::API|SerpAPI|https://serpapi.com/google-reverse-image]]
|-
! Domains
| [[domain::Computer_Vision]], [[domain::Information_Retrieval]], [[domain::Multimodal]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==

Reverse image search method that uses SerpAPI's Google reverse image engine to find visually similar images and their associated metadata.

=== Description ===

The `search_image_by_image_url()` method in the `VLSearchImage` class performs reverse image search using Google's visual similarity algorithms. It takes a publicly accessible image URL, queries the SerpAPI Google reverse image endpoint, and returns structured results containing similar images, text snippets, and source URLs.

The method includes:
- Retry logic (up to 10 attempts) for API resilience
- Configurable timeout for slow responses
- Result parsing to extract thumbnails, snippets, and URLs
- Top-5 result limiting for concise output

=== Usage ===

Use `search_image_by_image_url()` when:
- The agent needs to identify unknown visual entities
- Gathering web context about image content
- Finding source attribution for images
- Supplementing vision-language model knowledge

This method is typically called by the agent when VLSearchImage tool is invoked.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba-NLP/DeepResearch]
* '''File:''' WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/vl_search_image.py
* '''Lines:''' 75-108

=== Signature ===
<syntaxhighlight lang="python">
def search_image_by_image_url(
    self,
    download_url: str,
    img_save_path: str = None,
    byte: bool = True,
    retry_attempt: int = 10,
    timeout: int = 30
) -> List[Dict]:
    """
    Perform reverse image search using Google's reverse image API.

    Args:
        download_url: str - Publicly accessible URL of the image to search
        img_save_path: str - Optional path to save downloaded image
        byte: bool - Whether to handle image as bytes (default True)
        retry_attempt: int - Maximum retry attempts for API calls (default 10)
        timeout: int - Request timeout in seconds (default 30)

    Returns:
        List[Dict] - List of search results, each containing:
            - "image_path": str - URL of visually similar image thumbnail
            - "snippet": str - Text description from source page
            - "url": str - Source webpage URL

    Note:
        - Returns top 5 results by default
        - Requires valid SERPAPI_KEY environment variable
        - Results are cached to avoid redundant API calls
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from qwen_agent.tools.vl_search_image import VLSearchImage

vl_search = VLSearchImage()
results = vl_search.search_image_by_image_url("https://example.com/image.jpg")
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| download_url || str || Yes || Publicly accessible URL of image to search
|-
| img_save_path || str || No || Optional local path to save the image
|-
| byte || bool || No || Handle image as bytes (default: True)
|-
| retry_attempt || int || No || Max retry attempts (default: 10)
|-
| timeout || int || No || Request timeout in seconds (default: 30)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| results || List[Dict] || List of up to 5 search results
|-
| results[].image_path || str || URL of visually similar image thumbnail
|-
| results[].snippet || str || Text snippet from source page
|-
| results[].url || str || Source webpage URL
|}

== Usage Examples ==

=== Basic Reverse Image Search ===
<syntaxhighlight lang="python">
from qwen_agent.tools.vl_search_image import VLSearchImage

vl_search = VLSearchImage()

# Search for visually similar images
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa.jpg/800px-Mona_Lisa.jpg"
results = vl_search.search_image_by_image_url(image_url)

for result in results:
    print(f"Similar image: {result['image_path']}")
    print(f"Description: {result['snippet']}")
    print(f"Source: {result['url']}")
    print("---")
</syntaxhighlight>

=== With Custom Retry Settings ===
<syntaxhighlight lang="python">
from qwen_agent.tools.vl_search_image import VLSearchImage

vl_search = VLSearchImage()

# Use more retries for unreliable network
results = vl_search.search_image_by_image_url(
    download_url="https://example.com/image.jpg",
    retry_attempt=20,
    timeout=60
)

if results:
    print(f"Found {len(results)} similar images")
else:
    print("No results found")
</syntaxhighlight>

=== Integration with Agent Tool Call ===
<syntaxhighlight lang="python">
from qwen_agent.tools.vl_search_image import VLSearchImage

vl_search = VLSearchImage()

# Tool call parameters from agent
tool_params = {
    "image_urls": ["https://example.com/unknown_landmark.jpg"]
}

# Process through the tool's call method
formatted_results = vl_search.call(tool_params)
print(formatted_results)  # Formatted string with images and snippets
</syntaxhighlight>

=== Handling Search Results ===
<syntaxhighlight lang="python">
from qwen_agent.tools.vl_search_image import VLSearchImage

vl_search = VLSearchImage()

results = vl_search.search_image_by_image_url(image_url)

# Extract entity information from snippets
for result in results:
    snippet = result.get("snippet", "")
    if snippet:
        # Snippets often contain entity names, dates, locations
        print(f"Context: {snippet}")

    # Visit source URL for more details
    source_url = result.get("url")
    if source_url:
        print(f"More info at: {source_url}")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Alibaba_NLP_DeepResearch_Reverse_Image_Search]]
