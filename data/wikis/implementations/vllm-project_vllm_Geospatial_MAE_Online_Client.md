{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::Geospatial]], [[domain::Multimodal]], [[domain::Plugin]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Online inference client for geospatial satellite image segmentation using Prithvi EO model with GeoTIFF input/output.

=== Description ===
This example demonstrates online inference with vLLM's plugin system for geospatial foundation models. It processes GeoTIFF satellite imagery using the Prithvi-EO-2.0 model via vLLM's /pooling endpoint. The client sends a URL to a GeoTIFF image, which the server downloads, processes through the model, and returns base64-encoded segmentation results. This showcases vLLM's capability to handle domain-specific multimodal data formats beyond standard text and images, enabled by the terratorch_segmentation I/O processor plugin.

=== Usage ===
Use this example when performing geospatial AI tasks like satellite image segmentation, land use classification, or change detection using foundation models. It's particularly relevant for remote sensing applications requiring high-performance inference on multispectral GeoTIFF data.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/pooling/plugin/prithvi_geospatial_mae_client.py examples/pooling/plugin/prithvi_geospatial_mae_client.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Install requirements
pip install terratorch>=v1.1

# Start vLLM server with geospatial model and plugin
vllm serve christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM \
  --model-impl terratorch \
  --trust-remote-code \
  --skip-tokenizer-init \
  --enforce-eager \
  --io-processor-plugin terratorch_segmentation \
  --enable-mm-embeds

# Run online inference client
python prithvi_geospatial_mae_client.py
</syntaxhighlight>

== Key Concepts ==

=== I/O Processor Plugin ===
The terratorch_segmentation plugin handles GeoTIFF-specific data processing, including reading multispectral bands, applying normalization, and encoding output segmentation masks back to GeoTIFF format.

=== URL-Based Input ===
The client specifies a GeoTIFF URL in the request payload. The vLLM server downloads and processes the remote image, enabling distributed workflows without large file transfers.

=== Base64 Output Encoding ===
Segmentation results are returned as base64-encoded GeoTIFF data, which the client decodes and saves to disk. This enables binary data transmission over JSON API.

=== Model Implementation ===
The --model-impl terratorch flag tells vLLM to use TerraTorch's model implementation, which supports geospatial foundation models with specialized architectures.

=== Multimodal Embeddings ===
The --enable-mm-embeds flag enables processing of multimodal data, necessary for the model to handle both spatial image data and potential metadata.

== Usage Examples ==

<syntaxhighlight lang="python">
import base64
import os
import requests

# Configure request
image_url = "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/valencia_example_2024-10-26.tiff"
server_endpoint = "http://localhost:8000/pooling"

request_payload = {
    "data": {
        "data": image_url,           # URL to GeoTIFF
        "data_format": "url",         # Input from URL
        "image_format": "tiff",       # GeoTIFF format
        "out_data_format": "b64_json" # Base64 encoded output
    },
    "priority": 0,
    "model": "christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM"
}

# Send request
response = requests.post(server_endpoint, json=request_payload)

print(f"Status: {response.status_code}")
print(f"Reason: {response.reason}")

# Decode base64 output to GeoTIFF
response_data = response.json()
decoded_image = base64.b64decode(response_data["data"]["data"])

# Save to file
output_path = os.path.join(os.getcwd(), "online_prediction.tiff")
with open(output_path, "wb") as f:
    f.write(decoded_image)

print(f"Saved segmentation to: {output_path}")
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
* [[implements::Component:vLLM_Pooling_API]]
* [[uses::Tool:TerraTorch]]
* [[related::Implementation:vllm-project_vllm_Geospatial_MAE_Offline_Example]]
