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
Offline batch inference example for geospatial satellite image segmentation using vLLM's LLM class with plugin support.

=== Description ===
This example demonstrates offline batch inference for geospatial foundation models using vLLM's Python API. It processes GeoTIFF satellite imagery using the Prithvi-EO-2.0 model directly through the LLM class without requiring a separate server. The example uses the terratorch_segmentation I/O processor plugin to handle GeoTIFF-specific data formats and returns base64-encoded segmentation masks. This approach is suitable for batch processing workflows where you want direct Python integration rather than HTTP API calls.

=== Usage ===
Use this example for offline batch processing of satellite imagery, research experiments requiring direct model access, or when integrating geospatial AI into Python pipelines without HTTP overhead. It's particularly useful for processing large datasets where you want to control parallelism and memory usage through vLLM's max_num_seqs parameter.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/pooling/plugin/prithvi_geospatial_mae_io_processor.py examples/pooling/plugin/prithvi_geospatial_mae_io_processor.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Install requirements
pip install terratorch>=v1.1

# Run offline inference
python prithvi_geospatial_mae_io_processor.py
</syntaxhighlight>

== Key Concepts ==

=== LLM Class with Plugin ===
Initializes vLLM's LLM class with io_processor_plugin="terratorch_segmentation" and model_impl="terratorch" to enable geospatial data processing capabilities.

=== Prompt Dictionary Format ===
Input is specified as a dictionary with keys: data (URL/path), data_format (url/path), image_format (tiff), and out_data_format (b64_json). This structured format is processed by the I/O plugin.

=== Pooling Task Plugin ===
Uses llm.encode() with pooling_task="plugin" to invoke the custom plugin for generating segmentation outputs rather than standard embeddings or token classifications.

=== Memory Management ===
The max_num_seqs=32 parameter limits parallel requests to prevent OOM errors when processing large GeoTIFF files, allowing you to tune throughput based on available GPU memory.

=== Float16 Precision ===
Sets torch.set_default_dtype(torch.float16) for memory efficiency with geospatial models, which typically work well with half precision.

== Usage Examples ==

<syntaxhighlight lang="python">
import base64
import os
import torch
from vllm import LLM

# Set precision
torch.set_default_dtype(torch.float16)

# Prepare input
image_url = "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/valencia_example_2024-10-26.tiff"
img_prompt = dict(
    data=image_url,
    data_format="url",
    image_format="tiff",
    out_data_format="b64_json"
)

# Initialize model with plugin
llm = LLM(
    model="christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM",
    skip_tokenizer_init=True,
    trust_remote_code=True,
    enforce_eager=True,
    max_num_seqs=32,  # Limit parallel requests
    io_processor_plugin="terratorch_segmentation",
    model_impl="terratorch",
    enable_mm_embeds=True
)

# Run inference
pooler_output = llm.encode(img_prompt, pooling_task="plugin")
output = pooler_output[0].outputs

# Decode and save result
decoded_data = base64.b64decode(output.data)
file_path = os.path.join(os.getcwd(), "offline_prediction.tiff")
with open(file_path, "wb") as f:
    f.write(decoded_data)

print(f"Output file path: {file_path}")
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
* [[related::Implementation:vllm-project_vllm_Geospatial_MAE_Online_Client]]
