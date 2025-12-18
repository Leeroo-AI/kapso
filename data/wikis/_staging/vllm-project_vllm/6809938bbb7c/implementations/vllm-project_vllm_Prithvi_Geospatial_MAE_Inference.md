{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Geospatial AI]], [[domain::Computer Vision]], [[domain::Semantic Segmentation]], [[domain::Earth Observation]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Prithvi Geospatial MAE is a specialized vLLM plugin for running flood detection and semantic segmentation on satellite imagery using the Prithvi-EO-2.0 model.

=== Description ===
This implementation provides offline inference for the Prithvi-EO-V2-300M model fine-tuned on the Sen1Floods11 dataset for flood detection. Key features include:

* Integration with vLLM's pooling interface for geospatial embeddings
* Support for multi-spectral satellite imagery (6 bands: RGB, NIR, SWIR)
* Sliding window inference for large geotiff images
* Temporal and spatial coordinate encoding
* Automatic image preprocessing and normalization
* Output prediction masks and RGB overlays

The model uses a masked autoencoder architecture adapted for Earth observation data, processing Sentinel-2 satellite imagery to detect flood-affected areas.

=== Usage ===
Use this implementation when you need to:
* Detect floods in satellite imagery
* Process large-scale geospatial raster data
* Run semantic segmentation on multi-spectral Earth observation data
* Integrate geospatial AI into vLLM workflows
* Deploy Earth observation models with location-aware embeddings

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/pooling/plugin/prithvi_geospatial_mae_offline.py examples/pooling/plugin/prithvi_geospatial_mae_offline.py]

=== Signature ===
<syntaxhighlight lang="python">
class PrithviMAE:
    def __init__(self, model: str)

    def run(
        self,
        input_data: torch.Tensor,
        location_coords: torch.Tensor
    ) -> torch.Tensor

def main(
    data_file: str,
    model: str,
    output_dir: str,
    rgb_outputs: bool,
    input_indices: list[int] = None
) -> None

def load_example(
    file_paths: list[str],
    mean: list[float] = None,
    std: list[float] = None,
    indices: list[int] | None = None
) -> tuple[np.ndarray, list, list, list]

def run_model(
    input_data,
    temporal_coords,
    location_coords,
    model,
    datamodule,
    img_size: int
) -> torch.Tensor
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm import LLM
import rasterio
import albumentations
from terratorch.datamodules import Sen1Floods11NonGeoDataModule
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| data_file || str || Path to input GeoTIFF file with satellite imagery
|-
| model || str || HuggingFace model ID (default: christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM)
|-
| output_dir || str || Directory to save prediction outputs
|-
| input_indices || list[int] || 0-based band indices to select (default: [1,2,3,8,11,12] for S2L1C)
|-
| rgb_outputs || bool || Whether to output RGB visualizations
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| pred_{filename}.tiff || GeoTIFF || Binary flood prediction mask (0=no flood, 1=flood)
|-
| rgb_pred_{filename}.tiff || GeoTIFF || RGB image with flood overlay (70% original + 30% prediction)
|-
| original_rgb_{filename}.tiff || GeoTIFF || Original RGB visualization (if --rgb_outputs flag set)
|}

== Usage Examples ==

=== Basic Flood Detection ===
<syntaxhighlight lang="bash">
# Run inference on a single satellite image
python examples/pooling/plugin/prithvi_geospatial_mae_offline.py \
    --data_file ./India_900498_S2Hand.tif \
    --model christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM \
    --output_dir output
</syntaxhighlight>

=== Process with Custom Band Selection ===
<syntaxhighlight lang="bash">
# Select specific Sentinel-2 bands
python examples/pooling/plugin/prithvi_geospatial_mae_offline.py \
    --data_file ./satellite_image.tif \
    --model christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM \
    --output_dir predictions \
    --input_indices 0 1 2 3 4 5 \
    --rgb_outputs
</syntaxhighlight>

=== Programmatic Usage ===
<syntaxhighlight lang="python">
from vllm import LLM
import torch

# Initialize model
model = LLM(
    model="christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM",
    skip_tokenizer_init=True,
    dtype="float16",
    enforce_eager=True,
    model_impl="terratorch",
    enable_mm_embeds=True,
)

# Prepare input data
pixel_values = torch.randn(1, 6, 3, 224, 224, dtype=torch.float16)
location_coords = torch.tensor([[[-95.7129, 37.0902]]], dtype=torch.float16)

mm_data = {
    "pixel_values": pixel_values,
    "location_coords": location_coords,
}

prompt = {
    "prompt_token_ids": [1],
    "multi_modal_data": mm_data
}

# Run inference
outputs = model.encode(prompt, pooling_task="plugin", use_tqdm=False)
predictions = outputs[0].outputs.data
</syntaxhighlight>

=== Process Large Imagery with Sliding Windows ===
<syntaxhighlight lang="python">
import torch
import numpy as np
from einops import rearrange

# Input image with shape (C, T, H, W)
input_data = np.random.rand(1, 6, 3, 2048, 2048)

# Reflect pad to make divisible by 512
img_size = 512
h, w = input_data.shape[-2:]
pad_h = (img_size - (h % img_size)) % img_size
pad_w = (img_size - (w % img_size)) % img_size
padded = np.pad(input_data, ((0,0), (0,0), (0,0), (0,pad_h), (0,pad_w)), mode="reflect")

# Create sliding windows
batch = torch.tensor(padded)
windows = batch.unfold(3, img_size, img_size).unfold(4, img_size, img_size)
windows = rearrange(windows, "b c t h1 w1 h w -> (b h1 w1) c t h w", h=img_size, w=img_size)

# Process each window
predictions = []
for window in windows:
    pred = model.encode(window)
    predictions.append(pred)

# Reconstruct full image
output = rearrange(predictions, "(b h1 w1) c h w -> b c (h1 h) (w1 w)", h1=h1, w1=w1)
</syntaxhighlight>

== Data Format ==

=== Input Band Configuration ===
Default Sen1Floods11 bands:
* Band 0: BLUE
* Band 1: GREEN
* Band 2: RED
* Band 3: NIR_NARROW
* Band 4: SWIR_1
* Band 5: SWIR_2

=== GeoTIFF Processing ===
<syntaxhighlight lang="python">
import rasterio

# Read geotiff with metadata
with rasterio.open("satellite_image.tif") as src:
    img = src.read()  # Shape: (bands, height, width)
    meta = src.meta
    coords = src.lnglat()  # Longitude, latitude

# Save predictions
with rasterio.open("output.tif", "w", **meta) as dest:
    for i in range(image.shape[0]):
        dest.write(image[i, :, :], i + 1)
</syntaxhighlight>

== Performance Characteristics ==

* '''Model Size:''' 300M parameters
* '''Input Size:''' 512x512 patches per window
* '''Batch Processing:''' Automatically splits large images
* '''Precision:''' float16 for GPU efficiency
* '''Memory:''' ~2GB GPU memory for inference

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
* [[LLM]]
* [[Pooling Tasks]]
* [[Multi-Modal Embeddings]]
* [[Plugin Models]]
