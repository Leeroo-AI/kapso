# Prithvi Geospatial Segmentation - Offline Processor

**Source:** `examples/pooling/plugin/prithvi_geospatial_mae_io_processor.py`
**Repository:** [vllm-project/vllm](https://github.com/vllm-project/vllm)
**Lines:** 58

## Overview

This example demonstrates offline batch inference for geospatial segmentation using the Prithvi Earth Observation model with vLLM's LLM class. Unlike the online client version, this script directly instantiates the model for local processing, making it suitable for batch processing large datasets without network overhead.

## Implementation Pattern

### Architecture Design

**Offline Processing:**
- Instantiates vLLM LLM class directly (no API server)
- Processes GeoTIFF images through TerraTorch plugin
- Generates segmentation masks locally
- Writes results to disk

**Use Cases:**
- Batch processing large image collections
- Offline data processing pipelines
- Research and development workflows
- Air-gapped environments

### Advantages Over Online Approach

**Performance:**
- No network latency
- No HTTP serialization overhead
- Direct memory access to results

**Scalability:**
- Process multiple images in sequence
- Better for large-scale datasets
- Easier to parallelize across multiple GPUs

**Simplicity:**
- No server management
- Single-process execution
- Straightforward debugging

## Technical Implementation

### 1. Model Initialization

```python
import torch
from vllm import LLM

def main():
    torch.set_default_dtype(torch.float16)
    image_url = "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/valencia_example_2024-10-26.tiff"  # noqa: E501

    llm = LLM(
        model="christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM",
        skip_tokenizer_init=True,
        trust_remote_code=True,
        enforce_eager=True,
        # Limit the maximum number of parallel requests
        # to avoid the model going OOM.
        # The maximum number depends on the available GPU memory
        max_num_seqs=32,
        io_processor_plugin="terratorch_segmentation",
        model_impl="terratorch",
        enable_mm_embeds=True,
    )
```

**Configuration Parameters:**

**torch.set_default_dtype(torch.float16):**
Sets default tensor dtype to FP16 for memory efficiency.
- Reduces memory usage by ~50%
- Minimal impact on segmentation accuracy
- Essential for processing high-resolution imagery

**model:**
Model identifier pointing to Prithvi geospatial model.

**skip_tokenizer_init=True:**
Bypasses text tokenizer initialization.
- Not needed for image-only models
- Reduces initialization time
- Saves memory

**trust_remote_code=True:**
Allows execution of custom model code.
- Required for TerraTorch models
- Review code before enabling in production

**enforce_eager=True:**
Disables CUDA graph compilation.
- Faster startup for development
- Trades some inference speed for flexibility
- Recommended for batch processing scripts

**max_num_seqs=32:**
Maximum number of concurrent sequences.
- Controls peak memory usage
- Adjust based on available GPU memory
- Lower values prevent OOM errors

**io_processor_plugin="terratorch_segmentation":**
Loads the segmentation I/O processor.
- Handles GeoTIFF input/output
- Processes multi-band satellite imagery
- Formats segmentation predictions

**model_impl="terratorch":**
Uses TerraTorch model implementation.
- Specialized for geospatial models
- Provides satellite imagery processing layers

**enable_mm_embeds=True:**
Enables multimodal embedding support.
- Required for processing image inputs
- Allows non-text data flow

### 2. Input Preparation

```python
img_prompt = dict(
    data=image_url,
    data_format="url",
    image_format="tiff",
    out_data_format="b64_json",
)
```

**Input Dictionary Structure:**

**data:**
Source of the input image.
- Can be URL (as shown)
- Can be local file path: `data="/path/to/image.tiff"`
- Can be base64-encoded data

**data_format:**
Format of the data field.
- `"url"`: HTTP/HTTPS URL
- `"path"`: Local file path
- `"base64"`: Embedded base64 data

**image_format:**
Image file format.
- `"tiff"`: GeoTIFF with geospatial metadata
- Preserves CRS and geotransform

**out_data_format:**
Output format for predictions.
- `"b64_json"`: Base64-encoded binary
- Enables consistent handling with online version

### 3. Inference Execution

```python
pooler_output = llm.encode(img_prompt, pooling_task="plugin")
output = pooler_output[0].outputs
```

**llm.encode():**
Processes input through the model.
- Uses encoder-style forward pass
- Returns pooled/aggregated outputs
- Supports batch processing

**pooling_task="plugin":**
Specifies plugin-based pooling.
- Routes to TerraTorch segmentation plugin
- Different from standard embedding pooling
- Produces pixel-wise predictions

**Return Structure:**
```python
pooler_output: list[PoolingRequestOutput]
pooler_output[0].outputs: PoolingOutput
```

**PoolingOutput fields:**
- `data`: Base64-encoded segmentation mask
- `format`: Output format ("b64_json")
- Additional metadata from plugin

### 4. Output Processing

```python
import base64
import os

print(output)
decoded_data = base64.b64decode(output.data)

file_path = os.path.join(os.getcwd(), "offline_prediction.tiff")
with open(file_path, "wb") as f:
    f.write(decoded_data)

print(f"Output file path: {file_path}")
```

**Processing Steps:**
1. Print output object for debugging
2. Decode base64 data to binary TIFF
3. Construct output file path
4. Write binary data to file
5. Report completion with file path

## Batch Processing Patterns

### Sequential Processing

```python
def process_images_sequential(image_paths, output_dir):
    """Process multiple images sequentially."""
    llm = LLM(
        model="christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM",
        skip_tokenizer_init=True,
        trust_remote_code=True,
        enforce_eager=True,
        max_num_seqs=32,
        io_processor_plugin="terratorch_segmentation",
        model_impl="terratorch",
        enable_mm_embeds=True,
    )

    results = []

    for image_path in image_paths:
        print(f"Processing {image_path}...")

        img_prompt = dict(
            data=image_path,
            data_format="path",
            image_format="tiff",
            out_data_format="b64_json",
        )

        pooler_output = llm.encode(img_prompt, pooling_task="plugin")
        output = pooler_output[0].outputs

        # Decode and save
        decoded_data = base64.b64decode(output.data)
        output_filename = os.path.basename(image_path).replace('.tiff', '_seg.tiff')
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, "wb") as f:
            f.write(decoded_data)

        results.append({
            "input": image_path,
            "output": output_path,
            "success": True
        })

    return results
```

### Parallel Batch Processing

```python
def process_images_parallel_batches(image_paths, output_dir, batch_size=4):
    """Process multiple images in parallel batches."""
    llm = LLM(
        model="christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM",
        skip_tokenizer_init=True,
        trust_remote_code=True,
        enforce_eager=True,
        max_num_seqs=batch_size,  # Match batch size
        io_processor_plugin="terratorch_segmentation",
        model_impl="terratorch",
        enable_mm_embeds=True,
    )

    results = []

    # Process in batches
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}...")

        # Prepare batch prompts
        img_prompts = [
            dict(
                data=path,
                data_format="path",
                image_format="tiff",
                out_data_format="b64_json",
            )
            for path in batch
        ]

        # Process batch
        pooler_outputs = llm.encode(img_prompts, pooling_task="plugin")

        # Save results
        for path, pooler_output in zip(batch, pooler_outputs):
            output = pooler_output.outputs
            decoded_data = base64.b64decode(output.data)

            output_filename = os.path.basename(path).replace('.tiff', '_seg.tiff')
            output_path = os.path.join(output_dir, output_filename)

            with open(output_path, "wb") as f:
                f.write(decoded_data)

            results.append({
                "input": path,
                "output": output_path,
                "success": True
            })

    return results
```

### Multi-GPU Processing

```python
import torch.multiprocessing as mp

def process_on_gpu(gpu_id, image_paths, output_dir):
    """Process images on a specific GPU."""
    torch.cuda.set_device(gpu_id)

    llm = LLM(
        model="christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM",
        skip_tokenizer_init=True,
        trust_remote_code=True,
        enforce_eager=True,
        max_num_seqs=32,
        io_processor_plugin="terratorch_segmentation",
        model_impl="terratorch",
        enable_mm_embeds=True,
        tensor_parallel_size=1,  # Single GPU per process
    )

    for image_path in image_paths:
        img_prompt = dict(
            data=image_path,
            data_format="path",
            image_format="tiff",
            out_data_format="b64_json",
        )

        pooler_output = llm.encode(img_prompt, pooling_task="plugin")
        output = pooler_output[0].outputs

        # Save output
        decoded_data = base64.b64decode(output.data)
        output_filename = f"gpu{gpu_id}_{os.path.basename(image_path)}"
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, "wb") as f:
            f.write(decoded_data)

def process_images_multi_gpu(image_paths, output_dir, num_gpus=4):
    """Distribute processing across multiple GPUs."""
    # Split images across GPUs
    images_per_gpu = [
        image_paths[i::num_gpus]
        for i in range(num_gpus)
    ]

    # Spawn processes
    processes = []
    for gpu_id, gpu_images in enumerate(images_per_gpu):
        p = mp.Process(
            target=process_on_gpu,
            args=(gpu_id, gpu_images, output_dir)
        )
        p.start()
        processes.append(p)

    # Wait for completion
    for p in processes:
        p.join()
```

## Memory Management

### GPU Memory Optimization

**Memory Usage Factors:**
- Model size: ~300-600 MB for Prithvi
- Input image size: Varies with resolution
- Intermediate activations: Depends on max_num_seqs
- Output buffers: Minimal

**Optimization Strategies:**

**1. Reduce Batch Size:**
```python
llm = LLM(
    ...,
    max_num_seqs=16,  # Lower if OOM occurs
)
```

**2. Use FP16:**
```python
torch.set_default_dtype(torch.float16)
```

**3. Clear Cache Between Batches:**
```python
import gc

for batch in batches:
    process_batch(batch)
    gc.collect()
    torch.cuda.empty_cache()
```

**4. GPU Memory Utilization:**
```python
llm = LLM(
    ...,
    gpu_memory_utilization=0.8,  # Reserve 20% for safety
)
```

### Monitoring GPU Usage

```python
import subprocess

def get_gpu_memory_usage():
    """Get current GPU memory usage."""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
        capture_output=True,
        text=True
    )

    lines = result.stdout.strip().split('\n')
    gpu_memory = []

    for line in lines:
        used, total = map(int, line.split(','))
        gpu_memory.append({
            'used_mb': used,
            'total_mb': total,
            'usage_percent': (used / total) * 100
        })

    return gpu_memory

# Check before processing
memory_usage = get_gpu_memory_usage()
print(f"GPU 0: {memory_usage[0]['usage_percent']:.1f}% used")
```

## Error Handling

### Robust Processing Pipeline

```python
import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_with_error_handling(image_paths, output_dir):
    """Process images with comprehensive error handling."""
    try:
        llm = LLM(
            model="christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM",
            skip_tokenizer_init=True,
            trust_remote_code=True,
            enforce_eager=True,
            max_num_seqs=32,
            io_processor_plugin="terratorch_segmentation",
            model_impl="terratorch",
            enable_mm_embeds=True,
        )
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return []

    results = []

    for image_path in image_paths:
        try:
            logger.info(f"Processing {image_path}")

            # Validate input
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            img_prompt = dict(
                data=image_path,
                data_format="path",
                image_format="tiff",
                out_data_format="b64_json",
            )

            # Process
            pooler_output = llm.encode(img_prompt, pooling_task="plugin")
            output = pooler_output[0].outputs

            # Decode
            decoded_data = base64.b64decode(output.data)

            # Validate output
            if len(decoded_data) == 0:
                raise ValueError("Empty output received")

            # Save
            output_filename = os.path.basename(image_path).replace('.tiff', '_seg.tiff')
            output_path = os.path.join(output_dir, output_filename)

            with open(output_path, "wb") as f:
                f.write(decoded_data)

            logger.info(f"Saved to {output_path}")

            results.append({
                "input": image_path,
                "output": output_path,
                "success": True,
                "error": None
            })

        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            logger.error(traceback.format_exc())

            results.append({
                "input": image_path,
                "output": None,
                "success": False,
                "error": str(e)
            })

    return results
```

## Integration Examples

### Dataset Processing Pipeline

```python
import pandas as pd
from pathlib import Path

class GeospatialDatasetProcessor:
    """Process entire geospatial datasets."""

    def __init__(self, model_config):
        self.llm = LLM(**model_config)

    def process_dataset(self, dataset_dir, output_dir, metadata_path=None):
        """Process all images in a dataset directory."""
        dataset_dir = Path(dataset_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all TIFF files
        image_paths = list(dataset_dir.glob("*.tiff")) + list(dataset_dir.glob("*.tif"))

        print(f"Found {len(image_paths)} images")

        # Process
        results = []
        for image_path in image_paths:
            result = self.process_single_image(image_path, output_dir)
            results.append(result)

        # Save metadata
        if metadata_path:
            df = pd.DataFrame(results)
            df.to_csv(metadata_path, index=False)

        return results

    def process_single_image(self, image_path, output_dir):
        """Process single image and return metadata."""
        img_prompt = dict(
            data=str(image_path),
            data_format="path",
            image_format="tiff",
            out_data_format="b64_json",
        )

        start_time = time.time()
        pooler_output = self.llm.encode(img_prompt, pooling_task="plugin")
        processing_time = time.time() - start_time

        output = pooler_output[0].outputs
        decoded_data = base64.b64decode(output.data)

        output_filename = image_path.stem + "_seg.tiff"
        output_path = output_dir / output_filename

        with open(output_path, "wb") as f:
            f.write(decoded_data)

        return {
            "input_path": str(image_path),
            "output_path": str(output_path),
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
```

### Command-Line Tool

```python
#!/usr/bin/env python3
"""Command-line tool for geospatial segmentation."""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Process geospatial imagery for segmentation"
    )
    parser.add_argument(
        "input",
        help="Input image path or directory"
    )
    parser.add_argument(
        "output",
        help="Output directory"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for parallel processing"
    )
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=0.9,
        help="GPU memory utilization (0-1)"
    )

    args = parser.parse_args()

    # Check input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    # Get image list
    if input_path.is_dir():
        image_paths = list(input_path.glob("*.tif*"))
    else:
        image_paths = [input_path]

    print(f"Processing {len(image_paths)} images...")

    # Initialize model
    llm = LLM(
        model="christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM",
        skip_tokenizer_init=True,
        trust_remote_code=True,
        enforce_eager=True,
        max_num_seqs=args.batch_size,
        io_processor_plugin="terratorch_segmentation",
        model_impl="terratorch",
        enable_mm_embeds=True,
        gpu_memory_utilization=args.gpu_memory,
    )

    # Process
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = process_images_parallel_batches(
        [str(p) for p in image_paths],
        str(output_dir),
        args.batch_size
    )

    # Report
    success_count = sum(1 for r in results if r["success"])
    print(f"Completed: {success_count}/{len(results)} images processed successfully")

if __name__ == "__main__":
    main()
```

## Performance Benchmarking

### Timing Analysis

```python
import time

def benchmark_processing(image_paths, config_variations):
    """Benchmark different configurations."""
    results = {}

    for config_name, config in config_variations.items():
        print(f"\nTesting {config_name}...")

        llm = LLM(**config)

        start_time = time.time()

        for image_path in image_paths:
            img_prompt = dict(
                data=image_path,
                data_format="path",
                image_format="tiff",
                out_data_format="b64_json",
            )
            pooler_output = llm.encode(img_prompt, pooling_task="plugin")

        total_time = time.time() - start_time
        avg_time = total_time / len(image_paths)

        results[config_name] = {
            "total_time": total_time,
            "avg_time_per_image": avg_time,
            "images_per_second": len(image_paths) / total_time
        }

        print(f"  Total: {total_time:.2f}s")
        print(f"  Avg per image: {avg_time:.2f}s")

    return results
```

## Troubleshooting

### Common Issues

**Out of Memory:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**
- Reduce `max_num_seqs` parameter
- Use smaller batch sizes
- Enable `gpu_memory_utilization=0.8`
- Process sequentially instead of in batches

**TerraTorch Not Found:**
```
ModuleNotFoundError: No module named 'terratorch'
```

**Solution:**
```bash
pip install terratorch>=v1.1
```

**Invalid Image Format:**
```
ValueError: Unsupported image format
```

**Solution:**
- Ensure input is valid GeoTIFF
- Check file extension (`.tiff` or `.tif`)
- Verify file is not corrupted: `gdalinfo image.tiff`

## Related Examples

- **prithvi_geospatial_mae_client.py:** Online version using API server
- **openai_pooling_client.py:** Generic pooling endpoint

## References

- **TerraTorch:** [GitHub](https://github.com/IBM/terratorch)
- **Prithvi Model:** [HuggingFace](https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM)
- **vLLM Offline Inference:** Documentation on LLM class
- **GeoTIFF Processing:** GDAL/Rasterio tutorials
