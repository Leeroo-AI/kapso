# Environment: llama_cpp_Environment

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|save.py|https://github.com/unslothai/unsloth/blob/main/unsloth/save.py]]
* [[source::Repo|llama.cpp|https://github.com/ggerganov/llama.cpp]]
|-
! Domains
| [[domain::Deployment]], [[domain::Quantization]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2026-01-09 12:00 GMT]]
|}

== Overview ==
Environment for GGUF export and validation using llama.cpp tools for local LLM deployment.

=== Description ===
This environment provides the tools needed to convert trained models to GGUF format and validate them using llama.cpp. GGUF is the standard format for running LLMs locally with tools like Ollama, LM Studio, and llama.cpp.

The environment includes:
- `llama.cpp` conversion scripts for GGUF export
- Quantization tools for various precision levels (q4_k_m, q5_k_m, q8_0, etc.)
- `llama-cli` for validation and inference testing

=== Usage ===
Use this environment for the **GGUF_Export** workflow, specifically for the verification step using `llama-cli`. This environment can run on CPU-only systems for inference testing, though GPU acceleration is recommended for faster quantization.

== System Requirements ==
{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux, macOS, Windows || Cross-platform support
|-
| Hardware || CPU (GPU optional) || GPU accelerates quantization
|-
| RAM || >= 16GB || For loading models during conversion
|-
| Disk || 50GB SSD || For model files and GGUF outputs
|}

== Dependencies ==
=== System Packages ===
* `cmake` >= 3.14
* `gcc` / `clang` (C++ compiler)
* `make`
* CUDA toolkit (optional, for GPU-accelerated quantization)

=== Python Packages ===
* `sentencepiece` (for tokenizer conversion)
* `psutil` (for memory monitoring)
* `numpy`

=== External Tools ===
* `llama.cpp` (build from source or use pre-built binaries)
* `llama-cli` (included with llama.cpp)

== Credentials ==
* `HF_TOKEN`: HuggingFace API token for `push_to_hub_gguf` operations

== Quick Install ==
<syntaxhighlight lang="bash">
# Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j

# Or with CUDA support
make -j LLAMA_CUDA=1

# Install Python dependencies for Unsloth GGUF export
pip install sentencepiece psutil numpy
</syntaxhighlight>

== Code Evidence ==

GGUF quantization methods from `save.py:104-131`:
<syntaxhighlight lang="python">
ALLOWED_QUANTS = {
    "not_quantized": "No quantization (float16)",
    "fast_quantized": "Q8_0 quantization (8-bit)",
    "quantized": "Q4_K_M quantization (4-bit)",
    "q4_k_m": "Q4_K_M quantization",
    "q5_k_m": "Q5_K_M quantization",
    "q8_0": "Q8_0 quantization",
    "q2_k": "Q2_K quantization (2-bit)",
    "q3_k_m": "Q3_K_M quantization",
    "q6_k": "Q6_K quantization",
    "f16": "Float16 (no quantization)",
    "bf16": "BFloat16 (no quantization)",
}
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `llama.cpp not found` || llama.cpp not in PATH || Build llama.cpp and add to PATH
|-
|| `Unsupported quantization method` || Invalid quant string || Use one of: q4_k_m, q5_k_m, q8_0, q2_k, q3_k_m, q6_k, f16, bf16
|-
|| `sentencepiece not installed` || Missing Python package || `pip install sentencepiece`
|-
|| `Memory error during quantization` || Insufficient RAM || Close other applications or use machine with more RAM
|-
|| `GGUF validation failed` || Corrupted conversion || Re-run conversion with different quantization
|}

== Compatibility Notes ==

* **Cross-Platform:** llama.cpp works on Linux, macOS, and Windows.
* **CPU Inference:** GGUF files can run on CPU-only systems, making them ideal for deployment on edge devices.
* **GPU Acceleration:** For faster quantization, build llama.cpp with CUDA/Metal/OpenCL support.
* **Ollama Integration:** Unsloth automatically generates Ollama Modelfiles for easy deployment.
* **Model Size:** Quantization significantly reduces model size:
  - q4_k_m: ~4 bits per weight (~25% of original size)
  - q8_0: ~8 bits per weight (~50% of original size)
  - f16: ~16 bits per weight (full precision)

== Related Pages ==
* [[required_by::Implementation:Unslothai_Unsloth_llama_cli_validation]]
