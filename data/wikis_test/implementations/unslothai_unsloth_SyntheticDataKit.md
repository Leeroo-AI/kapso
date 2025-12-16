# Implementation: SyntheticDataKit

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai/en/latest/]]
* [[source::Blog|Synthetic Data Generation|https://unsloth.ai/blog]]
|-
! Domains
| [[domain::Data_Preparation]], [[domain::Synthetic_Data]], [[domain::vLLM]], [[domain::NLP]]
|-
! Last Updated
| [[last_updated::2025-12-15 20:00 GMT]]
|}

== Overview ==
Comprehensive toolkit for generating synthetic training data by managing a vLLM server process and orchestrating question-answer pair generation from source documents.

=== Description ===
The `SyntheticDataKit` class provides end-to-end infrastructure for creating synthetic training data, particularly useful for fine-tuning on domain-specific data where labeled examples are scarce. It manages the complete pipeline from launching a vLLM inference server to chunking documents and preparing QA generation configurations.

Key components:

1. **vLLM Server Management**
   - Spawns vLLM server as a subprocess with configurable engine arguments
   - Monitors stdout/stderr with non-blocking threaded capture (`PipeCapture`)
   - Detects server readiness via regex pattern matching
   - Handles graceful shutdown and process tree cleanup

2. **Document Processing**
   - `chunk_data()` splits large documents into overlapping chunks
   - Automatically sizes chunks to fit within model's max_seq_length minus generation headroom
   - Maintains context continuity across chunk boundaries via overlap

3. **QA Generation Pipeline**
   - `prepare_qa_generation()` creates output folder structure
   - Generates YAML configuration files for external QA generation tools
   - Configurable temperature, top_p, and cleanup parameters

The class implements context manager protocol (`__enter__`, `__exit__`) for automatic cleanup and includes `terminate_tree()` for robust process cleanup using psutil or platform-specific commands.

=== Usage ===
Import `SyntheticDataKit` when you need to generate synthetic training data from source documents (PDFs, text files, etc.) using LLM-based question-answer generation. This is essential for domain adaptation and data augmentation workflows.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/dataprep/synthetic.py#L153-L465 unsloth/dataprep/synthetic.py]
* '''Lines:''' 153-465 (SyntheticDataKit class)

=== Signature ===
<syntaxhighlight lang="python">
class SyntheticDataKit:
    """
    Toolkit for synthetic data generation using vLLM.

    Manages vLLM server lifecycle and provides methods for document
    chunking and QA pair generation configuration.
    """

    def __init__(
        self,
        model_name: str = "unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit",
        max_seq_length: int = 2048,
        gpu_memory_utilization: float = 0.98,
        float8_kv_cache: bool = False,
        conservativeness: float = 1.0,
        token: str = None,
        timeout: int = 1200,
        **kwargs,
    ):
        """
        Initialize SyntheticDataKit and start vLLM server.

        Args:
            model_name: HuggingFace model ID to use for generation
            max_seq_length: Maximum sequence length for the model
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
            float8_kv_cache: Enable FP8 KV cache for memory efficiency
            conservativeness: Memory allocation conservativeness (1.0 = default)
            token: HuggingFace API token for private models
            timeout: Server startup timeout in seconds
        """

    @staticmethod
    def from_pretrained(
        model_name: str = "unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit",
        max_seq_length: int = 2048,
        gpu_memory_utilization: float = 0.9,
        **kwargs,
    ) -> "SyntheticDataKit":
        """Alternative constructor with commonly used defaults."""

    def chunk_data(self, filename: str) -> list[str]:
        """
        Split document into overlapping chunks sized for generation.

        Args:
            filename: Path to input text file

        Returns:
            List of filenames for created chunk files
        """

    def prepare_qa_generation(
        self,
        output_folder: str = "data",
        max_generation_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        overlap: int = 64,
        default_num_pairs: int = 25,
        cleanup_threshold: float = 1.0,
        cleanup_batch_size: int = 4,
        cleanup_temperature: float = 0.3,
    ) -> None:
        """Configure QA generation pipeline and create output directories."""

    def cleanup(self) -> None:
        """Terminate vLLM server and free GPU memory."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth.dataprep import SyntheticDataKit
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model_name || str || Yes || HuggingFace model ID for generation (e.g., "unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit")
|-
| max_seq_length || int || Yes || Maximum context length for the model
|-
| gpu_memory_utilization || float || No || Fraction of GPU memory to use (default: 0.98)
|-
| float8_kv_cache || bool || No || Enable FP8 KV cache for memory efficiency (default: False)
|-
| token || str || No || HuggingFace API token for gated models
|-
| filename || str || Yes (for chunk_data) || Path to source document for chunking
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| chunk_data() returns || list[str] || List of filenames for created chunk files
|-
| config || File || YAML configuration file at `synthetic_data_kit_config.yaml`
|-
| output_folder || Directory || Created folder structure for data pipeline
|}

== Usage Examples ==

=== Basic Synthetic Data Generation ===
<syntaxhighlight lang="python">
from unsloth.dataprep import SyntheticDataKit

# Initialize with context manager for automatic cleanup
with SyntheticDataKit.from_pretrained(
    model_name="unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit",
    max_seq_length=4096,
    gpu_memory_utilization=0.9,
) as kit:
    # Prepare output folders and configuration
    kit.prepare_qa_generation(
        output_folder="my_data",
        max_generation_tokens=512,
        temperature=0.7,
        default_num_pairs=25,
    )

    # Chunk a large document
    chunk_files = kit.chunk_data("my_document.txt")
    print(f"Created {len(chunk_files)} chunks")

# vLLM server is automatically terminated
</syntaxhighlight>

=== Processing Multiple Documents ===
<syntaxhighlight lang="python">
from unsloth.dataprep import SyntheticDataKit
import os

# Initialize kit
kit = SyntheticDataKit(
    model_name="unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit",
    max_seq_length=4096,
)

try:
    # Setup generation configuration
    kit.prepare_qa_generation(
        output_folder="domain_data",
        overlap=64,  # Token overlap between chunks
        default_num_pairs=30,  # QA pairs per chunk
    )

    # Process multiple source files
    all_chunks = []
    for doc in os.listdir("source_docs/"):
        if doc.endswith(".txt"):
            chunks = kit.chunk_data(f"source_docs/{doc}")
            all_chunks.extend(chunks)
            print(f"Chunked {doc}: {len(chunks)} chunks")

    print(f"Total chunks: {len(all_chunks)}")
finally:
    kit.cleanup()
</syntaxhighlight>

=== Custom vLLM Configuration ===
<syntaxhighlight lang="python">
from unsloth.dataprep import SyntheticDataKit

# Advanced configuration for large models
kit = SyntheticDataKit(
    model_name="unsloth/Qwen2.5-72B-Instruct-bnb-4bit",
    max_seq_length=8192,
    gpu_memory_utilization=0.95,
    float8_kv_cache=True,  # Enable FP8 for memory savings
    conservativeness=0.9,  # More aggressive memory usage
    timeout=1800,  # Longer timeout for large models
)
</syntaxhighlight>

== Related Pages ==
=== Context & Requirements ===
* [[requires_env::Environment:unslothai_unsloth_GPU_CUDA_Environment]]

=== Tips and Tricks ===
* [[uses_heuristic::Heuristic:unslothai_unsloth_Memory_Management]]
