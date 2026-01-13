# Implementation: SyntheticDataKit

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::NLP]], [[domain::Data_Processing]], [[domain::Synthetic_Data]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==
SyntheticDataKit provides a high-level interface for generating synthetic training data using a vLLM inference server, enabling automated QA pair generation from text documents.

=== Description ===
SyntheticDataKit manages the lifecycle of a vLLM server subprocess for efficient synthetic data generation. It:

1. '''Server Management''': Launches vLLM as a subprocess with optimized settings (GPU memory utilization, FP8 KV cache, sequence length), monitors startup via regex-based pipe capture, and handles graceful shutdown.

2. '''Non-blocking I/O''': Uses the PipeCapture helper class for asynchronous stdout/stderr capture with thread-safe buffering, supporting timeout-based ready detection.

3. '''Data Chunking''': Provides chunk_data() to split large documents into appropriately sized segments respecting the model's context window minus generation headroom.

4. '''QA Generation Pipeline''': The prepare_qa_generation() method sets up configuration for synthetic question-answer pair generation including temperature, top_p, overlap, and cleanup parameters.

The class implements context manager protocol (__enter__/__exit__) and destructor cleanup to ensure the vLLM process is properly terminated.

=== Usage ===
Import SyntheticDataKit when you need to generate synthetic training data (e.g., QA pairs, instruction-response pairs) from raw documents using a local vLLM inference server. This is useful for data augmentation, distillation, or creating instruction-tuning datasets.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' unsloth/dataprep/synthetic.py
* '''Lines:''' 1-466

=== Signature ===
<syntaxhighlight lang="python">
class SyntheticDataKit:
    def __init__(
        self,
        model_name: str = "unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit",
        max_seq_length: int = 2048,
        gpu_memory_utilization: float = 0.98,
        float8_kv_cache: bool = False,
        conservativeness: float = 1.0,
        token: Optional[str] = None,
        timeout: int = 1200,
        **kwargs
    ) -> None: ...

    @staticmethod
    def from_pretrained(
        model_name: str = "unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit",
        max_seq_length: int = 2048,
        gpu_memory_utilization: float = 0.9,
        float8_kv_cache: bool = False,
        conservativeness: float = 1.0,
        token: Optional[str] = None,
        **kwargs
    ) -> "SyntheticDataKit": ...

    def chunk_data(self, filename: str) -> List[str]: ...
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
        cleanup_temperature: float = 0.3
    ) -> None: ...
    def cleanup(self) -> None: ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth.dataprep.synthetic import SyntheticDataKit
</syntaxhighlight>

== I/O Contract ==

=== Inputs (SyntheticDataKit.__init__) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model_name || str || No || HuggingFace model identifier for the vLLM server (default: "unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit")
|-
| max_seq_length || int || No || Maximum sequence length for the model (default: 2048)
|-
| gpu_memory_utilization || float || No || Fraction of GPU memory to use (default: 0.98)
|-
| float8_kv_cache || bool || No || Enable FP8 KV cache for memory efficiency (default: False)
|-
| conservativeness || float || No || Memory allocation conservativeness factor (default: 1.0)
|-
| token || str || No || HuggingFace API token for private models
|-
| timeout || int || No || Timeout in seconds for vLLM server startup (default: 1200)
|}

=== Inputs (prepare_qa_generation) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| output_folder || str || No || Base folder for generated data outputs (default: "data")
|-
| max_generation_tokens || int || No || Maximum tokens to generate per response (default: 512)
|-
| temperature || float || No || Sampling temperature for generation (default: 0.7)
|-
| top_p || float || No || Nucleus sampling probability threshold (default: 0.95)
|-
| overlap || int || No || Token overlap between chunks (default: 64)
|-
| default_num_pairs || int || No || Default number of QA pairs to generate per chunk (default: 25)
|-
| cleanup_threshold || float || No || Threshold for cleanup filtering (default: 1.0)
|-
| cleanup_batch_size || int || No || Batch size for cleanup pass (default: 4)
|-
| cleanup_temperature || float || No || Temperature for cleanup generation (default: 0.3)
|}

=== Inputs (chunk_data) ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| filename || str || Yes || Path to the text file to chunk
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| from_pretrained return || SyntheticDataKit || Initialized SyntheticDataKit instance with running vLLM server
|-
| chunk_data return || List[str] || List of filenames for the generated chunk files
|-
| check_vllm_status return || bool || True if vLLM server is responsive at localhost:8000/metrics
|}

== Usage Examples ==
<syntaxhighlight lang="python">
from unsloth.dataprep.synthetic import SyntheticDataKit

# Use as context manager for automatic cleanup
with SyntheticDataKit.from_pretrained(
    model_name="unsloth/Llama-3.1-8B-Instruct",
    max_seq_length=4096,
    gpu_memory_utilization=0.9,
) as kit:
    # Prepare QA generation configuration
    kit.prepare_qa_generation(
        output_folder="./synthetic_data",
        max_generation_tokens=512,
        temperature=0.7,
        top_p=0.95,
        default_num_pairs=25,
    )

    # Chunk a large document into processable segments
    chunk_files = kit.chunk_data(filename="./large_document.txt")
    print(f"Created {len(chunk_files)} chunk files")

    # Check if vLLM server is healthy
    if SyntheticDataKit.check_vllm_status():
        print("vLLM server is running at localhost:8000")

# vLLM server is automatically cleaned up after context exit

# Alternative: manual lifecycle management
kit = SyntheticDataKit(
    model_name="unsloth/Llama-3.1-8B-Instruct",
    max_seq_length=2048,
    timeout=1800,  # 30 min timeout for large model download
)

try:
    kit.prepare_qa_generation(output_folder="./output")
    # ... perform data generation ...
finally:
    kit.cleanup()  # Ensure vLLM process is terminated
</syntaxhighlight>

== Related Pages ==
* [[related::Implementation:Unslothai_Unsloth_RawTextDataLoader]]
* [[requires_env::Environment:Unslothai_Unsloth_VLLM]]
