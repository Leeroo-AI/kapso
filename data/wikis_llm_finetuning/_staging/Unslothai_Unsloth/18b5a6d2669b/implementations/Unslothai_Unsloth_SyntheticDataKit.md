# Implementation: SyntheticDataKit

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Data_Preparation]], [[domain::Synthetic_Data]], [[domain::NLP]]
|-
! Last Updated
| [[last_updated::2026-01-09 15:46 GMT]]
|}

== Overview ==
Tool for generating synthetic Q&A training data using a vLLM-powered language model server.

=== Description ===
`SyntheticDataKit` manages a vLLM server subprocess to generate synthetic question-answer pairs from raw text documents. It handles server lifecycle (startup, health checks, shutdown), text chunking based on model context limits, and configuration for Q&A generation parameters like temperature and top_p.

=== Usage ===
Use this class when you need to generate synthetic training data from raw documents (PDFs, HTML, text files) for fine-tuning. It's particularly useful for creating domain-specific Q&A datasets.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth Unslothai_Unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/dataprep/synthetic.py unsloth/dataprep/synthetic.py]
* '''Lines:''' 153-465

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
        **kwargs,
    ):
        """
        Initialize synthetic data generation kit with vLLM server.

        Args:
            model_name: HuggingFace model ID for generation
            max_seq_length: Maximum sequence length
            gpu_memory_utilization: GPU memory fraction for vLLM
            float8_kv_cache: Use FP8 KV cache for efficiency
            conservativeness: Memory allocation conservativeness
            token: HuggingFace token for gated models
            timeout: Server startup timeout in seconds
        """

    @staticmethod
    def from_pretrained(...) -> "SyntheticDataKit":
        """Alternative constructor matching HF API style."""

    def chunk_data(self, filename: str) -> List[str]:
        """Split a document into chunks respecting model context limits."""

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
        """Configure and prepare directories for Q&A generation."""

    def cleanup(self) -> None:
        """Terminate vLLM server and free GPU memory."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth import SyntheticDataKit
# or
from unsloth.dataprep.synthetic import SyntheticDataKit
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model_name || str || No || Model for generation (default: Llama-3.1-8B-Instruct)
|-
| max_seq_length || int || No || Maximum context length (default: 2048)
|-
| gpu_memory_utilization || float || No || GPU memory fraction (default: 0.98)
|-
| filename || str || Yes (for chunk_data) || Document path to process
|-
| output_folder || str || No || Output directory (default: "data")
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| chunk_data() || List[str] || List of chunk filenames created
|-
| prepare_qa_generation() || None || Creates config file and output directories
|-
| Output directories || Directories || pdf/, html/, txt/, output/, generated/, cleaned/, final/
|}

== Usage Examples ==

=== Basic Usage with Context Manager ===
<syntaxhighlight lang="python">
from unsloth import SyntheticDataKit

# Use context manager for automatic cleanup
with SyntheticDataKit(
    model_name="unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit",
    max_seq_length=4096,
    gpu_memory_utilization=0.9,
) as kit:
    # Chunk a document
    chunks = kit.chunk_data("./my_document.txt")
    print(f"Created {len(chunks)} chunks")

    # Prepare for Q&A generation
    kit.prepare_qa_generation(
        output_folder="./synthetic_data",
        max_generation_tokens=512,
        temperature=0.7,
        default_num_pairs=25,
    )
# Server automatically cleaned up
</syntaxhighlight>

=== Manual Lifecycle Management ===
<syntaxhighlight lang="python">
from unsloth import SyntheticDataKit

# Alternative constructor
kit = SyntheticDataKit.from_pretrained(
    model_name="unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit",
    max_seq_length=4096,
)

try:
    # Check server status
    if kit.check_vllm_status():
        print("vLLM server is ready")

    # Process documents
    chunks = kit.chunk_data("./corpus.txt")
    kit.prepare_qa_generation(output_folder="./data")
finally:
    # Manual cleanup
    kit.cleanup()
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_vLLM_Environment]]
