# Implementation: save_to_gguf

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|llama.cpp|https://github.com/ggerganov/llama.cpp]]
* [[source::Doc|GGUF Format|https://github.com/ggerganov/ggml/blob/master/docs/gguf.md]]
|-
! Domains
| [[domain::Model_Export]], [[domain::GGUF]], [[domain::Deployment]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2025-12-15 19:00 GMT]]
|}

== Overview ==
Concrete tool for converting fine-tuned models to GGUF format with various quantization methods provided by the Unsloth library.

=== Description ===
The GGUF export functionality in Unsloth provides seamless conversion of trained models to the GGUF (GPT-Generated Unified Format) format for deployment with llama.cpp, Ollama, LM Studio, and other inference engines.

Key features:
1. **Automatic llama.cpp Integration** - Downloads and compiles llama.cpp automatically
2. **Multiple Quantization Methods** - 20+ quantization options (q4_k_m, q5_k_m, q8_0, f16, etc.)
3. **Ollama Modelfile Generation** - Creates ready-to-use Modelfiles with correct chat templates
4. **Hub Upload** - Direct push to HuggingFace Hub in GGUF format

The conversion process:
1. Merge LoRA weights into base model (if applicable)
2. Save as HuggingFace format (16-bit)
3. Convert to GGUF using llama.cpp's convert script
4. Apply quantization method
5. Optionally generate Ollama Modelfile

=== Usage ===
Use these functions when you need to:
- Deploy models with llama.cpp for CPU/GPU inference
- Create Ollama-compatible models for local deployment
- Reduce model size through quantization
- Upload GGUF models to HuggingFace Hub

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unslothai/unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/save.py#L1000-L1500 unsloth/save.py]
* '''Lines:''' ~1000-1500 (save_to_gguf implementation)

Source Files: unsloth/save.py:L1000-L1500

=== Signature ===
<syntaxhighlight lang="python">
def save_to_gguf(
    model_directory: str,
    quantization_method: Union[str, List[str]] = "q4_k_m",
    first_conversion: bool = True,
) -> str:
    """
    Convert a HuggingFace model directory to GGUF format.

    Args:
        model_directory: Path to HuggingFace model directory
        quantization_method: Quantization type or list of types
        first_conversion: Whether to do HF->GGUF conversion first

    Returns:
        Path to converted GGUF file
    """

# Typically accessed via model method:
def save_pretrained_gguf(
    self,
    save_directory: Union[str, os.PathLike],
    tokenizer: PreTrainedTokenizer,
    quantization_method: Union[str, List[str]] = "q4_k_m",
    create_ollama_modelfile: bool = False,
    push_to_hub: bool = False,
    token: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Save model directly to GGUF format with optional Ollama Modelfile.

    Args:
        save_directory: Output directory for GGUF files
        tokenizer: Tokenizer for Modelfile generation
        quantization_method: GGUF quantization type(s)
        create_ollama_modelfile: Generate Ollama Modelfile
        push_to_hub: Upload to HuggingFace Hub
        token: HF API token

    Quantization Methods:
        - "f16": Float16 (100% accuracy, large)
        - "bf16": BFloat16 (100% accuracy, fast conversion)
        - "q8_0": 8-bit (high quality)
        - "q5_k_m": 5-bit mixed (recommended quality)
        - "q4_k_m": 4-bit mixed (recommended balance)
        - "q3_k_m": 3-bit mixed (aggressive compression)
        - "q2_k": 2-bit (extreme compression)
    """

def push_to_hub_gguf(
    self,
    repo_id: str,
    tokenizer: PreTrainedTokenizer,
    quantization_method: Union[str, List[str]] = "q4_k_m",
    token: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Convert to GGUF and push directly to HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID
        tokenizer: Tokenizer for conversion
        quantization_method: GGUF quantization type(s)
        token: HF API token
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth.save import save_to_gguf

# Or use patched model methods:
model.save_pretrained_gguf(...)
model.push_to_hub_gguf(...)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| save_directory || str || Yes || Output directory for GGUF files
|-
| tokenizer || PreTrainedTokenizer || Yes || Tokenizer for template detection
|-
| quantization_method || str/List[str] || No || Quantization type (default: "q4_k_m")
|-
| create_ollama_modelfile || bool || No || Generate Ollama Modelfile (default: False)
|-
| push_to_hub || bool || No || Upload to Hub (default: False)
|-
| token || str || No || HuggingFace API token
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| GGUF file || .gguf file || Quantized model (e.g., model-Q4_K_M.gguf)
|-
| Modelfile || text file || Ollama configuration (if requested)
|-
| tokenizer files || various || Copied tokenizer for reference
|}

=== Quantization Methods Reference ===
{| class="wikitable"
|-
! Method !! Bits !! Description !! Recommended Use
|-
| f16 || 16 || Full float16 precision || Maximum accuracy
|-
| bf16 || 16 || BFloat16 precision || Fast conversion, full accuracy
|-
| q8_0 || 8 || 8-bit quantization || High quality, reasonable size
|-
| q5_k_m || 5 || Mixed 5-bit with Q6_K || Balanced quality/size
|-
| q4_k_m || 4 || Mixed 4-bit with Q6_K || '''Default - Best balance'''
|-
| q4_k_s || 4 || Pure 4-bit || Smaller, slightly lower quality
|-
| q3_k_m || 3 || Mixed 3-bit || Aggressive compression
|-
| q2_k || 2 || 2-bit quantization || Extreme compression
|}

== Usage Examples ==

=== Basic GGUF Conversion ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load and train model...
model, tokenizer = FastLanguageModel.from_pretrained(...)
# ... training code ...

# Convert to GGUF with recommended quantization
model.save_pretrained_gguf(
    "model_gguf",
    tokenizer,
    quantization_method="q4_k_m",
)

# Output: model_gguf/model-Q4_K_M.gguf
</syntaxhighlight>

=== Multiple Quantization Methods ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Save multiple quantized versions
model.save_pretrained_gguf(
    "model_gguf_multi",
    tokenizer,
    quantization_method=["q4_k_m", "q5_k_m", "q8_0"],
)

# Output:
# - model_gguf_multi/model-Q4_K_M.gguf
# - model_gguf_multi/model-Q5_K_M.gguf
# - model_gguf_multi/model-Q8_0.gguf
</syntaxhighlight>

=== With Ollama Modelfile ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Create GGUF with Ollama-ready Modelfile
model.save_pretrained_gguf(
    "ollama_model",
    tokenizer,
    quantization_method="q4_k_m",
    create_ollama_modelfile=True,  # Generate Modelfile
)

# Output:
# - ollama_model/model-Q4_K_M.gguf
# - ollama_model/Modelfile  (ready for: ollama create my-model -f Modelfile)
</syntaxhighlight>

=== Push GGUF to Hub ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Convert and upload directly to HuggingFace Hub
model.push_to_hub_gguf(
    "username/my-model-GGUF",
    tokenizer,
    quantization_method=["q4_k_m", "q5_k_m"],
    token="hf_xxxx",
)

# Available at: https://huggingface.co/username/my-model-GGUF
</syntaxhighlight>

=== Deploy with Ollama ===
<syntaxhighlight lang="bash">
# After generating GGUF with Modelfile:

# Create model in Ollama
cd ollama_model
ollama create my-finetuned-model -f Modelfile

# Run inference
ollama run my-finetuned-model "Hello, how are you?"

# List models
ollama list
</syntaxhighlight>

=== Use with llama.cpp ===
<syntaxhighlight lang="bash">
# Direct llama.cpp usage
./llama.cpp/main \
    -m model_gguf/model-Q4_K_M.gguf \
    -p "Hello, my name is" \
    -n 50

# Server mode
./llama.cpp/server \
    -m model_gguf/model-Q4_K_M.gguf \
    --port 8080
</syntaxhighlight>

=== Full Workflow Example ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# 1. Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Llama-3.2-1B-Instruct",
    load_in_4bit=True,
)

# 2. Add LoRA
model = FastLanguageModel.get_peft_model(model, r=16)

# 3. Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(output_dir="outputs", max_steps=100),
)
trainer.train()

# 4. Save merged 16-bit (required for GGUF)
model.save_pretrained_merged(
    "merged_model",
    tokenizer,
    save_method="merged_16bit",
)

# 5. Convert to GGUF with Ollama Modelfile
model.save_pretrained_gguf(
    "gguf_output",
    tokenizer,
    quantization_method="q4_k_m",
    create_ollama_modelfile=True,
)

# 6. Optionally push to Hub
model.push_to_hub_gguf(
    "username/my-llama-GGUF",
    tokenizer,
    quantization_method="q4_k_m",
    token="hf_xxxx",
)
</syntaxhighlight>

== Related Pages ==
=== Context & Requirements ===
* Requires llama.cpp (auto-installed by Unsloth)
* Model must be in 16-bit format before conversion
* Requires cmake/make for llama.cpp compilation

=== Tips and Tricks ===
* Use q4_k_m for best balance of size/quality
* Use q8_0 for highest quality with acceptable size
* Always merge LoRA before GGUF conversion
* Ollama Modelfile includes correct chat template automatically
