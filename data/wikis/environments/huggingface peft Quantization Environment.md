# Environment: huggingface_peft_Quantization_Environment

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace PEFT|https://github.com/huggingface/peft]]
* [[source::Repo|bitsandbytes|https://github.com/bitsandbytes-foundation/bitsandbytes]]
* [[source::Doc|QLoRA Paper|https://huggingface.co/papers/2305.14314]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Quantization]], [[domain::QLoRA]]
|-
! Last Updated
| [[last_updated::2024-12-18 00:00 GMT]]
|}

== Overview ==
Extended PEFT environment with bitsandbytes for 4-bit/8-bit quantized model training (QLoRA).

=== Description ===
This environment extends the Core PEFT Environment with bitsandbytes support for quantized model loading and QLoRA training. It enables training large language models on consumer GPUs by quantizing base model weights to 4-bit or 8-bit precision while keeping LoRA adapters in full precision.

=== Usage ===
Use this environment when you need to:
- Train LoRA adapters on quantized models (QLoRA workflow)
- Load models in 4-bit (`load_in_4bit=True`) or 8-bit (`load_in_8bit=True`)
- Use `BitsAndBytesConfig` for quantization configuration
- Call `prepare_model_for_kbit_training()` to prepare quantized models

== System Requirements ==
{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux || Windows/macOS have limited bitsandbytes support
|-
| Python || >= 3.10.0 || Required by PEFT
|-
| Hardware || NVIDIA GPU with CUDA || Required for bitsandbytes quantization
|-
| CUDA || >= 11.0 || Check bitsandbytes compatibility matrix
|-
| VRAM || 8GB+ recommended || 4-bit allows 7B models on 8GB VRAM
|}

== Dependencies ==
=== System Packages ===
* CUDA toolkit (11.0+)
* cuDNN

=== Python Packages ===
* All packages from `huggingface_peft_Core_Environment`
* `bitsandbytes` (any recent version for 8-bit, must have `Linear4bit` for 4-bit)

== Credentials ==
* Same as `huggingface_peft_Core_Environment`
* `HF_TOKEN` often required for accessing quantization-friendly models (e.g., Llama)

== Quick Install ==
<syntaxhighlight lang="bash">
# Install PEFT with bitsandbytes
pip install peft bitsandbytes

# Verify 4-bit support
python -c "import bitsandbytes as bnb; print(hasattr(bnb.nn, 'Linear4bit'))"
</syntaxhighlight>

== Code Evidence ==

bitsandbytes availability check from `import_utils.py:24-35`:
<syntaxhighlight lang="python">
@lru_cache
def is_bnb_available() -> bool:
    return importlib.util.find_spec("bitsandbytes") is not None

@lru_cache
def is_bnb_4bit_available() -> bool:
    if not is_bnb_available():
        return False

    import bitsandbytes as bnb

    return hasattr(bnb.nn, "Linear4bit")
</syntaxhighlight>

QLoRA model preparation from `other.py:130-215`:
<syntaxhighlight lang="python">
def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=None):
    """
    This method wraps the entire protocol for preparing a model before running a training.
    This includes:
        1- Cast the layernorm in fp32
        2- making output embedding layer require grads
        3- Add the upcasting of the lm head to fp32
        4- Freezing the base model layers to ensure they are not updated during training
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
    # ...
</syntaxhighlight>

Conditional 4-bit layer definition from `bnb.py:322`:
<syntaxhighlight lang="python">
if is_bnb_4bit_available():

    class Linear4bit(torch.nn.Module, LoraLayer):
        # Lora implemented in a dense layer
</syntaxhighlight>

Clone required for 4-bit backprop from `bnb.py:548-553`:
<syntaxhighlight lang="python">
# As per Tim Dettmers, for 4bit, we need to defensively clone here.
# The reason is that in some cases, an error can occur that backprop
# does not work on a manipulated view. This issue may be solved with
# newer PyTorch versions but this would need extensive testing to be
# sure.
result = result.clone()
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `ImportError: bitsandbytes` || bitsandbytes not installed || `pip install bitsandbytes`
|-
|| `AttributeError: 'NoneType' object has no attribute 'Linear4bit'` || Old bitsandbytes version || `pip install -U bitsandbytes`
|-
|| `RuntimeError: CUDA error: no kernel image is available` || CUDA/bitsandbytes mismatch || Reinstall bitsandbytes matching CUDA version
|-
|| Merge warning: "may get different generations due to rounding errors" || Merging quantized weights || Expected behavior; verify outputs post-merge
|}

== Compatibility Notes ==

* **Linux only**: bitsandbytes has full support only on Linux with CUDA
* **Windows**: Limited support via WSL2
* **macOS**: Not officially supported for quantization
* **Multi-GPU**: Works with `device_map="auto"` and accelerate
* **Merge limitations**: Merging 4-bit/8-bit models may introduce rounding errors

== Related Pages ==
* [[requires_env::Implementation:huggingface_peft_BitsAndBytesConfig_4bit]]
* [[requires_env::Implementation:huggingface_peft_prepare_model_for_kbit_training]]
