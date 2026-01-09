# Environment: CUDA_GPU_vLLM_Environment

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|loader.py|https://github.com/unslothai/unsloth/blob/main/unsloth/models/loader.py]]
* [[source::Doc|rl.py|https://github.com/unslothai/unsloth/blob/main/unsloth/models/rl.py]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::LLMs]], [[domain::Reinforcement_Learning]], [[domain::Infrastructure]]
|-
! Last Updated
| [[last_updated::2026-01-09 12:00 GMT]]
|}

== Overview ==
Extended GPU environment with vLLM integration for fast batch inference, required for GRPO training and RL workflows.

=== Description ===
This environment extends the base CUDA_GPU_Environment with vLLM support for fast batch inference. vLLM enables efficient parallel generation which is critical for RL workflows like GRPO where multiple completions must be generated per prompt.

The environment requires additional GPU memory for vLLM's KV cache alongside model training. The `gpu_memory_utilization` parameter controls how much GPU memory vLLM reserves.

=== Usage ===
Use this environment for **GRPO Training** and any workflow requiring `fast_inference=True`. This is mandatory for `UnslothGRPOTrainer` and reinforcement learning fine-tuning.

== System Requirements ==
{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux (Ubuntu 20.04+) || Windows not supported for vLLM
|-
| Hardware || NVIDIA GPU || AMD ROCm has limited vLLM support
|-
| CUDA || >= 11.8 || Required for vLLM
|-
| VRAM || >= 24GB || Recommended for 7B models with vLLM overhead
|-
| Disk || 100GB SSD || For model caching and vLLM artifacts
|}

== Dependencies ==
=== System Packages ===
* `cuda-toolkit` >= 11.8
* All packages from [[Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]

=== Python Packages ===
* All packages from [[Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
* `vllm` >= 0.6.0
* `ray` (installed with vLLM)

== Credentials ==
The following environment variables may be required:
* `HF_TOKEN`: HuggingFace API token for private model access
* `WANDB_API_KEY`: Weights & Biases API key for training logging (optional)

== Quick Install ==
<syntaxhighlight lang="bash">
# Install base dependencies
pip install torch>=2.4.0 transformers>=4.45.0 bitsandbytes>=0.43.3 peft>=0.10.0 trl>=0.11.0 accelerate triton>=3.0.0

# Install vLLM for fast inference
pip install vllm>=0.6.0

# For Unsloth
pip install unsloth
</syntaxhighlight>

== Code Evidence ==

vLLM availability check from `loader.py:234-239`:
<syntaxhighlight lang="python">
if fast_inference:
    if importlib.util.find_spec("vllm") is None:
        raise ImportError(
            "Unsloth: Please install vLLM before enabling `fast_inference`!\n"
            "You can do this in a terminal via `pip install vllm`"
        )
</syntaxhighlight>

DGX Spark compatibility check from `loader.py:240-249`:
<syntaxhighlight lang="python">
if DEVICE_TYPE_TORCH == "cuda":
    for i in range(DEVICE_COUNT):
        # [TODO] DGX Spark vLLM breaks
        if "NVIDIA GB10" in str(torch.cuda.get_device_name(i)).upper():
            print(
                "Unsloth: DGX Spark detected - `fast_inference=True` is currently broken as of January 2026.\n"
                "Defaulting to native Unsloth inference."
            )
            fast_inference = False
            break
</syntaxhighlight>

FP8 requires vLLM from `loader.py:252-256`:
<syntaxhighlight lang="python">
if load_in_fp8 != False:
    if not fast_inference:
        raise NotImplementedError(
            "Unsloth: set `fast_inference = True` when doing `load_in_fp8`."
        )
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `Please install vLLM before enabling fast_inference` || vLLM not installed || `pip install vllm`
|-
|| `set fast_inference = True when doing load_in_fp8` || FP8 requires vLLM || Enable `fast_inference=True` when using `load_in_fp8`
|-
|| `DGX Spark detected - fast_inference is currently broken` || DGX Spark GPU limitation || Use native Unsloth inference (automatic fallback)
|-
|| `CUDA out of memory` || Insufficient VRAM for model + vLLM || Reduce `gpu_memory_utilization` to 0.3-0.5
|-
|| `vLLM initialization failed` || vLLM version incompatibility || Upgrade: `pip install --upgrade vllm`
|}

== Compatibility Notes ==

* **DGX Spark (GB10):** vLLM is currently broken on DGX Spark as of January 2026. Unsloth auto-falls back to native inference.
* **AMD ROCm:** vLLM has limited support on AMD GPUs. Check vLLM documentation for compatibility.
* **Memory Usage:** vLLM reserves GPU memory for KV cache. Use `gpu_memory_utilization` (default 0.5) to control allocation.
* **LoRA Rank:** When using vLLM with LoRA, ensure `r` <= `max_lora_rank` (default 64).
* **Blackwell GPUs (SM100):** PDL (Programmatic Dependent Launch) fix applied automatically via `TRITON_DISABLE_PDL=1`.

== Related Pages ==
* [[required_by::Implementation:Unslothai_Unsloth_FastLanguageModel_from_pretrained_vllm]]
* [[required_by::Implementation:Unslothai_Unsloth_get_peft_model_rl]]
* [[required_by::Implementation:Unslothai_Unsloth_dataset_mapping_pattern]]
* [[required_by::Implementation:Unslothai_Unsloth_reward_function_pattern]]
* [[required_by::Implementation:Unslothai_Unsloth_train_on_responses_only]]
* [[required_by::Implementation:Unslothai_Unsloth_UnslothGRPOConfig]]
* [[required_by::Implementation:Unslothai_Unsloth_UnslothGRPOTrainer]]
