{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Multimodal Inference]], [[domain::Audio Processing]], [[domain::Language Models]], [[domain::Example Code]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
A comprehensive example demonstrating offline inference with 12 different audio-language models using vLLM's multimodal API.

=== Description ===
This example showcases vLLM's audio-language model support through unified interfaces for 12 different architectures: AudioFlamingo3, Gemma3N, Granite Speech, MiDashengLM, MiniCPM-O, Phi-4-multimodal, Qwen2-Audio, Qwen2.5-Omni, Ultravox, Voxtral, and Whisper. Each model has specific prompt formatting requirements, audio placeholder tokens, and configuration needs (including LoRA adapters for Granite Speech and Phi-4). The example handles single and multiple audio inputs per prompt, demonstrates batch inference, and supports various audio formats through vLLM's AudioAsset utilities. It includes proper stop token handling, tensor parallelism configuration, and model-specific chat templates.

=== Usage ===
Use this example to understand how to run audio-language model inference with vLLM, learn prompt formatting for specific audio models, or as a template for building audio-enabled applications.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/audio_language.py examples/offline_inference/audio_language.py]

=== Signature ===
<syntaxhighlight lang="python">
class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompt: str | None = None
    prompt_token_ids: dict[str, list[int]] | None = None
    multi_modal_data: dict[str, Any] | None = None
    stop_token_ids: list[int] | None = None
    lora_requests: list[LoRARequest] | None = None

def run_audioflamingo3(question: str, audio_count: int) -> ModelRequestData
def run_gemma3n(question: str, audio_count: int) -> ModelRequestData
def run_granite_speech(question: str, audio_count: int) -> ModelRequestData
# ... (functions for other 9 models)
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Run with specific model
python examples/offline_inference/audio_language.py \
    --model-type ultravox \
    --num-audios 1 \
    --num-prompts 1

# Run with different model
python examples/offline_inference/audio_language.py \
    --model-type qwen2_audio \
    --num-audios 2 \
    --num-prompts 4
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| --model-type || str || Model to use: audioflamingo3, gemma3n, granite_speech, etc.
|-
| --num-prompts || int || Number of prompts to generate (for batch inference)
|-
| --num-audios || int || Number of audio inputs per prompt (0, 1, or 2)
|-
| --seed || int || Random seed for generation (default 0)
|-
| --tensor-parallel-size || int || Tensor parallelism size (optional)
|-
| audio_assets || list[AudioAsset] || Pre-configured audio files (mary_had_lamb, winning_call)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| generated_text || str || Model's text response to audio+question
|-
| outputs || list[RequestOutput] || vLLM output objects with generated sequences
|}

== Supported Models ==

{| class="wikitable"
|-
! Model !! Audio Token !! LoRA Required !! Special Features
|-
| AudioFlamingo3 || <sound> || No || Multi-audio support
|-
| Gemma3N || <audio_soft_token> || No || Efficient small model
|-
| Granite Speech || <|audio|> || Yes (speech-lora) || Beam search recommended
|-
| MiDashengLM || <|AUDIO|> || No || Chinese audio support
|-
| MiniCPM-O || (<audio>./</audio>) || No || Omni capabilities
|-
| Phi-4-multimodal || <|audio_N|> || Yes (speech-lora) || Numbered audio placeholders
|-
| Qwen2-Audio || <|AUDIO|> || No || Strong transcription
|-
| Qwen2.5-Omni || <|AUDIO|> || No || Latest Qwen audio
|-
| Ultravox || <|audio|> || No || Llama-based
|-
| Voxtral || (mistral format) || No || Requires mistral_common[audio]
|-
| Whisper || (transcript only) || No || Transcription-only, single audio
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Run Ultravox with single audio
python examples/offline_inference/audio_language.py \
    --model-type ultravox \
    --num-audios 1 \
    --num-prompts 1

# Run Qwen2-Audio with multiple audios and batch inference
python examples/offline_inference/audio_language.py \
    --model-type qwen2_audio \
    --num-audios 2 \
    --num-prompts 8

# Run Granite Speech (requires LoRA)
python examples/offline_inference/audio_language.py \
    --model-type granite_speech \
    --num-audios 1 \
    --num-prompts 1

# Run with custom tensor parallelism
python examples/offline_inference/audio_language.py \
    --model-type phi4_mm \
    --num-audios 1 \
    --tensor-parallel-size 2

# Programmatic usage
from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset

# Load model
llm = LLM(
    model="fixie-ai/ultravox-v0_5-llama-3_2-1b",
    max_model_len=4096,
    max_num_seqs=5,
    trust_remote_code=True,
    limit_mm_per_prompt={"audio": 1, "image": 0, "video": 0}
)

# Prepare audio and prompt
audio_asset = AudioAsset("mary_had_lamb")
audio_data = audio_asset.audio_and_sample_rate

prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<|audio|>\nWhat is recited?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

# Generate
outputs = llm.generate(
    {
        "prompt": prompt,
        "multi_modal_data": {"audio": [audio_data]}
    },
    sampling_params=SamplingParams(temperature=0.0, max_tokens=64)
)

print(outputs[0].outputs[0].text)
</syntaxhighlight>

== Model-Specific Notes ==

=== Granite Speech ===
* Requires enabling LoRA and specifying max_lora_rank=64
* Audio-specific LoRA is in the model directory
* Recommended to use beam search (not shown in basic example)

=== Phi-4-multimodal ===
* Has separate vision-lora and speech-lora
* Must manually specify speech-lora path
* Supports max_lora_rank=320

=== Voxtral ===
* Requires mistral-common[audio] package
* Uses Mistral's custom tokenization format
* Requires tokenizer_mode="mistral", config_format="mistral", load_format="mistral"

=== Whisper ===
* Transcription-only (no chat/instruction following)
* Only supports single audio per prompt
* Uses special start token: "<|startoftranscript|>"

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[Concept:Multimodal_Inference]]
* [[API:vLLM_Audio_API]]
* [[Example:Offline_Inference]]
* [[Tool:Audio_Processing]]
