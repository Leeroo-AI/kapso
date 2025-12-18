{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::Online Serving]], [[domain::Audio]], [[domain::Whisper]], [[domain::Transcription]], [[domain::OpenAI API]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Demonstrates audio transcription using vLLM's OpenAI-compatible API with Whisper models, supporting both synchronous and streaming transcription workflows.

=== Description ===
This example shows how to use vLLM to perform audio-to-text transcription using OpenAI's Whisper models through the OpenAI-compatible API. The script demonstrates both synchronous (batch) and asynchronous (streaming) transcription, including how to pass additional sampling parameters beyond the standard OpenAI API.

The example uses the <code>openai/whisper-large-v3</code> model and vLLM's built-in audio assets for testing. It showcases vLLM's multimodal capabilities for processing audio inputs alongside text generation tasks.

Key features demonstrated:
* Using OpenAI's Python client with vLLM
* Synchronous transcription with <code>client.audio.transcriptions.create()</code>
* Streaming transcription with async client
* Passing custom parameters via <code>extra_body</code>
* Language specification and response format options

=== Usage ===
Use this approach when:
* Building audio transcription services with vLLM
* Processing voice recordings, meetings, or podcasts
* Creating subtitles or closed captions for videos
* Integrating speech recognition into applications
* Building voice-to-text interfaces for accessibility
* Processing multilingual audio content with automatic language detection

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_transcription_client.py examples/online_serving/openai_transcription_client.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
# First, start vLLM server with Whisper model
vllm serve openai/whisper-large-v3

# In another terminal, run the example
python examples/online_serving/openai_transcription_client.py

# The example will:
# 1. Perform synchronous transcription of "mary_had_lamb.wav"
# 2. Perform streaming transcription of "winning_call.wav"
</syntaxhighlight>

== Key Concepts ==

=== Whisper Model Support ===
vLLM supports OpenAI's Whisper models for audio transcription:
* <code>openai/whisper-large-v3</code>: Most accurate, slower
* <code>openai/whisper-medium</code>: Balanced accuracy/speed
* <code>openai/whisper-small</code>: Faster, less accurate
* <code>openai/whisper-base</code>: Fastest, basic accuracy
* <code>openai/whisper-tiny</code>: Minimal resources

Models handle:
* Multilingual transcription (99 languages)
* Automatic language detection
* Timestamp generation
* Speaker change detection (with proper prompting)

=== OpenAI Transcriptions API ===
The API endpoint: <code>/v1/audio/transcriptions</code>

Standard parameters:
* <code>file</code>: Audio file (MP3, WAV, M4A, etc.)
* <code>model</code>: Whisper model name
* <code>language</code>: ISO-639-1 language code (optional)
* <code>response_format</code>: json, text, srt, vtt, verbose_json
* <code>temperature</code>: Sampling temperature (0-1)

vLLM extensions via <code>extra_body</code>:
* <code>seed</code>: Random seed for reproducibility
* <code>repetition_penalty</code>: Penalty for repeated tokens
* <code>top_p</code>: Nucleus sampling parameter
* Other standard vLLM sampling parameters

=== Audio File Formats ===
Supported formats:
* WAV (recommended for quality)
* MP3 (compressed)
* M4A (Apple format)
* FLAC (lossless)
* OGG (Vorbis/Opus)

Recommendations:
* Sample rate: 16kHz (Whisper's native rate)
* Channels: Mono (stereo is automatically mixed)
* Bit depth: 16-bit or higher

=== Streaming vs. Non-Streaming ===
'''Non-streaming (sync):'''
* Processes entire audio file
* Returns complete transcription
* Simpler API, easier error handling
* Better for short audio clips

'''Streaming (async):'''
* Returns tokens as they're generated
* Lower perceived latency
* Better UX for long audio files
* Requires async/await patterns

== Usage Examples ==

=== Basic Synchronous Transcription ===
<syntaxhighlight lang="python">
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

# Transcribe audio file
with open("audio.wav", "rb") as f:
    transcription = client.audio.transcriptions.create(
        file=f,
        model="openai/whisper-large-v3",
        language="en",  # Optional: specify language
        response_format="json",
        temperature=0.0,  # Deterministic output
    )

print("Transcription:", transcription.text)
</syntaxhighlight>

=== With Custom Sampling Parameters ===
<syntaxhighlight lang="python">
with open("audio.wav", "rb") as f:
    transcription = client.audio.transcriptions.create(
        file=f,
        model="openai/whisper-large-v3",
        language="en",
        response_format="json",
        temperature=0.0,
        # Additional vLLM-specific parameters
        extra_body=dict(
            seed=4419,  # Reproducible outputs
            repetition_penalty=1.3,  # Reduce repetition
            top_p=0.95,  # Nucleus sampling
        ),
    )
    print(transcription.text)
</syntaxhighlight>

=== Async Streaming Transcription ===
<syntaxhighlight lang="python">
import asyncio
from openai import AsyncOpenAI

async def stream_transcription():
    client = AsyncOpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
    )

    print("Transcription:", end=" ", flush=True)

    with open("audio.wav", "rb") as f:
        transcription = await client.audio.transcriptions.create(
            file=f,
            model="openai/whisper-large-v3",
            language="en",
            response_format="json",
            temperature=0.0,
            stream=True,  # Enable streaming
            extra_body=dict(
                seed=420,
                top_p=0.6,
            ),
        )

        async for chunk in transcription:
            if chunk.choices:
                content = chunk.choices[0].get("delta", {}).get("content")
                if content:
                    print(content, end="", flush=True)

    print()  # Final newline

asyncio.run(stream_transcription())
</syntaxhighlight>

=== Multilingual Transcription ===
<syntaxhighlight lang="python">
# Automatic language detection (omit language parameter)
with open("multilingual_audio.wav", "rb") as f:
    transcription = client.audio.transcriptions.create(
        file=f,
        model="openai/whisper-large-v3",
        response_format="verbose_json",  # Includes detected language
        temperature=0.0,
    )

print(f"Detected language: {transcription.language}")
print(f"Transcription: {transcription.text}")

# Or specify language explicitly
languages = {
    "es": "spanish_audio.wav",
    "fr": "french_audio.wav",
    "de": "german_audio.wav",
}

for lang_code, audio_file in languages.items():
    with open(audio_file, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=f,
            model="openai/whisper-large-v3",
            language=lang_code,
            response_format="json",
        )
    print(f"{lang_code}: {transcription.text}")
</syntaxhighlight>

=== Subtitle Generation ===
<syntaxhighlight lang="python">
# Generate SRT format subtitles
with open("video_audio.wav", "rb") as f:
    subtitles = client.audio.transcriptions.create(
        file=f,
        model="openai/whisper-large-v3",
        language="en",
        response_format="srt",  # SRT format with timestamps
    )

# Save to file
with open("subtitles.srt", "w") as f:
    f.write(subtitles)

# Generate WebVTT format
with open("video_audio.wav", "rb") as f:
    subtitles = client.audio.transcriptions.create(
        file=f,
        model="openai/whisper-large-v3",
        language="en",
        response_format="vtt",  # WebVTT format
    )

with open("subtitles.vtt", "w") as f:
    f.write(subtitles)
</syntaxhighlight>

=== Batch Processing Multiple Files ===
<syntaxhighlight lang="python">
import os
from pathlib import Path

audio_dir = Path("audio_files")
transcriptions = {}

for audio_file in audio_dir.glob("*.wav"):
    print(f"Processing {audio_file.name}...")

    with open(audio_file, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=f,
            model="openai/whisper-large-v3",
            response_format="json",
            temperature=0.0,
        )

    transcriptions[audio_file.name] = transcription.text

# Save results
import json
with open("transcriptions.json", "w") as f:
    json.dump(transcriptions, f, indent=2)
</syntaxhighlight>

=== With Error Handling ===
<syntaxhighlight lang="python">
from openai import OpenAI, APIError, APIConnectionError
import time

def transcribe_with_retry(audio_path, max_retries=3):
    client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

    for attempt in range(max_retries):
        try:
            with open(audio_path, "rb") as f:
                transcription = client.audio.transcriptions.create(
                    file=f,
                    model="openai/whisper-large-v3",
                    language="en",
                    response_format="json",
                    temperature=0.0,
                )
            return transcription.text

        except APIConnectionError as e:
            print(f"Connection error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise

        except APIError as e:
            print(f"API error: {e}")
            raise

        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

    return None
</syntaxhighlight>

=== Using vLLM Audio Assets ===
<syntaxhighlight lang="python">
from vllm.assets.audio import AudioAsset

# vLLM provides sample audio files for testing
mary_had_lamb = str(AudioAsset("mary_had_lamb").get_local_path())
winning_call = str(AudioAsset("winning_call").get_local_path())

# Transcribe sample audio
with open(mary_had_lamb, "rb") as f:
    transcription = client.audio.transcriptions.create(
        file=f,
        model="openai/whisper-large-v3",
        language="en",
        response_format="json",
        temperature=0.0,
    )

print(f"Mary had a lamb transcription: {transcription.text}")
</syntaxhighlight>

== Advanced Features ==

=== Verbose Response Format ===
<syntaxhighlight lang="python">
with open("audio.wav", "rb") as f:
    transcription = client.audio.transcriptions.create(
        file=f,
        model="openai/whisper-large-v3",
        response_format="verbose_json",
    )

print("Text:", transcription.text)
print("Language:", transcription.language)
print("Duration:", transcription.duration)

# Segments include timestamps
for segment in transcription.segments:
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")
</syntaxhighlight>

=== Prompt-Based Context ===
<syntaxhighlight lang="python">
# Provide context to improve accuracy
with open("technical_audio.wav", "rb") as f:
    transcription = client.audio.transcriptions.create(
        file=f,
        model="openai/whisper-large-v3",
        language="en",
        prompt="Discussion about machine learning, neural networks, and transformers.",
        response_format="json",
    )
</syntaxhighlight>

== Performance Considerations ==

=== Model Selection ===
* '''whisper-large-v3''': Best accuracy, ~10GB VRAM, ~2x real-time
* '''whisper-medium''': Good accuracy, ~5GB VRAM, ~4x real-time
* '''whisper-small''': Decent accuracy, ~2GB VRAM, ~8x real-time

=== Optimization Tips ===
* Convert audio to 16kHz mono WAV for best performance
* Use batch processing for multiple files
* Enable streaming for long audio (>10 minutes)
* Consider quantization for faster inference
* Use GPU with sufficient VRAM for model size

=== Throughput Scaling ===
<syntaxhighlight lang="bash">
# Use tensor parallelism for larger models
vllm serve openai/whisper-large-v3 --tensor-parallel-size 2

# Data parallelism for higher throughput
vllm serve openai/whisper-large-v3 --data-parallel-size 2
</syntaxhighlight>

== Server Configuration ==

=== Basic Server Setup ===
<syntaxhighlight lang="bash">
# Start Whisper transcription server
vllm serve openai/whisper-large-v3 \
    --host 0.0.0.0 \
    --port 8000

# With specific GPU
CUDA_VISIBLE_DEVICES=0 vllm serve openai/whisper-large-v3

# With custom settings
vllm serve openai/whisper-large-v3 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 448  # Whisper's max length
</syntaxhighlight>

=== Multi-Modal Server ===
<syntaxhighlight lang="bash">
# Serve both text and audio models
# (vLLM can serve multiple model types)
vllm serve openai/whisper-large-v3 --port 8000
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[related::Implementation:vllm-project_vllm_OpenAITranslationClient]]
* [[related::Concept:vllm-project_vllm_Multimodal_Inference]]
* [[related::Concept:vllm-project_vllm_Audio_Processing]]
* [[related::API:vllm-project_vllm_OpenAI_Compatible_API]]
* [[related::Model:OpenAI_Whisper]]
