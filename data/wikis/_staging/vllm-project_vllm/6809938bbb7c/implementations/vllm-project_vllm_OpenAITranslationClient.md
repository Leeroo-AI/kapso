{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::Online Serving]], [[domain::Audio]], [[domain::Whisper]], [[domain::Translation]], [[domain::OpenAI API]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Demonstrates audio translation using vLLM's OpenAI-compatible API with Whisper models, translating speech from any language to English with streaming support.

=== Description ===
This example shows how to use vLLM to translate audio from any language to English using OpenAI's Whisper models through the OpenAI-compatible <code>/v1/audio/translations</code> endpoint. Unlike transcription (which outputs text in the same language), translation always produces English text regardless of the input language.

The script demonstrates:
* Synchronous translation using OpenAI's Python client
* Streaming translation using raw HTTP requests (OpenAI client doesn't support streaming for translations)
* Custom sampling parameters via <code>extra_body</code>
* Language specification for better translation quality

This is useful for creating multilingual voice applications, international content localization, or accessibility features that require English output from diverse audio sources.

=== Usage ===
Use this approach when:
* Building multilingual voice interfaces that need English output
* Creating international meeting transcription systems
* Translating foreign language podcasts or videos to English
* Implementing real-time interpretation services
* Processing customer support calls in multiple languages
* Creating accessibility tools for English speakers listening to foreign content

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_translation_client.py examples/online_serving/openai_translation_client.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
# First, start vLLM server with Whisper model
vllm serve openai/whisper-large-v3

# In another terminal, run the example
python examples/online_serving/openai_translation_client.py

# The example will:
# 1. Perform synchronous translation of Italian audio
# 2. Perform streaming translation of the same audio
</syntaxhighlight>

== Key Concepts ==

=== Translation vs. Transcription ===
'''Transcription:''' Audio → Text (same language)
* Italian audio → Italian text
* Spanish audio → Spanish text
* Endpoint: <code>/v1/audio/transcriptions</code>

'''Translation:''' Audio → English Text (always)
* Italian audio → English text
* Spanish audio → English text
* Endpoint: <code>/v1/audio/translations</code>

Translation is effectively transcription + translation to English in one step, optimized for Whisper's architecture.

=== OpenAI Translations API ===
The API endpoint: <code>/v1/audio/translations</code>

Standard parameters:
* <code>file</code>: Audio file in any language
* <code>model</code>: Whisper model name
* <code>response_format</code>: json, text, srt, vtt, verbose_json
* <code>temperature</code>: Sampling temperature (0-1)

vLLM extensions via <code>extra_body</code>:
* <code>language</code>: Source language hint (improves quality)
* <code>seed</code>: Random seed for reproducibility
* <code>repetition_penalty</code>: Penalty for repeated tokens
* <code>top_p</code>: Nucleus sampling parameter

=== Streaming Translation ===
'''Important:''' OpenAI's official Python client doesn't support streaming for translations. The example shows how to use raw HTTP requests with <code>httpx</code> for streaming:

<syntaxhighlight lang="python">
async with httpx.AsyncClient() as client:
    with open(audio_path, "rb") as f:
        async with client.stream(
            "POST",
            url + "/audio/translations",
            files={"file": f},
            data={"stream": True, "model": "openai/whisper-large-v3"},
            headers={"Authorization": f"Bearer {api_key}"}
        ) as response:
            async for line in response.aiter_lines():
                # Process streaming chunks
</syntaxhighlight>

=== Audio Asset ===
The example uses vLLM's built-in audio asset:
* <code>azacinto_foscolo</code>: Italian poetry recitation
* Demonstrates translation from Italian to English

== Usage Examples ==

=== Basic Synchronous Translation ===
<syntaxhighlight lang="python">
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

# Translate audio to English
with open("italian_audio.wav", "rb") as f:
    translation = client.audio.translations.create(
        file=f,
        model="openai/whisper-large-v3",
        response_format="json",
        temperature=0.0,
    )

print("English translation:", translation.text)
</syntaxhighlight>

=== With Source Language Hint ===
<syntaxhighlight lang="python">
# Specify source language for better translation quality
with open("spanish_audio.wav", "rb") as f:
    translation = client.audio.translations.create(
        file=f,
        model="openai/whisper-large-v3",
        response_format="json",
        temperature=0.0,
        extra_body=dict(
            language="es",  # Spanish source
            seed=4419,
            repetition_penalty=1.3,
        ),
    )

print(translation.text)
</syntaxhighlight>

=== Streaming Translation with httpx ===
<syntaxhighlight lang="python">
import asyncio
import json
import httpx

async def stream_translation(audio_path: str, base_url: str, api_key: str):
    data = {
        "language": "it",  # Italian
        "stream": True,
        "model": "openai/whisper-large-v3",
        "temperature": 0.0,
    }

    url = base_url + "/audio/translations"
    headers = {"Authorization": f"Bearer {api_key}"}

    print("Translation:", end=" ", flush=True)

    async with httpx.AsyncClient() as client:
        with open(audio_path, "rb") as f:
            async with client.stream(
                "POST", url, files={"file": f}, data=data, headers=headers
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        if line.startswith("data: "):
                            line = line[len("data: "):]

                        if line.strip() == "[DONE]":
                            break

                        chunk = json.loads(line)
                        content = chunk["choices"][0].get("delta", {}).get("content")
                        if content:
                            print(content, end="", flush=True)

    print()  # Final newline

# Run
asyncio.run(stream_translation(
    "italian_audio.wav",
    "http://localhost:8000/v1",
    "EMPTY"
))
</syntaxhighlight>

=== Batch Translation ===
<syntaxhighlight lang="python">
from pathlib import Path

audio_dir = Path("multilingual_audio")
translations = {}

# Translate all audio files to English
for audio_file in audio_dir.glob("*.wav"):
    print(f"Translating {audio_file.name}...")

    with open(audio_file, "rb") as f:
        translation = client.audio.translations.create(
            file=f,
            model="openai/whisper-large-v3",
            response_format="json",
            temperature=0.0,
        )

    translations[audio_file.name] = translation.text

# Save results
import json
with open("translations.json", "w") as f:
    json.dump(translations, f, indent=2, ensure_ascii=False)
</syntaxhighlight>

=== Multi-Language Support ===
<syntaxhighlight lang="python">
# Dictionary of audio files with known source languages
audio_files = {
    "french": {"file": "french_audio.wav", "lang": "fr"},
    "german": {"file": "german_audio.wav", "lang": "de"},
    "japanese": {"file": "japanese_audio.wav", "lang": "ja"},
    "spanish": {"file": "spanish_audio.wav", "lang": "es"},
}

for name, info in audio_files.items():
    with open(info["file"], "rb") as f:
        translation = client.audio.translations.create(
            file=f,
            model="openai/whisper-large-v3",
            response_format="json",
            extra_body=dict(language=info["lang"]),
        )

    print(f"{name.title()} → English:")
    print(f"  {translation.text}\n")
</syntaxhighlight>

=== Subtitle Translation ===
<syntaxhighlight lang="python">
# Translate foreign video audio to English subtitles
with open("foreign_video_audio.wav", "rb") as f:
    subtitles = client.audio.translations.create(
        file=f,
        model="openai/whisper-large-v3",
        response_format="srt",  # SRT format with timestamps
        extra_body=dict(
            language="fr",  # French source
        ),
    )

# Save English subtitles
with open("english_subtitles.srt", "w") as f:
    f.write(subtitles)

# Or WebVTT format
with open("foreign_video_audio.wav", "rb") as f:
    subtitles = client.audio.translations.create(
        file=f,
        model="openai/whisper-large-v3",
        response_format="vtt",
        extra_body=dict(language="fr"),
    )

with open("english_subtitles.vtt", "w") as f:
    f.write(subtitles)
</syntaxhighlight>

=== Real-Time Voice Translation ===
<syntaxhighlight lang="python">
import asyncio
import json
import httpx
from queue import Queue
from threading import Thread

class RealtimeTranslator:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key
        self.audio_queue = Queue()

    async def translate_chunk(self, audio_chunk):
        """Translate a single audio chunk"""
        data = {
            "stream": True,
            "model": "openai/whisper-large-v3",
        }

        url = self.base_url + "/audio/translations"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                url,
                files={"file": audio_chunk},
                data=data,
                headers=headers,
            ) as response:
                async for line in response.aiter_lines():
                    if line and line.startswith("data: "):
                        chunk_data = line[len("data: "):]
                        if chunk_data.strip() != "[DONE]":
                            chunk = json.loads(chunk_data)
                            content = chunk["choices"][0].get("delta", {}).get("content")
                            if content:
                                yield content

# Usage
translator = RealtimeTranslator("http://localhost:8000/v1", "EMPTY")

async def process_audio_stream():
    with open("streaming_audio.wav", "rb") as f:
        async for text in translator.translate_chunk(f):
            print(text, end="", flush=True)

asyncio.run(process_audio_stream())
</syntaxhighlight>

=== Using vLLM Audio Assets ===
<syntaxhighlight lang="python">
from vllm.assets.audio import AudioAsset

# vLLM provides sample Italian audio
foscolo = str(AudioAsset("azacinto_foscolo").get_local_path())

# Translate Italian poetry to English
with open(foscolo, "rb") as f:
    translation = client.audio.translations.create(
        file=f,
        model="openai/whisper-large-v3",
        response_format="json",
        temperature=0.0,
        extra_body=dict(language="it"),
    )

print("Italian → English:")
print(translation.text)
</syntaxhighlight>

=== Error Handling ===
<syntaxhighlight lang="python">
from openai import OpenAI, APIError, APIConnectionError

def translate_with_retry(audio_path, source_lang=None, max_retries=3):
    client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

    for attempt in range(max_retries):
        try:
            extra = {"language": source_lang} if source_lang else {}

            with open(audio_path, "rb") as f:
                translation = client.audio.translations.create(
                    file=f,
                    model="openai/whisper-large-v3",
                    response_format="json",
                    temperature=0.0,
                    extra_body=extra,
                )

            return translation.text

        except APIConnectionError as e:
            print(f"Connection error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(2 ** attempt)
            else:
                raise

        except APIError as e:
            print(f"API error: {e}")
            raise

    return None
</syntaxhighlight>

== Supported Languages ==

Whisper can translate from these languages to English:
* European: French, German, Spanish, Italian, Portuguese, Dutch, Polish, etc.
* Asian: Chinese, Japanese, Korean, Vietnamese, Thai, etc.
* Middle Eastern: Arabic, Hebrew, Turkish, Persian, etc.
* Others: Russian, Hindi, Indonesian, and 90+ more

Full list: See OpenAI Whisper documentation for all 99 supported languages.

== Performance Considerations ==

=== Model Selection ===
* '''whisper-large-v3''': Best translation quality, ~10GB VRAM
* '''whisper-medium''': Good quality, ~5GB VRAM
* '''whisper-small''': Decent quality, ~2GB VRAM

=== Optimization Tips ===
* Specify source language with <code>extra_body={"language": "fr"}</code> for better quality
* Use lower temperature (0.0-0.3) for more literal translations
* Convert audio to 16kHz mono WAV for best performance
* Enable streaming for long audio files
* Use batch processing for multiple files

=== Quality Improvements ===
<syntaxhighlight lang="python">
# High-quality translation settings
translation = client.audio.translations.create(
    file=audio_file,
    model="openai/whisper-large-v3",
    response_format="json",
    temperature=0.0,  # Deterministic
    extra_body=dict(
        language="fr",  # Specify source language
        repetition_penalty=1.3,  # Reduce repetition
        top_p=0.9,  # More focused sampling
    ),
)
</syntaxhighlight>

== Server Configuration ==

=== Basic Server Setup ===
<syntaxhighlight lang="bash">
# Start Whisper translation server
vllm serve openai/whisper-large-v3 \
    --host 0.0.0.0 \
    --port 8000

# With tensor parallelism for faster processing
vllm serve openai/whisper-large-v3 \
    --tensor-parallel-size 2 \
    --port 8000

# With optimized settings
vllm serve openai/whisper-large-v3 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 448
</syntaxhighlight>

== Limitations ==

=== Translation Direction ===
* Only supports translation TO English
* Cannot translate English to other languages
* Cannot translate between non-English languages
* For other translation directions, use transcription + separate translation model

=== Streaming Support ===
* OpenAI Python client doesn't support streaming for translations
* Must use raw HTTP requests with httpx or similar
* Streaming response format: newline-delimited JSON with "data:" prefix

== Comparison with Transcription ==

{| class="wikitable"
|-
! Feature !! Transcription !! Translation
|-
| Output Language || Same as input || Always English
|-
| Use Case || Same-language text || Cross-language conversion
|-
| Endpoint || /v1/audio/transcriptions || /v1/audio/translations
|-
| Language Parameter || Optional (detected or specified) || Optional hint for quality
|-
| Streaming Support || Official client + HTTP || HTTP only
|}

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[related::Implementation:vllm-project_vllm_OpenAITranscriptionClient]]
* [[related::Concept:vllm-project_vllm_Multimodal_Inference]]
* [[related::Concept:vllm-project_vllm_Audio_Processing]]
* [[related::API:vllm-project_vllm_OpenAI_Compatible_API]]
* [[related::Model:OpenAI_Whisper]]
