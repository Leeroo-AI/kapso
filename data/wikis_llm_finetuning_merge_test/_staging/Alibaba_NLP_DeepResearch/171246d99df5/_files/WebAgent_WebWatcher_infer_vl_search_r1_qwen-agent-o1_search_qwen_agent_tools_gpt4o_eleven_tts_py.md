# File: `WebAgent/WebWatcher/infer/vl_search_r1/qwen-agent-o1_search/qwen_agent/tools/gpt4o/eleven_tts.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 30 |
| Functions | `tts` |
| Imports | json, os, requests |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** Explored

**Purpose:** Provides text-to-speech (TTS) functionality by interfacing with the ElevenLabs TTS API through a proxy server.

**Mechanism:** The `tts()` function:
- Takes parameters: `text`, `language` (default "en"), `model_id` (default 'eleven_multilingual_v2'), `voice_id` (default '9BWtsMINqrJLrRacOk9x')
- Sends POST requests to an internal proxy server (http://47.88.8.18:8088/elevenlabs/v1/text-to-speech/{voice_id})
- Uses the MIT_SPIDER_TOKEN environment variable for Bearer authentication
- Selects model based on language: 'eleven_multilingual_v2' for English, 'eleven_turbo_v2_5' for other languages
- Configures voice settings (stability: 0.5, similarity_boost: 0.5) and output format (pcm_16000)
- Adds language_code for non-English text to enable proper pronunciation

**Significance:** Utility component enabling voice synthesis capabilities for the agent system. Supports multimodal interactions by converting text responses to audio, which can be used for audio output in conversational AI applications.
