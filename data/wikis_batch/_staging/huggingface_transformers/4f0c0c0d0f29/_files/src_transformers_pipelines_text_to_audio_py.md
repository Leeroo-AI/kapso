# File: `src/transformers/pipelines/text_to_audio.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 311 |
| Classes | `AudioOutput`, `TextToAudioPipeline` |
| Imports | audio_utils, base, generation, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements text-to-audio/speech generation pipeline that converts text input into audio waveforms using models like Bark, MusicGen, and SpeechT5.

**Mechanism:** The `TextToAudioPipeline` class extends the base Pipeline with generation capabilities. It preprocesses text through tokenizers/processors, uses models (either text-to-waveform or text-to-spectrogram with vocoder conversion), and handles various model types with special formatting (Bark, CSM, DIA). The pipeline automatically determines sampling rates from model configs and optionally uses a vocoder (SpeechT5HifiGan) for spectrogram-to-audio conversion. Returns `AudioOutput` TypedDict containing the audio array and sampling rate.

**Significance:** Core pipeline component enabling text-to-speech and music generation tasks. Provides unified interface for multiple model architectures (Bark, MusicGen, SpeechT5) with consistent API, supporting both direct waveform generation and spectrogram-based approaches. Essential for voice synthesis and audio generation applications.
