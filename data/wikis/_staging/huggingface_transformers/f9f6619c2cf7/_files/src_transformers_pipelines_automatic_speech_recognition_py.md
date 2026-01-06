# File: `src/transformers/pipelines/automatic_speech_recognition.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 684 |
| Classes | `AutomaticSpeechRecognitionPipeline` |
| Functions | `rescale_stride`, `chunk_iter` |
| Imports | audio_utils, base, collections, generation, httpx, numpy, tokenization_python, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Transcribes speech from audio files or raw waveforms into text using automatic speech recognition models (CTC or seq2seq like Whisper).

**Mechanism:** The AutomaticSpeechRecognitionPipeline extends ChunkPipeline to support long-form audio through chunking with configurable stride for overlapping context. It detects model type (CTC, CTC with language model, seq2seq, or Whisper) and adapts processing accordingly. Audio preprocessing involves resampling if needed, converting to feature extractor format (mel spectrograms for Whisper), and optionally chunking long audio. For CTC models, it uses greedy decoding or beam search with optional language model boosting via pyctcdecode. For Whisper, it leverages the model's native generate() with support for timestamps at word or segment level via dynamic time warping. The pipeline stitches together overlapping chunk predictions using stride-aware logic.

**Significance:** This is one of the most widely used pipelines, enabling speech-to-text capabilities essential for voice interfaces, transcription services, and accessibility tools. Its sophisticated handling of long-form audio, multiple model architectures, and timestamp generation makes it production-ready for real-world applications ranging from meeting transcription to voice assistants.
