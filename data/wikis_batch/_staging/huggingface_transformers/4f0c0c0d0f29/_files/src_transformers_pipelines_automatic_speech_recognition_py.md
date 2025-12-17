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

**Purpose:** Transcribes audio to text using CTC or seq2seq speech recognition models with support for long-form audio via chunking.

**Mechanism:** AutomaticSpeechRecognitionPipeline extends ChunkPipeline supporting three model types (CTC, CTC with language model, seq2seq/Whisper). Implements audio chunking with configurable stride via chunk_iter(), uses rescale_stride() to map audio space to token space, supports optional pyctcdecode for LM-boosted decoding, and handles timestamp prediction for both word and character-level granularity.

**Significance:** Core pipeline for speech-to-text applications supporting various model architectures (Wav2Vec2, Whisper, HuBERT), enabling real-time transcription with streaming, accurate long-form audio processing through intelligent chunking, and timestamp extraction for accessibility and alignment use cases.
