# File: `src/transformers/pipelines/audio_classification.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 259 |
| Classes | `AudioClassificationPipeline` |
| Functions | `ffmpeg_read` |
| Imports | base, httpx, numpy, subprocess, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Classifies audio inputs into predefined categories using AutoModelForAudioClassification models.

**Mechanism:** AudioClassificationPipeline accepts raw audio waveforms, audio file paths, or URLs, uses ffmpeg_read() to decode audio files at correct sampling rate, preprocesses with feature extractor, runs through classification model, and applies softmax/sigmoid to logits to return top-k scored labels.

**Significance:** Provides easy-to-use interface for audio classification tasks like keyword spotting, emotion recognition, and sound event detection, supporting multiple input formats and handling audio resampling automatically via torchaudio when needed.
