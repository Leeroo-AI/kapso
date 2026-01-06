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

**Purpose:** Classifies audio files or raw waveforms into predefined categories using audio classification models (like Wav2Vec2).

**Mechanism:** The AudioClassificationPipeline accepts audio in multiple formats: URLs, local file paths, raw bytes, or numpy arrays. The ffmpeg_read() helper function uses ffmpeg subprocess calls to decode audio files at the correct sampling rate. Audio is resampled if needed using torchaudio, then processed through a feature extractor that converts waveforms to model inputs (e.g., mel spectrograms). Model outputs are logits that get converted to probabilities via softmax/sigmoid, and top-k predictions are returned with scores and labels.

**Significance:** Provides essential audio understanding capability for tasks like keyword spotting, music genre classification, and sound event detection. It bridges the gap between raw audio signals and high-level semantic categories, making audio AI accessible through a simple interface similar to image or text classification.
