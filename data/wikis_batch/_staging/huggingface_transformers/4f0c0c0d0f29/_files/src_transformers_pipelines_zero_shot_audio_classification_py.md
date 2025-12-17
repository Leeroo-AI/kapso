# File: `src/transformers/pipelines/zero_shot_audio_classification.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 161 |
| Classes | `ZeroShotAudioClassificationPipeline` |
| Imports | audio_classification, base, collections, httpx, numpy, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements zero-shot audio classification using CLAP (Contrastive Language-Audio Pretraining) models to classify audio against arbitrary text labels without task-specific training.

**Mechanism:** The `ZeroShotAudioClassificationPipeline` processes audio inputs (URLs, local files, or numpy arrays) through feature extraction, formats candidate labels using a hypothesis template (default: "This is a sound of {}."), encodes both audio and text labels through the CLAP model, computes similarity via `logits_per_audio`, applies softmax for probability distribution, and returns ranked label predictions. Uses `ffmpeg_read` to decode audio files and ensures single-channel audio input.

**Significance:** Enables flexible audio classification without pre-defined categories or retraining. Particularly valuable for custom sound classification tasks where training data is limited or categories change dynamically. Leverages vision-language model paradigm applied to audio-text modality for zero-shot transfer learning.
