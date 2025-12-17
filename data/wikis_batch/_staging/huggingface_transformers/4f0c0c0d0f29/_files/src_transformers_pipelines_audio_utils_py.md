# File: `src/transformers/pipelines/audio_utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 296 |
| Functions | `ffmpeg_read`, `ffmpeg_microphone`, `ffmpeg_microphone_live`, `chunk_bytes_iter` |
| Imports | datetime, numpy, platform, subprocess |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides utility functions for audio processing including file reading, microphone streaming, and audio chunking using ffmpeg.

**Mechanism:** Implements ffmpeg_read() to decode audio bytes to numpy arrays, ffmpeg_microphone() for platform-specific (Linux/Darwin/Windows) microphone input via ffmpeg, ffmpeg_microphone_live() for streaming with overlapping chunks and stride support, and chunk_bytes_iter() for managing audio chunk iteration with configurable overlap.

**Significance:** Essential utility layer that abstracts ffmpeg integration for audio pipelines, enabling support for multiple audio formats, real-time microphone input, and efficient streaming with context preservation through striding - critical for automatic speech recognition and audio processing tasks.
