# File: `src/transformers/pipelines/audio_utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 296 |
| Functions | `ffmpeg_read`, `ffmpeg_microphone`, `ffmpeg_microphone_live`, `chunk_bytes_iter` |
| Imports | datetime, numpy, platform, subprocess |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides audio I/O utilities for reading audio files and streaming from microphones using ffmpeg, shared across audio-related pipelines.

**Mechanism:** Core functions include: (1) ffmpeg_read() which decodes audio bytes through ffmpeg subprocess at specified sampling rate, (2) ffmpeg_microphone() which captures audio chunks from system microphone using platform-specific devices (alsa/avfoundation/dshow), (3) ffmpeg_microphone_live() which streams microphone audio with overlapping chunks and stride for continuous processing, and (4) chunk_bytes_iter() which manages chunk buffering with configurable stride for overlapping context windows.

**Significance:** This is the foundational audio infrastructure that enables real-time speech recognition and audio processing capabilities. By abstracting ffmpeg complexity and providing cross-platform microphone access with sophisticated chunking/striding logic, it makes live audio AI applications possible without requiring users to understand low-level audio programming.
