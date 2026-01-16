# File: `inference/file_tools/video_analysis.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 619 |
| Classes | `AnalysisResult`, `VideoAnalysis` |
| Functions | `temp_directory` |
| Imports | PIL, base64, contextlib, io, json, openai, os, pathlib, qwen_agent, requests, ... +4 more |

## Understanding

**Status:** Explored

**Purpose:** Core video and audio analysis tool that extracts audio transcriptions and visual keyframes from media files, then uses AI models to analyze the content and answer user queries.

**Mechanism:** The `VideoAnalysis` class implements a multi-step pipeline: 1) Downloads/validates media files (supports MP4, MOV, MKV, WEBM for video; MP3, WAV, AAC for audio); 2) Extracts audio using ffmpeg and transcribes it via Qwen Omni model with base64-encoded audio; 3) For videos, extracts keyframes either using scene detection (SceneDetect library) or uniform sampling; 4) Sends transcript and keyframes to an analysis model (qwen-plus-latest) that answers the user's prompt. Includes robust retry logic, temporary directory management, and fallbacks between ffmpeg-python and subprocess.

**Significance:** Essential component for multimedia understanding in DeepResearch. Enables the system to process audio/video content from user uploads or URLs, transcribe speech, extract visual information, and provide AI-powered analysis of multimedia content.
