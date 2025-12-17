# File: `src/transformers/pipelines/video_classification.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 191 |
| Classes | `VideoClassificationPipeline` |
| Functions | `read_video_pyav` |
| Imports | base, httpx, io, typing, utils, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements video classification pipeline that assigns class labels to video inputs using video understanding models.

**Mechanism:** The `VideoClassificationPipeline` accepts videos as URLs or local paths, uses PyAV library (via `read_video_pyav`) to decode and sample frames at specified intervals, extracts a fixed number of frames (from model config) using linear sampling, processes frames through the image processor, runs inference, and applies softmax/sigmoid to produce class probabilities. Supports configurable frame sampling rate and customizable output function (softmax, sigmoid, none). Requires the 'av' backend for video decoding.

**Significance:** Enables video understanding capabilities for action recognition, content classification, and video analysis tasks. Handles the complexity of temporal sampling and video decoding, providing a simple API for video-to-label predictions without requiring users to manually extract and process frames.
