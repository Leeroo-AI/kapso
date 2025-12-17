# File: `src/transformers/video_utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 893 |
| Classes | `VideoMetadata` |
| Functions | `is_valid_video_frame`, `is_valid_video`, `valid_videos`, `is_batched_video`, `is_scaled_video`, `convert_pil_frames_to_video`, `make_batched_videos`, `make_batched_metadata`, `... +13 more` |
| Imports | collections, contextlib, dataclasses, httpx, image_transforms, image_utils, io, numpy, os, typing, ... +3 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides low-level utilities for video loading, validation, format conversion, and metadata extraction, supporting multiple video backends (OpenCV, PyAV, decord, torchvision, torchcodec) for the video processing pipeline.

**Mechanism:** Implements comprehensive video handling functionality: VideoMetadata dataclass stores video properties (total_num_frames, fps, width, height, duration, video_backend, frames_indices) with computed properties for timestamps and sampled_fps; validation functions (is_valid_video checks 4D arrays, is_valid_video_frame checks 3D arrays, valid_videos handles batches, is_batched_video detects batch dimension); format conversion (make_batched_videos flattens nested structures and converts PIL frames to arrays, convert_pil_frames_to_video stacks image lists, make_batched_metadata wraps metadata in consistent format); video loading functions supporting 5 backends - read_video_opencv/pyav/decord/torchvision/torchcodec each with custom frame sampling, optional audio extraction, and metadata return; utility functions (get_video_size extracts dimensions, get_uniform_frame_indices computes sampling indices, default_sample_indices_fn provides uniform/fps-based sampling); URL/path handling via load_video dispatcher; and frame grouping/reordering for batch processing (group_videos_by_shape, reorder_videos).

**Significance:** Foundational layer for video input handling that abstracts backend-specific differences behind a uniform interface. Enables video models to work across diverse sources (files, URLs, arrays) and backends without model-specific code changes. Critical for video preprocessing pipeline reliability and performance, handling edge cases like variable frame rates, partial downloads, and format conversions. Provides the flexibility to choose optimal backends for different use cases (torchcodec for GPU decoding, OpenCV for CPU, PyAV for precise seeking).
