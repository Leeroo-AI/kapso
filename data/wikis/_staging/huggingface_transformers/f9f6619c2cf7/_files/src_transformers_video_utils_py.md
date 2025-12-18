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

**Purpose:** Provides low-level utilities for video data handling, including video loading from multiple sources, format validation, metadata management, and backend-specific decoders for various video libraries.

**Mechanism:** Defines VideoMetadata dataclass to track video properties (fps, frame count, duration, dimensions). Implements validation functions (is_valid_video, is_valid_video_frame, is_batched_video) to check video formats. Provides batching utilities (make_batched_videos, make_batched_metadata) to normalize input formats. Implements backend-specific decoders (read_video_opencv, read_video_decord, read_video_pyav, read_video_torchvision, read_video_torchcodec) that load videos and sample frames using custom sampling functions. The main load_video() function dispatches to appropriate decoder based on backend parameter, handles URLs (including YouTube via yt_dlp), and returns decoded frames with metadata. Includes utilities for grouping videos by shape, format conversion, padding, and RGB conversion.

**Significance:** Foundation layer for video processing that abstracts away the complexity of different video decoding libraries and input formats. Enables the video processing pipeline to work seamlessly with various video sources (local files, HTTP URLs, YouTube) and decoding backends, providing flexibility and robustness for video-language models.
