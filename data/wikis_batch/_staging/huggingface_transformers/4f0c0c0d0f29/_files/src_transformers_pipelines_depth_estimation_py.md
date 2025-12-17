# File: `src/transformers/pipelines/depth_estimation.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 145 |
| Classes | `DepthEstimationPipeline` |
| Imports | base, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Predicts depth maps from images using AutoModelForDepthEstimation models.

**Mechanism:** DepthEstimationPipeline loads images via load_image(), preprocesses with image processor, runs through depth estimation model, post-processes outputs with image_processor.post_process_depth_estimation() to match target sizes, normalizes depth values to 0-255 range, and returns both raw tensor and PIL image depth map.

**Significance:** Enables computer vision applications requiring depth perception like 3D reconstruction, augmented reality, and robotics by providing easy access to depth estimation models (DPT, ZoeDepth, Depth Anything) with automatic image loading and depth map visualization.
