# File: `src/transformers/pipelines/depth_estimation.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 145 |
| Classes | `DepthEstimationPipeline` |
| Imports | base, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Estimates per-pixel depth information from single images using monocular depth estimation models.

**Mechanism:** The DepthEstimationPipeline loads images via URL or local path, preprocesses them through an image processor that handles resizing and normalization, and feeds them to depth estimation models (like DPT or Depth Anything). The model outputs predicted depth maps as tensors. Postprocessing uses the image processor's post_process_depth_estimation() to resize depth maps back to original image dimensions, then normalizes values to 0-255 range and converts to PIL Images for visualization. The pipeline returns both the raw depth tensor (for downstream use) and a normalized PIL image (for display).

**Significance:** Enables 3D scene understanding from 2D images, which is crucial for robotics, AR/VR, autonomous vehicles, and photo editing applications. By democratizing access to sophisticated depth estimation models through a simple interface, it makes advanced computer vision capabilities accessible to developers without deep learning expertise.
