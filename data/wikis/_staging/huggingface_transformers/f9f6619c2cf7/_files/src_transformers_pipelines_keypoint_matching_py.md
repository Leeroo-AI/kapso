# File: `src/transformers/pipelines/keypoint_matching.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 176 |
| Classes | `Keypoint`, `Match`, `KeypointMatchingPipeline` |
| Functions | `validate_image_pairs` |
| Imports | base, collections, image_utils, typing, typing_extensions, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements a pipeline for finding and matching corresponding keypoints between pairs of images using keypoint detection and matching models.

**Mechanism:** The `KeypointMatchingPipeline` validates input image pairs using `validate_image_pairs()`, preprocesses both images together while preserving their target sizes, runs them through an `AutoModelForKeypointMatching` model to detect keypoints and compute matching scores, and postprocesses results using the image processor's keypoint matching post-processor to produce matched keypoint coordinates in original image space. Results are filtered by a configurable threshold and sorted by matching confidence scores.

**Significance:** This pipeline enables computer vision applications that require establishing correspondences between images, such as image stitching, 3D reconstruction, visual SLAM, object tracking, and augmented reality. It provides a high-level interface to sophisticated keypoint matching models, abstracting away the complexity of keypoint detection, descriptor extraction, and matching algorithms into a simple API that returns coordinate pairs with confidence scores.
