# File: `tests/test_processing_common.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 1880 |
| Classes | `ProcessorTesterMixin` |
| Functions | `prepare_image_inputs`, `floats_list` |
| Imports | fetch_hub_objects_for_ci, huggingface_hub, inspect, json, numpy, os, parameterized, pathlib, random, shutil, ... +3 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Multimodal processor testing framework providing `ProcessorTesterMixin` for testing unified processors that handle multiple modalities (text, images, video, audio).

**Mechanism:** Tests processor functionality including saving/loading, JSON serialization, modality-specific preprocessing, and proper handling of combined inputs. Uses `prepare_image_inputs()` and `floats_list()` helpers to generate test data for various modalities.

**Significance:** Core test infrastructure for multimodal models. Ensures processors correctly handle combined inputs from different modalities and maintain consistency across save/load cycles.
