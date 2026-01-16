# File: `WebAgent/WebWatcher/infer/scripts_eval/download_image.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 46 |
| Functions | `download_images_from_jsonl` |
| Imports | json, os, requests, urllib |

## Understanding

**Status:** ✅ Explored

**Purpose:** Utility script for downloading evaluation images from URLs specified in JSONL dataset files to a local directory for offline processing.

**Mechanism:** The single function `download_images_from_jsonl()` performs:
1. Creates output directory if it does not exist using `os.makedirs()`
2. Reads JSONL file line by line, parsing each JSON record
3. Extracts `file_path` field from each record (containing image URL)
4. Derives local filename from URL using `urlparse()` and `os.path.basename()`
5. Downloads images via `requests.get()` with streaming enabled
6. Writes image data in 1024-byte chunks to local file
7. Provides console feedback for successful downloads and error handling for failed requests
8. Default configuration downloads HLE benchmark images (hle_50.jsonl) to scripts_eval/images/hle_50/

**Significance:** Data preparation utility that enables local evaluation without repeated network requests. By pre-downloading evaluation images, the agent_eval.py script can run inference using local image paths via the IMAGE_DIR environment variable, improving reliability and speed during benchmark evaluation. Essential preprocessing step for vision-language evaluation workflows.
