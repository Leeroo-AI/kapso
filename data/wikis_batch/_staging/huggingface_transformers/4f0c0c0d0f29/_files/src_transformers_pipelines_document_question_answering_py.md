# File: `src/transformers/pipelines/document_question_answering.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 546 |
| Classes | `ModelType`, `DocumentQuestionAnsweringPipeline` |
| Functions | `normalize_box`, `apply_tesseract` |
| Imports | base, generation, numpy, question_answering, re, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Answers questions about documents by combining OCR text extraction with visual layout understanding.

**Mechanism:** DocumentQuestionAnsweringPipeline supports three model types (LayoutLM, LayoutLMv2/v3, VisionEncoderDecoder/Donut). Uses apply_tesseract() with pytesseract for OCR when word_boxes not provided, normalize_box() to standardize bounding boxes to 0-1000 scale, extends ChunkPipeline for long documents with doc_stride, and extracts answers with start/end positions from the OCR'd text.

**Significance:** Critical for document understanding applications like invoice processing, form extraction, and contract analysis where spatial layout matters. Bridges computer vision and NLP by combining text, bounding boxes, and visual features to answer questions about scanned documents or images.
