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

**Purpose:** Answers questions about document images by combining visual layout understanding with text extraction, enabling QA over forms, invoices, and structured documents.

**Mechanism:** The DocumentQuestionAnsweringPipeline supports two model architectures: (1) LayoutLM-family models that require OCR'd words and bounding boxes, and (2) VisionEncoderDecoder models like Donut that process images directly without OCR. For LayoutLM models, it optionally runs Tesseract OCR via apply_tesseract() to extract words and normalized bounding boxes (0-1000 scale), then tokenizes questions and words together with spatial information encoded in bbox features. The pipeline uses extractive QA logic (finding start/end positions) or generative decoding depending on model type. It extends ChunkPipeline to handle long documents that exceed model context windows through doc_stride chunking.

**Significance:** Bridges the gap between document understanding and question answering, enabling automated information extraction from scanned documents, forms, and receipts. This is critical for document automation in industries like finance, healthcare, and logistics where structured document parsing is essential but traditional OCR alone is insufficient for understanding semantic relationships.
