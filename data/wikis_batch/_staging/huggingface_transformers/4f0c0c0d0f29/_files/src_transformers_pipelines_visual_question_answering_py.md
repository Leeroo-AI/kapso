# File: `src/transformers/pipelines/visual_question_answering.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 212 |
| Classes | `VisualQuestionAnsweringPipeline` |
| Imports | base, generation, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements visual question answering (VQA) pipeline that answers natural language questions about image content using vision-language models like ViLT.

**Mechanism:** The `VisualQuestionAnsweringPipeline` processes both image and question inputs together - loads images via `load_image`, tokenizes questions, processes images through image processor, combines features, and either generates text answers (for generative models with max_new_tokens=256 default) or classifies to answer classes. Supports flexible input formats: single image/question pairs, broadcasting (one image to multiple questions or vice versa), Cartesian product of image and question lists, and dataset integration via `KeyDataset`. Outputs ranked answer candidates with confidence scores.

**Significance:** Core multimodal pipeline enabling visual reasoning and image understanding through natural language. Essential for applications requiring scene comprehension, visual content query, and image-based question answering. Supports both classification-based VQA models (fixed answer vocabulary) and generative models (free-form answers).
