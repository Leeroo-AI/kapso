# File: `src/transformers/pipelines/text_classification.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 235 |
| Classes | `ClassificationFunction`, `TextClassificationPipeline` |
| Functions | `sigmoid`, `softmax` |
| Imports | base, inspect, numpy, typing, utils, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements text classification pipeline for categorizing text into predefined labels. Handles sentiment analysis, topic classification, intent detection, and other text categorization tasks using sequence classification models.

**Mechanism:** The TextClassificationPipeline processes text through preprocess() which tokenizes inputs (supporting text pairs via dict format), _forward() for model inference with use_cache disabled, and postprocess() which applies appropriate activation functions based on problem type: sigmoid for multi-label/single-label cases, softmax for multi-class classification, or none for regression. Returns top-k results with label names (via id2label) and confidence scores, with backward compatibility for legacy return_all_scores parameter.

**Significance:** Fundamental NLP component powering countless text understanding applications. Essential for sentiment analysis, content moderation, customer feedback categorization, email routing, news classification, and intent recognition in chatbots. One of the most widely-used pipeline types due to its broad applicability across industries for organizing and understanding textual data at scale.
