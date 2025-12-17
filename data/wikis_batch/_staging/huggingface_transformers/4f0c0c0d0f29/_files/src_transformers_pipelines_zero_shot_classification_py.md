# File: `src/transformers/pipelines/zero_shot_classification.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 267 |
| Classes | `ZeroShotClassificationArgumentHandler`, `ZeroShotClassificationPipeline` |
| Imports | base, inspect, numpy, tokenization_python, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements zero-shot text classification using NLI (natural language inference) models to classify text into arbitrary categories without task-specific training.

**Mechanism:** The `ZeroShotClassificationPipeline` (extends ChunkPipeline) transforms classification into NLI by pairing each input sequence with hypotheses formatted from candidate labels (default template: "This example is {}."), creating premise-hypothesis pairs. It identifies the entailment label ID from model config, runs NLI inference on all pairs, extracts entailment logits as classification scores, and supports both single-label (softmax normalization) and multi-label (independent sigmoid) modes. The `ZeroShotClassificationArgumentHandler` manages sequence-label pair generation and validates template formatting.

**Significance:** Highly flexible text classification without requiring labeled training data for specific categories. Enables runtime category definition, making it ideal for dynamic classification scenarios, exploratory analysis, or when training data is unavailable. Leverages NLI models' semantic understanding to generalize across arbitrary label sets.
