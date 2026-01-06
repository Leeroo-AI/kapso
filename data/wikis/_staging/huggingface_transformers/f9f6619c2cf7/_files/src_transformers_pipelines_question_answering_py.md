# File: `src/transformers/pipelines/question_answering.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 685 |
| Classes | `QuestionAnsweringArgumentHandler`, `QuestionAnsweringPipeline` |
| Functions | `decode_spans`, `select_starts_ends` |
| Imports | base, collections, data, inspect, modelcard, numpy, tokenization_python, types, typing, utils, ... +1 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements extractive question answering pipeline that finds answer spans within context text. Uses models fine-tuned on SQuAD-style datasets to predict start and end positions of answers.

**Mechanism:** Converts question-context pairs into SquadExample format, tokenizes with stride-based chunking for long contexts, runs model inference to get start/end logits, and decodes top-k answer spans using decode_spans function. Handles both fast and slow tokenizers with different preprocessing paths. Supports impossible answer detection and word-aligned answer extraction.

**Significance:** Core pipeline enabling extractive QA capabilities. Essential for document search and information extraction tasks where answers exist verbatim in source text.
