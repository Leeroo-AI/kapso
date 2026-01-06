# File: `src/transformers/pipelines/table_question_answering.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 382 |
| Classes | `TableQuestionAnsweringArgumentHandler`, `TableQuestionAnsweringPipeline` |
| Imports | base, collections, generation, numpy, types, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements table question answering pipeline for answering queries about structured tabular data. Supports both TAPAS-style models (with cell selection) and seq2seq models (with text generation).

**Mechanism:** Converts pandas DataFrames and queries into model inputs via tokenizer. For TAPAS models, supports sequential inference for conversational queries tracking previous answers, and batch inference for single queries. Outputs coordinates, cell values, and aggregation operations. For seq2seq models, generates textual answers directly.

**Significance:** Specialized pipeline enabling natural language queries over structured tables. Critical for business intelligence and data analysis applications where users query databases in plain language.
