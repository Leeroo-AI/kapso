# File: `src/transformers/pipelines/table_question_answering.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 382 |
| Classes | `TableQuestionAnsweringArgumentHandler`, `TableQuestionAnsweringPipeline` |
| Imports | base, collections, generation, numpy, types, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements specialized question answering pipeline for tabular data using models like TAPAS. Answers natural language questions about structured data in pandas DataFrames or dictionaries by identifying relevant cells and applying aggregation operations.

**Mechanism:** The TableQuestionAnsweringPipeline uses TableQuestionAnsweringArgumentHandler to normalize table inputs (dict/DataFrame) with queries into consistent format. Processing tokenizes table structure with questions, then offers two inference modes: batch_inference() for simple queries or sequential_inference() for conversational SQA models that track previous answers through token_type_ids to handle follow-up questions. Postprocess() extracts answer cells by coordinates, applies aggregation functions (SUM, AVERAGE, COUNT) if model supports it, and formats results with answer text, cell coordinates, cell values, and aggregator type.

**Significance:** Specialized component enabling natural language interfaces to structured data. Critical for business intelligence, data analysis automation, spreadsheet querying, and making tabular information accessible without SQL or programming knowledge. Bridges the gap between unstructured natural language questions and structured database queries, democratizing data access for non-technical users.
