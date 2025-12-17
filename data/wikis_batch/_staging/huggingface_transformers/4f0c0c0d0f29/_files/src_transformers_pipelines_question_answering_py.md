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

**Purpose:** Implements extractive question answering pipeline that finds answer spans within context passages. Uses models like BERT and RoBERTa fine-tuned on SQuAD to locate text segments that answer natural language questions.

**Mechanism:** The QuestionAnsweringPipeline extends ChunkPipeline and uses QuestionAnsweringArgumentHandler to normalize diverse input formats (question+context pairs) into SquadExample objects. Processing converts examples to SquadFeatures with squad_convert_examples_to_features(), runs inference to get start/end logits, then decode_spans() computes probabilities for all valid answer spans using matrix multiplication of start/end probabilities. select_starts_ends() applies p_mask to exclude invalid tokens, normalizes with softmax, handles impossible answers, and returns top-k spans with scores and character offsets.

**Significance:** Core NLP component enabling natural language understanding applications to extract specific information from documents. Essential for search engines, chatbots, document analysis systems, customer service automation, and any application requiring precise information retrieval from text. Represents key capability in building knowledge-seeking AI systems that can read and comprehend documents to answer user questions.
