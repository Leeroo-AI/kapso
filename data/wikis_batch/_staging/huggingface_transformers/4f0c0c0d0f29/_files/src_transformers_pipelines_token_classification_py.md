# File: `src/transformers/pipelines/token_classification.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 646 |
| Classes | `TokenClassificationArgumentHandler`, `AggregationStrategy`, `TokenClassificationPipeline` |
| Imports | base, models, numpy, types, typing, utils, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements named entity recognition (NER) and token classification pipeline that labels individual tokens in text sequences with classes like person, organization, location, or part-of-speech tags.

**Mechanism:** The `TokenClassificationPipeline` (extends ChunkPipeline) processes text through tokenization with offset mapping, runs model inference to get token-level predictions, and offers five aggregation strategies: NONE (raw tokens), SIMPLE (groups consecutive entities), FIRST (uses first token's label for words), AVERAGE (averages scores across word tokens), and MAX (uses highest-scoring token per word). Uses `BasicTokenizer` for word boundary detection and `TokenClassificationArgumentHandler` to manage inputs. Supports chunking with stride for long sequences.

**Significance:** Critical pipeline for information extraction tasks. Handles complex entity grouping challenges (e.g., preventing "Microsoft" from splitting into "Micro" + "soft") and provides flexible aggregation strategies to match different model architectures and linguistic requirements. Widely used for NER, POS tagging, and similar token-level classification tasks.
