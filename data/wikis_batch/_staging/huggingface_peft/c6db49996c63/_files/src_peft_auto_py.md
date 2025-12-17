# File: `src/peft/auto.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 184 |
| Classes | `_BaseAutoPeftModel`, `AutoPeftModel`, `AutoPeftModelForCausalLM`, `AutoPeftModelForSeq2SeqLM`, `AutoPeftModelForSequenceClassification`, `AutoPeftModelForTokenClassification`, `AutoPeftModelForQuestionAnswering`, `AutoPeftModelForFeatureExtraction` |
| Imports | __future__, config, importlib, os, peft_model, transformers, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides automatic model class selection for loading pretrained PEFT models based on task type.

**Mechanism:** Implements a family of AutoPeftModel classes that mirror Transformers' Auto classes. When from_pretrained is called, it loads the PeftConfig, infers the appropriate PeftModel class based on task_type, loads the base model using Transformers' Auto classes, and wraps it with the correct PEFT wrapper. Handles tokenizer resizing when needed.

**Significance:** Simplifies the user experience by allowing users to load PEFT models without knowing the specific model class needed. Follows the Transformers library pattern, making PEFT integration seamless for existing Transformers users. Critical for the library's usability.
