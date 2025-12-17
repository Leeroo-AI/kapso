# File: `src/transformers/training_args_seq2seq.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 89 |
| Classes | `Seq2SeqTrainingArguments` |
| Imports | dataclasses, generation, logging, pathlib, training_args, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Extends TrainingArguments with sequence-to-sequence specific parameters for controlling text generation during evaluation and prediction, enabling proper metric computation for generation tasks.

**Mechanism:** Seq2SeqTrainingArguments inherits all TrainingArguments fields and adds four seq2seq-specific fields: sortish_sampler (bool, enables SortishSampler for grouping similar-length sequences), predict_with_generate (bool, uses model.generate() instead of forward pass for metrics like ROUGE/BLEU), generation_max_length (optional int, sets max_length for generate() defaulting to model config), generation_num_beams (optional int, sets beam search width defaulting to model config), and generation_config (GenerationConfig or path, loads custom generation parameters from file/Hub). The to_dict() method serializes the config while converting GenerationConfig objects to dictionaries for JSON compatibility. Uses @add_start_docstrings to inherit parent class documentation.

**Significance:** Essential configuration class for seq2seq models that bridges training infrastructure with text generation capabilities. Enables proper evaluation of translation, summarization, and other generation tasks where success is measured by generated output quality rather than next-token prediction loss. Works in tandem with Seq2SeqTrainer to provide a complete seq2seq training solution.
