# File: `src/transformers/training_args_seq2seq.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 89 |
| Classes | `Seq2SeqTrainingArguments` |
| Imports | dataclasses, generation, logging, pathlib, training_args, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Extends TrainingArguments with sequence-to-sequence specific configuration options for generation-based evaluation and SortishSampler support.

**Mechanism:** Inherits from TrainingArguments and adds five specialized fields: sortish_sampler (enables bucketing by sequence length for efficiency), predict_with_generate (toggles generation mode during evaluation), generation_max_length (controls output sequence length), generation_num_beams (configures beam search width), and generation_config (loads GenerationConfig from pretrained models). Overrides to_dict() to properly serialize GenerationConfig objects to dictionaries for JSON compatibility.

**Significance:** Provides the configuration complement to Seq2SeqTrainer, enabling users to control generation behavior during training through the same argument-based interface used for other training parameters. Essential for seq2seq tasks where generation quality is the primary evaluation metric.
