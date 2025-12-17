# File: `src/transformers/trainer_seq2seq.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 386 |
| Classes | `Seq2SeqTrainer` |
| Imports | collections, contextlib, copy, generation, integrations, pathlib, torch, trainer, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Extends the base Trainer class to support sequence-to-sequence models by integrating text generation capabilities for evaluation and prediction, enabling proper metric computation for tasks like translation and summarization.

**Mechanism:** Seq2SeqTrainer inherits from Trainer and overrides key methods: __init__ loads GenerationConfig from args.generation_config; evaluate() and predict() accept generation kwargs (max_length, num_beams) from training args or method parameters; prediction_step() uses model.generate() instead of forward pass when predict_with_generate=True, generates tokens with configured parameters (respecting DeepSpeed Zero3 and FSDP contexts via summon_full_params), pads generated outputs to consistent length, computes loss separately from labels, and returns (loss, generated_tokens, labels) tuple. The _pad_tensors_to_max_len helper pads sequences using pad_token_id or eos_token_id. Handles both cases where labels are present (computes metrics) and absent (generation only).

**Significance:** Critical specialization for seq2seq tasks where evaluation requires generating complete sequences rather than computing next-token loss. Enables proper BLEU, ROUGE, and other generation-based metrics computation. Essential for training and evaluating translation, summarization, question answering, and other text generation models while maintaining compatibility with the broader Trainer ecosystem.
