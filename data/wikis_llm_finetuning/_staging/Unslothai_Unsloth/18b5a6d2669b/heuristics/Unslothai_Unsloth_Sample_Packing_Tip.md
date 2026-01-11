# Heuristic: Sample_Packing_Tip

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|trainer.py|https://github.com/unslothai/unsloth/blob/main/unsloth/trainer.py]]
|-
! Domains
| [[domain::Optimization]], [[domain::Training]], [[domain::Efficiency]]
|-
! Last Updated
| [[last_updated::2026-01-09 12:00 GMT]]
|}

== Overview ==
Enable sample packing with `packing=True` for >2x faster training and reduced VRAM usage.

=== Description ===
Sample packing concatenates multiple shorter training samples into a single sequence up to `max_seq_length`. Instead of padding each sample individually (wasting compute on pad tokens), packing fills sequences efficiently by combining multiple samples.

Unsloth auto-enables padding-free batching when packing is not explicitly set, providing similar efficiency gains with automatic setup.

=== Usage ===
Use this heuristic when:
- **Training on short samples:** Average sample length << `max_seq_length`
- **Optimizing training speed:** Want >2x faster training throughput
- **Reducing VRAM usage:** Packing reduces wasted compute on padding

**Do NOT use when:**
- Training vision-language models (VLMs)
- Using custom data collators
- Model type is in blocklist (gemma2, gpt_oss)
- Need to return logits (`UNSLOTH_RETURN_LOGITS=1`)

== The Insight (Rule of Thumb) ==
* **Action:** Set `packing=True` in `SFTConfig` or `SFTTrainer`
* **Value:** Boolean `True`
* **Trade-off:**
  - Faster training (2x or more)
  - May slightly change loss computation due to cross-contamination between packed samples
  - Requires `dataset_text_field` or formatted dataset
* **Alternative:** If not setting `packing`, Unsloth auto-enables `padding_free` mode

== Reasoning ==
Training on short sequences with individual padding wastes significant compute:
- A batch of 8 samples with lengths [128, 256, 192, 64, 320, 96, 160, 224] padded to 512 wastes ~68% of compute
- Packing combines these into fewer, fuller sequences

Unsloth's implementation:
1. Auto-detects when packing is beneficial
2. Falls back gracefully if packing fails
3. Auto-enables padding-free mode as alternative

**Model Blocklist:** Some models don't work correctly with packing:
- `gemma2`: Uses slow_attention_softcapping with torch.compile issues
- `gpt_oss`: Uses Flex Attention which doesn't handle padding_free correctly

== Code Evidence ==

Packing blocklist from `trainer.py:57-60`:
<syntaxhighlight lang="python">
PADDING_FREE_BLOCKLIST = {
    "gemma2",  # - gemma2:  Uses slow_attention_softcapping which has torch.compile issues
    "gpt_oss",  # - gpt_oss: Uses Flex Attention which doesn't handle padding_free correctly
}
</syntaxhighlight>

Auto packing detection from `trainer.py:346-349`:
<syntaxhighlight lang="python">
if _should_pack(config_arg) and not blocked:
    configure_sample_packing(config_arg)
    packing_active = True
    logger.info("Unsloth: Sample packing enabled for SFTTrainer instance.")
</syntaxhighlight>

User-facing message from `trainer.py:393-396`:
<syntaxhighlight lang="python">
if trainer_packing and (packing_active or _should_pack(trainer_args)):
    enable_sample_packing(self.model, self)
    print(
        "🦥 Unsloth: Packing enabled - training is >2x faster and uses less VRAM!"
    )
</syntaxhighlight>

== Related Pages ==
* [[used_by::Implementation:Unslothai_Unsloth_SFTTrainer_train]]
* [[used_by::Implementation:Unslothai_Unsloth_UnslothTrainingArguments]]
