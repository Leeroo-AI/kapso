# Principle: Vision_Training_Setup

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|TRL SFTTrainer|https://huggingface.co/docs/trl/sft_trainer]]
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Computer_Vision]], [[domain::Training]], [[domain::Data_Collation]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

Mechanism for configuring vision-specific data collation and training settings for VLM fine-tuning.

=== Description ===

Vision training setup differs from language-only training because:

1. **Variable-Size Images**: Each image produces different numbers of tokens
2. **Data Collation**: Must handle mixed image and text tokens
3. **Processor vs Tokenizer**: Vision models use processors that handle both modalities
4. **Column Handling**: Different column removal and text field settings

The `UnslothVisionDataCollator` handles:
* Processing conversation messages with embedded images
* Creating proper attention masks for image tokens
* Padding sequences with variable image token counts
* Batching samples with different image resolutions

=== Usage ===

Use this principle when:
* Setting up the training loop for vision models
* Configuring data collation for multimodal batches
* The dataset contains images of varying sizes
* Standard SFTConfig needs vision-specific overrides

This step follows data preparation and precedes the training loop.

== Theoretical Basis ==

'''Vision Data Collation:'''
<syntaxhighlight lang="python">
# Pseudo-code for vision data collation
def collate_vision_batch(samples):
    # 1. Process each sample's conversation
    processed = []
    for sample in samples:
        # Apply chat template with images
        result = processor.apply_chat_template(
            sample["messages"],
            add_generation_prompt=False,
            return_tensors=None,
        )
        processed.append(result)

    # 2. Find max length (varies due to image tokens)
    max_len = max(len(p["input_ids"]) for p in processed)

    # 3. Pad to max length
    batch = {
        "input_ids": pad([p["input_ids"] for p in processed], max_len),
        "attention_mask": pad([p["attention_mask"] for p in processed], max_len),
        "pixel_values": stack([p["pixel_values"] for p in processed]),
    }

    return batch
</syntaxhighlight>

'''Key Configuration Differences:'''
{| class="wikitable"
|-
! Setting !! Language Model !! Vision Model
|-
| processing_class || tokenizer || processor
|-
| data_collator || Default/None || UnslothVisionDataCollator
|-
| remove_unused_columns || True (default) || False (required)
|-
| dataset_text_field || "text" || "" (empty)
|}

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_UnslothVisionDataCollator]]

