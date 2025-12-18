# Heuristic: huggingface_transformers_Warning_Deprecated_ModelCard

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|HuggingFace Hub Cards|https://huggingface.co/docs/hub/model-cards]]
|-
! Domains
| [[domain::Deprecation]], [[domain::Documentation]], [[domain::Migration]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==
**DEPRECATION WARNING:** The `ModelCard` class in `modelcard.py` is deprecated and will be removed in Transformers version 5. Use HuggingFace Hub's model card functionality instead.

=== Description ===
The legacy `ModelCard` class was originally designed to store and generate model cards (documentation) for trained models. However, this functionality has been superseded by the HuggingFace Hub's more robust model card system, which supports YAML front-matter, markdown rendering, and direct Hub integration.

The deprecation warning states: "The class `ModelCard` is deprecated and will be removed in version 5 of Transformers."

=== Usage ===
**Do not use this class for new code.** If you encounter code using `ModelCard`, migrate to the HuggingFace Hub's `huggingface_hub.ModelCard` class or create a `README.md` file directly.

== The Insight (Rule of Thumb) ==

* **Action:** Migrate from `transformers.ModelCard` to `huggingface_hub.ModelCard` or direct README.md creation
* **Value:** N/A (Migration guidance)
* **Trade-off:** None - the new approach is strictly better
* **Timeline:** `ModelCard` will be removed in Transformers v5.x

== Reasoning ==

The HuggingFace Hub has evolved to provide a comprehensive model card system that:
1. Supports YAML metadata front-matter for structured information
2. Integrates directly with Hub repositories
3. Provides rich markdown rendering
4. Enables automatic card validation

The legacy `ModelCard` class predates this system and duplicates functionality now better handled by the Hub.

== Migration Guide ==

**Before (deprecated):**
<syntaxhighlight lang="python">
from transformers import ModelCard

# Old approach - don't use
card = ModelCard(
    model_details={"name": "my-model"},
    model_index=[{"name": "accuracy", "value": 0.95}]
)
card.save("./model_card.json")
</syntaxhighlight>

**After (recommended):**
<syntaxhighlight lang="python">
from huggingface_hub import ModelCard, ModelCardData

# New approach - use HuggingFace Hub
card_data = ModelCardData(
    language='en',
    license='mit',
    model_name='my-model'
)
card = ModelCard.from_template(
    card_data,
    template_path='modelcard_template.md',
    model_id='username/my-model'
)
card.push_to_hub('username/my-model')
</syntaxhighlight>

**Or simply create a README.md:**
<syntaxhighlight lang="markdown">
---
language: en
license: mit
tags:
- text-classification
---

# My Model

Model description goes here...
</syntaxhighlight>

== Related Pages ==
* [[uses_heuristic::Implementation:huggingface_transformers_ModelCard]]
