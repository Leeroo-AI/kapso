# Principle: Weight_Conversion

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Language Models are Unsupervised Multitask Learners|https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf]]
* [[source::Doc|TensorFlow Checkpoint Format|https://www.tensorflow.org/guide/checkpoint]]
* [[source::Repo|PicoGPT|https://github.com/jaymody/picoGPT]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Model_Management]], [[domain::Data_Engineering]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==

Process of transforming model weights from one serialization format to another while preserving parameter values and structure.

=== Description ===

Weight Conversion is essential when using pre-trained models that were saved in a different framework than the one used for inference. GPT-2 was originally released by OpenAI as TensorFlow checkpoints, but PicoGPT performs inference using pure NumPy. This requires converting the TensorFlow checkpoint format into NumPy arrays organized in a specific nested dictionary structure.

The conversion must:
1. **Parse variable names** - TensorFlow checkpoints use hierarchical naming (e.g., `model/h0/attn/c_proj/w`)
2. **Load arrays** - Extract the actual tensor values from the checkpoint
3. **Restructure** - Organize into a nested dictionary matching the expected forward pass structure
4. **Handle shapes** - Some variables may have extra singleton dimensions that need squeezing

=== Usage ===

Use this principle when:
- Loading pre-trained weights from a different framework
- Converting between checkpoint formats (TF → PyTorch, TF → NumPy, etc.)
- Building framework-agnostic inference implementations

Weight conversion is typically done once at model load time, not during inference.

== Theoretical Basis ==

The conversion follows a parsing and restructuring pattern:

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# Abstract algorithm description
params = initialize_structure()

for variable_name, shape in list_checkpoint_variables():
    # Load the actual tensor data
    array = load_variable(checkpoint_path, variable_name)

    # Remove singleton dimensions
    array = squeeze(array)

    # Parse hierarchical name into path
    path = parse_variable_name(variable_name)  # "model/h0/attn/w" -> ["blocks", 0, "attn", "w"]

    # Set in nested structure
    set_nested(params, path, array)

return params
</syntaxhighlight>

Key data structures for GPT-2:
- '''wte''' - Token embeddings [n_vocab, n_embd]
- '''wpe''' - Positional embeddings [n_ctx, n_embd]
- '''blocks''' - List of transformer block weight dicts
- '''ln_f''' - Final layer normalization parameters

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Jaymody_PicoGPT_Load_Gpt2_Params_From_Tf_Ckpt]]
