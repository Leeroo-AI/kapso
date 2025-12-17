{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Model_Loading]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Concrete tool for establishing parameter sharing relationships between model components, typically linking input embeddings and output projection layers provided by the HuggingFace Transformers library.

=== Description ===

`PreTrainedModel.tie_weights()` implements weight tying by making multiple model parameters reference the same underlying tensor storage. The most common use case is sharing embeddings between the token input embedding layer and the language modeling head output projection. This reduces model size (eliminating duplicate parameters) and can improve training through shared gradient updates. The function handles complex scenarios including: symmetric tying (swapping source/target if only one exists in checkpoint), multi-way tying (more than 2 parameters sharing storage), and validation (warning if checkpoint contains both parameters when tying is configured). It also adjusts bias tensors when tied weight shapes change.

=== Usage ===

Use this when you need to:
* Finalize model loading by establishing configured parameter sharing relationships
* Reduce model memory footprint through embedding/projection weight sharing
* Implement custom model architectures with tied parameters
* Handle checkpoints where only one of a tied pair exists
* Debug weight tying configurations and checkpoint compatibility

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/modeling_utils.py (lines 2383-2461)

=== Signature ===
<syntaxhighlight lang="python">
def tie_weights(
    self,
    missing_keys: Optional[set[str]] = None,
    recompute_mapping: bool = True
):
    """
    Tie the model weights. If recompute_mapping=False (default when called internally),
    it will rely on the model.all_tied_weights_keys attribute, containing the
    {target: source} mapping for the tied params. If recompute_mapping=True, it will
    re-check all internal submodels and their config to determine the params that need
    to be tied.

    Note that during from_pretrained, tying is *symmetric*: if the mapping says
    "tie target -> source" but source is missing in the checkpoint while target exists,
    we *swap* source and target so we can still tie everything to the parameter that
    actually exists.

    Args:
        missing_keys (Optional[set[str]]): Set of parameter names missing from checkpoint.
            If provided, enables symmetric tying logic.
        recompute_mapping (bool): If True, recompute tied weight mapping from config.
            If False, use cached mapping from model.all_tied_weights_keys.
            Default True when called externally, False during from_pretrained.

    Returns:
        None (modifies model parameters in-place)

    Side Effects:
        - Target parameters replaced with references to source parameters
        - missing_keys updated to remove successfully tied targets
        - model.all_tied_weights_keys may be updated if conflicts detected
        - Bias tensors adjusted if output embeddings have bias attribute
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import AutoModel

# tie_weights is a method on PreTrainedModel instances
model = AutoModel.from_pretrained("google-bert/bert-base-uncased")
model.tie_weights()  # Usually called automatically during loading
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| missing_keys || set[str] | None || No || Set of parameter names missing from loaded checkpoint; enables symmetric tying logic during from_pretrained
|-
| recompute_mapping || bool || No || Whether to recompute tied weight mappings from config (True) or use cached mapping (False). Default: True
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| None || None || Function modifies model in-place; no return value
|}

'''Post-Execution State:'''
* Target parameters now reference source parameter's storage (tied)
* Missing keys set updated (tied targets removed as they're no longer missing)
* Model memory footprint reduced by size of tied parameters
* Gradient updates to tied parameters affect all references

'''Common Tied Parameter Pairs:'''
* Input embeddings ↔ Output projection (LM head): `model.embed_tokens.weight` ↔ `lm_head.weight`
* Encoder embeddings ↔ Decoder embeddings: `encoder.embed_tokens.weight` ↔ `decoder.embed_tokens.weight`

== Usage Examples ==

=== Basic Usage ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM, AutoConfig
import torch

# Example 1: Automatic tying during model loading
# (happens internally in from_pretrained)
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Check if weights are tied
input_embeddings = model.transformer.wte.weight
output_embeddings = model.lm_head.weight

# For GPT-2, these should be the same object (tied)
print(f"Same object: {input_embeddings is output_embeddings}")
print(f"Same data pointer: {input_embeddings.data_ptr() == output_embeddings.data_ptr()}")

# Example 2: Manual tying after model creation
config = AutoConfig.from_pretrained("gpt2")
config.tie_word_embeddings = True  # Enable tying in config

# Create model from config (weights not loaded yet)
from accelerate import init_empty_weights
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

# Load weights...
# (simplified - actual loading omitted)

# Manually trigger weight tying
model.tie_weights()

# Verify tying
input_emb = model.transformer.wte.weight
output_emb = model.lm_head.weight
print(f"Tied: {input_emb.data_ptr() == output_emb.data_ptr()}")

# Example 3: Symmetric tying during checkpoint loading
# Simulating scenario where checkpoint has lm_head but not embed_tokens
config = AutoConfig.from_pretrained("gpt2")
config.tie_word_embeddings = True

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

# Simulate missing_keys from checkpoint loading
missing_keys = {"transformer.wte.weight"}  # Input embeddings missing

# tie_weights will swap: since source is missing but target exists,
# it will tie lm_head.weight -> transformer.wte.weight (backwards)
model.tie_weights(missing_keys=missing_keys)

# missing_keys now empty (transformer.wte.weight no longer missing - it's tied)
print(f"Missing keys after tying: {missing_keys}")

# Example 4: Disable tying via config
config = AutoConfig.from_pretrained("gpt2")
config.tie_word_embeddings = False  # Disable tying

model = AutoModelForCausalLM.from_config(config)
# Now model.transformer.wte.weight and model.lm_head.weight are separate

# Manually retie if needed
config.tie_word_embeddings = True
model.tie_weights(recompute_mapping=True)  # Recompute from updated config

# Example 5: Check tied weight mappings
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Inspect which weights are tied
print("Tied weight mappings:")
for target, source in model.all_tied_weights_keys.items():
    print(f"  {target} -> {source}")

# Example 6: Memory savings from tying
config = AutoConfig.from_pretrained("gpt2")

# Model with tying enabled
config.tie_word_embeddings = True
model_tied = AutoModelForCausalLM.from_config(config)
model_tied.tie_weights()

# Count unique parameters
def count_unique_params(model):
    seen_ptrs = set()
    unique_params = 0
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
        if param.data_ptr() not in seen_ptrs:
            unique_params += param.numel()
            seen_ptrs.add(param.data_ptr())
    return total_params, unique_params

total, unique = count_unique_params(model_tied)
print(f"Total param count: {total:,}")
print(f"Unique param count: {unique:,}")
print(f"Savings from tying: {total - unique:,} parameters")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Model_Post_Processing]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_PyTorch]]
