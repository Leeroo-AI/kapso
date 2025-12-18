{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|CPT|https://arxiv.org/abs/2410.17222]]
|-
! Domains
| [[domain::NLP]], [[domain::PEFT]], [[domain::Prompt_Tuning]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==

Configuration class for CPT (Context-aware Prompt Tuning) that stores parameters for context-aware soft prompt fine-tuning with token type masks and projection settings.

=== Description ===

CPTConfig extends PromptLearningConfig for context-aware prompt tuning. It supports token type masks (cpt_tokens_type_mask), weighted loss with decay (opt_weighted_loss_type, opt_loss_decay_factor), and projection settings (opt_projection_epsilon). CPT is specifically designed for causal language models.

=== Usage ===

Use CPTConfig for context-aware prompt tuning. Only supports task_type=CAUSAL_LM. Configure token IDs, masks, and projection parameters for context-aware adaptation.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft PEFT]
* '''File:''' [https://github.com/huggingface/peft/blob/main/src/peft/tuners/cpt/config.py src/peft/tuners/cpt/config.py]
* '''Lines:''' 1-100

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class CPTConfig(PromptLearningConfig):
    """
    Configuration for CPT (Context-aware Prompt Tuning).

    Args:
        cpt_token_ids: Token IDs for CPT prompts
        cpt_mask: Mask applied to CPT tokens
        cpt_tokens_type_mask: Type mask for each token
        opt_weighted_loss_type: Weighted loss type ('none', 'decay')
        opt_loss_decay_factor: Decay factor for loss weighting
        opt_projection_epsilon: Epsilon for input projection
        tokenizer_name_or_path: Tokenizer for initialization
    """
    cpt_token_ids: Optional[list[int]] = None
    cpt_mask: Optional[list[int]] = None
    cpt_tokens_type_mask: Optional[list[int]] = None
    opt_weighted_loss_type: Optional[Literal["none", "decay"]] = "none"
    opt_loss_decay_factor: Optional[float] = 1.0
    opt_projection_epsilon: Optional[float] = 0.1
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft import CPTConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| cpt_token_ids || list[int] || No || Token IDs for prompts
|-
| cpt_tokens_type_mask || list[int] || No || Token type mask
|-
| opt_weighted_loss_type || str || No || Loss weighting ('none', 'decay')
|-
| task_type || TaskType || Yes || Must be CAUSAL_LM
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| CPTConfig || dataclass || Configuration for get_peft_model
|}

== Usage Examples ==

=== Basic CPT Configuration ===
<syntaxhighlight lang="python">
from peft import CPTConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

config = CPTConfig(
    task_type=TaskType.CAUSAL_LM,  # Required
    cpt_token_ids=[1, 2, 3, 4, 5],
    cpt_tokens_type_mask=[1, 1, 1, 1, 1],
)

model = get_peft_model(model, config)
</syntaxhighlight>

=== CPT with Weighted Loss ===
<syntaxhighlight lang="python">
from peft import CPTConfig, TaskType

config = CPTConfig(
    task_type=TaskType.CAUSAL_LM,
    cpt_token_ids=[1, 2, 3, 4, 5],
    opt_weighted_loss_type="decay",
    opt_loss_decay_factor=0.9,
    opt_projection_epsilon=0.1,
)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
