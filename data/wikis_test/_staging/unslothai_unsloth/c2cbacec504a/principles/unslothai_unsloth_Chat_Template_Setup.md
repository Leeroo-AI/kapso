# Principle: unslothai_unsloth_Chat_Template_Setup

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|HuggingFace Chat Templates|https://huggingface.co/docs/transformers/chat_templating]]
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::NLP]], [[domain::Tokenization]], [[domain::Data_Preprocessing]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Technique for configuring tokenizer chat templates specifically for reinforcement learning workflows.

=== Description ===

Chat Template Setup in RL contexts focuses on:
1. **Prompt-only formatting**: RL datasets typically contain prompts without responses
2. **Generation prompt handling**: Ensuring proper assistant prefix for generation
3. **Token boundary awareness**: Critical for reward computation on completions

This is similar to Data_Formatting but optimized for RL data requirements.

=== Usage ===

Use this in GRPO/RL workflows when:
- Preparing prompt datasets for RL training
- Configuring generation prompts for sampling
- Ensuring consistency between training and inference

== Theoretical Basis ==

=== RL-Specific Template Considerations ===

<syntaxhighlight lang="python">
# For RL, we format prompts only (not responses)
def format_prompt_for_rl(prompt, tokenizer):
    messages = [{"role": "user", "content": prompt}]

    # Always add generation prompt for RL
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # Critical for RL
    )
    return formatted
</syntaxhighlight>

The generation prompt ensures the model knows to start generating an assistant response.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_get_chat_template]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_GRPO_Reinforcement_Learning]]
