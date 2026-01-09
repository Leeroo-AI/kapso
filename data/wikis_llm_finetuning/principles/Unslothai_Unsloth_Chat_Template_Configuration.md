# Principle: Chat_Template_Configuration

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Blog|HuggingFace Chat Templates|https://huggingface.co/docs/transformers/chat_templating]]
* [[source::Paper|GRPO|https://arxiv.org/abs/2402.03300]]
|-
! Domains
| [[domain::NLP]], [[domain::Reinforcement_Learning]], [[domain::Data_Preprocessing]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Configuration of chat templates for reinforcement learning workflows, ensuring consistent formatting between generation prompts and reward computation.

=== Description ===

Chat Template Configuration for RL training ensures that:
1. Prompts for generation are correctly formatted
2. Generated completions can be properly extracted
3. Reward functions receive consistently formatted text

For GRPO specifically, the template must support reasoning formats (chain-of-thought) with clear delimiters for the reward function to parse.

This principle shares the same implementation as Data_Formatting (get_chat_template) but is applied in the RL context.

=== Usage ===

Configure chat templates after model loading, before dataset preparation. For RL, consider:
* Templates supporting reasoning tags (e.g., `<think>...</think>`)
* Clear delimiters for reward function parsing
* Consistency with base model's pre-training format

== Theoretical Basis ==

=== Template Requirements for RL ===

RL generation requires templates that:

1. **Clearly separate roles**: Generation starts at assistant turn
2. **Support reasoning markers**: For step-by-step thinking
3. **Define stop conditions**: EOS token for generation termination

<math>
\text{Prompt} = \text{Template}(\text{system}, \text{user\_turn})
</math>

<math>
\text{Completion} = \text{Model}.\text{generate}(\text{Prompt})
</math>

=== Reasoning Format Example ===

For math reasoning tasks:
```
<think>
Let me solve this step by step...
1. First observation...
2. Key insight...
</think>
The answer is \boxed{42}
```

The reward function parses this structure to verify:
* Reasoning is present (`<think>` tags)
* Final answer is correctly formatted (`\boxed{}`)

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_get_chat_template]]

(Note: Same implementation as Data_Formatting, different context)
