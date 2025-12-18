== MultitaskPromptTuningConfig ==

=== Knowledge Sources ===
* [https://github.com/huggingface/peft PEFT Repository]
* [https://huggingface.co/papers/2303.02861 Multitask Prompt Tuning Paper]

=== Domains ===
[[Category:NLP]]
[[Category:PEFT]]
[[Category:Prompt_Tuning]]
[[Category:Multitask_Learning]]
[[Category:Configuration]]

=== Overview ===

==== Description ====
'''MultitaskPromptTuningConfig''' is a configuration class for multitask prompt tuning, extending the PromptTuningConfig to support training prompts across multiple tasks. It implements the approach described in the paper [https://huggingface.co/papers/2303.02861 "Multitask Prompt Tuning Enables Parameter-Efficient Transfer Learning"].

This configuration enables different initialization strategies for prompt tuning parameters, including:
* Random initialization
* Text-based initialization
* Transfer learning from source tasks (average, exact, or shared embeddings only)

The configuration manages task-specific parameters including the number of tasks, ranks for low-rank decomposition, and paths to pretrained source prompts for transfer learning scenarios.

==== Usage ====
Used to configure multitask prompt tuning models when adapting pre-trained language models to multiple related tasks simultaneously. It's particularly useful for parameter-efficient transfer learning where knowledge from source tasks can be leveraged to improve performance on target tasks.

=== Code Reference ===

==== Source Location ====
<code>src/peft/tuners/multitask_prompt_tuning/config.py</code>

==== Signature ====
<syntaxhighlight lang="python">
@dataclass
class MultitaskPromptTuningConfig(PromptTuningConfig):
    prompt_tuning_init: Union[MultitaskPromptTuningInit, str] = field(
        default=MultitaskPromptTuningInit.RANDOM,
        metadata={
            "help": (
                "How to initialize the prompt tuning parameters. Can be one of TEXT, RANDOM, AVERAGE_SOURCE_TASKS, "
                "EXACT_SOURCE_TASK, ONLY_SOURCE_SHARED."
            ),
        },
    )
    prompt_tuning_init_state_dict_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The path of source state dict. This is required when training the downstream target prompt from "
                "the pretrained source prompt"
            ),
        },
    )
    prompt_tuning_init_task: Optional[int] = field(default=0, metadata={"help": "source task id for initialization"})
    num_ranks: Optional[int] = field(default=1, metadata={"help": "ranks"})
    num_tasks: Optional[int] = field(default=1, metadata={"help": "number of tasks"})
</syntaxhighlight>

==== Import ====
<syntaxhighlight lang="python">
from peft.tuners.multitask_prompt_tuning import MultitaskPromptTuningConfig
</syntaxhighlight>

=== I/O Contract ===

==== Initialization Parameters ====
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| prompt_tuning_init || Union[MultitaskPromptTuningInit, str] || RANDOM || Initialization method: TEXT, RANDOM, AVERAGE_SOURCE_TASKS, EXACT_SOURCE_TASK, or ONLY_SOURCE_SHARED
|-
| prompt_tuning_init_state_dict_path || Optional[str] || None || Path to source state dict for transfer learning
|-
| prompt_tuning_init_task || Optional[int] || 0 || Source task ID for initialization when using EXACT_SOURCE_TASK
|-
| num_ranks || Optional[int] || 1 || Number of ranks for low-rank decomposition
|-
| num_tasks || Optional[int] || 1 || Number of tasks to handle
|-
| colspan="4" | ''Inherits all parameters from PromptTuningConfig''
|}

==== Initialization Enum Values ====
{| class="wikitable"
! Value !! Description
|-
| TEXT || Initialize prompt with text
|-
| RANDOM || Initialize prompt with random matrix
|-
| AVERAGE_SOURCE_TASKS || Average the prefix and column matrices from all source tasks
|-
| EXACT_SOURCE_TASK || Use prefix and column matrices from a specific source task
|-
| ONLY_SOURCE_SHARED || Use only the shared prompt embeddings from source training
|}

==== Returns ====
Configuration object ready to be used with MultitaskPromptEmbedding model.

==== Side Effects ====
* Sets peft_type to PeftType.MULTITASK_PROMPT_TUNING in __post_init__

=== Usage Examples ===

==== Basic Configuration ====
<syntaxhighlight lang="python">
from peft import MultitaskPromptTuningConfig

# Create configuration for multitask prompt tuning with random initialization
config = MultitaskPromptTuningConfig(
    task_type="SEQ_2_SEQ_LM",
    num_virtual_tokens=20,
    num_tasks=5,
    num_ranks=4,
    prompt_tuning_init="RANDOM"
)
</syntaxhighlight>

==== Transfer Learning from Source Tasks ====
<syntaxhighlight lang="python">
from peft import MultitaskPromptTuningConfig

# Initialize from averaged source tasks
config = MultitaskPromptTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=10,
    num_tasks=3,
    num_ranks=2,
    prompt_tuning_init="AVERAGE_SOURCE_TASKS",
    prompt_tuning_init_state_dict_path="path/to/source_model.pt"
)
</syntaxhighlight>

==== Exact Source Task Initialization ====
<syntaxhighlight lang="python">
from peft import MultitaskPromptTuningConfig

# Initialize from a specific source task
config = MultitaskPromptTuningConfig(
    task_type="SEQ_CLS",
    num_virtual_tokens=15,
    num_tasks=4,
    num_ranks=3,
    prompt_tuning_init="EXACT_SOURCE_TASK",
    prompt_tuning_init_state_dict_path="path/to/source_model.pt",
    prompt_tuning_init_task=2  # Use task ID 2 from source
)
</syntaxhighlight>

=== Related Pages ===
* [[huggingface_peft_MultitaskPromptTuningModel|MultitaskPromptTuningModel]] - The model implementation using this configuration
* [[huggingface_peft_PromptTuningConfig|PromptTuningConfig]] - Parent configuration class
* [[huggingface_peft_PeftConfig|PeftConfig]] - Base PEFT configuration class
* [[PEFT|Parameter-Efficient Fine-Tuning]]
* [[Prompt_Tuning|Prompt Tuning]]
