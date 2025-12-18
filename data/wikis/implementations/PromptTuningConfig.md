= PromptTuningConfig =

== Knowledge Sources ==
* Source Repository: https://github.com/huggingface/peft
* Related: [https://arxiv.org/abs/2104.08691 The Power of Scale for Parameter-Efficient Prompt Tuning]

== Domains ==
* [[NLP]]
* [[PEFT]] (Parameter-Efficient Fine-Tuning)
* [[Prompt Engineering]]
* [[Soft Prompts]]
* [[Few-Shot Learning]]

== Overview ==

=== Description ===
PromptTuningConfig is the configuration class for storing the configuration of a PromptEmbedding. It implements prompt tuning, a parameter-efficient fine-tuning technique where continuous, task-specific vectors (soft prompts) are prepended to the input embeddings while keeping the language model frozen.

The configuration supports three initialization strategies: TEXT (initialize from text tokens), SAMPLE_VOCAB (randomly sample from vocabulary), and RANDOM (random continuous vectors). This allows flexible initialization based on the use case and available information.

=== Usage ===
PromptTuningConfig is used to configure prompt tuning for language models. It's typically used with models that support prompt embeddings to enable task-specific adaptation without modifying model weights.

== Code Reference ==

=== Source Location ===
<code>src/peft/tuners/prompt_tuning/config.py</code>

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class PromptTuningConfig(PromptLearningConfig):
    def __init__(
        self,
        prompt_tuning_init: Union[PromptTuningInit, str] = PromptTuningInit.RANDOM,
        prompt_tuning_init_text: Optional[str] = None,
        tokenizer_name_or_path: Optional[str] = None,
        tokenizer_kwargs: Optional[dict] = None,
        # Inherited from PromptLearningConfig:
        # num_virtual_tokens, task_type, inference_mode, etc.
    )
</syntaxhighlight>

=== Import Statement ===
<syntaxhighlight lang="python">
from peft.tuners.prompt_tuning.config import PromptTuningConfig, PromptTuningInit
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
! Parameter !! Type !! Default !! Description
|-
| prompt_tuning_init || Union[PromptTuningInit, str] || PromptTuningInit.RANDOM || Initialization strategy: TEXT, SAMPLE_VOCAB, or RANDOM
|-
| prompt_tuning_init_text || Optional[str] || None || Text to initialize prompt embedding (required if init is TEXT)
|-
| tokenizer_name_or_path || Optional[str] || None || Name or path of tokenizer (required if init is TEXT)
|-
| tokenizer_kwargs || Optional[dict] || None || Keyword arguments for AutoTokenizer.from_pretrained (only used with TEXT init)
|}

=== PromptTuningInit Enum Values ===
{| class="wikitable"
! Value !! Description
|-
| TEXT || Initialize prompt embedding with embeddings of provided text tokens
|-
| SAMPLE_VOCAB || Initialize with randomly sampled tokens from model's vocabulary
|-
| RANDOM || Initialize with random continuous soft tokens (may fall outside embedding manifold)
|}

=== Outputs ===
{| class="wikitable"
! Return Type !! Description
|-
| PromptTuningConfig || A configured PromptTuningConfig instance with peft_type set to PeftType.PROMPT_TUNING
|}

=== Validation ===
The <code>__post_init__</code> method performs validation:
* When <code>prompt_tuning_init</code> is TEXT, <code>tokenizer_name_or_path</code> must be provided
* When <code>prompt_tuning_init</code> is TEXT, <code>prompt_tuning_init_text</code> must be provided
* <code>tokenizer_kwargs</code> can only be used when <code>prompt_tuning_init</code> is TEXT

== Usage Examples ==

=== Random Initialization ===
<syntaxhighlight lang="python">
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Configure with random initialization (default)
config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    prompt_tuning_init=PromptTuningInit.RANDOM,
    num_virtual_tokens=20
)

# Apply prompt tuning
peft_model = get_peft_model(model, config)
print(f"Trainable parameters: {peft_model.num_parameters(only_trainable=True)}")
</syntaxhighlight>

=== Text Initialization ===
<syntaxhighlight lang="python">
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model
from transformers import AutoModelForSeq2SeqLM

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

# Initialize prompt from text
config = PromptTuningConfig(
    task_type="SEQ_2_SEQ_LM",
    prompt_tuning_init=PromptTuningInit.TEXT,
    prompt_tuning_init_text="Classify the sentiment of this text as positive or negative:",
    tokenizer_name_or_path="t5-base",
    num_virtual_tokens=20
)

peft_model = get_peft_model(model, config)
</syntaxhighlight>

=== Vocabulary Sampling Initialization ===
<syntaxhighlight lang="python">
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model

# Sample random tokens from vocabulary
config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    prompt_tuning_init=PromptTuningInit.SAMPLE_VOCAB,
    num_virtual_tokens=50,
    inference_mode=False
)

peft_model = get_peft_model(model, config)
</syntaxhighlight>

=== With Tokenizer Kwargs ===
<syntaxhighlight lang="python">
from peft import PromptTuningConfig, PromptTuningInit

# Provide additional tokenizer arguments
config = PromptTuningConfig(
    task_type="SEQ_CLS",
    prompt_tuning_init=PromptTuningInit.TEXT,
    prompt_tuning_init_text="Question: Is this statement true or false?",
    tokenizer_name_or_path="bert-base-uncased",
    tokenizer_kwargs={
        "use_fast": True,
        "add_special_tokens": True,
        "max_length": 512
    },
    num_virtual_tokens=10
)
</syntaxhighlight>

=== Multi-Task Setup ===
<syntaxhighlight lang="python">
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

# Configure different prompts for different tasks
task_configs = {
    "summarization": PromptTuningConfig(
        task_type="SEQ_2_SEQ_LM",
        prompt_tuning_init=PromptTuningInit.TEXT,
        prompt_tuning_init_text="summarize: ",
        tokenizer_name_or_path="t5-base",
        num_virtual_tokens=8
    ),
    "translation": PromptTuningConfig(
        task_type="SEQ_2_SEQ_LM",
        prompt_tuning_init=PromptTuningInit.TEXT,
        prompt_tuning_init_text="translate English to French: ",
        tokenizer_name_or_path="t5-base",
        num_virtual_tokens=8
    )
}

# Apply first adapter
peft_model = get_peft_model(model, task_configs["summarization"], adapter_name="summarization")

# Add second adapter
peft_model.add_adapter("translation", task_configs["translation"])
</syntaxhighlight>

=== Saving and Loading Configuration ===
<syntaxhighlight lang="python">
from peft import PromptTuningConfig, PromptTuningInit
import json

# Create configuration
config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    prompt_tuning_init=PromptTuningInit.TEXT,
    prompt_tuning_init_text="Generate a creative story:",
    tokenizer_name_or_path="gpt2",
    num_virtual_tokens=15
)

# Save configuration
config.save_pretrained("./prompt_tuning_config")

# Load configuration
loaded_config = PromptTuningConfig.from_pretrained("./prompt_tuning_config")
print(f"Loaded init method: {loaded_config.prompt_tuning_init}")
</syntaxhighlight>

== Initialization Strategies ==

=== TEXT Initialization ===
* '''Pros''': Uses meaningful token embeddings, may converge faster
* '''Cons''': Requires tokenizer, limited to vocabulary tokens
* '''Use case''': When you have a good textual description of the task

=== SAMPLE_VOCAB Initialization ===
* '''Pros''': Uses real vocabulary embeddings, no tokenizer needed for config
* '''Cons''': Random selection may not be meaningful
* '''Use case''': When you want embeddings from vocabulary but no specific text

=== RANDOM Initialization ===
* '''Pros''': No constraints, maximum flexibility
* '''Cons''': May start outside embedding manifold, potentially slower convergence
* '''Use case''': When you want the model to learn prompts from scratch

== Related Pages ==
* [[huggingface_peft_PromptEmbedding|PromptEmbedding]] - The embedding layer using this config
* [[huggingface_peft_PromptEncoder|PromptEncoder]] - Related prompt encoding approach
* [[huggingface_peft_PrefixTuningConfig|PrefixTuningConfig]] - Related prefix tuning configuration
* [[huggingface_peft_MultitaskPromptTuningConfig|MultitaskPromptTuningConfig]] - Multi-task variant
* [[PEFT]] - Parameter-Efficient Fine-Tuning overview
* [[Prompt Engineering]]

== Categories ==
[[Category:PEFT]]
[[Category:Configuration]]
[[Category:Prompt Tuning]]
[[Category:NLP]]
[[Category:HuggingFace]]
