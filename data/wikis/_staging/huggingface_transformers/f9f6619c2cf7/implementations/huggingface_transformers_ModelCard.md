{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|huggingface_transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Model_Architecture]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
**⚠️ DEPRECATED:** ModelCard is deprecated and will be removed in Transformers v5. Use `huggingface_hub.ModelCard` instead.

ModelCard provides structured model documentation and automated model card generation for transformer models, supporting both manual creation and auto-generation from training metadata.

=== Description ===
**Deprecation Notice:** This class emits a `FutureWarning` on instantiation. Migrate to `huggingface_hub.ModelCard` for new code.

The ModelCard class implements the Model Cards for Model Reporting framework, providing a structured way to document model details, intended use, training data, evaluation metrics, and ethical considerations. It includes the TrainingSummary dataclass which automatically generates model cards from Trainer instances by parsing training logs, extracting hyperparameters, and formatting metadata for the Hugging Face Hub. The module supports loading and saving model cards in JSON format, with special handling for YAML frontmatter metadata that includes task tags, dataset information, and evaluation results.

=== Usage ===
Use ModelCard when you need to document a pre-trained model or automatically generate documentation from training runs. The TrainingSummary.from_trainer() method is particularly useful for creating standardized model cards that include training hyperparameters, evaluation results, and framework versions. ModelCard supports loading from local files or the Hugging Face Hub, making it suitable for both manual documentation and automated model card generation workflows.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers huggingface_transformers]
* '''File:''' src/transformers/modelcard.py

=== Signature ===
<syntaxhighlight lang="python">
class ModelCard:
    def __init__(self, **kwargs):
        # Stores model card sections based on Model Cards paper
        pass

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # Load model card from Hub or local path
        pass

    def save_pretrained(self, save_directory_or_file):
        # Save model card to directory or file
        pass

@dataclass
class TrainingSummary:
    model_name: str
    language: Optional[Union[str, list[str]]] = None
    license: Optional[str] = None
    tasks: Optional[Union[str, list[str]]] = None
    dataset: Optional[Union[str, list[str]]] = None
    eval_results: Optional[dict[str, float]] = None
    hyperparameters: Optional[dict[str, Any]] = None

    @classmethod
    def from_trainer(cls, trainer, language=None, license=None, **kwargs):
        # Generate training summary from Trainer instance
        pass

    def to_model_card(self) -> str:
        # Convert training summary to formatted model card
        pass
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import ModelCard
from transformers.modelcard import TrainingSummary
</syntaxhighlight>

== I/O Contract ==

=== ModelCard Inputs ===
{| class="wikitable"
! Parameter !! Type !! Description
|-
| pretrained_model_name_or_path || str || Model ID on Hub or local path to model card file/directory
|-
| cache_dir || str (optional) || Directory for caching downloaded model cards
|-
| **kwargs || dict || Key-value pairs to update model card attributes
|}

=== ModelCard Outputs ===
{| class="wikitable"
! Return !! Type !! Description
|-
| modelcard || ModelCard || Instance with loaded model card data and metadata sections
|}

=== TrainingSummary Inputs ===
{| class="wikitable"
! Parameter !! Type !! Description
|-
| trainer || Trainer || Transformers Trainer instance with training state and logs
|-
| model_name || str (optional) || Name of the model (defaults to output directory name)
|-
| language || str/list (optional) || Language(s) the model was trained on
|-
| license || str (optional) || Model license (auto-inferred if possible)
|-
| tasks || str/list (optional) || ML tasks the model is designed for
|-
| dataset || str/list (optional) || Dataset name(s) used for training
|-
| finetuned_from || str (optional) || Base model that was fine-tuned
|}

=== TrainingSummary Outputs ===
{| class="wikitable"
! Return !! Type !! Description
|-
| model_card || str || Formatted markdown model card with YAML metadata frontmatter
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Example 1: Load existing model card from Hub
from transformers import ModelCard

modelcard = ModelCard.from_pretrained("google-bert/bert-base-uncased")
print(modelcard.model_details)

# Example 2: Create model card from local JSON file
modelcard = ModelCard.from_pretrained("./saved_model/modelcard.json")

# Example 3: Auto-generate model card from Trainer
from transformers import Trainer, TrainingSummary

# After training a model with Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)
trainer.train()

# Generate model card
training_summary = TrainingSummary.from_trainer(
    trainer,
    language="en",
    license="apache-2.0",
    tags=["text-classification", "sentiment-analysis"],
    finetuned_from="bert-base-uncased",
    dataset="imdb"
)

# Convert to markdown format
model_card_text = training_summary.to_model_card()

# Save to file
with open("README.md", "w") as f:
    f.write(model_card_text)

# Example 4: Parse training logs for evaluation results
from transformers.modelcard import parse_log_history

log_history = trainer.state.log_history
train_log, eval_lines, eval_results = parse_log_history(log_history)
print(f"Final evaluation results: {eval_results}")

# Example 5: Extract hyperparameters from Trainer
from transformers.modelcard import extract_hyperparameters_from_trainer

hyperparameters = extract_hyperparameters_from_trainer(trainer)
print(f"Training hyperparameters: {hyperparameters}")
</syntaxhighlight>

== Related Pages ==
* [[uses_heuristic::Heuristic:huggingface_transformers_Warning_Deprecated_ModelCard]]
