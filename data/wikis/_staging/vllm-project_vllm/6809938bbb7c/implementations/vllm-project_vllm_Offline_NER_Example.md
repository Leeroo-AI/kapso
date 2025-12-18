{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::NER]], [[domain::Token Classification]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Offline batch inference example for Named Entity Recognition using vLLM's token classification pooling task.

=== Description ===
This example demonstrates offline NER inference using vLLM's LLM class with the token_classify pooling task. It uses the NeuroBERT-NER model to identify named entities in text, predicting entity labels for each token. The example shows how to extract logits from the pooling output, map them to entity labels, and display results in a human-readable format. This offline approach is ideal for batch processing scenarios where you need direct Python API access rather than HTTP endpoints.

=== Usage ===
Use this example for batch NER processing, research experiments requiring direct model access, or when integrating NER into Python pipelines. It's suitable for extracting person names, organizations, locations, and other entities from text documents at scale.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/pooling/token_classify/ner.py examples/pooling/token_classify/ner.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Run offline NER inference
python ner.py

# With custom model
python ner.py --model boltuix/NeuroBERT-NER
</syntaxhighlight>

== Key Concepts ==

=== Token Classification Task ===
Uses llm.encode() with pooling_task="token_classify" to generate per-token predictions rather than sequence-level embeddings or classifications.

=== Logits and Predictions ===
The output contains logits for each token-label pair. Taking argmax along the label dimension gives the predicted entity label for each token.

=== Label Mapping ===
The model's config contains an id2label mapping that converts predicted label IDs to human-readable entity tags (e.g., B-PER for beginning of person name).

=== Special Token Filtering ===
The example filters out special tokens (like [CLS], [SEP], [PAD]) when displaying results, showing only meaningful content tokens.

=== FlexibleArgumentParser ===
Uses vLLM's FlexibleArgumentParser with EngineArgs.add_cli_args() to support all engine configuration options while providing example-specific defaults.

== Usage Examples ==

<syntaxhighlight lang="python">
from vllm import LLM

# Initialize model for token classification
llm = LLM(
    model="boltuix/NeuroBERT-NER",
    runner="pooling",
    enforce_eager=True,
    trust_remote_code=True
)

# Get tokenizer and label mapping
tokenizer = llm.get_tokenizer()
label_map = llm.llm_engine.vllm_config.model_config.hf_config.id2label

# Prepare input text
prompts = [
    "Barack Obama visited Microsoft headquarters in Seattle on January 2025."
]

# Run NER inference
outputs = llm.encode(prompts, pooling_task="token_classify")

# Process results
for prompt, output in zip(prompts, outputs):
    # Get predicted labels
    logits = output.outputs.data
    predictions = logits.argmax(dim=-1)

    # Map to tokens and labels
    tokens = tokenizer.convert_ids_to_tokens(output.prompt_token_ids)
    labels = [label_map[p.item()] for p in predictions]

    # Display results (filtering special tokens)
    for token, label in zip(tokens, labels):
        if token not in tokenizer.all_special_tokens:
            print(f"{token:15} → {label}")
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
* [[related::Implementation:vllm-project_vllm_NER_API_Client]]
