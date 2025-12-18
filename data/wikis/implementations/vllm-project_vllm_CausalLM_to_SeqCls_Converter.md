{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::Model Conversion]], [[domain::Reranking]], [[domain::Classification]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Utility script to convert CausalLM reranker models to SequenceClassification format for efficient vLLM inference.

=== Description ===
This conversion tool transforms models trained as causal language models for reranking (like bge-reranker-v2-gemma) into proper sequence classification models that vLLM can serve more efficiently. Many reranking models are trained by finetuning causal LMs to output specific tokens (like "Yes"/"No" or "0"/"1") for relevance scoring. This script extracts the relevant weights from the language model head and creates a lightweight classification head, significantly reducing inference overhead. It supports two conversion methods: no_post_processing (direct token extraction) and from_2_way_softmax (for binary classification scenarios).

=== Usage ===
Use this script before deploying reranking models to vLLM when the model is a finetuned causal LM that outputs classification tokens. This is essential for models like BAAI/bge-reranker-v2-gemma, mixedbread-ai/mxbai-rerank-base-v2, and Qwen/Qwen3-Reranker-0.6B. The converted model will be faster and use less memory during inference.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/pooling/score/convert_model_to_seq_cls.py examples/pooling/score/convert_model_to_seq_cls.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Convert bge-reranker-v2-gemma (single token "Yes")
python convert_model_to_seq_cls.py \
  --model_name BAAI/bge-reranker-v2-gemma \
  --classifier_from_tokens '["Yes"]' \
  --method no_post_processing \
  --path ./bge-reranker-v2-gemma-seq-cls

# Convert mxbai-rerank (binary tokens "0"/"1")
python convert_model_to_seq_cls.py \
  --model_name mixedbread-ai/mxbai-rerank-base-v2 \
  --classifier_from_tokens '["0", "1"]' \
  --method from_2_way_softmax \
  --path ./mxbai-rerank-base-v2-seq-cls

# Convert Qwen3-Reranker (binary tokens "no"/"yes")
python convert_model_to_seq_cls.py \
  --model_name Qwen/Qwen3-Reranker-0.6B \
  --classifier_from_tokens '["no", "yes"]' \
  --method from_2_way_softmax \
  --path ./Qwen3-Reranker-0.6B-seq-cls
</syntaxhighlight>

== Key Concepts ==

=== Conversion Methods ===
Two methods are supported: (1) no_post_processing extracts weights for specified tokens directly, (2) from_2_way_softmax computes the difference between true/false token weights for binary classification.

=== Token ID Extraction ===
The script identifies the language model head weights corresponding to the classification tokens (e.g., "Yes", "0"/"1", "no"/"yes") and extracts them to form the classification layer.

=== Binary Classification Optimization ===
For binary cases with from_2_way_softmax, the script computes score_weight = LM_head[true_token] - LM_head[false_token], producing a single score that increases with relevance.

=== Model Architecture ===
The converted model uses AutoModelForSequenceClassification architecture, which vLLM's pooling runner handles more efficiently than running full causal LM inference.

=== Configuration Preservation ===
The script saves both the converted model and tokenizer, along with configuration specifying pad_token behavior, ensuring the converted model works correctly with vLLM.

== Usage Examples ==

<syntaxhighlight lang="python">
import transformers
import torch
import json

def convert_reranker(model_name, classifier_tokens, method, output_path):
    # Load original causal LM
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    causal_lm = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu"
    )

    # Determine number of labels
    num_labels = 1 if method == "from_2_way_softmax" else len(classifier_tokens)

    # Create sequence classification model
    seq_cls_model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        device_map="cpu"
    )

    # Extract weights from LM head
    lm_head_weights = causal_lm.lm_head.weight
    token_ids = [tokenizer.convert_tokens_to_ids(t) for t in classifier_tokens]

    if method == "from_2_way_softmax":
        # Compute difference for binary classification
        false_id, true_id = token_ids
        score_weight = (lm_head_weights[true_id].to(torch.float32) -
                       lm_head_weights[false_id].to(torch.float32))
        score_weight = score_weight.unsqueeze(0)
    else:
        # Direct extraction
        score_weight = lm_head_weights[token_ids]

    # Copy weights to classification head
    with torch.no_grad():
        seq_cls_model.score.weight.copy_(score_weight)
        if seq_cls_model.score.bias is not None:
            seq_cls_model.score.bias.zero_()

    # Configure and save
    seq_cls_model.config.use_pad_token = False
    seq_cls_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

# Example usage
convert_reranker(
    model_name="BAAI/bge-reranker-v2-gemma",
    classifier_tokens=["Yes"],
    method="no_post_processing",
    output_path="./bge-reranker-v2-gemma-seq-cls"
)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_Python_Environment]]
* [[uses::Tool:Transformers]]
* [[related::Implementation:vllm-project_vllm_Cross_Encoder_Scoring]]
* [[related::Implementation:vllm-project_vllm_Cohere_Rerank_Client]]
