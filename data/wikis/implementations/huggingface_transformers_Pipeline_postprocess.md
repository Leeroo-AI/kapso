{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Transformers Documentation|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Concrete tool for pipeline postprocessing provided by HuggingFace Transformers.

=== Description ===
The `postprocess` method is an abstract method in the `Pipeline` base class that subclasses must implement to transform raw model outputs into user-facing results. Each task-specific pipeline (TextClassificationPipeline, QuestionAnsweringPipeline, ImageClassificationPipeline, etc.) provides its own implementation that handles the task's specific output format requirements. This method receives tensors from the forward pass (already transferred to CPU) and performs Python-level operations to create structured, human-readable results.

The postprocessing implementation varies widely by task: classification pipelines apply softmax and map indices to labels, generation pipelines decode token IDs to text, token classification pipelines aggregate subword predictions and align with character offsets, question answering pipelines extract answer spans from contexts, and object detection pipelines apply non-maximum suppression and format bounding boxes. The method also handles filtering (by confidence threshold or top-k), ranking, formatting outputs as dictionaries or lists, and unwrapping single results from batch outputs when appropriate.

=== Usage ===
Implement this method when:
* Creating custom pipeline subclasses for new task types
* Adapting pipelines to return non-standard output formats
* Building specialized postprocessing for domain-specific applications
* Implementing custom filtering, ranking, or aggregation strategies
* Creating pipelines that return multiple result formats based on parameters

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/transformers transformers]
* '''File:''' src/transformers/pipelines/base.py:L1160-1167

=== Signature ===
<syntaxhighlight lang="python">
@abstractmethod
def postprocess(self, model_outputs: ModelOutput, **postprocess_parameters: dict) -> Any:
    """
    Postprocess will receive the raw outputs of the `_forward` method, generally tensors,
    and reformat them into something more friendly. Generally it will output a list or a
    dict or results (containing just strings and numbers).

    Args:
        model_outputs: Raw outputs from _forward method (logits, token IDs, etc.)
        **postprocess_parameters: Task-specific formatting and filtering parameters

    Returns:
        User-facing results in task-appropriate format (dict, list, str, etc.)

    Raises:
        NotImplementedError: This is an abstract method that must be overridden.

    Notes:
        - Subclasses must implement this method with task-specific logic
        - Input tensors are already on CPU (transferred by forward wrapper)
        - Should handle both batch and single-input cases
        - Output format should match task documentation
    """
    raise NotImplementedError("postprocess not implemented")
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# For implementing custom pipelines
from transformers.pipelines.base import Pipeline
from transformers.modeling_outputs import ModelOutput
from typing import Any

class CustomPipeline(Pipeline):
    def postprocess(self, model_outputs: ModelOutput, **postprocess_parameters: dict) -> Any:
        # Custom implementation here
        pass
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model_outputs || ModelOutput or Tensor or dict || Yes || Raw model outputs from forward pass. Structure varies by task: logits for classification, token IDs for generation, etc. Tensors already on CPU.
|-
| top_k || int || No || Number of top predictions to return for ranking tasks. Defaults vary by task (often 5 or None).
|-
| threshold || float || No || Minimum confidence score for filtering results. Defaults to 0.0 (no filtering).
|-
| return_all_scores || bool || No || (Classification) Return scores for all classes instead of just top-k. Defaults to False.
|-
| aggregation_strategy || str || No || (Token classification) How to aggregate subword predictions: "simple", "first", "average", "max". Defaults to "simple".
|-
| return_full_text || bool || No || (Generation) Include prompt in output text. Defaults to True.
|-
| clean_up_tokenization_spaces || bool || No || (Generation) Remove extra spaces from decoded text. Defaults to True.
|-
| **postprocess_parameters || dict || No || Additional task-specific formatting parameters.
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| results || dict or list or str || User-facing results. Format varies by task. For single inputs, may unwrap from list. Common formats: list of dicts with labels and scores, single dict for QA, string for generation.
|}

== Usage Examples ==

=== Example 1: Text Classification Postprocessing Implementation ===
<syntaxhighlight lang="python">
from transformers.pipelines.base import Pipeline
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
from typing import Any

class TextClassificationPipeline(Pipeline):
    def postprocess(
        self,
        model_outputs: SequenceClassifierOutput,
        **postprocess_parameters: dict
    ) -> list[dict[str, Any]]:
        """
        Convert logits to labeled predictions with scores.

        Args:
            model_outputs: Contains logits [batch, num_labels]
            postprocess_parameters: top_k, threshold, return_all_scores
        """
        # Extract parameters
        top_k = postprocess_parameters.get("top_k", None)
        threshold = postprocess_parameters.get("threshold", 0.0)
        return_all_scores = postprocess_parameters.get("return_all_scores", False)

        # Convert logits to probabilities
        logits = model_outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Process each item in batch
        results = []
        for batch_probs in probs:
            if return_all_scores:
                # Return all class scores
                scores = [
                    {
                        "label": self.model.config.id2label[i],
                        "score": prob.item()
                    }
                    for i, prob in enumerate(batch_probs)
                ]
            else:
                # Get top-k predictions
                k = top_k or len(batch_probs)
                top_k_probs, top_k_indices = torch.topk(batch_probs, k=min(k, len(batch_probs)))
                scores = [
                    {
                        "label": self.model.config.id2label[idx.item()],
                        "score": prob.item()
                    }
                    for idx, prob in zip(top_k_indices, top_k_probs)
                ]

            # Filter by threshold
            scores = [s for s in scores if s["score"] >= threshold]
            results.append(scores)

        # Unwrap single result
        if len(results) == 1:
            return results[0]
        return results

# Usage
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=0)
result = pipeline("This is great!", top_k=2)
# Output: [{"label": "POSITIVE", "score": 0.998}, {"label": "NEGATIVE", "score": 0.002}]
</syntaxhighlight>

=== Example 2: Text Generation Postprocessing Implementation ===
<syntaxhighlight lang="python">
from transformers.pipelines.base import Pipeline
import torch
from typing import Any

class TextGenerationPipeline(Pipeline):
    def postprocess(
        self,
        model_outputs: dict,
        **postprocess_parameters: dict
    ) -> list[dict[str, str]]:
        """
        Decode token IDs to text.

        Args:
            model_outputs: {"generated_sequence": Tensor, "input_ids": Tensor}
            postprocess_parameters: return_full_text, clean_up_tokenization_spaces
        """
        # Extract parameters
        return_full_text = postprocess_parameters.get("return_full_text", True)
        clean_up_spaces = postprocess_parameters.get("clean_up_tokenization_spaces", True)

        generated_sequences = model_outputs["generated_sequence"]
        prompt_ids = model_outputs["input_ids"]

        results = []
        for i, sequence in enumerate(generated_sequences):
            if return_full_text:
                # Decode entire sequence including prompt
                text = self.tokenizer.decode(
                    sequence,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=clean_up_spaces
                )
            else:
                # Decode only generated part
                prompt_length = prompt_ids[i].shape[0]
                generated_tokens = sequence[prompt_length:]
                text = self.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=clean_up_spaces
                )

            results.append({"generated_text": text})

        return results

# Usage
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, device="cuda:0")
result = pipeline("Once upon a time", max_new_tokens=20, return_full_text=False)
# Output: [{"generated_text": ", there was a brave knight who lived in a castle."}]
</syntaxhighlight>

=== Example 3: Question Answering Postprocessing Implementation ===
<syntaxhighlight lang="python">
from transformers.pipelines.base import Pipeline
from transformers.modeling_outputs import QuestionAnsweringModelOutput
import torch
from typing import Any

class QuestionAnsweringPipeline(Pipeline):
    def postprocess(
        self,
        model_outputs: QuestionAnsweringModelOutput,
        **postprocess_parameters: dict
    ) -> dict[str, Any]:
        """
        Extract answer span from logits.

        Args:
            model_outputs: Contains start_logits and end_logits [batch, seq_len]
            postprocess_parameters: max_answer_length, top_k, min_score
        """
        # Extract parameters
        max_answer_length = postprocess_parameters.get("max_answer_length", 15)
        top_k = postprocess_parameters.get("top_k", 1)
        min_score = postprocess_parameters.get("min_score", 0.0)
        context = postprocess_parameters["context"]
        offset_mapping = postprocess_parameters["offset_mapping"][0]

        start_logits = model_outputs.start_logits[0]
        end_logits = model_outputs.end_logits[0]

        # Find all valid start-end pairs
        candidates = []
        start_indexes = torch.argsort(start_logits, descending=True)[:20]
        end_indexes = torch.argsort(end_logits, descending=True)[:20]

        for start_idx in start_indexes:
            for end_idx in end_indexes:
                # Skip invalid spans
                if end_idx < start_idx:
                    continue
                if end_idx - start_idx + 1 > max_answer_length:
                    continue
                if offset_mapping[start_idx] is None or offset_mapping[end_idx] is None:
                    continue

                score = start_logits[start_idx] + end_logits[end_idx]
                candidates.append({
                    "start_idx": start_idx.item(),
                    "end_idx": end_idx.item(),
                    "score": score.item()
                })

        # Sort by score and filter
        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k]
        candidates = [c for c in candidates if c["score"] >= min_score]

        if not candidates:
            return {"answer": "", "score": 0.0, "start": 0, "end": 0}

        # Extract best answer
        best = candidates[0]
        char_start = offset_mapping[best["start_idx"]][0]
        char_end = offset_mapping[best["end_idx"]][1]
        answer = context[char_start:char_end]

        return {
            "answer": answer,
            "score": best["score"],
            "start": char_start,
            "end": char_end
        }

# Usage
pipeline = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer, device=0)
result = pipeline(
    question="What is the capital?",
    context="Paris is the capital of France.",
    max_answer_length=10
)
# Output: {"answer": "Paris", "score": 12.5, "start": 0, "end": 5}
</syntaxhighlight>

=== Example 4: Token Classification (NER) Postprocessing Implementation ===
<syntaxhighlight lang="python">
from transformers.pipelines.base import Pipeline
from transformers.modeling_outputs import TokenClassifierOutput
import torch
from typing import Any

class TokenClassificationPipeline(Pipeline):
    def postprocess(
        self,
        model_outputs: TokenClassifierOutput,
        **postprocess_parameters: dict
    ) -> list[dict[str, Any]]:
        """
        Convert token predictions to entity annotations.

        Args:
            model_outputs: Contains logits [batch, seq_len, num_labels]
            postprocess_parameters: aggregation_strategy, ignore_labels
        """
        # Extract parameters
        aggregation = postprocess_parameters.get("aggregation_strategy", "simple")
        ignore_labels = postprocess_parameters.get("ignore_labels", ["O"])
        sentence = postprocess_parameters["sentence"]
        offset_mapping = postprocess_parameters["offset_mapping"][0]
        special_tokens_mask = postprocess_parameters["special_tokens_mask"][0]

        # Get predictions
        logits = model_outputs.logits[0]
        predictions = torch.argmax(logits, dim=-1)
        scores = torch.nn.functional.softmax(logits, dim=-1).max(dim=-1).values

        # Build entities
        entities = []
        current_entity = None

        for idx, (pred_idx, score, (start, end), is_special) in enumerate(
            zip(predictions, scores, offset_mapping, special_tokens_mask)
        ):
            # Skip special tokens
            if is_special or start == end:
                continue

            label = self.model.config.id2label[pred_idx.item()]

            # Skip ignored labels
            if label in ignore_labels:
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
                continue

            word = sentence[start:end]

            # Handle entity continuity (for aggregation strategies)
            if aggregation == "simple":
                entities.append({
                    "entity": label,
                    "word": word,
                    "score": score.item(),
                    "start": start,
                    "end": end,
                    "index": idx
                })
            else:
                # Aggregate subword tokens
                is_continuation = label.startswith("I-") or word.startswith("##")

                if is_continuation and current_entity is not None:
                    # Continue current entity
                    current_entity["word"] += word.replace("##", "")
                    current_entity["end"] = end
                    if aggregation == "average":
                        current_entity["scores"].append(score.item())
                    elif aggregation == "max":
                        current_entity["score"] = max(current_entity["score"], score.item())
                else:
                    # Start new entity
                    if current_entity is not None:
                        # Finalize previous entity
                        if aggregation == "average":
                            current_entity["score"] = sum(current_entity["scores"]) / len(current_entity["scores"])
                            del current_entity["scores"]
                        entities.append(current_entity)

                    current_entity = {
                        "entity": label.replace("B-", "").replace("I-", ""),
                        "word": word,
                        "score": score.item(),
                        "start": start,
                        "end": end
                    }
                    if aggregation == "average":
                        current_entity["scores"] = [score.item()]

        # Don't forget last entity
        if current_entity is not None:
            if aggregation == "average":
                current_entity["score"] = sum(current_entity["scores"]) / len(current_entity["scores"])
                del current_entity["scores"]
            entities.append(current_entity)

        return entities

# Usage
pipeline = TokenClassificationPipeline(model=model, tokenizer=tokenizer, device=0)
result = pipeline(
    "John Smith works at Microsoft.",
    aggregation_strategy="max"
)
# Output: [
#   {"entity": "PER", "word": "John Smith", "score": 0.99, "start": 0, "end": 10},
#   {"entity": "ORG", "word": "Microsoft", "score": 0.98, "start": 20, "end": 29}
# ]
</syntaxhighlight>

=== Example 5: Image Classification Postprocessing Implementation ===
<syntaxhighlight lang="python">
from transformers.pipelines.base import Pipeline
from transformers.modeling_outputs import ImageClassifierOutput
import torch
from typing import Any

class ImageClassificationPipeline(Pipeline):
    def postprocess(
        self,
        model_outputs: ImageClassifierOutput,
        **postprocess_parameters: dict
    ) -> list[dict[str, Any]]:
        """
        Convert image classification logits to labeled predictions.

        Args:
            model_outputs: Contains logits [batch, num_classes]
            postprocess_parameters: top_k, threshold
        """
        # Extract parameters
        top_k = postprocess_parameters.get("top_k", 5)
        threshold = postprocess_parameters.get("threshold", 0.0)

        # Convert logits to probabilities
        logits = model_outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)

        results = []
        for batch_probs in probs:
            # Get top-k predictions
            top_k_probs, top_k_indices = torch.topk(batch_probs, k=min(top_k, len(batch_probs)))

            scores = [
                {
                    "label": self.model.config.id2label[idx.item()],
                    "score": prob.item()
                }
                for idx, prob in zip(top_k_indices, top_k_probs)
            ]

            # Filter by threshold
            scores = [s for s in scores if s["score"] >= threshold]
            results.append(scores)

        # Unwrap single result
        if len(results) == 1:
            return results[0]
        return results

# Usage
from PIL import Image
pipeline = ImageClassificationPipeline(
    model=model,
    image_processor=image_processor,
    device="cuda:0"
)
result = pipeline(Image.open("cat.jpg"), top_k=3)
# Output: [
#   {"label": "Egyptian cat", "score": 0.89},
#   {"label": "tabby cat", "score": 0.07},
#   {"label": "tiger cat", "score": 0.03}
# ]
</syntaxhighlight>

=== Example 6: Feature Extraction Postprocessing Implementation ===
<syntaxhighlight lang="python">
from transformers.pipelines.base import Pipeline
from transformers.modeling_outputs import BaseModelOutput
import torch
import numpy as np
from typing import Any

class FeatureExtractionPipeline(Pipeline):
    def postprocess(
        self,
        model_outputs: BaseModelOutput,
        **postprocess_parameters: dict
    ) -> np.ndarray:
        """
        Extract embeddings from model outputs.

        Args:
            model_outputs: Contains last_hidden_state [batch, seq_len, hidden_dim]
            postprocess_parameters: pooling_strategy, normalize
        """
        # Extract parameters
        pooling = postprocess_parameters.get("pooling_strategy", "mean")
        normalize = postprocess_parameters.get("normalize", False)

        # Get hidden states
        hidden_states = model_outputs.last_hidden_state  # [batch, seq_len, hidden_dim]

        # Apply pooling
        if pooling == "mean":
            # Mean pooling over sequence dimension
            embeddings = hidden_states.mean(dim=1)
        elif pooling == "max":
            # Max pooling over sequence dimension
            embeddings = hidden_states.max(dim=1).values
        elif pooling == "cls":
            # Use [CLS] token embedding
            embeddings = hidden_states[:, 0, :]
        else:
            # Return all tokens
            embeddings = hidden_states

        # Normalize if requested
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        # Convert to numpy
        return embeddings.cpu().numpy()

# Usage
pipeline = FeatureExtractionPipeline(model=model, tokenizer=tokenizer, device=0)
result = pipeline(
    "This is a sentence to encode.",
    pooling_strategy="mean",
    normalize=True
)
# Output: array([[0.023, -0.145, 0.678, ...]])  # shape: (1, 768)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Pipeline_Postprocessing]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Pipeline_Environment]]
