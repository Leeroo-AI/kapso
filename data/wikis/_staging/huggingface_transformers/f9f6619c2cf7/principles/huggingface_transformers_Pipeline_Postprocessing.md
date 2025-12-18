{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Documentation|https://huggingface.co/docs/transformers]]
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Pipeline postprocessing transforms raw model outputs into user-friendly, task-specific results with semantic meaning and appropriate formatting.

=== Description ===
Pipeline postprocessing is the principle of converting low-level model outputs (logits, token IDs, embeddings) into interpretable, human-readable results that match user expectations for the task. Raw model outputs are typically numerical tensors: classification logits are unnormalized scores, generation produces integer token IDs, detection yields bounding box coordinates, and embeddings are high-dimensional vectors. Users need label names, decoded text, annotated images, structured JSON, confidence scores, and ranked results instead of raw tensors.

This transformation involves multiple operations: applying softmax or sigmoid to convert logits to probabilities, mapping class indices to human-readable labels, decoding token IDs back to text with proper handling of special tokens, computing confidence scores and rankings, filtering results by threshold or top-k, formatting outputs as dictionaries or structured objects, handling special cases like empty results or ties, and optionally cleaning up formatting artifacts. Postprocessing runs on CPU after outputs are transferred from the accelerator, allowing Python-level operations without blocking GPU resources. This separation enables different postprocessing strategies for the same model (e.g., return top-1 vs top-5 predictions).

=== Usage ===
Apply this principle when:
* Designing inference APIs where users expect semantic results, not raw tensors
* Implementing custom pipelines for new tasks with specific output formats
* Building systems that need different result formats for different consumers
* Creating evaluation pipelines that compare model outputs to ground truth
* Optimizing throughput by offloading non-GPU operations to separate processing stages

== Theoretical Basis ==

=== Postprocessing Interface Contract ===

All pipeline postprocessing methods follow a standard signature:

<pre>
FUNCTION postprocess(model_outputs: ModelOutput, **postprocess_parameters: Dict) -> Any:
    """
    Args:
        model_outputs: Raw outputs from model forward pass (logits, token IDs, etc.)
        postprocess_parameters: Task-specific formatting and filtering options

    Returns:
        User-facing results in task-appropriate format (dict, list, string, etc.)

    Contract Requirements:
        1. Input tensors are on CPU (transferred by forward wrapper)
        2. Output format documented per task
        3. Must handle batch and single-input cases
        4. Should filter/rank results as needed
        5. Include confidence scores when applicable
    """
</pre>

=== Task-Specific Output Formats ===

Different tasks produce different output structures:

<pre>
Text Classification:
    Input: SequenceClassifierOutput(logits=[batch, num_labels])
    Output: List[Dict{"label": str, "score": float}]
    Example: [{"label": "POSITIVE", "score": 0.998}, {"label": "NEGATIVE", "score": 0.002}]

Token Classification (NER):
    Input: TokenClassifierOutput(logits=[batch, seq_len, num_labels])
    Output: List[Dict{"entity": str, "word": str, "score": float, "start": int, "end": int}]
    Example: [{"entity": "B-PER", "word": "John", "score": 0.99, "start": 0, "end": 4}]

Question Answering:
    Input: QuestionAnsweringModelOutput(start_logits=[batch, seq_len], end_logits=[batch, seq_len])
    Output: Dict{"answer": str, "score": float, "start": int, "end": int}
    Example: {"answer": "Paris", "score": 0.95, "start": 0, "end": 5}

Text Generation:
    Input: Tensor[batch, generated_length] (token IDs)
    Output: List[Dict{"generated_text": str}]
    Example: [{"generated_text": "Once upon a time, there was a brave knight..."}]

Image Classification:
    Input: ImageClassifierOutput(logits=[batch, num_classes])
    Output: List[Dict{"label": str, "score": float}]
    Example: [{"label": "Egyptian cat", "score": 0.89}, {"label": "tabby cat", "score": 0.07}]

Object Detection:
    Input: Dict{
        "logits": Tensor[batch, num_queries, num_classes],
        "boxes": Tensor[batch, num_queries, 4]
    }
    Output: List[Dict{"label": str, "score": float, "box": Dict{"xmin": int, "ymin": int, "xmax": int, "ymax": int}}]
    Example: [{"label": "dog", "score": 0.95, "box": {"xmin": 10, "ymin": 20, "xmax": 150, "ymax": 200}}]

Feature Extraction:
    Input: BaseModelOutput(last_hidden_state=[batch, seq_len, hidden_dim])
    Output: np.ndarray[batch, hidden_dim] or List[List[float]]
    Example: [[0.023, -0.145, 0.678, ...]]  # 768-dim embedding
</pre>

=== Postprocessing Operations by Task ===

'''Classification Postprocessing:'''
<pre>
1. Convert Logits to Probabilities
   logits = model_outputs.logits  # [batch, num_labels]
   probs = softmax(logits, dim=-1)  # Normalize to [0, 1]

2. Map Indices to Labels
   predicted_idx = argmax(probs, dim=-1)
   predicted_label = id2label[predicted_idx]

3. Extract Top-K Predictions
   top_k_probs, top_k_indices = torch.topk(probs, k=top_k)
   results = [
       {"label": id2label[idx], "score": prob}
       for idx, prob in zip(top_k_indices, top_k_probs)
   ]

4. Filter by Threshold (optional)
   results = [r for r in results if r["score"] >= threshold]

5. Sort by Score
   results = sorted(results, key=lambda x: x["score"], reverse=True)
</pre>

'''Token Classification Postprocessing:'''
<pre>
1. Convert Logits to Predictions
   logits = model_outputs.logits  # [batch, seq_len, num_labels]
   predictions = argmax(logits, dim=-1)  # [batch, seq_len]
   scores = softmax(logits, dim=-1).max(dim=-1).values  # [batch, seq_len]

2. Align Tokens with Characters
   # Use offset_mapping from preprocessing
   FOR token_idx, (start, end) IN enumerate(offset_mapping):
       IF special_token_mask[token_idx]:
           CONTINUE  # Skip [CLS], [SEP], [PAD]

       entity = id2label[predictions[token_idx]]
       word = text[start:end]
       score = scores[token_idx]

3. Aggregate Subwords
   # Combine WordPiece/BPE subwords into full words
   IF current_word is continuation:
       current_entity["word"] += word.replace("##", "")
       current_entity["end"] = end
   ELSE:
       entities.append(current_entity)
       current_entity = new_entity(entity, word, score, start, end)

4. Filter and Format
   entities = [e for e in entities if e["entity"] != "O"]  # Remove non-entities
   RETURN entities
</pre>

'''Generation Postprocessing:'''
<pre>
1. Decode Token IDs to Text
   generated_ids = model_outputs  # [batch, generated_length]

   FOR sequence IN generated_ids:
       # Remove prompt tokens (if included in output)
       generated_tokens = sequence[prompt_length:]

       # Decode to text
       text = tokenizer.decode(
           generated_tokens,
           skip_special_tokens=True,  # Remove [PAD], [EOS], etc.
           clean_up_tokenization_spaces=clean_up_spaces
       )

2. Handle Multiple Sequences (num_return_sequences > 1)
   results = [{"generated_text": text} for text in decoded_texts]

3. Return Prompt + Generation (optional)
   IF return_full_text:
       results = [{"generated_text": prompt + text} for text in decoded_texts]
   ELSE:
       results = [{"generated_text": text} for text in decoded_texts]
</pre>

'''Question Answering Postprocessing:'''
<pre>
1. Find Best Answer Span
   start_logits = model_outputs.start_logits  # [batch, seq_len]
   end_logits = model_outputs.end_logits  # [batch, seq_len]

   # Find all valid start-end pairs
   FOR start_idx IN top_k_starts:
       FOR end_idx IN top_k_ends WHERE end_idx >= start_idx:
           score = start_logits[start_idx] + end_logits[end_idx]
           candidates.append((start_idx, end_idx, score))

2. Select Best Candidate
   best_start, best_end, best_score = max(candidates, key=lambda x: x[2])

3. Extract Answer Text
   # Use offset_mapping to get character span
   char_start = offset_mapping[best_start][0]
   char_end = offset_mapping[best_end][1]
   answer = context[char_start:char_end]

4. Compute Confidence Score
   score = sigmoid(best_score) OR softmax(best_score)

5. Return Structured Result
   RETURN {
       "answer": answer,
       "score": score,
       "start": char_start,
       "end": char_end
   }
</pre>

'''Image Classification Postprocessing:'''
<pre>
1. Convert Logits to Probabilities
   logits = model_outputs.logits  # [batch, num_classes]
   probs = softmax(logits, dim=-1)

2. Get Top-K Predictions
   top_k_probs, top_k_indices = torch.topk(probs, k=top_k)

3. Map to Class Names
   results = [
       {
           "label": model.config.id2label[idx.item()],
           "score": prob.item()
       }
       for idx, prob in zip(top_k_indices[0], top_k_probs[0])
   ]

4. Return Formatted Results
   RETURN results
</pre>

=== Parameter Configuration ===

Postprocessing parameters control output format:

<pre>
Common Postprocessing Parameters:

Classification:
    - top_k: int (number of top predictions to return)
    - threshold: float (minimum confidence score)
    - return_all_scores: bool (return scores for all labels)

Token Classification:
    - aggregation_strategy: str ("simple", "first", "average", "max")
        - simple: Return all tokens
        - first: Use first subword prediction for whole word
        - average: Average scores across subwords
        - max: Use maximum score across subwords
    - ignore_labels: List[str] (labels to exclude, e.g., ["O"])

Generation:
    - return_full_text: bool (include prompt in output)
    - clean_up_tokenization_spaces: bool (remove extra spaces)
    - handle_long_generation: str (truncate, error, or allow)

Question Answering:
    - max_answer_length: int (maximum answer span length)
    - top_k: int (return top k answer candidates)
    - min_score: float (minimum confidence threshold)

Image Classification:
    - top_k: int (number of classes to return)

Object Detection:
    - threshold: float (confidence threshold for detections)
    - nms_threshold: float (non-maximum suppression threshold)
</pre>

=== Error Handling ===

Postprocessing must handle edge cases:

<pre>
Common Edge Cases:

1. Empty Results
   IF no predictions meet threshold:
       RETURN [] OR {"answer": "", "score": 0.0}

2. Batch vs Single Input
   IF batch_size == 1 AND not return_batch:
       RETURN results[0]  # Unwrap single result
   ELSE:
       RETURN results  # List of results

3. Missing Labels
   IF class_idx not in id2label:
       label = f"LABEL_{class_idx}"  # Fallback

4. Invalid Spans (QA)
   IF best_end < best_start:
       RETURN {"answer": "", "score": 0.0}

5. Decoding Errors (Generation)
   TRY:
       text = tokenizer.decode(tokens)
   EXCEPT:
       text = "<decoding error>"
</pre>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_Pipeline_postprocess]]
