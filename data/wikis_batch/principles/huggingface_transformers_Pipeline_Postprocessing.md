{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Docs|https://huggingface.co/docs/transformers]]
|-
! Domains
| [[domain::NLP]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2025-12-17 00:00 GMT]]
|}

== Overview ==

Pipeline postprocessing is the transformation of raw model outputs into human-interpretable, structured results.

=== Description ===

Pipeline postprocessing addresses the semantic gap between neural network outputs (tensors of logits, token IDs, or embeddings) and the information users actually want (predicted labels, confidence scores, generated text, bounding boxes with class names). This transformation is task-specific and often domain-specific: classification tasks require softmax and label mapping, generation tasks require token decoding and special token removal, object detection requires coordinate rescaling and non-maximum suppression, and named entity recognition requires token aggregation and span extraction.

The postprocessing principle establishes the output contract: transform ModelOutput objects into JSON-serializable Python data structures (lists, dicts, primitives) that can be easily consumed by applications, logged, or displayed to users. This contract ensures that pipeline outputs are self-documenting (containing field names like "label" and "score" rather than raw indices) and consistent across tasks (similar tasks return similar structures).

Postprocessing also handles result filtering and ranking: returning top-k predictions, applying confidence thresholds, sorting by score, and aggregating results. The transformation must be deterministic and lossless where appropriate (e.g., generated text should be exactly what the model produced, with special tokens removed). For some tasks, postprocessing includes decoding strategies like beam search result selection, constraint satisfaction for structured outputs, or format conversion for domain-specific representations.

=== Usage ===

Use pipeline postprocessing when you need to:
* Convert raw model outputs to human-readable formats
* Apply softmax to logits and map indices to labels
* Decode token IDs to text strings
* Filter and rank predictions by confidence
* Aggregate token-level predictions into spans or entities
* Format outputs for JSON serialization and API responses
* Apply task-specific transformations (NMS, coordinate rescaling)

== Theoretical Basis ==

Pipeline postprocessing follows a task-specific transformation and formatting pattern:

```
Input: model_outputs (ModelOutput or dict), postprocess_params

For Classification Tasks:
  Step 1: Extract Logits
    logits = model_outputs.logits  # Shape: [batch_size, num_classes]

  Step 2: Apply Softmax
    probs = softmax(logits, dim=-1)  # Convert to probabilities

  Step 3: Get Top-K Predictions
    top_k = postprocess_params.get("top_k", 1)
    scores, indices = topk(probs, k=top_k, dim=-1)

  Step 4: Map Indices to Labels
    id2label = model.config.id2label
    results = []
    for i in range(batch_size):
      item_results = []
      for j in range(top_k):
        item_results.append({
          "label": id2label[indices[i, j].item()],
          "score": scores[i, j].item()
        })
      results.append(item_results if top_k > 1 else item_results[0])

  Output: list[dict] or list[list[dict]]
    [{"label": "POSITIVE", "score": 0.9998}]

For Text Generation Tasks:
  Step 1: Extract Generated IDs
    generated_ids = model_outputs["generated_ids"]  # Shape: [batch_size, seq_len]

  Step 2: Decode Token IDs
    generated_texts = tokenizer.batch_decode(
      generated_ids,
      skip_special_tokens=True,  # Remove [CLS], [SEP], [PAD]
      clean_up_tokenization_spaces=True
    )

  Step 3: Format Results
    results = []
    for text in generated_texts:
      results.append({"generated_text": text})

  Output: list[dict]
    [{"generated_text": "Once upon a time, there was a princess..."}]

For Token Classification (NER):
  Step 1: Extract Logits
    logits = model_outputs.logits  # Shape: [batch_size, seq_len, num_tags]

  Step 2: Get Predicted Tags
    predictions = argmax(logits, dim=-1)  # Shape: [batch_size, seq_len]

  Step 3: Map to Label Names
    id2label = model.config.id2label
    pred_labels = [[id2label[p.item()] for p in pred] for pred in predictions]

  Step 4: Aggregate Tokens into Entities
    aggregation_strategy = postprocess_params.get("aggregation_strategy", "simple")
    entities = []

    for tokens, labels, scores in zip(token_texts, pred_labels, confidence_scores):
      current_entity = None
      for token, label, score in zip(tokens, labels, scores):
        if label.startswith("B-"):  # Begin entity
          if current_entity:
            entities.append(current_entity)
          current_entity = {
            "entity_group": label[2:],
            "word": token,
            "start": token_start,
            "end": token_end,
            "score": score
          }
        elif label.startswith("I-") and current_entity:  # Inside entity
          current_entity["word"] += " " + token
          current_entity["end"] = token_end
          current_entity["score"] = (current_entity["score"] + score) / 2
        else:  # O label or end of entity
          if current_entity:
            entities.append(current_entity)
            current_entity = None

  Output: list[dict]
    [
      {"entity_group": "PER", "word": "John", "start": 11, "end": 15, "score": 0.995},
      {"entity_group": "LOC", "word": "New York", "start": 29, "end": 37, "score": 0.992}
    ]

For Question Answering:
  Step 1: Extract Start/End Logits
    start_logits = model_outputs.start_logits  # Shape: [batch_size, seq_len]
    end_logits = model_outputs.end_logits

  Step 2: Find Best Answer Span
    start_idx = argmax(start_logits, dim=-1)
    end_idx = argmax(end_logits, dim=-1)

  Step 3: Extract Answer Text
    answer_tokens = input_ids[0, start_idx:end_idx+1]
    answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True)

  Step 4: Calculate Confidence
    start_score = softmax(start_logits, dim=-1)[0, start_idx]
    end_score = softmax(end_logits, dim=-1)[0, end_idx]
    score = (start_score * end_score).item()

  Output: dict
    {
      "answer": "Paris",
      "score": 0.9845,
      "start": 45,
      "end": 50
    }

For Object Detection:
  Step 1: Extract Predictions
    logits = model_outputs.logits  # Shape: [batch_size, num_queries, num_classes]
    boxes = model_outputs.pred_boxes  # Shape: [batch_size, num_queries, 4]

  Step 2: Apply Softmax and Threshold
    probs = softmax(logits, dim=-1)
    scores, labels = probs.max(dim=-1)
    threshold = postprocess_params.get("threshold", 0.5)
    keep = scores > threshold

  Step 3: Rescale Bounding Boxes
    # Convert from normalized [0, 1] to pixel coordinates
    boxes_scaled = boxes * torch.tensor([img_width, img_height, img_width, img_height])

  Step 4: Apply NMS (Non-Maximum Suppression)
    keep_indices = nms(boxes_scaled, scores, iou_threshold=0.5)

  Step 5: Format Results
    results = []
    for idx in keep_indices:
      results.append({
        "label": id2label[labels[idx].item()],
        "score": scores[idx].item(),
        "box": {
          "xmin": boxes_scaled[idx, 0].item(),
          "ymin": boxes_scaled[idx, 1].item(),
          "xmax": boxes_scaled[idx, 2].item(),
          "ymax": boxes_scaled[idx, 3].item()
        }
      })

  Output: list[dict]
    [
      {"label": "car", "score": 0.982, "box": {"xmin": 123, "ymin": 45, "xmax": 456, "ymax": 234}},
      {"label": "person", "score": 0.952, "box": {"xmin": 67, "ymin": 89, "xmax": 234, "ymax": 567}}
    ]

Output: JSON-serializable data structures
```

Key principles:

1. **Human Readability**: Outputs use natural language labels, not indices
2. **Confidence Reporting**: Include scores/probabilities for transparency
3. **Structured Format**: Consistent field names across similar tasks
4. **JSON Serializable**: Only primitives, lists, and dicts in output
5. **Lossless Where Appropriate**: Generated text matches model output exactly
6. **Filtering and Ranking**: Support top-k, thresholds, and sorting
7. **Task Specificity**: Each task has appropriate output schema

Postprocessing completes the pipeline by bridging the gap between neural network mathematics and application requirements.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_Pipeline_postprocess]]

=== Part Of ===
* [[part_of::Principle:huggingface_transformers_Pipeline_Instantiation]]

=== Receives From ===
* [[receives_from::Principle:huggingface_transformers_Pipeline_Model_Forward]]
