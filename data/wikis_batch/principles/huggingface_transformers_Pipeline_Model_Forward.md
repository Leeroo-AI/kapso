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

Pipeline model forward is the execution of neural network inference to transform preprocessed tensors into raw model predictions.

=== Description ===

Pipeline model forward encapsulates the core computational step of ML inference: passing preprocessed tensors through a neural network to generate predictions. This principle defines the "hot path" of the inference pipeline—the performance-critical section where most computation occurs. The forward pass must be device-agnostic (working identically on CPU, GPU, or distributed configurations), framework-specific (utilizing PyTorch, TensorFlow, or JAX optimizations), and task-aware (understanding whether to call model() for classification or model.generate() for text generation).

The forward principle establishes isolation: it receives tensors and returns tensors, with no knowledge of preprocessing or postprocessing. This isolation enables several optimizations: gradient disabling (since inference doesn't need backpropagation), mixed precision (using float16/bfloat16 for speed), model compilation (JIT or graph optimization), and batching (processing multiple inputs simultaneously). The forward method is wrapped in infrastructure that handles device synchronization, inference mode context, and error handling, but the core logic remains pure tensor transformation.

For generative tasks, the forward pass includes decoding strategies: greedy search, beam search, sampling, or nucleus sampling. These strategies operate within the forward phase because they require iterative model calls with intermediate state management. The forward contract must support both discriminative models (single forward pass producing logits) and generative models (autoregressive decoding producing sequences).

=== Usage ===

Use pipeline model forward when you need to:
* Execute neural network inference on preprocessed tensors
* Generate raw model predictions (logits, embeddings, hidden states)
* Implement task-specific generation strategies (beam search, sampling)
* Optimize inference with device placement and precision control
* Isolate computation-heavy logic from preprocessing/postprocessing
* Support both single-forward (classification) and autoregressive (generation) tasks

== Theoretical Basis ==

Pipeline model forward follows a device-agnostic inference execution pattern:

```
Input: preprocessed_tensors (dict[str, torch.Tensor]), forward_params

Infrastructure Wrapper (applied by Pipeline.forward):
  # Disable gradient computation for inference
  with torch.no_grad():
    # Enter inference mode (PyTorch optimization)
    with torch.inference_mode():
      # Move tensors to model device
      for key, tensor in preprocessed_tensors.items():
        if isinstance(tensor, torch.Tensor):
          preprocessed_tensors[key] = tensor.to(model.device)

      # Call core forward logic
      output = _forward(preprocessed_tensors, **forward_params)

      return output

Core Forward Logic (task-specific):

For Discriminative Tasks (Classification, NER, etc.):
  Step 1: Single Forward Pass
    outputs = model(**preprocessed_tensors)
    # model() returns ModelOutput with logits, hidden_states, attentions

  Step 2: Extract Relevant Outputs
    # Most tasks only need logits
    return outputs  # Full ModelOutput object

For Generative Tasks (Text Generation, Summarization, Translation):
  Step 1: Extract Generation Parameters
    max_length = forward_params.get("max_length", 50)
    num_beams = forward_params.get("num_beams", 1)
    do_sample = forward_params.get("do_sample", False)
    temperature = forward_params.get("temperature", 1.0)
    top_k = forward_params.get("top_k", 50)
    top_p = forward_params.get("top_p", 1.0)

  Step 2: Call Generation Method
    generated_ids = model.generate(
      input_ids=preprocessed_tensors["input_ids"],
      attention_mask=preprocessed_tensors.get("attention_mask"),
      max_length=max_length,
      num_beams=num_beams,
      do_sample=do_sample,
      temperature=temperature,
      top_k=top_k,
      top_p=top_p,
      **forward_params
    )

  Step 3: Package Outputs
    return {
      "generated_ids": generated_ids,
      "input_ids": preprocessed_tensors["input_ids"]
    }

For Encoder-Decoder Tasks (Translation, Summarization):
  # Similar to generative but with encoder_outputs
  encoder_outputs = model.encoder(**preprocessed_tensors)
  decoder_outputs = model.decoder(
    encoder_hidden_states=encoder_outputs.last_hidden_state,
    **decoder_inputs
  )
  return decoder_outputs

For Embedding Tasks (Feature Extraction):
  outputs = model(**preprocessed_tensors)
  # Extract embeddings from appropriate layer
  embeddings = outputs.last_hidden_state  # or outputs.pooler_output
  return {"embeddings": embeddings}

Output: ModelOutput or dict[str, torch.Tensor]
```

Generation strategies (for generative tasks):

```
Greedy Decoding (num_beams=1, do_sample=False):
  for step in range(max_length):
    logits = model(input_ids)[:, -1, :]
    next_token = argmax(logits, dim=-1)
    input_ids = cat([input_ids, next_token], dim=1)
    if next_token == eos_token:
      break

Beam Search (num_beams>1):
  beams = [(input_ids, score=0.0)]
  for step in range(max_length):
    candidates = []
    for beam_ids, beam_score in beams:
      logits = model(beam_ids)[:, -1, :]
      log_probs = log_softmax(logits, dim=-1)
      top_k_probs, top_k_ids = topk(log_probs, k=num_beams)
      for prob, token_id in zip(top_k_probs, top_k_ids):
        new_beam = (cat([beam_ids, token_id]), beam_score + prob)
        candidates.append(new_beam)
    beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:num_beams]

Sampling (do_sample=True):
  for step in range(max_length):
    logits = model(input_ids)[:, -1, :] / temperature
    probs = softmax(logits, dim=-1)
    if top_k > 0:
      probs = filter_top_k(probs, k=top_k)
    if top_p < 1.0:
      probs = filter_nucleus(probs, p=top_p)
    next_token = sample(probs)
    input_ids = cat([input_ids, next_token], dim=1)
```

Key principles:

1. **Gradient-Free**: All inference runs without gradient tracking
2. **Device Consistency**: All tensors on same device as model
3. **Task Specialization**: Different forward logic for different task types
4. **Parameter Forwarding**: Generation parameters flow through cleanly
5. **Output Standardization**: Return ModelOutput or structured dict
6. **Isolation**: No preprocessing or postprocessing in forward logic

The forward pass is performance-critical and often benefits from hardware-specific optimizations like CUDA graphs, TensorRT, or ONNX conversion.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_Pipeline_forward]]

=== Part Of ===
* [[part_of::Principle:huggingface_transformers_Pipeline_Instantiation]]

=== Receives From ===
* [[receives_from::Principle:huggingface_transformers_Pipeline_Preprocessing]]

=== Feeds Into ===
* [[feeds_into::Principle:huggingface_transformers_Pipeline_Postprocessing]]
