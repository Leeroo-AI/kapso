# Workflow: Text_Generation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PicoGPT|https://github.com/jaymody/picoGPT]]
* [[source::Blog|GPT in 60 Lines of NumPy|https://jaykmody.com/blog/gpt-from-scratch/]]
* [[source::Paper|GPT-2 Paper|https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf]]
|-
! Domains
| [[domain::LLMs]], [[domain::Inference]], [[domain::Education]]
|-
! Last Updated
| [[last_updated::2026-01-14 10:00 GMT]]
|}

== Overview ==
End-to-end process for generating text using a minimal, educational GPT-2 implementation in pure NumPy.

=== Description ===
This workflow demonstrates how to run GPT-2 text generation using PicoGPT, a minimal implementation designed for educational purposes. It covers the complete pipeline from loading pre-trained OpenAI GPT-2 weights to auto-regressive text generation using greedy decoding. The implementation uses pure NumPy for all transformer operations, making the underlying mathematics explicit and easy to understand.

'''Key characteristics:'''
* Pure NumPy inference (no PyTorch/TensorFlow runtime)
* Greedy decoding (argmax token selection)
* Support for all GPT-2 model sizes (124M, 355M, 774M, 1558M)
* Automatic download of OpenAI's pre-trained weights

=== Usage ===
Execute this workflow when you want to:
* Generate text completions from a prompt using GPT-2
* Understand how transformer-based language models work at a low level
* Study the GPT-2 architecture through readable, well-commented code
* Run GPT-2 inference without heavy framework dependencies

'''Input:''' A text prompt string (e.g., "Alan Turing theorized that computers would one day become")

'''Output:''' Generated text continuation (default: 40 tokens)

== Execution Steps ==

=== Step 1: Model Loading ===

Load the GPT-2 encoder (tokenizer), hyperparameters, and model weights. If the model files are not present locally, they are automatically downloaded from OpenAI's public Azure blob storage.

'''What happens:'''
* Check if model checkpoint exists in the specified models directory
* If not present, download all required files: checkpoint, encoder.json, hparams.json, vocab.bpe, and TensorFlow checkpoint files
* Load the BPE tokenizer with vocabulary and merge rules
* Parse hyperparameters (n_layer, n_head, n_embd, n_ctx, n_vocab)
* Load TensorFlow checkpoint and restructure weights into nested NumPy arrays

'''Key considerations:'''
* Supported model sizes: 124M, 355M, 774M, 1558M
* First run requires downloading ~500MB for 124M model
* Weights are organized as nested dicts: `params["blocks"][i]["attn"]["c_attn"]["w"]`

=== Step 2: Input Tokenization ===

Convert the input text prompt into a sequence of token IDs using GPT-2's Byte Pair Encoding (BPE) tokenizer.

'''What happens:'''
* Split input text using regex pattern for words, numbers, and special tokens
* Encode each character to bytes, then to unicode representation
* Apply BPE merges iteratively based on learned merge rankings
* Map resulting subword tokens to vocabulary IDs

'''Key considerations:'''
* The tokenizer handles contractions ('s, 't, 're, etc.) specially
* UTF-8 bytes are mapped to unicode characters to avoid UNK tokens
* Maximum context length is limited by model's n_ctx (1024 for GPT-2)

=== Step 3: Transformer Forward Pass ===

Run the input token sequence through the GPT-2 transformer architecture to produce logits (unnormalized probabilities) for the next token.

'''What happens:'''
* Look up token embeddings (wte) and add positional embeddings (wpe)
* Process through N transformer blocks, each containing:
  - Layer normalization
  - Multi-head causal self-attention with masking
  - Residual connection
  - Layer normalization
  - Feed-forward network (GELU activation, project up then down)
  - Residual connection
* Apply final layer normalization
* Project to vocabulary size using transposed embedding matrix

'''Pseudocode:'''
  1. x = wte[input_ids] + wpe[positions]
  2. for each block in transformer_blocks:
       x = x + attention(layer_norm(x))
       x = x + ffn(layer_norm(x))
  3. x = layer_norm(x)
  4. logits = x @ wte.T

=== Step 4: Auto-regressive Generation ===

Generate new tokens one at a time using greedy decoding, appending each prediction to the input sequence for the next iteration.

'''What happens:'''
* Run forward pass to get logits for entire sequence
* Select token with highest probability at the last position (argmax)
* Append selected token to input sequence
* Repeat until desired number of tokens generated

'''Key considerations:'''
* Uses greedy decoding (no temperature, top-k, or top-p sampling)
* Each new token requires a full forward pass through all previous tokens
* Generation loop shows progress via tqdm progress bar

=== Step 5: Output Decoding ===

Convert the generated token IDs back into human-readable text using the BPE tokenizer's decode function.

'''What happens:'''
* Map each token ID back to its BPE token string
* Concatenate all token strings
* Decode unicode characters back to UTF-8 bytes
* Return final decoded text string

== Execution Diagram ==
{{#mermaid:graph TD
    A[Model Loading] --> B[Input Tokenization]
    B --> C[Transformer Forward Pass]
    C --> D[Auto-regressive Generation]
    D --> E[Output Decoding]
    D --> |next token| C
}}

== GitHub URL ==

[[github_url::PENDING_REPO_BUILD]]
