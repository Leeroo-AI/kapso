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
| [[domain::LLMs]], [[domain::Inference]], [[domain::Transformers]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==

End-to-end process for generating text from a prompt using GPT-2 implemented in pure NumPy.

=== Description ===

This workflow demonstrates autoregressive text generation using a minimal GPT-2 implementation. The process loads official OpenAI GPT-2 weights (converted from TensorFlow checkpoints to NumPy arrays), tokenizes the input prompt using Byte-Pair Encoding (BPE), performs forward passes through the transformer architecture, and generates tokens via greedy sampling. The entire inference pipeline runs without PyTorch or TensorFlow, relying solely on NumPy for computation.

The implementation supports four model sizes (124M, 355M, 774M, 1558M) and serves primarily as an educational resource for understanding transformer internals.

=== Usage ===

Execute this workflow when you want to:
* Generate text completions from a prompt using GPT-2
* Understand how transformer inference works at a low level
* Study the GPT-2 architecture without framework abstractions
* Learn about BPE tokenization and autoregressive generation

Prerequisites:
* Python 3.9+ environment
* TensorFlow (for loading checkpoint files only, not inference)
* Sufficient disk space for model weights (124M-1558M)

== Execution Steps ==

=== Step 1: Model Download ===

Check if model files exist locally, and if not, download them from OpenAI's public Azure blob storage. The model files include the TensorFlow checkpoint (weights), hyperparameters configuration, BPE vocabulary, and encoder mappings.

'''Required files:'''
* `checkpoint` - TensorFlow checkpoint pointer
* `model.ckpt.*` - Model weight files
* `hparams.json` - Model hyperparameters (n_layer, n_head, n_embd, etc.)
* `encoder.json` - BPE token-to-id mappings
* `vocab.bpe` - BPE merge rules

=== Step 2: Weight Conversion ===

Convert TensorFlow checkpoint variables to NumPy arrays. The loader reads variable names from the checkpoint, strips the "model/" prefix, and organizes weights into a nested dictionary structure matching the transformer architecture (embedding tables, layer blocks with attention and MLP weights, final layer norm).

'''Key transformations:'''
* Parse checkpoint variable names to identify layer indices
* Squeeze singleton dimensions from weight tensors
* Structure weights as: `{wte, wpe, blocks: [{ln_1, attn, ln_2, mlp}, ...], ln_f}`

=== Step 3: Tokenizer Initialization ===

Initialize the BPE tokenizer by loading the encoder vocabulary and merge rules. The tokenizer maps text to tokens using the same encoding scheme as the original GPT-2, handling Unicode through byte-to-unicode mappings.

'''Tokenizer components:'''
* Encoder dictionary mapping BPE tokens to integer IDs
* BPE merge rules defining the subword vocabulary
* Regex pattern for initial text splitting (handles contractions, numbers, whitespace)

=== Step 4: Input Encoding ===

Encode the user's text prompt into a sequence of token IDs. The BPE algorithm splits text into subword units based on learned merge rules, converting arbitrary text into the fixed vocabulary the model understands.

'''Process:'''
1. Split text into initial tokens using regex
2. Convert characters to byte-level unicode representation
3. Apply iterative BPE merging based on merge priorities
4. Map final BPE tokens to integer IDs

=== Step 5: Autoregressive Generation ===

Generate new tokens one at a time by repeatedly running the forward pass and selecting the next token. Each iteration appends the predicted token to the input sequence and re-runs the model with the extended context.

'''Generation loop:'''
1. Run GPT-2 forward pass on current input sequence
2. Extract logits for the last position
3. Select token with highest probability (greedy sampling)
4. Append selected token to input sequence
5. Repeat until desired number of tokens generated

=== Step 6: Forward Pass (GPT-2 Transformer) ===

Execute the core transformer computation: embed tokens with positional encodings, pass through N transformer blocks (each containing multi-head attention and feed-forward network with residual connections and layer normalization), then project to vocabulary logits.

'''Architecture details:'''
* Token embeddings + learned positional embeddings
* Pre-norm transformer blocks (LayerNorm before attention/FFN)
* Multi-head causal self-attention with scaled dot-product
* GELU-activated feed-forward network (4x expansion)
* Final layer norm and projection to vocabulary via embedding transpose

=== Step 7: Output Decoding ===

Convert the generated token IDs back into human-readable text. The decoder reverses the BPE encoding by mapping token IDs to their string representations and reconstructing the original character bytes.

'''Process:'''
1. Map each token ID to its BPE string
2. Concatenate all BPE strings
3. Convert byte-level unicode back to UTF-8 bytes
4. Decode bytes to final text string

== Execution Diagram ==

{{#mermaid:graph TD
    A[Model Download] --> B[Weight Conversion]
    B --> C[Tokenizer Initialization]
    C --> D[Input Encoding]
    D --> E[Autoregressive Generation]
    E --> F[Forward Pass]
    F --> G{More tokens?}
    G -->|Yes| E
    G -->|No| H[Output Decoding]
}}

== GitHub URL ==

[[github_url::https://github.com/leeroo-coder/workflow-jaymody-picogpt-text-generation]]
