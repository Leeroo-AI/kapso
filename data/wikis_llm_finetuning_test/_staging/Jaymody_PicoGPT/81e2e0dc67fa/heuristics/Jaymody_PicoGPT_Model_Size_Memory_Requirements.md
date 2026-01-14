# Heuristic: Model_Size_Memory_Requirements

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|picoGPT|https://github.com/jaymody/picoGPT]]
* [[source::Doc|OpenAI GPT-2|https://openai.com/blog/better-language-models/]]
|-
! Domains
| [[domain::NLP]], [[domain::Infrastructure]], [[domain::Memory]]
|-
! Last Updated
| [[last_updated::2026-01-14 10:00 GMT]]
|}

== Overview ==
GPT-2 model sizes range from 124M to 1558M parameters, requiring approximately 500MB to 6GB of RAM for inference.

=== Description ===
PicoGPT supports four GPT-2 model sizes: 124M, 355M, 774M, and 1558M parameters. Each size requires proportionally more memory for weights and activations. Since PicoGPT uses NumPy with float32 precision, memory usage is straightforward to estimate: roughly 4 bytes per parameter plus activation memory during forward pass.

=== Usage ===
Use this heuristic when:
- Choosing which model size to download
- Debugging memory errors or slow performance
- Running on resource-constrained systems
- Planning disk space for model storage

== The Insight (Rule of Thumb) ==
* **Action:** Choose model size based on available system RAM
* **Values:**
  - 124M: ~500MB RAM, ~500MB disk
  - 355M: ~1.5GB RAM, ~1.4GB disk
  - 774M: ~3GB RAM, ~3GB disk
  - 1558M: ~6GB RAM, ~6GB disk
* **Trade-off:**
  - Larger models produce higher quality text
  - Larger models require more memory and run slower
  - 124M is sufficient for educational purposes and testing

== Reasoning ==
Memory estimation:
1. **Parameters:** Model params * 4 bytes (float32) = base memory
2. **Activations:** Roughly 2x params for forward pass intermediate tensors
3. **Disk:** TensorFlow checkpoint is approximately equal to param count in bytes

The model sizes are validated at load time:
```python
assert model_size in ["124M", "355M", "774M", "1558M"]
```

Practical guidance:
- **Laptops (8GB RAM):** Use 124M or 355M
- **Workstations (16GB RAM):** All sizes work
- **Minimal systems (4GB RAM):** 124M only

== Code Evidence ==

Model size validation from `utils.py:14`:
<syntaxhighlight lang="python">
assert model_size in ["124M", "355M", "774M", "1558M"]
</syntaxhighlight>

Model files downloaded from `utils.py:15-23`:
<syntaxhighlight lang="python">
for filename in [
    "checkpoint",
    "encoder.json",
    "hparams.json",
    "model.ckpt.data-00000-of-00001",
    "model.ckpt.index",
    "model.ckpt.meta",
    "vocab.bpe",
]:
</syntaxhighlight>

Default model size in CLI from `gpt2.py:97`:
<syntaxhighlight lang="python">
def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
</syntaxhighlight>

== Model Architecture Details ==

{| class="wikitable"
|-
! Size !! Parameters !! Layers !! Heads !! Embedding Dim !! Context
|-
| 124M || 124,439,808 || 12 || 12 || 768 || 1024
|-
| 355M || 354,823,168 || 24 || 16 || 1024 || 1024
|-
| 774M || 774,030,080 || 36 || 20 || 1280 || 1024
|-
| 1558M || 1,557,611,200 || 48 || 25 || 1600 || 1024
|}

== Related Pages ==
* [[used_by::Implementation:Jaymody_PicoGPT_Load_Encoder_Hparams_And_Params]]
* [[used_by::Principle:Jaymody_PicoGPT_Model_Loading]]
