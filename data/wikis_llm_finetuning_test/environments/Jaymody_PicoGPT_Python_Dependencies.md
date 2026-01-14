# Environment: Python_Dependencies

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|picoGPT|https://github.com/jaymody/picoGPT]]
* [[source::Doc|requirements.txt|https://github.com/jaymody/picoGPT/blob/main/requirements.txt]]
|-
! Domains
| [[domain::NLP]], [[domain::Deep_Learning]], [[domain::Educational]]
|-
! Last Updated
| [[last_updated::2026-01-14 10:00 GMT]]
|}

== Overview ==
Python 3.9+ environment with NumPy, TensorFlow, and BPE tokenization dependencies for running GPT-2 inference.

=== Description ===
This environment provides a CPU-based Python runtime for running GPT-2 inference using pure NumPy operations. TensorFlow is required only for loading pre-trained weights from OpenAI's TensorFlow checkpoint format. The environment supports both standard x86 systems and Apple Silicon (M1/M2) Macs with appropriate TensorFlow variants.

=== Usage ===
Use this environment for running PicoGPT's text generation workflow. It is required for:
- Loading GPT-2 model weights from TensorFlow checkpoints
- BPE tokenization of input text
- NumPy-based transformer forward pass
- Autoregressive text generation

== System Requirements ==
{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux, macOS, Windows || Cross-platform Python environment
|-
| Python || 3.9+ || Tested on Python 3.9.10
|-
| RAM || 2-8 GB || Depends on model size: 124M (~500MB), 1558M (~6GB)
|-
| Disk || 1-6 GB || For model weights download and storage
|-
| Network || Internet access || Required for initial model download from OpenAI
|}

== Dependencies ==
=== System Packages ===
* No special system packages required (pure Python)

=== Python Packages ===
* `numpy` == 1.24.1 — Core tensor operations, model weights, forward pass
* `regex` == 2017.4.5 — Unicode-aware BPE tokenizer patterns
* `requests` == 2.27.1 — Download GPT-2 files from OpenAI blob storage
* `tqdm` == 4.64.0 — Progress bars for downloads and generation
* `fire` == 0.5.0 — CLI argument parsing
* `tensorflow` == 2.11.0 — TF checkpoint loading (x86 systems)
* `tensorflow-macos` == 2.11.0 — TF checkpoint loading (Apple Silicon)

== Credentials ==
No credentials or API keys are required. Model weights are downloaded from OpenAI's public blob storage without authentication.

== Quick Install ==
<syntaxhighlight lang="bash">
# Standard installation
pip install numpy==1.24.1 regex==2017.4.5 requests==2.27.1 tqdm==4.64.0 fire==0.5.0 tensorflow==2.11.0

# Apple Silicon (M1/M2 Macs)
pip install numpy==1.24.1 regex==2017.4.5 requests==2.27.1 tqdm==4.64.0 fire==0.5.0 tensorflow-macos==2.11.0

# Or install from requirements.txt
pip install -r requirements.txt
</syntaxhighlight>

== Code Evidence ==

Model size validation from `utils.py:14` and `utils.py:69`:
<syntaxhighlight lang="python">
assert model_size in ["124M", "355M", "774M", "1558M"]
</syntaxhighlight>

Context length validation from `gpt2.py:107`:
<syntaxhighlight lang="python">
# make sure we are not surpassing the max sequence length of our model
assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]
</syntaxhighlight>

TensorFlow checkpoint loading from `utils.py:54-55`:
<syntaxhighlight lang="python">
for name, _ in tf.train.list_variables(tf_ckpt_path):
    array = np.squeeze(tf.train.load_variable(tf_ckpt_path, name))
</syntaxhighlight>

BPE tokenizer regex pattern from `encoder.py:58`:
<syntaxhighlight lang="python">
self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `AssertionError` on model_size || Invalid model size specified || Use one of: "124M", "355M", "774M", "1558M"
|-
|| `AssertionError` on input length || Input + generation length exceeds n_ctx || Reduce input length or n_tokens_to_generate (max context is 1024)
|-
|| `ModuleNotFoundError: No module named 'tensorflow'` || TensorFlow not installed || Run `pip install tensorflow` or `tensorflow-macos` for Apple Silicon
|-
|| `ModuleNotFoundError: No module named 'regex'` || Using built-in `re` instead of `regex` || Run `pip install regex` (needed for Unicode patterns)
|-
|| Connection error during download || Network issues || Ensure internet connectivity to openaipublic.blob.core.windows.net
|}

== Compatibility Notes ==

* '''Apple Silicon (M1/M2):''' Requires `tensorflow-macos` instead of standard `tensorflow`. The requirements.txt handles this automatically with platform markers.
* '''GPU Acceleration:''' Not supported. PicoGPT uses pure NumPy for educational clarity. For GPU acceleration, use PyTorch or TensorFlow implementations.
* '''Windows:''' Fully supported with standard TensorFlow installation.
* '''Memory Usage:''' The 1558M model requires ~6GB RAM. Smaller models (124M) work on systems with 2GB RAM.

== Related Pages ==
* [[required_by::Implementation:Jaymody_PicoGPT_Load_Encoder_Hparams_And_Params]]
* [[required_by::Implementation:Jaymody_PicoGPT_Encoder_Encode]]
* [[required_by::Implementation:Jaymody_PicoGPT_Gpt2]]
* [[required_by::Implementation:Jaymody_PicoGPT_Generate]]
* [[required_by::Implementation:Jaymody_PicoGPT_Encoder_Decode]]
