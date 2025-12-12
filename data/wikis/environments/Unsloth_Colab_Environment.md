{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth GitHub|https://github.com/unslothai/unsloth]]
* [[source::Doc|Google Colab|https://colab.research.google.com/]]
* [[source::Blog|Unsloth Colab Notebooks|https://docs.unsloth.ai/get-started/unsloth-notebooks]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Cloud]], [[domain::LLMs]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Google Colab notebook environment with T4/A100 GPU runtime pre-configured for Unsloth fine-tuning workflows.

=== Description ===
This environment leverages Google Colab's free and Pro GPU runtimes for running Unsloth fine-tuning workflows. Colab provides accessible GPU compute (T4 free tier, A100 for Pro users) without local hardware requirements. Unsloth's memory optimizations are particularly valuable here, enabling fine-tuning of 7B+ models on the free T4 GPU (16GB VRAM) that would otherwise be impossible with standard training methods.

=== Usage ===
Use this environment when you want to **quickly prototype** fine-tuning workflows, **lack local GPU hardware**, or need to **share reproducible notebooks**. Ideal for educational purposes, experimentation, and small-scale fine-tuning jobs. The free T4 tier can fine-tune models up to 7B parameters with Unsloth's optimizations.

== System Requirements ==
{| class="wikitable"
! Category !! Requirement !! Notes
|-
|| Runtime || GPU Runtime || Must select GPU in Runtime > Change runtime type
|-
|| GPU Type || T4 (free) / A100 (Pro) || T4: 16GB VRAM, A100: 40GB VRAM
|-
|| RAM || 12-52GB || Depends on Colab tier (free/Pro/Pro+)
|-
|| Disk || ~100GB || Colab provides temporary storage
|-
|| Session || Up to 12 hours || Free tier disconnects after idle/usage limits
|}

== Dependencies ==
=== Installation Commands ===
Run these cells at the start of your Colab notebook:

<syntaxhighlight lang="python">
# Install Unsloth (run first)
%%capture
!pip install unsloth
# Also get the latest nightly Unsloth for bleeding edge updates
!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
</syntaxhighlight>

=== Python Packages (Auto-installed) ===
* `unsloth` (latest)
* `torch` (Colab pre-installed)
* `transformers`
* `trl`
* `peft`
* `bitsandbytes`
* `accelerate`
* `datasets`

== Credentials ==
The following may be required for gated models:
* `HF_TOKEN`: Set via `huggingface_hub.login()` or Colab Secrets
* `WANDB_API_KEY`: For experiment tracking (optional)

== Related Pages ==
=== Required By ===
* [[required_by::Implementation:Unsloth_FastLanguageModel]]
* [[required_by::Implementation:Unsloth_FastModel]]
* [[required_by::Implementation:TRL_SFTTrainer]]
* [[required_by::Implementation:Unsloth_get_peft_model]]

