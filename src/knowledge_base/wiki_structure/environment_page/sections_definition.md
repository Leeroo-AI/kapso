# Environment Page Sections Guide

This document defines the schema, purpose, and detailed writing instructions for an **Environment** page. Every section is mandatory to ensuring the graph remains executable and reproducible.

---

## Page Title Requirements (WikiMedia Compliance)

### Naming Format
```
{repo_namespace}_{Environment_Name}.md
```

### WikiMedia Syntax Rules
1. **First character capitalized** — Auto-converted by system
2. **Underscores only** — Use `_` as word separator (NO hyphens, NO spaces)
3. **Case-sensitive after first character**

### Forbidden Characters
Never use: `#`, `<`, `>`, `[`, `]`, `{`, `}`, `|`, `+`, `:`, `/`, `-` (hyphen)

### Examples
| Correct | Incorrect | Issue |
|---------|-----------|-------|
| `Owner_Repo_CUDA_11_8_Requirements.md` | `Owner-Repo_CUDA-11.8.md` | Hyphens, dot |
| `Owner_Repo_Python_3_10_Environment.md` | `owner_repo_python_3_10_environment.md` | Lowercase |
| `Owner_Repo_Docker_GPU.md` | `Owner_Repo_Docker/GPU.md` | Slash |

---

## 0. Page Title (REQUIRED - First Line)

**Goal:** Provide a human-readable H1 title as the very first line of the page.

**Format:** `# Environment: {Page_Name}`

Where `{Page_Name}` is the page name WITHOUT the repo namespace prefix.

**Sample:**
```mediawiki
# Environment: GitHub_Actions_Runner
```

For a file named `Owner_Repo_GitHub_Actions_Runner.md`, the title is:
- ✅ `# Environment: GitHub_Actions_Runner` (correct - no repo prefix)
- ❌ `# Environment: Owner_Repo_GitHub_Actions_Runner` (wrong - includes repo prefix)

---

## 1. Metadata Block
**Goal:** Provide structured, machine-readable context for the graph parser and search index.
**Format:** Semantic MediaWiki Table (Right-aligned).

### Fields Explanation
1.  **Knowledge Sources:** The provenance of this definition.
    *   *Why:* establishes credibility and allows users to trace back to the original repo or paper.
    *   *Syntax:* `[[source::{Type}|{Title}|{URL}]]`
    *   *Types:* `Repo` (GitHub), `Doc` (Official Documentation), `Dockerfile` (Source Image), `Blog` (Tutorial).
2.  **Domains:** Categorization tags for filtering.
    *   *Why:* Allows queries like "Show me all Infrastructure environments".
    *   *Syntax:* `[[domain::{Tag}]]`
    *   *Examples:* `Infrastructure`, `NLP`, `Computer_Vision`, `Reinforcement_Learning`.
3.  **Last Updated:** Freshness marker.
    *   *Why:* Agents use this to decide if the environment definition needs a refresh.
    *   *Syntax:* `[[last_updated::{YYYY-MM-DD HH:MM GMT}]]`

**Sample:**
```mediawiki
{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PyTorch Lightning|https://github.com/Lightning-AI/lightning]]
* [[source::Doc|NVIDIA NGC|https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Deep_Learning]]
|-
! Last Updated
| [[last_updated::2023-10-27 14:00 GMT]]
|}
```

---

## 2. Overview Block (The "Card")
This section is the "Executive Summary". It is highly weighted in search embeddings.

### `== Overview ==`
**Instruction:** Write a single, concise sentence summary of the stack.
*   **Purpose:** The "Snippet" shown in search results.
*   **Do:** Mention OS, key Accelerator (CUDA/TPU), and primary Language/Library version.
*   **Don't:** Be vague ("A training environment") or overly verbose (full paragraph).

**Sample:**
```mediawiki
== Overview ==
Ubuntu 20.04 environment with CUDA 11.8, Python 3.9, and PyTorch 2.0+.
```

### `=== Description ===` (The "What")
**Instruction:** Detail the **Configuration State**.
*   **Purpose:** Describes the "Container". An agent reads this to determine compatibility.
*   **Content:** Explain the container/OS base, specific hardware optimizations (Ampere/Hopper), and the scope of the software stack.
*   **Edge Case:** If generic, state "Standard CPU-based Python environment".

**Sample:**
```mediawiki
=== Description ===
This environment provides a standard GPU-accelerated context for deep learning. It is built on top of the NVIDIA NGC base image and includes the full CUDA 11.8 toolkit, cuDNN 8.6, and a Python 3.9 runtime. It is optimized for Ampere (A100) and Hopper (H100) architectures.
```

### `=== Usage ===` (The "When")
**Instruction:** Define the **Dependency Trigger**.
*   **Purpose:** Describes the "Switch". An agent reads this to know *when* to activate this node.
*   **Content:** Specify the *condition* or *tasks* that require this environment.
*   **Goal:** Answer "Why should I switch to this context instead of the default?"

**Sample:**
```mediawiki
=== Usage ===
Use this environment for any **Model Training** or **Fine-Tuning** workflow that requires GPU acceleration. It is the mandatory prerequisite for running the `LightningTrainer` and `HF_Accelerator` implementations.
```

---

## 3. Technical Specifications
This section contains the hard constraints checked by deployment agents.

### `== System Requirements ==`
**Instruction:** define **Hard Constraints** in a table.
*   **Purpose:** Pre-flight checks before attempting to build the environment.
*   **Columns:** `Category`, `Requirement`, `Notes`.
*   **Rows:**
    *   `OS`: Distribution and Kernel (e.g., "Ubuntu 20.04").
    *   `Hardware`: GPU type/VRAM, CPU cores. Be specific (e.g., "NVIDIA A100 40GB").
    *   `Disk`: Storage type (SSD/HDD) and size.

**Sample:**
```mediawiki
== System Requirements ==
{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Ubuntu 20.04 LTS || Kernel 5.15+ recommended
|-
| Hardware || NVIDIA GPU || Minimum 16GB VRAM (A100 preferred)
|-
| Disk || 50GB SSD || High IOPS required for dataset caching
|}
```

### `== Dependencies ==`
**Instruction:** List all required software packages.
*   **Purpose:** The "Bill of Materials" for building the Docker image or Conda environment.
*   **System Packages:** OS-level libs (apt/brew). e.g., `cuda-toolkit`, `git-lfs`, `ffmpeg`.
*   **Python Packages:** Language-level libs (pip/conda). **Must** include major version constraints (`>=`).

**Sample:**
```mediawiki
== Dependencies ==
=== System Packages ===
* `cuda-toolkit` = 11.8
* `cudnn` = 8.6
* `git-lfs`
* `ffmpeg`

=== Python Packages ===
* `torch` >= 2.0.1
* `torchvision` >= 0.15.2
* `lightning` >= 2.0.0
* `transformers`
```

### `== Credentials ==`
**Instruction:** List required environment variables by **Name Only**.
*   **Purpose:** Notifies the user/agent of secrets that must be injected at runtime.
*   **Warning:** **NEVER** include actual secret values (tokens, keys, passwords).
*   **Content:** Variable Name + Description of purpose.

**Sample:**
```mediawiki
== Credentials ==
The following environment variables must be set in `.env`:
* `HF_TOKEN`: HuggingFace API token (Read access).
* `WANDB_API_KEY`: Weights & Biases API key for logging.
* `AWS_ACCESS_KEY_ID`: For S3 checkpoint storage.
```

---

## 4. Validation & Troubleshooting (NEW SECTIONS)

### `== Quick Install ==`
**Instruction:** Provide a **single copy-pasteable command** to install all dependencies.
*   **Purpose:** Allows engineers to set up the environment in one step.
*   **Content:** Combined pip install command with all required packages.
*   **Format:** Use `syntaxhighlight` with `lang="bash"`.

**Sample:**
```mediawiki
== Quick Install ==
<syntaxhighlight lang="bash">
# Install all required packages
pip install torch>=2.4.0 transformers>=4.37 bitsandbytes>=0.43.3 peft>=0.10.0 trl accelerate

# For GGUF export (optional)
pip install sentencepiece psutil
</syntaxhighlight>
```

### `== Code Evidence ==`
**Instruction:** Show **actual code snippets** from the repository that validate requirements.
*   **Purpose:** Proves these requirements come from the source code, not assumptions.
*   **Content:** Include:
    1. **Version checks** - Code that validates minimum versions
    2. **Detection logic** - Code that auto-detects hardware/configuration
    3. **Error handling** - Code that raises errors when requirements not met
*   **Format:** Include file path and line numbers for each snippet.

**Sample:**
```mediawiki
== Code Evidence ==

Version validation from `loader.py:55-62`:
<syntaxhighlight lang="python">
SUPPORTS_FOURBIT = transformers_version >= Version("4.37")
if not SUPPORTS_FOURBIT:
    raise ImportError(
        "Unsloth requires transformers >= 4.37 for 4-bit loading support."
    )
</syntaxhighlight>

Hardware detection from `device_type.py:37-45`:
<syntaxhighlight lang="python">
if hasattr(torch, "cuda") and torch.cuda.is_available():
    return "cuda"
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    return "xpu"
raise NotImplementedError("Unsloth requires NVIDIA, AMD, or Intel GPU.")
</syntaxhighlight>
```

### `== Common Errors ==`
**Instruction:** Document **error messages** and their solutions.
*   **Purpose:** Helps engineers self-diagnose installation issues.
*   **Content:** Table with columns: `Error Message`, `Cause`, `Solution`.
*   **Goal:** Reduce support burden by documenting known failure modes.

**Sample:**
```mediawiki
== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `ImportError: vLLM not found` || vLLM not installed || `pip install vllm`
|-
|| `CUDA out of memory` || Insufficient VRAM || Reduce `gpu_memory_utilization` to 0.5
|-
|| `Model is in float16 but you want bfloat16` || Dtype mismatch || Set `fp16=True, bf16=False` in config
|}
```

### `== Compatibility Notes ==`
**Instruction:** Document **platform-specific limitations or differences**.
*   **Purpose:** Warns engineers about edge cases before they encounter them.
*   **Content:** Any limitations for:
    - Different GPU vendors (NVIDIA vs AMD vs Intel)
    - Different operating systems (Linux vs Windows vs Mac)
    - Different model architectures (some models need special handling)
*   **Format:** Bullet list or table.

**Sample:**
```mediawiki
== Compatibility Notes ==

* '''AMD GPUs (ROCm):''' Requires bitsandbytes >= 0.48.3. Pre-quantized models may not work due to blocksize differences (AMD uses 128 vs NVIDIA's 64).
* '''Intel XPU:''' Requires PyTorch >= 2.6.0.
* '''Windows:''' Not officially supported; use WSL2.
* '''Colab/Kaggle:''' Auto-frees cached models to manage limited disk space.
```

---

## 5. Graph Connections

### `== Related Pages ==`
**Instruction:** Document which Implementation pages require this environment using semantic backlinks.

Environments are **Leaf Nodes** — they receive incoming connections from Implementation pages. Use the `required_by` edge type to document these references.

**Sample:**
```mediawiki
== Related Pages ==
* [[required_by::Implementation:Owner_Repo_Awesome_Lint_Action_Execution]]
* [[required_by::Implementation:Owner_Repo_Issue_To_PR_Conversion]]
* [[required_by::Implementation:Owner_Repo_GitHub_Actions_Cron_Schedule]]
* [[required_by::Implementation:Owner_Repo_Git_Config_Add_Commit_Push]]
```

**Edge Type for Environment Backlinks:**

| Edge Property | Meaning | Source Page Types |
|:--------------|:--------|:------------------|
| `required_by` | "This environment is required by X" | Implementation |

**Note:** Do NOT use `[[requires_env::...]]` on Environment pages — that edge type belongs on Implementation pages pointing TO this environment.

---

## 6. Extraction Guidance for Phase 3 (Enrichment)

When creating Environment pages during the Enrichment phase, **search the source code for these patterns**:

### What to Extract from Source Code

| Pattern to Search | What It Reveals | Section to Fill |
|:------------------|:----------------|:----------------|
| `Version(...)` comparisons | Minimum package versions | Dependencies, Code Evidence |
| `ImportError`, `raise RuntimeError` | Error messages when requirements not met | Common Errors |
| `is_available()`, `find_spec()` | Auto-detection logic | Code Evidence |
| `os.environ.get()` | Required environment variables | Credentials |
| `if DEVICE_TYPE == "hip"` | Platform-specific code paths | Compatibility Notes |
| `pip install` in docstrings/comments | Install commands | Quick Install |
| `blocksize`, `dtype` conditionals | Hardware-specific limitations | Compatibility Notes |

### Extraction Checklist

Before marking an Environment page as complete, verify:

- [ ] **Dependencies**: All packages with version constraints extracted from import checks
- [ ] **Quick Install**: Combined pip command covering all packages
- [ ] **Code Evidence**: At least 2 code snippets showing version/hardware validation
- [ ] **Common Errors**: At least 2 error messages with solutions documented
- [ ] **Compatibility Notes**: Any if-else branches for different platforms documented

### Search Patterns (grep/regex)

```bash
# Find version checks
grep -rn "Version\(" --include="*.py"

# Find import errors
grep -rn "raise ImportError\|raise RuntimeError" --include="*.py"

# Find environment variable usage
grep -rn "os.environ.get\|os.environ\[" --include="*.py"

# Find platform conditionals
grep -rn "DEVICE_TYPE\|is_cuda\|is_hip\|is_xpu" --include="*.py"

# Find minimum version comments
grep -rn "requires\|minimum\|>=" --include="*.py"
```
