# Environment Page Sections Guide

This document defines the schema, purpose, and detailed writing instructions for an **Environment** page. Every section is mandatory to ensuring the graph remains executable and reproducible.

---

## 1. Metadata Block (Top of Page)
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

## 4. Graph Connections

### `== Related Pages ==`
**Instruction:** List the incoming connections (backlinks) using semantic wiki links.

Environments are **Leaf Nodes** — they only receive connections. List which Implementations require this environment:

*   *Syntax:* `* [[requires_env::Implementation:{Implementation_Name}]]`
*   *Source Type:* `Implementation`
*   *Meaning:* "This Implementation requires this environment to run"

**Sample:**
```mediawiki
== Related Pages ==
* [[requires_env::Implementation:LightningTrainer]]
* [[requires_env::Implementation:HF_Accelerator]]
```

**Connection Types for Environment (Incoming Only):**
| Edge Property | Source Node | Meaning |
|:--------------|:------------|:--------|
| `requires_env` | Implementation | "This code needs this environment to run" |
