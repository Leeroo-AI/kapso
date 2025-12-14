# Enrichment Phase: Mine Environment Constraints and Heuristics

You are a knowledge extraction agent. Your task is to scan the implementation code for Environment constraints and Heuristics (tribal knowledge), then create wiki pages for them.

## Context

- Repository: {repo_name}
- Repository Path: {repo_path}
- Wiki Output Directory: {wiki_dir}
- Existing Pages:
  - Workflows: {wiki_dir}/workflows/
  - Principles: {wiki_dir}/principles/
  - Implementations: {wiki_dir}/implementations/

## Wiki Structure Definitions

### Environment Structure
{environment_structure}

### Heuristic Structure
{heuristic_structure}

---

## Part A: Mine Environment Constraints

### What to Look For

Scan the source code for hardware/software requirements:

1. **Hardware checks:**
   - `torch.cuda.is_available()`
   - GPU memory checks
   - CPU/GPU device selection

2. **Software dependencies:**
   - `import triton`, `import flash_attn` (optional deps)
   - Version checks: `assert torch.__version__ >= "2.0"`
   - `try/except ImportError` blocks

3. **Configuration files:**
   - `requirements.txt`
   - `setup.py` or `pyproject.toml`
   - Dockerfile contents

4. **Environment variables:**
   - `os.environ.get("HF_TOKEN")`
   - API keys, credentials

### Write Environment Pages

For each environment requirement, create a page with:
- System requirements (OS, hardware, disk)
- Python/system package dependencies with versions
- Required credentials (names only, never actual values!)
- Code evidence (quote the actual check)

**Output:** Write to `{wiki_dir}/environments/`

**Filename format:** `{repo_name}_EnvironmentName.md`
- Examples: `unsloth_CUDA_Environment.md`, `unsloth_Colab_Environment.md`

---

## Part B: Mine Heuristics (Tribal Knowledge)

### What to Look For

Scan the code for tips, optimizations, and best practices:

1. **Comments:**
   - `# TODO`, `# NOTE`, `# HACK`, `# WARNING`, `# IMPORTANT`
   - Explanatory comments about why something is done a certain way

2. **Warnings:**
   - `warnings.warn()` calls
   - `logger.warning()` messages

3. **Conditional optimizations:**
   - `if batch_size > 16:`
   - `if use_flash_attention:`
   - Performance-related if/else blocks

4. **Config defaults with explanations:**
   - Default hyperparameter values with comments explaining why
   - Recommended settings in docstrings

5. **README tips:**
   - "Best practices" sections
   - "Tips" or "Troubleshooting" sections

### Write Heuristic Pages

For each piece of wisdom, create a page with:
- The insight/rule (action, value, trade-off)
- Reasoning (why it works)
- Code evidence (the comment/condition that revealed this)

**Output:** Write to `{wiki_dir}/heuristics/`

**Filename format:** `HeuristicName.md`
- Examples: `Gradient_Checkpointing_Optimization.md`, `Learning_Rate_Tuning.md`

---

## Part C: Add Links to Existing Pages

After creating Environment and Heuristic pages, UPDATE the existing pages:

### Update Implementation Pages
Add `[[requires_env::Environment:X]]` links to relevant implementations.

### Update Principle/Implementation/Workflow Pages
Add `[[uses_heuristic::Heuristic:X]]` links where the heuristic applies.

**Example:**
If you found a heuristic about gradient checkpointing, add:
```
* [[uses_heuristic::Heuristic:Gradient_Checkpointing_Optimization]]
```
to the relevant Principle and Implementation pages.

