# Enrichment Phase: Mine Environment Constraints and Heuristics

You are a knowledge extraction agent. Your task is to scan the implementation code for Environment constraints and Heuristics (tribal knowledge), then create wiki pages for them.

## ⚠️ FILE PLACEMENT RULES (CRITICAL)

**Only create files in these directories:**
- `{wiki_dir}/environments/` - Environment pages
- `{wiki_dir}/heuristics/` - Heuristic pages
- `{wiki_dir}/_reports/` - Execution reports

**DO NOT create:**
- Summary files at the root of `{wiki_dir}`
- Documentation files outside the designated directories
- Any file that doesn't follow the `{repo_name}_Page_Name.md` naming convention
- "Notes", "summaries", or "completion reports" outside `_reports/`

## 📛 PAGE NAMING RULES (WikiMedia Compliance)

All page names must follow WikiMedia technical syntax:

### Syntax Rules
1. **First character capitalized** — System auto-converts (e.g., `cuda_11` → `Cuda_11`)
2. **Underscores only** — Use `_` as the sole word separator (NO hyphens, NO spaces)
3. **Case-sensitive after first character** — `CUDA_11` ≠ `Cuda_11`

### Forbidden Characters (NEVER use in page names)

| Character | Name | Why Forbidden |
|-----------|------|---------------|
| `#` | Hash | Section anchors |
| `< >` | Angle brackets | HTML tags |
| `[ ]` | Square brackets | Wiki links |
| `{{ }}` | Curly brackets | Templates |
| `\|` | Pipe | Link separators |
| `+` | Plus | URL encoding |
| `:` | Colon | Namespaces |
| `/` | Slash | Subpages |
| `-` | Hyphen | Use underscore instead |

### Naming Examples

```
✅ CORRECT:
   {repo_name}_CUDA_11_Requirements
   {repo_name}_Batch_Size_Tips
   {repo_name}_Gradient_Checkpointing

❌ WRONG:
   {repo_name}_CUDA-11              (hyphen)
   {repo_name}_batch_size_tips      (lowercase after prefix)
   {repo_name}_GPU/Memory           (slash)
```

## Context

- Repository: {repo_name}
- Repository Path: {repo_path}
- Wiki Output Directory: {wiki_dir}
- **Repository Map (Index):** {repo_map_path}
- **File Details:** {wiki_dir}/_files/
- Existing Pages:
  - Workflows: {wiki_dir}/workflows/
  - Principles: {wiki_dir}/principles/
  - Implementations: {wiki_dir}/implementations/

## IMPORTANT: Read Previous Phase Reports

**FIRST**, read the previous phase reports:
- `{wiki_dir}/_reports/phase2_excavation_synthesis.md` - Implementation-Principle pairs created, hints for environments/heuristics

This report tells you which files may have environment requirements or tribal knowledge.

## IMPORTANT: Use the Repository Map

**THEN**, read the Repository Map index at `{repo_map_path}`.

The index contains:
- **Purpose column:** Hints about which files might have env requirements
- **Coverage column:** Which files are already documented
- Import lists in file details reveal dependencies

For files likely to have environment requirements, read their detail pages in `_files/`.

## IMPORTANT: Check the Page Indexes

**Also read:**
- `{wiki_dir}/_ImplementationIndex.md` — See `⬜Env:{repo_name}_X` references (Environments needed)

**How to read the Connections column:**
- `✅Env:{repo_name}_CUDA_11` = Environment page EXISTS (don't create duplicate)
- `⬜Env:{repo_name}_Triton` = Environment page MISSING (you should create it)
- `⬜Heuristic:{repo_name}_BatchSize` = Heuristic page MISSING (you should create it)

**Your job:** Create pages for all `⬜Env:{repo_name}_X` and `⬜Heuristic:{repo_name}_X` references you see.

## Wiki Structure Definitions

### Environment Structure
{environment_structure}

### Heuristic Structure
{heuristic_structure}

---

## Part A: Mine Environment Constraints

### What to Look For

Using the Repository Map, identify files likely to have requirements:
- Files with Purpose mentioning "CUDA", "GPU", "kernel"
- Files importing `triton`, `flash_attn`, `bitsandbytes`

Then scan those files for:

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

---

## Part B: Mine Heuristics (Tribal Knowledge)

### What to Look For

Scan code (prioritize files with high Coverage) for:

1. **Comments:**
   - `# TODO`, `# NOTE`, `# HACK`, `# WARNING`, `# IMPORTANT`
   - Explanatory comments about why something is done a certain way

2. **Warnings:**
   - `warnings.warn()` calls
   - `logger.warning()` messages

3. **Conditional optimizations:**
   - `if batch_size > 16:`
   - `if use_flash_attention:`

4. **README tips:**
   - "Best practices" sections
   - "Tips" or "Troubleshooting" sections

### Write Heuristic Pages

For each piece of wisdom, create a page with:
- The insight/rule (action, value, trade-off)
- Reasoning (why it works)
- Code evidence

**Output:** Write to `{wiki_dir}/heuristics/`

**Filename format:** `{repo_name}_HeuristicName.md`

---

## Part C: Add Links to Existing Pages (CRITICAL)

After creating Environment and Heuristic pages, UPDATE existing pages to add links.

### Link Naming Rule

The link target must EXACTLY match the filename (without .md extension).

### Update Implementation Pages
Add `[[requires_env::Environment:{repo_name}_X]]` links.

### Update Implementation and Principle Pages
Add `[[uses_heuristic::Heuristic:{repo_name}_X]]` links.

**Note:** Heuristics are practical tips that belong at the Implementation and Principle level. **Do NOT add heuristic links to Workflow pages** — Workflows no longer have wiki link connections.

### Verification Step

Before adding a link, verify the target page exists in the directory.

---

## Part D: Update Coverage in Repository Map

After creating Environment and Heuristic pages, **update the index** at `{repo_map_path}`:

```markdown
| ✅ | `unsloth/kernels/utils.py` | 150 | Kernel utilities | Impl: kernel_utils; Env: CUDA_Environment | [→](...) |
```

For files where you found heuristics:
```markdown
| ✅ | `unsloth/trainer.py` | 300 | Training wrapper | Impl: UnslothTrainer; Heur: Batch_Size_Tips | [→](...) |
```

## Part E: Update the Environment and Heuristic Indexes (IMMEDIATELY)

**⚠️ CRITICAL:** Update indexes **IMMEDIATELY after creating each page**.

### Update Environment Index

After creating Environment pages, add entries to `{wiki_dir}/_EnvironmentIndex.md`:

| Column | Content |
|--------|---------|
| Page | Environment page name (without .md) |
| File | Link: `[→](./environments/{repo_name}_X.md)` |
| Connections | All links with **per-reference status** |
| Notes | Brief description of the requirement |

**Connections Format (use FULL page names with `{repo_name}_` prefix):**
- `✅Impl:{repo_name}_FastLanguageModel` = Implementation exists
- `⬜Impl:{repo_name}_triton_kernels` = Implementation not created yet

**Example row:**
```
| {repo_name}_CUDA_11 | [→](./environments/...) | ✅Impl:{repo_name}_FastLanguageModel, ⬜Impl:{repo_name}_rope_kernel | GPU with CUDA 11.8+ |
```

### Update Heuristic Index

After creating Heuristic pages, add entries to `{wiki_dir}/_HeuristicIndex.md`:

| Column | Content |
|--------|---------|
| Page | Heuristic page name (without .md) |
| File | Link: `[→](./heuristics/{repo_name}_X.md)` |
| Connections | All links with **per-reference status** |
| Notes | Brief summary of the tip/advice |

**Connections Format (use FULL page names with `{repo_name}_` prefix):**
- `✅Impl:{repo_name}_UnslothTrainer` = Implementation exists
- `✅Principle:{repo_name}_Gradient_Optimization` = Principle exists

**Example row:**
```
| {repo_name}_Batch_Size_Tips | [→](./heuristics/...) | ✅Impl:{repo_name}_UnslothTrainer, ✅Principle:{repo_name}_Memory_Optimization | Use batch_size=1 with grad accum |
```

### Update Other Indexes (Bi-directional)

When you create an Environment or Heuristic, update references in OTHER indexes:

**For Environments (e.g., `{repo_name}_CUDA_11`):**
1. Search `_ImplementationIndex.md` for `⬜Env:{repo_name}_CUDA_11`
2. Change to `✅Env:{repo_name}_CUDA_11`

**For Heuristics (e.g., `{repo_name}_Batch_Size_Tips`):**
1. Search `_ImplementationIndex.md` and `_PrincipleIndex.md` for `⬜Heuristic:{repo_name}_Batch_Size_Tips`
2. Change to `✅Heuristic:{repo_name}_Batch_Size_Tips`

## ⚠️ Leaf Node Rule (Environment & Heuristic Pages)

**Environment and Heuristic pages are LEAF NODES** — they receive incoming connections only.

When creating the Related Pages section on these pages, use **backlink edge types**:

**⚠️ IMPORTANT:** Only add backlinks for pages that ACTUALLY have a forward link pointing to this leaf page. Do NOT add backlinks speculatively.

**For Heuristic pages** (use `used_by`):
```mediawiki
== Related Pages ==
* [[used_by::Implementation:{repo_name}_Git_Fork_Edit_Workflow]]
* [[used_by::Principle:{repo_name}_Memory_Optimization]]
```

**For Environment pages** (use `required_by`):
```mediawiki
== Related Pages ==
* [[required_by::Implementation:{repo_name}_Awesome_Lint_Action_Execution]]
* [[required_by::Implementation:{repo_name}_GitHub_Actions_Cron_Schedule]]
```

**DO NOT** use outgoing edge types (`[[uses_heuristic::...]]` or `[[requires_env::...]]`) on leaf pages — those belong on the source pages (Implementation, Principle) pointing TO the leaf pages.

---

## Repo Scoping Rule (CRITICAL)

Only create/update pages whose filenames start with `{repo_name}_`.

## ⚠️ File Editing Tip

When updating index files:
- **Use Write tool** (read entire file → modify → write back)
- **Avoid Edit tool** — it often fails on markdown tables

## 📝 Execution Report (REQUIRED)

When finished, write a summary report to `{wiki_dir}/_reports/phase3_enrichment.md`:

```markdown
# Phase 3: Enrichment Report

## Environments Created
| Environment | Required By |
|-------------|-------------|
| [Name] | [implementations] |

## Heuristics Created
| Heuristic | Applies To |
|-----------|------------|
| [Name] | [pages] |

## Links Added
- Environment links added: X
- Heuristic links added: X

## Notes for Audit Phase
- [Any potential broken links]
- [Pages that may need review]
```
