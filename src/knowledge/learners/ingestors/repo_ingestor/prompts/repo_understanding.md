# Phase 0: Repository Understanding

You are a Code Analyst. Your task is to understand this repository and fill in the Understanding sections for **EVERY file**.

## ⚠️ FILE PLACEMENT RULES (CRITICAL)

**Only modify/create files in these locations:**
- `{wiki_dir}/_files/` - Per-file detail pages
- `{wiki_dir}/_RepoMap_{repo_name}.md` - Repository index
- `{wiki_dir}/_reports/` - Execution reports

**DO NOT create:**
- Summary files at the root of `{wiki_dir}`
- Documentation files outside the designated directories
- "Notes", "summaries", or "completion reports" outside `_reports/`

## ⛔ CRITICAL: COMPLETION CRITERIA

**This task is NOT complete until ALL files are explored.**

```
❌ WRONG: "Explored 31/116 files. Task complete."
✅ RIGHT: "Explored 116/116 files. Task complete."
```

**You MUST continue until the index shows `Explored | X/X |` where both numbers match.**
Do NOT stop early. Do NOT provide a summary until 100% complete.

---

## Context

- Repository: {repo_name}
- Repository Path: {repo_path}
- Repository Map (Index): {repo_map_path}
- File Details Directory: {wiki_dir}/_files/

## File Structure

The repository understanding is split into:

1. **Index file** (`_RepoMap_{repo_name}.md`): Compact table with columns:
   - **Status**: ⬜ pending → ✅ explored
   - **Purpose**: Brief 3-5 word description (you fill this)
   - **Coverage**: Which wiki pages cover this file (filled by later phases)
   - **Details**: Link to per-file detail page

2. **Detail files** (`_files/*.md`): One file per Python file with full Understanding

## Your Task

### Step 1: Read the Index

Read `{repo_map_path}` to see the list of files organized by category:
- 📦 Package files (core library code) - **prioritize these**
- 📝 Example files (usage patterns) - **important for workflows**
- 🧪 Test files (brief descriptions are fine)

### Step 2: Explore Files in Batches (with Index Sync)

You may explore files in batches for efficiency. **But you MUST sync the index after every 5-10 files.**

#### Batch Workflow:

1. **Read batch** (5-10 source files + their detail files)
2. **Write detail files** with Understanding filled in
3. **SYNC INDEX** — Update all rows for files you just explored

**Example batch:**
```
# Batch 1: Read 5 files
Read: loader.py, __init__.py, save.py, trainer.py, _utils.py
Read: their detail files

# Write 5 detail files
Write: loader_py.md, __init___py.md, save_py.md, trainer_py.md, _utils_py.md

# SYNC INDEX — update 5 rows at once
Write: _RepoMap_xxx.md (with 5 rows changed from ⬜ to ✅ and Purpose filled)

# Batch 2: next 5-10 files...
```

#### Index Row Format:
```markdown
# BEFORE:
| ⬜ | `unsloth/models/loader.py` | 320 | — | — | [→](...) |

# AFTER:
| ✅ | `unsloth/models/loader.py` | 320 | Main model loader | — | [→](...) |
```

**Good Purpose values (3-5 words):**
- "Main model loader"
- "Triton attention kernel"  
- "LoRA weight patching"
- "QLoRA training example"

**Leave Coverage as `—`** — later phases fill this.

⚠️ **CRITICAL: Do NOT leave index sync until the end!** Sync after every batch of 5-10 files.

### Step 3: Continue Until 100% Complete

**Keep processing batches until ALL files are explored.**

Check your progress:
- Read the index header: `| Explored | X/Y |`
- If X < Y, you are NOT done — continue with more batches
- Only stop when X = Y (e.g., `| Explored | 116/116 |`)

## Priority Order (process ALL, but start with these)

1. **Core entry points first:** `__init__.py`, main loaders, primary APIs
2. **Example files:** These show intended usage patterns
3. **Kernel/utility files:** Supporting code
4. **Test files:** Just note what they test (brief)

⚠️ **All categories MUST be explored** — priority only affects order, not whether to include.

## Quality Bar

**Good Detail File:**
```markdown
**Status:** ✅ Explored

**Purpose:** Main model loader that wraps HuggingFace models with memory optimizations.

**Mechanism:** Intercepts model loading to apply 4-bit quantization via bitsandbytes, patches attention layers with custom Triton kernels.

**Significance:** Core entry point - this is what users import. All examples use this.
```

**Good Index Row:**
```
| ✅ | `unsloth/models/loader.py` | 320 | Main model loader | — | [→](...) |
```

## Efficiency Tips (for completing ALL files)

- **Use Task tool** to parallelize: spawn sub-agents for different directories
- Read files in batches (5-10 at a time) to build context
- Use the AST info in detail files (Classes, Functions) to know what to look for
- For test files, brief is OK: "Tests for X functionality"
- For similar model files (e.g., `gemma.py`, `mistral.py`), copy patterns: "Gemma model patching (similar to Llama)"
- **Don't overthink** — brief descriptions are fine, just cover ALL files

## Output

1. Edit ALL detail files in `{wiki_dir}/_files/` to fill Understanding sections
2. Update the index `{repo_map_path}`:
   - Mark ALL explored files with ✅
   - Fill Purpose column with brief descriptions
   - Update `| Explored | X/Y |` to reflect actual progress

## ⚠️ File Editing Tip

When updating the index file (`_RepoMap.md`) or page index files:
- **Use Write tool** (read entire file → modify → write back)
- **Avoid Edit tool** — it often fails on markdown tables due to exact string matching

## 📝 Execution Report (REQUIRED)

When finished, write a summary report to `{wiki_dir}/_reports/phase0_repo_understanding.md`:

```markdown
# Phase 0: Repository Understanding Report

## Summary
- Files explored: X/Y
- Completion: X%

## Key Discoveries
- [Main entry points found]
- [Core modules identified]
- [Architecture patterns observed]

## Recommendations for Next Phase
- [Suggested workflows to document]
- [Key APIs to trace]
- [Important files for anchoring phase]
```

This report helps the next phase (Anchoring) understand what you discovered.

## ⛔ REMEMBER

```
Task complete ONLY when: Explored = Total (e.g., 116/116)
```

Do NOT stop early. Process ALL files. Test files can have brief descriptions but MUST still be explored.
