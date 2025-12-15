# Excavation Phase: Trace Imports to Source Code

You are a knowledge extraction agent. Your task is to trace the imports from the Workflow pages to find the actual implementation code, then create Implementation wiki pages.

## Context

- Repository: {repo_name}
- Repository Path: {repo_path}
- Wiki Output Directory: {wiki_dir}
- **Repository Map (Index):** {repo_map_path}
- **File Details:** {wiki_dir}/_files/
- Workflow Pages Written: {wiki_dir}/workflows/

## IMPORTANT: Read Previous Phase Reports

**FIRST**, read the previous phase reports:
- `{wiki_dir}/_reports/phase0_repo_understanding.md` - Repository structure insights
- `{wiki_dir}/_reports/phase1_anchoring.md` - Workflows created, APIs to trace

These reports tell you what workflows exist and which APIs need Implementation pages.

## IMPORTANT: Use the Repository Map

**THEN**, read the Repository Map index at `{repo_map_path}`.

The index contains:
- **Purpose column:** What each file does (helps identify which files contain the code you need)
- **Coverage column:** Which files are already covered (avoid duplicate work)
- Classes and functions listed in each file

For detailed info on any file, read its detail page in `_files/`.

## Wiki Structure Definition

{implementation_structure}

## Your Task

### Step 1: Read the Index and Workflow Pages

1. Read `{repo_map_path}` to see file purposes and find relevant source files
2. Read the Workflow pages in `{wiki_dir}/workflows/`
3. Identify which classes/functions from the index are used in the workflows

### Step 2: Match Imports to Source Files

Use the Repository Map to locate source files. The **Purpose** column tells you what each file does.

Example: If a workflow imports `FastLanguageModel`:
- Look in index for files with Purpose like "Main model loader"
- Check the detail page for that file to confirm it has `FastLanguageModel` class

### Step 3: Write Implementation Pages

For EACH significant class/function you find, create an Implementation wiki page.

**Required Sections:**
1. Metadata block (wikitable with sources, domains, last_updated)
   - ⚠️ **Sources must be HIGH-LEVEL only:** repo URLs, docs, papers
   - ❌ WRONG: `[[source::Repo|Loader|unsloth/models/loader.py]]`
   - ✅ RIGHT: `[[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]`
2. `== Overview ==` - One sentence: "Concrete tool for X provided by Y library"
3. `=== Description ===` - What this code does, its role in the stack
4. `=== Usage ===` - When to import/use this
5. `== Code Reference ==` - **Source location, signature, and import** (file paths go HERE)
6. `== I/O Contract ==` - Inputs/outputs as structured tables
7. `== Usage Examples ==` - **Complete, runnable code examples**
8. `== Related Pages ==` - Context & Requirements, Tips and Tricks

#### Code Reference Format (CRITICAL)

```mediawiki
== Code Reference ==

=== Source Location ===
* '''Repository:''' [{repo_url} {repo_name}]
* '''File:''' [{repo_url}/blob/main/path/to/file.py#L50-L150 path/to/file.py]
* '''Lines:''' 50-150

=== Signature ===
<syntaxhighlight lang="python">
class FastLanguageModel:
    @staticmethod
    def from_pretrained(
        model_name: str,
        max_seq_length: int = 2048,
        dtype: Optional[torch.dtype] = None,
        load_in_4bit: bool = True,
        ...
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load and optimize a language model."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
</syntaxhighlight>
```

#### Usage Examples Format (CRITICAL)

```mediawiki
== Usage Examples ==

=== Basic Usage ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Use model for generation
inputs = tokenizer("Hello, ", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
</syntaxhighlight>
```

**Important:**
- Include the ACTUAL code signature from the source file
- Use REAL GitHub URLs with line numbers: `{repo_url}/blob/main/file.py#L50-L150`
- Provide COMPLETE, RUNNABLE examples
- Do NOT add `[[requires_env::...]]` or `[[uses_heuristic::...]]` links yet

### Step 4: Update Coverage in Repository Map

After creating Implementation pages, **update the index** at `{repo_map_path}`:

For each source file your Implementation covers, update its **Coverage column**:

```markdown
| ✅ | `unsloth/models/loader.py` | 320 | Main model loader | Impl: FastLanguageModel | [→](...) |
```

If a file is covered by multiple pages:
```markdown
| ✅ | `unsloth/save.py` | 450 | Model saving utilities | Impl: save_pretrained, save_gguf | [→](...) |
```

If a file already has Workflow coverage, append to it:
```markdown
| ✅ | `examples/qlora.py` | 150 | QLoRA example | Workflow: QLoRA_Finetuning; Impl: — | [→](...) |
```

### Step 5: Update the Implementation Index

**IMPORTANT:** After creating Implementation pages, add entries to `{wiki_dir}/_ImplementationIndex.md`:

| Column | Content |
|--------|---------|
| Page | Implementation page name (without .md) |
| File | Link to the implementation file: `[→](./implementations/{repo_name}_X.md)` |
| Source | Source file(s) with line ranges: `loader.py:L50-200` |
| Implements (Principle) | Which Principle this implements (filled in Synthesis phase) |
| Notes | Brief description of what this implementation does |

**Example row:**
```
| {repo_name}_FastLanguageModel | [→](./implementations/...) | `loader.py:L50-200` | — | Main model loader API |
```

## Output Instructions

Write .md files to: `{wiki_dir}/implementations/`

**Filename format:** `{repo_name}_ClassName.md` or `{repo_name}_function_name.md`

**Examples:**
- `{repo_name}_FastLanguageModel.md`
- `{repo_name}_get_peft_model.md`
- `{repo_name}_save_pretrained_gguf.md`

Focus on the most important user-facing APIs (the ones used in the workflow examples).

## Repo Scoping Rule (CRITICAL)

Only read Workflow pages whose filenames start with `{repo_name}_`.
Only create/update Implementation pages whose filenames start with `{repo_name}_`.

## Provenance Hook (IMPORTANT)

In each Implementation page, include the *exact* source evidence:
- The repo-relative file path(s) where the class/function is defined
- The line range(s) if you can determine them

Use this exact text pattern somewhere in the Implementation page (for later automation):
`Source Files: path/to/file.py:L10-L120; other/file.py:L5-L40`

## ⚠️ File Editing Tip

When updating index files (`_RepoMap.md`, `_ImplementationIndex.md`):
- **Use Write tool** (read entire file → modify → write back)
- **Avoid Edit tool** — it often fails on markdown tables

## 📝 Execution Report (REQUIRED)

When finished, write a summary report to `{wiki_dir}/_reports/phase2_excavation.md`:

```markdown
# Phase 2: Excavation Report

## Implementations Created
| Implementation | Source File | Lines |
|----------------|-------------|-------|
| [Name] | [file:L#-#] | [count] |

## API Coverage
- Classes documented: X
- Functions documented: X
- Total source files covered: X

## Notes for Synthesis Phase
- [Concepts that need Principle pages]
- [Patterns observed across implementations]
```
