# Excavation Phase: Trace Imports to Source Code

You are a knowledge extraction agent. Your task is to trace the imports from the Workflow pages to find the actual implementation code, then create Implementation wiki pages.

## Context

- Repository: {repo_name}
- Repository Path: {repo_path}
- Wiki Output Directory: {wiki_dir}
- Workflow Pages Written: {wiki_dir}/workflows/

## Wiki Structure Definition

{implementation_structure}

## Your Task

### Step 1: Read the Workflow Pages

First, read the Workflow pages that were created in the previous phase:
- List files in `{wiki_dir}/workflows/`
- For each workflow, identify the code examples and imports used

### Step 2: Trace Imports to Source

For each major class/function used in the workflows:

1. **Find the import statement** in the example code
2. **Locate the source file** in the repository
3. **Read the actual implementation** (class/function definition)
4. **Extract the code signature** and understand I/O contract

Example trace:
```
from unsloth import FastLanguageModel
→ unsloth/__init__.py
→ unsloth/models/loader.py::FastLanguageModel
```

### Step 3: Write Implementation Pages

For EACH significant class/function you find, create an Implementation wiki page.

**Required Sections:**
1. Metadata block (wikitable with sources, domains, last_updated)
2. `== Overview ==` - One sentence: "Concrete tool for X provided by Y library"
3. `=== Description ===` - What this code does, its role in the stack
4. `=== Usage ===` - When to import/use this
5. `== Code Signature ==` - The actual function/class signature with `<syntaxhighlight>`
6. `== I/O Contract ==` - What it consumes (inputs) and produces (outputs)
7. `== Related Pages ==` - Context & Requirements, Tips and Tricks

**Important:**
- Include the ACTUAL code signature from the source file
- Document real parameters with their types
- Note file path and line numbers if helpful
- Add `[[requires_env::Environment:X]]` placeholders for any hardware/software requirements you notice

## Output Instructions

Write .md files to: `{wiki_dir}/implementations/`

**Filename format:** `{repo_name}_ClassName.md` or `{repo_name}_function_name.md`

**Examples:**
- `unsloth_FastLanguageModel.md`
- `unsloth_get_peft_model.md`
- `unsloth_save_pretrained_gguf.md`

Focus on the most important user-facing APIs (the ones used in the workflow examples).

