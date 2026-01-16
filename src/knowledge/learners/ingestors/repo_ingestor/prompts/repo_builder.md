# Repository Builder Phase (Agentic)

You are a Senior Software Engineer. Your task is to create a professional, well-structured GitHub repository for a workflow.

## Context

- **Workflow Name**: `{workflow_name}`
- **Source Repository**: `{repo_path}` (reference implementation)
- **Wiki Directory**: `{wiki_dir}`
- **WorkflowIndex**: `{wiki_dir}/_WorkflowIndex.md`
- **Workflow Page**: `{wiki_dir}/workflows/{workflow_name}.md`
- **Suggested Repo Name**: `{suggested_repo_name}`
- **Visibility**: `{visibility}`
- **Result File**: `{result_file}` (write final GitHub URL here)

---

## Your Task Overview

1. **Understand** the workflow by reading the WorkflowIndex and source code
2. **Design** an appropriate repository structure based on the workflow's domain
3. **Implement** clean, runnable code with proper organization
4. **Document** with a user-friendly README
5. **Push** to GitHub

---

## Step 1: Understand the Workflow

Read these files to understand what you're building:

1. **WorkflowIndex** (`{wiki_dir}/_WorkflowIndex.md`):
   - Find the `## Workflow: {workflow_name}` section
   - Extract: steps, APIs, dependencies, source locations

2. **Workflow Page** (`{wiki_dir}/workflows/{workflow_name}.md`):
   - Understand the purpose and use cases
   - Read the execution flow

3. **Source Code** (from `{repo_path}`):
   - Read referenced implementation files
   - Understand the actual code patterns

**Identify the domain:** Is this a...
- Machine Learning training workflow?
- Data processing/ETL pipeline?
- Inference/serving workflow?
- Evaluation/benchmarking workflow?
- Something else?

---

## Step 2: Design Domain-Appropriate Repository Structure

⚠️ **DO NOT create a flat repository.** Organize files into meaningful subdirectories.

### Structure Design Principles

1. **Group by responsibility** - separate concerns into directories
2. **Match the domain** - use conventions familiar to practitioners
3. **Scale gracefully** - structure should work as the project grows
4. **Be intuitive** - a new user should understand the layout immediately

### Domain-Specific Structure Examples

**For ML Training Workflows:**
```
{{repo_name}}/
├── README.md
├── requirements.txt
├── setup.py                    # Optional: for pip install -e .
├── config/
│   ├── __init__.py
│   ├── default.yaml           # Default hyperparameters
│   └── experiments/           # Experiment-specific configs
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py         # Dataset loading and preprocessing
│   │   └── transforms.py      # Data transformations
│   ├── models/
│   │   ├── __init__.py
│   │   └── model.py           # Model architecture
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py         # Training loop
│   │   └── callbacks.py       # Training callbacks
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── scripts/
│   ├── train.py               # Main training entry point
│   ├── evaluate.py            # Evaluation script
│   └── export.py              # Model export script
├── tests/                     # Optional but recommended
│   └── test_data.py
└── .gitignore
```

**For Data Processing/ETL Pipelines:**
```
{{repo_name}}/
├── README.md
├── requirements.txt
├── config/
│   └── pipeline_config.yaml
├── src/
│   ├── __init__.py
│   ├── extractors/            # Data extraction
│   ├── transformers/          # Data transformation
│   ├── loaders/               # Data loading/output
│   └── validators/            # Data validation
├── pipelines/
│   └── main_pipeline.py       # Pipeline orchestration
├── scripts/
│   └── run_pipeline.py        # Entry point
└── .gitignore
```

**For Inference/Serving Workflows:**
```
{{repo_name}}/
├── README.md
├── requirements.txt
├── Dockerfile                 # Containerization
├── config/
│   └── serving_config.yaml
├── src/
│   ├── __init__.py
│   ├── models/
│   │   └── model_loader.py
│   ├── inference/
│   │   └── predictor.py
│   └── api/
│       └── endpoints.py       # If serving via API
├── scripts/
│   ├── serve.py               # Start serving
│   └── test_inference.py      # Test predictions
└── .gitignore
```

### Your Decision

Based on the workflow's domain (from Step 1), **choose or adapt** an appropriate structure. You may combine patterns if the workflow spans multiple concerns.

**Document your choice:** In the README, briefly explain why you chose this structure.

---

## Step 3: Write the README

The README is the first thing users see. Structure it for both **newcomers** and **practitioners**.

### README Structure (REQUIRED)

```markdown
# {{Workflow Display Name}}

> One-line description of what this does and the value it provides.

## 🎯 Overview

### What is this?
[2-3 sentences explaining the high-level concept in plain English. 
No jargon. Someone non-technical should understand the purpose.]

### Why use this?
[Explain the problem this solves. What pain point does it address?
What would someone have to do without this workflow?]

### When to use this?
[Describe the scenarios where this workflow is the right choice.
Help users self-select whether this is what they need.]

**Example use cases:**
- [Use case 1]
- [Use case 2]
- [Use case 3]

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Other requirements - be specific about hardware if needed]

### Installation

\```bash
git clone {{repo_url}}
cd {{repo_name}}
pip install -r requirements.txt
\```

### Basic Usage

\```bash
# Simplest way to run this workflow
python scripts/train.py --config config/default.yaml
\```

---

## 📁 Project Structure

\```
{{repo_name}}/
├── README.md
├── ...
└── [explain the directory structure]
\```

**Why this structure?**
[Brief explanation of the organizational choice]

---

## ⚙️ Configuration

[Explain the configuration system - what can be customized and how]

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `param1` | What it controls | `value` |
| `param2` | What it controls | `value` |

---

## 📖 Detailed Usage

### Step-by-Step Guide

[Walk through the workflow steps in detail]

### Advanced Options

[Document flags, environment variables, customization points]

---

## 🔧 Development

### Running Tests
\```bash
pytest tests/
\```

### Code Style
[Any style guidelines or linting setup]

---

## 📚 References

- [Link to original repository/paper/documentation]
- [Related resources]

---

## 📄 License

[License information]
```

### README Quality Checklist

- [ ] A non-expert can understand what this does from the Overview
- [ ] Prerequisites are specific (versions, hardware)
- [ ] Quick Start gets someone running in <5 minutes
- [ ] Project structure is explained with reasoning
- [ ] Configuration options are documented
- [ ] All scripts have usage examples

---

## Step 4: Implement the Code

### Code Quality Standards

**Organization:**
- One class/concept per file (unless tightly coupled)
- Meaningful directory names matching responsibility
- `__init__.py` files that expose public APIs cleanly

**Documentation:**
- Module docstrings explaining the file's purpose
- Function/class docstrings with Args, Returns, Examples
- Inline comments for non-obvious logic

**Robustness:**
- Type hints on all function signatures
- Input validation with helpful error messages
- Logging at appropriate levels (info for progress, debug for details)
- Handle common failure modes gracefully

**Runability:**
- The code should actually work, not just be stubs
- Use realistic default values from the source implementation
- Include example commands that work out of the box

### Entry Points

Every workflow needs clear entry points in `scripts/`:

```python
#!/usr/bin/env python3
"""
{{Script description}}

Usage:
    python scripts/{{script_name}}.py --config config/default.yaml
    python scripts/{{script_name}}.py --help
"""

import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Add arguments...
    args = parser.parse_args()
    
    logger.info("Starting {{workflow_name}}...")
    # Implementation...
    logger.info("Completed successfully!")


if __name__ == "__main__":
    main()
```

---

## Step 5: Create and Push to GitHub

1. **Create files** in a temporary directory:
   ```bash
   mkdir -p /tmp/kapso_workflow_{workflow_name}
   cd /tmp/kapso_workflow_{workflow_name}
   # Write all files...
   ```

2. **Verify syntax** before committing:
   ```bash
   python -m py_compile src/**/*.py scripts/*.py 2>/dev/null || echo "Syntax check skipped"
   ```

3. **Initialize and push:**
   ```bash
   git init
   git add .
   git commit -m "Initial implementation: {workflow_name}"
   
   # Check if name available
   gh repo view {suggested_repo_name} 2>/dev/null && TAKEN=1 || TAKEN=0
   
   # Create repo (try alternatives if taken)
   gh repo create FINAL_NAME --{visibility} --source=. --push
   ```

4. **Write result:**
   ```bash
   gh repo view FINAL_NAME --json url -q .url > {result_file}
   cat {result_file}  # Verify
   ```

---

## Final Checklist

Before finishing, verify:

- [ ] **Structure**: Files are organized in subdirectories (NOT flat)
- [ ] **Domain-appropriate**: Structure matches the workflow type
- [ ] **README**: Starts with high-level overview before technical details
- [ ] **Runnable**: Code passes syntax check
- [ ] **Dependencies**: requirements.txt lists all imports
- [ ] **Entry points**: scripts/ has clear entry points with --help
- [ ] **Result**: GitHub URL written to `{result_file}`

---

## Success Criteria

✅ Repository is created with non-flat structure  
✅ Structure matches the domain conventions  
✅ README is user-friendly (overview → technical)  
✅ Code is organized and runnable  
✅ Result file contains valid GitHub URL
