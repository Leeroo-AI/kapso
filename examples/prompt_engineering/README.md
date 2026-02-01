# Prompt Engineering Example

This example demonstrates using Kapso's `evolve()` function to iteratively improve a prompt for solving American Invitational Mathematics Examination (AIME) problems.

## Environment Setup

Before running this example, install the required dependencies:

```bash
# Create and activate conda environment (recommended)
conda create -n kapso python=3.10
conda activate kapso

# Install dependencies
pip install openai datasets

# Install Kapso from the project root
cd /path/to/kapso
pip install -e .
```

You'll also need an OpenAI API key:

```bash
export OPENAI_API_KEY=your_key_here
```

## Problem Description

The baseline implementation (`optimize.py`) contains a prompt template for solving AIME math problems. The goal is to optimize this prompt to improve accuracy on a subset of AIME 2024 problems.

### Constraints

- Must produce answers in `\boxed{XXX}` format (3-digit integer 000-999)
- Must work with the specified OpenAI model
- Prompt modifications only (no changes to evaluation logic)

## Files

- `run_evolve.py` - Main script that uses Kapso to optimize the prompt
- `initial_repo/optimize.py` - Baseline prompt template to be optimized
- `initial_repo/evaluate.py` - Evaluation script that measures accuracy
- `initial_repo/requirements.txt` - Python dependencies

## Usage

### Run Kapso Evolution

```bash
cd examples/prompt_engineering
python run_evolve.py
```

This will:
1. Initialize Kapso
2. Run multiple iterations to find optimized prompts
3. Output the best solution to `./prompt_optimized`

### Manual Evaluation

To evaluate a specific implementation:

```bash
cd initial_repo
python evaluate.py
```

## Success Criteria

- **Accuracy**: Higher is better (baseline ~0.10-0.20)
- **Format Compliance**: Answers must be in `\boxed{XXX}` format

A good optimization should achieve 0.30-0.50 accuracy through improved prompting techniques.
