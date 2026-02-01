# Agentic Scaffold Example

This example demonstrates using Kapso's `evolve()` function to optimize an AI workflow that extracts tabular data from chart images using a Vision Language Model (VLM).

## Environment Setup

Before running this example, install the required dependencies:

```bash
# Create and activate conda environment (recommended)
conda create -n kapso python=3.10
conda activate kapso

# Install dependencies
pip install openai huggingface_hub

# Install Kapso from the project root
cd /path/to/kapso
pip install -e . # or pip install leeroo-kapso
```

You'll also need an OpenAI API key:

```bash
export OPENAI_API_KEY=your_key_here
```

## Problem Description

The baseline implementation (`optimize.py`) contains a VLM-based extractor that converts chart images to CSV tables. The goal is to optimize the prompt and extraction logic to improve accuracy.

### Optimization Opportunities

1. **Prompt Engineering**: Improve instructions for data extraction
2. **Output Parsing**: Better handling of VLM output formats
3. **Multi-Step Extraction**: Break down complex charts into steps
4. **Error Handling**: Add validation and correction logic
5. **Format Specification**: More precise CSV formatting rules

### Constraints

- Must output valid CSV format
- Must work with the specified OpenAI vision model
- Average cost per query must stay under $0.02

## Files

- `run_evolve.py` - Main script that uses Kapso to optimize the extractor
- `initial_repo/optimize.py` - Baseline VLM extractor to be optimized
- `initial_repo/evaluate.py` - Evaluation script that measures accuracy
- `initial_repo/prepare_data.py` - Script to download and prepare test data
- `initial_repo/requirements.txt` - Python dependencies

## Usage

### Prepare the Data

First, download the test dataset:

```bash
cd examples/agentic_scaffold/initial_repo
python prepare_data.py
```

This creates a `subset_line_100/` directory with:
- `index.csv`: mapping of example IDs to image and table paths
- `images/`: chart images (PNG/JPEG)
- `tables/`: ground truth CSV tables

### Run Kapso Evolution

```bash
cd examples/agentic_scaffold
python run_evolve.py
```

This will:
1. Initialize Kapso
2. Run multiple iterations to find optimized implementations
3. Output the best solution to `./agentic_optimized`

### Manual Evaluation

To evaluate a specific implementation:

```bash
cd initial_repo
python evaluate.py --max-samples 10 --num-workers 4
```