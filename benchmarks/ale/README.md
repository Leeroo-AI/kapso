# ALE-Bench Integration

This module provides integration with [ALE-Bench](https://github.com/SakanaAI/ALE-Bench), a benchmark for evaluating AI agents on AtCoder Heuristic Contests (algorithmic optimization problems).

## Prerequisites

Before installing ALE-Bench, ensure you have:

1. **Core Expert Agent installed** (from repository root):
   ```bash
   pip install -r requirements.txt
   ```

2. **API Keys configured** in `.env` or environment:
   ```bash
   OPENAI_API_KEY=your-openai-api-key
   GOOGLE_API_KEY=your-google-api-key
   ```

3. **System dependencies**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install -y libcairo2-dev
   ```

4. **Docker** (required for code evaluation):
   ```bash
   # Ubuntu/Debian
   sudo apt-get install -y docker.io
   sudo usermod -aG docker $USER
   # Log out and back in for group changes to take effect
   ```

## Installation

### Step 1: Clone and Install ALE-Bench

```bash
# Clone the repository
git clone https://github.com/SakanaAI/ALE-Bench.git
cd ALE-Bench

# Install the package
pip install .
pip install ".[eval]"

# Return to expert agent directory
cd ..
```

### Step 2: Build Docker Container

The Docker container is required to evaluate C++ solutions:

```bash
cd ALE-Bench
bash ./scripts/docker_build_202301.sh $(id -u) $(id -g)
cd ..
```

## Usage

### Run ALE-Bench

```bash
cd /path/to/mle_expert_coding
PYTHONPATH=. python -m benchmarks.ale.runner
```

### Configure Problem

Edit `benchmarks/ale/runner.py` to select which problem to run:

```python
problems_list = ["ahc039"]  # AtCoder Heuristic Contest problem
```

## Available Problems

The lite version includes these AtCoder Heuristic Contest problems:

| Problem ID | Contestants | Scoring | Description |
|------------|-------------|---------|-------------|
| ahc008 | 824 | Maximize | - |
| ahc011 | 926 | Maximize | - |
| ahc015 | 779 | Maximize | - |
| ahc016 | 1047 | Maximize | - |
| ahc024 | 664 | Maximize | - |
| ahc025 | 879 | Minimize | - |
| ahc026 | 740 | Maximize | - |
| ahc027 | 999 | Minimize | - |
| ahc039 | 683 | Maximize | - |
| ahc046 | 939 | Maximize | - |

Use `ale_bench.list_problem_ids()` for all available problems.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `GOOGLE_API_KEY` | Google API key for Gemini (required) | - |

## Output

The agent generates C++ solutions (`main.cpp`) which are evaluated in Docker containers against test cases. Final rankings are computed based on the score achieved.

## Domain Knowledge

This benchmark includes built-in domain knowledge for common algorithmic optimization techniques:

- **Simulated Annealing** - State representation, neighborhood design, temperature scheduling
- **Beam Search** - Beam width, evaluation functions, diversity
- **Monte Carlo Tree Search (MCTS)** - Exploration vs exploitation
- **Random Simulation** - Heuristic scoring, depth control
- **Dynamic Programming** - Memoization, state space reduction

See `benchmarks/ale/handler.py:_get_domain_knowledge()` for details.
