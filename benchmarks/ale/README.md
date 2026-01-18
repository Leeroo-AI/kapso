# ALE-Bench Integration

This module provides integration with [ALE-Bench](https://github.com/SakanaAI/ALE-Bench), a benchmark for evaluating AI agents on AtCoder Heuristic Contests (algorithmic optimization problems).

Kapso achieved **#1 on ALE-Bench**.

![ALE-Bench Results](https://api.leeroo.com/storage/v1/object/public/opensource/ale_benchmark.png)

## Prerequisites

Before installing ALE-Bench, ensure you have:

1. **Core Kapso Agent installed** (from repository root):
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

# Return to Kapso directory
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

```bash
# List available problems
PYTHONPATH=. python -m benchmarks.ale.runner --list

# List lite benchmark problems
PYTHONPATH=. python -m benchmarks.ale.runner --lite

# Solve a problem
PYTHONPATH=. python -m benchmarks.ale.runner -p ahc039

# With options
PYTHONPATH=. python -m benchmarks.ale.runner \
    -p ahc039 \
    -i 14 \
    -m ALE_CONFIGS \
    -d aider
```

## CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `-p, --problem` | Problem ID (e.g., `ahc039`) | Required |
| `-i, --iterations` | Max experiment iterations | 14 |
| `-m, --mode` | Config mode | `ALE_CONFIGS` |
| `-d, --coding-agent` | Coding agent | From config |
| `--list` | List all problems | - |
| `--lite` | List lite problems | - |
| `--list-agents` | List coding agents | - |

## Available Problems

`ahc008`, `ahc011`, `ahc015`, `ahc016`, `ahc024`, `ahc025`, `ahc026`, `ahc027`, `ahc039`, `ahc046`

Use `ale_bench.list_problem_ids()` for all available problems.

## Output Structure

The agent generates:

```
experiment_workspace/{uuid}/
├── main.cpp          # C++ solution
├── pre_run.cpp       # Optional precomputation (max 1 min)
└── sessions/         # Experiment branches
```

## Evaluation

The evaluation process works as follows:

1. **Code Submission**: The `main.cpp` file is read from the experiment workspace
2. **Docker Evaluation**: Code is sent to `ale_bench.public_eval()` which compiles and runs in an isolated Docker container
3. **Test Execution**: Solution runs against all test cases with strict time limits
4. **Validation**: Each test case must return `ACCEPTED` with a non-zero score
5. **Score Stabilization**: If all tests pass, the solution runs **4 additional times** and scores are averaged for stability
6. **Final Ranking**: Private evaluation compares against original contest participants

## Code Requirements

Generated C++ must:
- Be time-aware (limit: time_limit - 100ms for I/O)
- Handle all input constraints
- Use efficient algorithms and data structures
- Include compiler optimization pragmas if helpful

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `GOOGLE_API_KEY` | Google API key for Gemini (required) | - |

## Domain Knowledge

This benchmark includes built-in domain knowledge for common algorithmic optimization techniques:

- **Simulated Annealing** - Design good state representation, balance small and large moves, avoid recomputation in legality checks, keep regret mechanism for constrained problems
- **Beam / Random Search** - Balance diversity and quality in beams, fast-stop bad solutions, use strong heuristic scoring
- **Random Simulation** - Define strong heuristic scoring, consider average and std of scores, balance greedy vs long-horizon moves

See `benchmarks/ale/handler.py:_get_domain_knowledge()` for details.
