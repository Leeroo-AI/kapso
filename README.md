<h1 align="center">Kapso</h1>

<h4 align="center">A Knowledge-grounded framework for Autonomous Program Synthesis and Optimization</h4>

<p align="center">
  <a href="https://docs.leeroo.com">Learn more</a> ·
  <a href="https://discord.gg/hqVbPNNEZM">Join Discord</a> ·
  <a href="https://leeroo.com">Website</a>
</p>

<p align="center">
  <a href="https://discord.gg/hqVbPNNEZM"><img src="https://dcbadge.limes.pink/api/server/hqVbPNNEZM?style=flat" alt="Discord"></a>
  <a href="https://github.com/leeroo-ai/kapso"><img src="https://img.shields.io/github/commit-activity/m/leeroo-ai/kapso" alt="GitHub commit activity"></a>
  <a href="https://www.ycombinator.com/companies/leeroo"><img src="https://img.shields.io/badge/Y%20Combinator-X25-orange?logo=ycombinator&logoColor=white" alt="Y Combinator X25"></a>
</p>

<p align="center">
  If you like this project, please support us by giving it a star ⭐
</p>

> **Early Access**: [Sign up](https://docs.google.com/forms/d/e/1FAIpQLSfk0RjtZaZFXq3-tclZhnz40E_mNzPSI1RHhBQWzswbNwp8Ug/viewform) for **[Leeroopedia](https://leeroopedia.com)** and the **hosted version of Kapso** : Leeroopedia is a centralized ML & Data knowledge wiki with best practices and expert-level implementation patterns, written by Kapso and human experts.

<p align="center">
  <img src="https://api.leeroo.com/storage/v1/object/public/opensource/framework.png" alt="Kapso Framework Architecture" width="800">
</p>

---

## News

- **Technical Report**: Our technical report is now available! [Read the paper](https://arxiv.org/abs/2601.21526)
- **#1 on [MLE-Bench](benchmarks/mle/README.md)**: KAPSO achieved top ranking among open-source systems on Kaggle ML competitions (MLE Benchmark).

  <img src="https://api.leeroo.com/storage/v1/object/public/opensource/mle_benchmark.png" alt="MLE-Bench Results" width="600">

- **#1 on [ALE-Bench](benchmarks/ale/README.md)**: KAPSO achieved top ranking on long-horizon algorithmic discovery problems (ALE Benchmark).

  <img src="https://api.leeroo.com/storage/v1/object/public/opensource/ale_benchmark.png" alt="ALE-Bench Results" width="600">

## What is KAPSO?

KAPSO combines **iterative experimentation** with a **knowledge base** of best practices and tricks to discover code improvements.

It automates the cycle of **designing**, **testing**, and **refining** algorithms, eventually adapting the optimized solution for **deployment** on your chosen infrastructure.

### The Four Pillars

| Pillar | Method | Description |
|--------|--------|-------------|
| **Evolve** | `.evolve()` | Run iterative experiments to build software for a goal. Uses tree search, coding agents, and KG context to generate and refine solutions. |
| **Learn** | `.learn()` | Ingest knowledge from repositories, past solutions, or research results. Extracts patterns and best practices into the Knowledge Graph. |
| **Research** | `.research()` | Run deep web research to gather ideas and implementation references. Returns structured findings you can feed into the knowledge base or use as context for evolving solutions. |
| **Deploy** | `.deploy()` | Turn a solution into running software. Supports local execution, Docker containers, or cloud platforms like Modal. |

## 🚀 Quickstart

### Installation

**From PyPI (recommended)**

```bash
pip install leeroo-kapso
```

**From source (for development or to access wiki knowledge data)**

```bash
git clone https://github.com/leeroo-ai/kapso.git
cd kapso

# Pull Git LFS files (wiki knowledge data)
git lfs install
git lfs pull

# Create conda environment (recommended)
conda create -n kapso python=3.12
conda activate kapso

# Install in development mode
pip install -e .
```

### Set Up API Keys

Create `.env` in project root:

```bash
OPENAI_API_KEY=your-openai-api-key
GOOGLE_API_KEY=your-google-api-key       # For Gemini
ANTHROPIC_API_KEY=your-anthropic-api-key # For Claude Code
```

### Basic Usage

```python
from kapso import Kapso, Source, DeployStrategy

# Initialize Kapso
# If you have a Knowledge Graph, pass kg_index; otherwise just use Kapso()
kapso = Kapso(kg_index="data/indexes/legal_contracts.index")

# Research: Gather domain-specific techniques from the web
# mode: "idea" | "implementation" | "study" (can pass multiple as list)
# depth: "light" | "deep" (default: "deep")

findings = kapso.research(
    "RLHF and DPO fine-tuning for legal contract analysis",
    mode=["idea", "implementation"],
    depth="deep",
)

# Learn: Ingest knowledge from repositories and research into the KG
kapso.learn(
    Source.Repo("https://github.com/huggingface/trl"),
    *findings.ideas,           # List[Source.Idea]
    *findings.implementations, # List[Source.Implementation]
    wiki_dir="data/wikis",
)

# Evolve: Build a solution through experimentation
# Use research results as context via to_string()
solution = kapso.evolve(
    goal="Fine-tune Llama-3.1-8B for legal clause risk classification, target F1 > 0.85",
    data_dir="./data/cuad_dataset", 
    output_path="./models/legal_risk_v1",
    context=[findings.to_string()],
)

# Deploy: Turn solution into running deployed_program
deployed_program = kapso.deploy(solution, strategy=DeployStrategy.MODAL)
deployed_program.stop()
```

For detailed integration steps, see the [Quickstart](https://docs.leeroo.com/docs/quickstart) and [Installation](https://docs.leeroo.com/docs/installation) guides.

## Examples

| Example | Description |
|---------|-------------|
| [**CUDA Optimization**](examples/cuda_optimization/README.md) | Optimize multi-head self-attention for CUDA using Flash Attention and Triton kernels |
| [**PyTorch Optimization**](examples/pytorch_optimization/README.md) | Optimize PyTorch operations via kernel fusion, torch.compile, and custom kernels |

## Supported Benchmarks

| Benchmark | Description |
|-----------|-------------|
| [**MLE-Bench**](benchmarks/mle/README.md) | Kaggle ML competitions — tabular, image, text, audio problems |
| [**ALE-Bench**](benchmarks/ale/README.md) | AtCoder algorithmic optimization — C++ solution generation |

## 📚 Documentation & Support

- **Full Documentation**: [docs.leeroo.com](https://docs.leeroo.com)
- **Community**: [Discord](https://discord.gg/hqVbPNNEZM)
- **Website**: [leeroo.com](https://leeroo.com)


## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to get started.

## Citation

If you use Kapso in your research, please cite:

```bibtex
@misc{nadaf2026kapsoknowledgegroundedframeworkautonomous,
      title={KAPSO: A Knowledge-grounded framework for Autonomous Program Synthesis and Optimization}, 
      author={Alireza Nadaf and Alireza Mohammadshahi and Majid Yazdani},
      year={2026},
      eprint={2601.21526},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.21526}, 
}
```

---