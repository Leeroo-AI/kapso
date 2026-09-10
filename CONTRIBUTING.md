# Contributing to Kapso

Thank you for your interest in contributing to Kapso! This guide will help you get started.

## Getting Started

### Prerequisites

- Python 3.10 or newer (Python 3.12 is recommended for development)
- Git
- A virtual environment tool such as `venv` or Conda

Kapso uses Unix process and file-locking primitives. On Windows, use WSL2 for
development and testing.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/leeroo-ai/kapso.git
cd kapso

# Create conda environment
conda create -n kapso-dev python=3.12
conda activate kapso-dev

# Install with dev dependencies
python -m pip install -e ".[dev]"
```

Git LFS is not required for the core Kapso checkout. Some optional benchmark
datasets may require it; follow the setup guide for the benchmark you are
working on.

### Environment Variables

The default unit test suite does not require API keys. Configure credentials
only for the integrations you intend to exercise:

```bash
# Embeddings used by memory and knowledge search
OPENAI_API_KEY=your-openai-api-key

# Direct API-backed agents, when selected
GOOGLE_API_KEY=your-google-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
```

The Codex adapter uses its CLI login. Claude Code can use its stored CLI login
or an API key when that authentication mode is selected. Run `kapso doctor`
(or, for example, `kapso doctor evolve`) to see what the current configuration
requires.

## Making Changes

### Code Style

- Write clean, simple, readable code
- Keep files small and focused (<200 lines when possible)
- Use clear, consistent naming
- Add helpful comments to explain non-obvious logic

### Linting

```bash
# Check formatting
python -m black --check src/ tests/

# Check style
python -m flake8 src/ tests/
```

### Testing

```bash
# Example: run the non-live test module relevant to your change
python -m pytest tests/test_cli_agent_choices.py

# Optional coverage for the same tests
python -m pytest --cov=kapso tests/test_cli_agent_choices.py
```

Non-live tests do not require API keys. Some benchmark and integration tests
need their corresponding optional dependencies or services.

Tests marked `live` launch real coding-agent sessions and consume subscription
quota. Run them only when the change requires it and you have configured the
necessary CLIs and credentials:

```bash
python -m pytest tests/live/test_inbox_live.py --run-live
```

Before opening a pull request, run the test suite, formatting check, style
check, and `git diff --check`.

## Submitting Changes

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run tests and linting
5. Commit with a clear message
6. Push to your fork
7. Open a Pull Request

### Commit Messages

Write clear, concise commit messages:

- Use present tense ("Add feature" not "Added feature")
- Keep the first line under 72 characters
- Reference issues when applicable (`Fixes #123`)

### PR Guidelines

- Keep PRs focused on a single change
- Include a description of what changed and why
- Update documentation if needed
- Ensure all tests pass

## Project Structure

```
kapso/
├── src/kapso/          # Main Python package
│   ├── core/           # Core configuration and utilities
│   ├── deployment/     # Deployment strategies
│   ├── execution/      # Experiment execution
│   ├── knowledge_base/ # Knowledge ingestion and storage
│   ├── learning/       # Learning and experiment-bank workflows
│   └── researcher/     # Research workflows
├── benchmarks/         # MLE-Bench and ALE-Bench
├── tests/              # Test suite
├── docs/               # Documentation
└── services/           # Infrastructure services
```

## Getting Help

- **Discord**: [Join our community](https://discord.gg/hqVbPNNEZM)
- **Issues**: Open a GitHub issue for bugs or feature requests
- **Documentation**: [docs.leeroo.com](https://docs.leeroo.com)

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
