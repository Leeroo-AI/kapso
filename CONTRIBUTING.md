# Contributing to Kapso

Thank you for your interest in contributing to Kapso! This guide will help you get started.

## Getting Started

### Prerequisites

- Python 3.12+
- Git with LFS support
- Conda (recommended)

### Development Setup

```bash
# Clone the repository
git clone https://github.com/leeroo-ai/kapso.git
cd kapso

# Pull Git LFS files
git lfs install
git lfs pull

# Create conda environment
conda create -n kapso-dev python=3.12
conda activate kapso-dev

# Install with dev dependencies
pip install -e ".[dev]"
```

### Environment Variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your-openai-api-key
GOOGLE_API_KEY=your-google-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
```

## Making Changes

### Code Style

- Write clean, simple, readable code
- Keep files small and focused (<200 lines when possible)
- Use clear, consistent naming
- Add helpful comments to explain non-obvious logic

### Linting

```bash
# Format code
black src/ tests/

# Check style
flake8 src/ tests/
```

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
├── src/              # Main source code
│   ├── core/         # Core utilities
│   ├── deployment/   # Deployment strategies
│   ├── execution/    # Experiment execution
│   ├── knowledge/    # Knowledge pipeline
│   ├── memory/       # Cognitive memory
│   └── repo_memory/  # Repository memory
├── benchmarks/       # MLE-Bench and ALE-Bench
├── tests/            # Test suite
├── docs/             # Documentation
└── services/         # Infrastructure services
```

## Getting Help

- **Discord**: [Join our community](https://discord.gg/hqVbPNNEZM)
- **Issues**: Open a GitHub issue for bugs or feature requests
- **Documentation**: [docs.leeroo.com](https://docs.leeroo.com)

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
