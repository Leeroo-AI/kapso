# Examples

Each example is a self-contained Kapso project: a starter repo, a goal,
and a driver that runs `evolve()` against it. Install and authenticate
per the [main quickstart](../README.md#-quickstart), verify with
`kapso doctor`, then follow the example's own README.

| Example | Description |
|---------|-------------|
| [**CUDA Optimization**](cuda_optimization/README.md) | Optimize CUDA kernels for GPU performance |
| [**PyTorch Optimization**](pytorch_optimization/README.md) | Cut wall-clock and memory — fuse ops, kill sync points and host-device chatter, saturate the GPU without changing numerics |
| [**ML Model Development**](ml_model_development/README.md) | End-to-end delivery of prediction models — data prep, features, training, and validation evolved into a deployable artifact |
| [**Harness Optimization**](prompt_engineering/README.md) | Evolve the harness around a model — prompts, decoding, parsing, and scoring tuned against a measurable target |
| [**Agent Optimization**](agentic_scaffold/README.md) | Agents improving agents — workflows, tools, and prompts evolved until the metric climbs |

ML Model Development doubles as the reference for the complete
production loop — research → `learn_knowledge()` → `evolve()` →
`learn()` → evolve again, served the lessons it just earned.
