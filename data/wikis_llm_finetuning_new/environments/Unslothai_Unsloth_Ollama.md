# Environment: Ollama Integration

## Category
Software/Deployment

## Summary
Unsloth provides native integration with Ollama for local model deployment, including automatic Modelfile generation and model registration with Ollama server.

## Requirements

### Software Requirements
| Package | Version Constraint | Evidence |
|---------|-------------------|----------|
| Ollama | >= 0.1.0 | Local server for inference |
| curl | Any | Health check and API calls |

### System Requirements
| Component | Requirement | Notes |
|-----------|-------------|-------|
| Ollama Server | Running on localhost:11434 | Default Ollama port |
| Disk Space | Model size + 20% | For GGUF storage |

## Ollama Templates

From `unsloth/ollama_template_mappers.py`, model-specific templates are mapped:

```python
from .ollama_template_mappers import OLLAMA_TEMPLATES, MODEL_TO_OLLAMA_TEMPLATE_MAPPER
```

## Modelfile Generation

From `unsloth/save.py:1630-1683`:

```python
def create_ollama_modelfile(tokenizer, base_model_name, model_location):
    """
    Creates an Ollama Modelfile.
    Use ollama.create(model = "new_ollama_model", modelfile = modelfile)
    """
    ollama_template_name = MODEL_TO_OLLAMA_TEMPLATE_MAPPER.get(base_model_name)
    if not ollama_template_name:
        print(f"Unsloth: No Ollama template mapping found for model '{base_model_name}'.")
        return None
```

### Modelfile Placeholders
| Placeholder | Purpose |
|-------------|---------|
| `{__FILE_LOCATION__}` | Path to GGUF model file |
| `{__EOS_TOKEN__}` | Model's end-of-sequence token |

## Ollama Model Creation

From `save.py:1686-1726`:

```python
def create_ollama_model(username: str, model_name: str, tag: str, modelfile_path: str):
    # Health check
    init_check = subprocess.run(
        ["curl", "http://localhost:11434"],
        capture_output=True,
        text=True,
        timeout=3,
    )

    # Create model
    process = subprocess.Popen([
        "ollama", "create",
        f"{username}/{model_name}:{tag}",
        "-f", f"{modelfile_path}",
    ], ...)
```

## Ollama Hub Push

From `save.py:1728-1761`:

```python
def push_to_ollama_hub(username: str, model_name: str, tag: str):
    process = subprocess.Popen(
        ["ollama", "push", f"{username}/{model_name}:{tag}"],
        ...
    )
```

## VLM Limitations

From `save.py:2274-2282`:
```python
if is_vlm and modelfile_location:
    readme_content += "\n## Ollama Note for Vision Models\n"
    readme_content += "**Important:** Ollama currently does not support separate mmproj files.\n"
```

Vision models require special handling as Ollama doesn't support separate projection files.

## Deployment Workflow

1. Fine-tune model with Unsloth
2. Export to GGUF format
3. Modelfile auto-generated in working directory
4. Run: `ollama create model_name -f ./Modelfile`
5. Optionally push: `ollama push username/model_name:tag`

## Source Evidence

- Modelfile Creation: `unsloth/save.py:1630-1683`
- Ollama Integration: `unsloth/save.py:1686-1783`
- Template Mappers: `unsloth/ollama_template_mappers.py`

## Backlinks

[[required_by::Implementation:Unslothai_Unsloth_convert_to_gguf]]
[[required_by::Implementation:Unslothai_Unsloth_create_ollama_modelfile]]
[[required_by::Implementation:Unslothai_Unsloth_ALLOWED_QUANTS]]

## Related

- [[Environment:Unslothai_Unsloth_VLLM]]
- [[Environment:Unslothai_Unsloth_Vision]]
