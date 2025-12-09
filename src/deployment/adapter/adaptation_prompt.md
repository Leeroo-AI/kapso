# Deployment Adaptation Task

## Original Goal
{goal}

## Target Deployment
- **Strategy**: {strategy}
- **Provider**: {provider}
- **Interface**: {interface}
- **Resources**: {resources}

---

## CRITICAL: COMPLETE DEPLOYMENT REQUIRED

**Your task is NOT just to create deployment files. You MUST actually deploy the solution and verify it works.**

1. Create all necessary deployment files
2. Run the DEPLOY COMMAND (specified in target instructions below)
3. Verify the deployment is successful
4. Report the deployment endpoint/URL

**DO NOT stop until the deployment is live and working. If deployment fails, debug and retry.**

---

## Base Requirements (Apply to ALL deployments)

### Entry Point

Every deployable solution MUST have a main entry point with a `predict` function:

```python
# main.py
def predict(inputs: dict) -> dict:
    """
    Main entry point for predictions/processing.
    
    Args:
        inputs: Input dictionary with data to process
        
    Returns:
        Dictionary with results
    """
    try:
        # Your logic here
        result = process(inputs)
        return {"status": "success", "output": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}

# For CLI usage
if __name__ == "__main__":
    import json
    import sys
    
    input_data = json.loads(sys.stdin.read()) if not sys.stdin.isatty() else {}
    result = predict(input_data)
    print(json.dumps(result))
```

### Dependency Management

- ALWAYS have a `requirements.txt` with all dependencies
- Include version pins for reproducibility (e.g., `torch==2.0.0`)
- Include ALL dependencies, including transitive ones

### Configuration

- Use environment variables for secrets (API keys, passwords)
- NEVER hardcode sensitive values in code
- Use `os.environ.get("VAR_NAME")` to read env vars

### Testing

After adaptation, the following should work:
1. `pip install -r requirements.txt` - installs all deps
2. `python -c "from main import predict"` - imports without error
3. `echo '{}' | python main.py` - runs without error

---

## Target Instructions for {strategy}

{target_instructions}

---

## Your Task

Adapt this repository for {strategy} deployment following the instructions above.

Make the MINIMAL necessary changes. Do not over-engineer.
Ensure the main prediction/processing logic is accessible via a `predict()` function.

---

## Deployment Output Format

After successful deployment, you MUST output the following in your final response:

### 1. Run Interface (REQUIRED)

Output the run interface configuration as JSON in XML tags:

```
<run_interface>{"type": "function", "module": "main", "callable": "predict"}</run_interface>
```

The JSON should include:
- `type`: Interface type (e.g., "function", "http", "modal", "bentocloud", "langgraph")
- `module`: Python module name (usually "main")
- `callable`: Function/method name (usually "predict")
- Any strategy-specific fields (e.g., `path`, `endpoint`, `app_name`)

### 2. Endpoint URL (if applicable)

If your deployment creates an HTTP endpoint, also output:

```
<endpoint_url>https://your-deployed-endpoint.com</endpoint_url>
```

This allows the system to automatically extract and configure the runner.
