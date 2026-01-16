# Local Deployment Instructions

Local deployment runs the solution directly as a Python process.
Best for development, testing, and simple use cases.

## DEPLOY COMMAND

```bash
python -c "from main import predict; print(predict({'test': True}))"
```

Run this command to verify the local deployment works. If it fails, debug and fix the error.

## RUN INTERFACE
- type: function
- module: main
- callable: predict

After successful deployment, output this JSON:
```
<run_interface>{"type": "function", "module": "main", "callable": "predict"}</run_interface>
```

## CRITICAL: YOU MUST VERIFY THE CODE WORKS

**Do NOT just create files. You MUST test that the predict function works.**

After creating main.py:
1. Run: `python -c "from main import predict; print(predict({'test': True}))"`
2. Verify it returns a valid response without errors
3. If there are import errors or runtime errors, fix them

**If testing fails, debug the error and fix it. Do not give up.**

## Required Structure

```
solution/
├── main.py           # Entry point with predict() function
├── requirements.txt  # Dependencies
└── ...               # Other modules
```

## Entry Point (main.py)

Create or ensure `main.py` has this structure:

```python
"""
Main entry point for local deployment.
"""

def predict(inputs: dict) -> dict:
    """
    Process inputs and return results.
    
    Args:
        inputs: Dictionary with input data
        
    Returns:
        Dictionary with results
    """
    # Import your modules here (lazy loading)
    # from my_model import MyModel
    
    # Process inputs
    result = process_data(inputs)
    
    return {"status": "success", "output": result}


def process_data(inputs: dict):
    """Your actual processing logic."""
    # Implement your logic here
    return inputs


# CLI support
if __name__ == "__main__":
    import json
    import sys
    
    # Read from stdin or use empty dict
    if sys.stdin.isatty():
        input_data = {}
    else:
        input_data = json.loads(sys.stdin.read())
    
    result = predict(input_data)
    print(json.dumps(result, indent=2))
```

## Testing Locally

```bash
# Test with empty input
echo '{}' | python main.py

# Test with sample input
echo '{"text": "hello world"}' | python main.py

# Interactive test
python -c "from main import predict; print(predict({'test': True}))"
```

## Notes

- Keep `main.py` lightweight - import heavy modules inside functions
- Avoid global state that could cause issues with multiple calls
- Handle missing inputs gracefully with defaults

