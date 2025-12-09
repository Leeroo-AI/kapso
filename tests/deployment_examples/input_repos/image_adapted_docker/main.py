"""
Main entry point for image processing solution.
"""

import json
import sys
from processor import predict


if __name__ == "__main__":
    input_data = json.loads(sys.stdin.read()) if not sys.stdin.isatty() else {}
    result = predict(input_data)
    print(json.dumps(result))
