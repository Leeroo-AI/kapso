"""
Deployment smoke test for the deployed LangGraph agent.

This is an **optional** test:
- It requires a running deployment server (default: http://127.0.0.1:8123)
- It will be skipped unless you explicitly opt in.

Enable with:
  PRAXIUM_RUN_DEPLOYMENT_TESTS=1 PYTHONPATH=. pytest -q <this_file>
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict

import pytest
import requests


RUN_DEPLOYMENT_TESTS = os.getenv("PRAXIUM_RUN_DEPLOYMENT_TESTS") == "1"
BASE_URL = os.getenv("PRAXIUM_LANGGRAPH_BASE_URL", "http://127.0.0.1:8123").rstrip("/")


@pytest.mark.skipif(not RUN_DEPLOYMENT_TESTS, reason="Set PRAXIUM_RUN_DEPLOYMENT_TESTS=1 to enable deployment smoke tests.")
def test_langgraph_deployment_smoke() -> None:
    # Create a thread
    try:
        response = requests.post(f"{BASE_URL}/threads", json={}, timeout=2)
    except requests.RequestException as e:
        pytest.skip(f"Deployment server not reachable at {BASE_URL}: {e}")

    assert response.ok, f"Failed to create thread: {response.status_code} {response.text}"
    thread: Dict[str, Any] = response.json()
    thread_id = thread["thread_id"]

    # Send a message
    input_data = {
        "assistant_id": "agent",
        "input": {"messages": [{"role": "user", "content": "Hello! What's your name?"}]},
    }

    response = requests.post(
        f"{BASE_URL}/threads/{thread_id}/runs",
        json=input_data,
        timeout=5,
    )
    assert response.ok, f"Failed to create run: {response.status_code} {response.text}"
    run: Dict[str, Any] = response.json()
    run_id = run["run_id"]

    # Wait for run to complete
    deadline = time.time() + 30
    status = None
    while time.time() < deadline:
        response = requests.get(f"{BASE_URL}/threads/{thread_id}/runs/{run_id}", timeout=5)
        assert response.ok, f"Failed to read run status: {response.status_code} {response.text}"
        run_status: Dict[str, Any] = response.json()
        status = run_status.get("status")
        if status in ["success", "error", "cancelled"]:
            break
        time.sleep(2)

    assert status == "success", f"Run did not succeed (status={status})"

    # Get the thread state to see the messages
    response = requests.get(f"{BASE_URL}/threads/{thread_id}/state", timeout=5)
    assert response.ok, f"Failed to read thread state: {response.status_code} {response.text}"
