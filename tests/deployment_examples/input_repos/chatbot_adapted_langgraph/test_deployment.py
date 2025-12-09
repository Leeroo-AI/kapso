"""Test the deployed LangGraph agent."""
import requests
import json

BASE_URL = "http://127.0.0.1:8123"

# Create a thread
response = requests.post(f"{BASE_URL}/threads", json={})
print(f"Response status: {response.status_code}")
print(f"Response text: {response.text}")
thread = response.json()
thread_id = thread["thread_id"]
print(f"Created thread: {thread_id}")

# Send a message
input_data = {
    "assistant_id": "agent",
    "input": {
        "messages": [{"role": "user", "content": "Hello! What's your name?"}]
    }
}

response = requests.post(
    f"{BASE_URL}/threads/{thread_id}/runs",
    json=input_data
)

run = response.json()
run_id = run["run_id"]
print(f"Run created: {run_id}")

# Wait for run to complete
import time
max_wait = 30
waited = 0
while waited < max_wait:
    response = requests.get(f"{BASE_URL}/threads/{thread_id}/runs/{run_id}")
    run_status = response.json()
    status = run_status["status"]
    print(f"Run status: {status}")

    if status in ["success", "error", "cancelled"]:
        break

    time.sleep(2)
    waited += 2

# Get the thread state to see the messages
response = requests.get(f"{BASE_URL}/threads/{thread_id}/state")
state = response.json()
print(f"\nFinal state:")
print(json.dumps(state, indent=2, default=str))
