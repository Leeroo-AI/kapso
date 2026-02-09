#!/bin/bash

# Stop Backend Search Service

cd "$(dirname "$0")"

if [ -f backend_search.pid ]; then
    PID=$(cat backend_search.pid)
    if kill -0 $PID 2>/dev/null; then
        echo "Stopping Backend Search (PID $PID)..."
        kill $PID
        rm backend_search.pid
        echo "Stopped."
    else
        echo "Process $PID not running, cleaning up pid file."
        rm backend_search.pid
    fi
else
    echo "No pid file found. Service may not be running."
    # Try to find and kill by name
    pkill -f "uvicorn backend_search.app:app" && echo "Killed by process name." || echo "No process found."
fi
