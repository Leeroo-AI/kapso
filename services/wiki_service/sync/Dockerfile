FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY sync/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the tools directory (for mw_client)
COPY tools/ /app/tools/

# Copy the sync service
COPY sync/ /app/sync/

# Create directories for state and conflicts
RUN mkdir -p /state /wikis/_conflicts

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python", "-m", "sync.service"]
