{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth GitHub|https://github.com/unslothai/unsloth]]
* [[source::Doc|Unsloth Docker Guide|https://docs.unsloth.ai/]]
* [[source::Doc|NVIDIA Container Toolkit|https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::DevOps]], [[domain::LLMs]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Containerized Unsloth environment using Docker with NVIDIA GPU support for reproducible fine-tuning deployments.

=== Description ===
This environment provides a fully containerized setup for running Unsloth fine-tuning workflows. Using Docker with NVIDIA Container Toolkit ensures reproducibility across different machines and simplifies deployment to cloud instances. The official Unsloth Docker image includes all dependencies pre-configured, with Jupyter Lab for interactive development.

=== Usage ===
Use this environment when you need **reproducible deployments**, **cloud server setups** (AWS, GCP, Azure), or **team collaboration** with consistent environments. Ideal for production workflows, automated training pipelines, and situations where environment consistency is critical.

== System Requirements ==
{| class="wikitable"
! Category !! Requirement !! Notes
|-
|| OS || Linux with Docker || Ubuntu 20.04/22.04 recommended
|-
|| Docker || Docker Engine 20.10+ || With Docker Compose support
|-
|| NVIDIA Driver || 525+ || Must support CUDA 11.8+
|-
|| NVIDIA Container Toolkit || Latest || For GPU passthrough to containers
|-
|| Hardware || NVIDIA GPU || Minimum 8GB VRAM
|-
|| Disk || 100GB+ SSD || For Docker images and model storage
|}

== Dependencies ==
=== Docker Setup ===

<syntaxhighlight lang="bash">
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
</syntaxhighlight>

=== Running Unsloth Container ===

<syntaxhighlight lang="bash">
# Pull and run official Unsloth Docker image
docker pull unsloth/unsloth:latest

# Run with GPU support and Jupyter Lab
docker run --gpus all -it -p 8888:8888 unsloth/unsloth:latest

# Access Jupyter Lab at http://localhost:8888
</syntaxhighlight>

=== Docker Compose (Recommended) ===

<syntaxhighlight lang="yaml">
version: '3.8'
services:
  unsloth:
    image: unsloth/unsloth:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - HF_TOKEN=${HF_TOKEN}
    ports:
      - "8888:8888"
    volumes:
      - ./data:/workspace/data
      - ./outputs:/workspace/outputs
</syntaxhighlight>

== Credentials ==
The following environment variables should be set:
* `HF_TOKEN`: HuggingFace API token (pass via `-e HF_TOKEN=...` or `.env` file)
* `WANDB_API_KEY`: Weights & Biases API key (optional)

== Related Pages ==
=== Required By ===
* [[required_by::Implementation:Unsloth_FastLanguageModel]]
* [[required_by::Implementation:Unsloth_FastModel]]
* [[required_by::Implementation:TRL_SFTTrainer]]
* [[required_by::Implementation:TRL_DPOTrainer]]
* [[required_by::Implementation:TRL_GRPOTrainer]]

