# Modal Deployment Strategy
#
# Deploys to Modal.com serverless infrastructure.
# Best for GPU workloads, auto-scaling, and serverless execution.

from src.deployment.strategies.modal.runner import ModalRunner

__all__ = ["ModalRunner"]

