# Modal Deployment Strategy
#
# Deploys to Modal.com serverless infrastructure.
# Best for GPU workloads, auto-scaling, and serverless execution.

from kapso.deployment.strategies.modal.runner import ModalRunner

__all__ = ["ModalRunner"]

