# Local Deployment Strategy
#
# Runs solutions directly as Python processes.
# Best for development, testing, and simple use cases.

from src.deployment.strategies.local.runner import LocalRunner

__all__ = ["LocalRunner"]

