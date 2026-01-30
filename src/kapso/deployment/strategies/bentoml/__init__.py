# BentoML Deployment Strategy
#
# Deploys to BentoCloud for production ML services.
# Best for production ML APIs with batching, monitoring, and auto-scaling.

from kapso.deployment.strategies.bentoml.runner import BentoMLRunner

__all__ = ["BentoMLRunner"]

