# CLI Entry Point
#
# Allows running the knowledge learners as a module:
#   python -m src.knowledge.learners https://github.com/user/repo

from src.knowledge.learners.knowledge_learner_pipeline import main

if __name__ == "__main__":
    main()

