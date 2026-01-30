# CLI Entry Point
#
# Allows running the knowledge learners as a module:
#   python -m src.knowledge_base.learners https://github.com/user/repo

from src.knowledge_base.learners.knowledge_learner_pipeline import main

if __name__ == "__main__":
    main()

