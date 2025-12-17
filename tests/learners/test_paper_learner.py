from src.knowledge.learners.paper_learner import PaperLearner


def test_paper_learner():
    learner = PaperLearner(params={})
    
    test_data = {"url": "https://arxiv.org/pdf/1706.03762"}
    chunks = learner.learn(test_data)
    print(chunks)

if __name__ == "__main__":
    test_paper_learner()
