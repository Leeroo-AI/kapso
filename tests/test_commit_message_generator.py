# The commit-message prompt carries the whole diff and the whole solution
# summary (Rule 6): a clipped diff yields a message about its first files only.

from kapso.execution.coding_agents.commit_message_generator import CommitMessageGenerator


class RecordingInference:
    def __init__(self):
        self.prompts = []

    def llm_completion(self, messages, role):
        assert role == "commit_message"
        self.prompts.append(messages[0]["content"])
        return "feat: add the thing"


def test_prompt_carries_the_whole_diff_and_summary():
    inference = RecordingInference()
    diff = "diff --git a/first.py b/first.py\n" + "+x = 1\n" * 5000
    diff += "diff --git a/last_file.py b/last_file.py\n+marker_at_the_very_end\n"
    summary = "s" * 2000 + "summary_tail_marker"

    message = CommitMessageGenerator(llm=inference).generate(diff, solution_summary=summary)

    assert message == "feat: add the thing"
    (prompt,) = inference.prompts
    assert diff in prompt
    assert summary in prompt
