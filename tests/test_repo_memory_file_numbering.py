import json
import re
from pathlib import Path
from typing import Any, Callable, List

import pytest

from kapso.execution.memories.repo_memory.builders import (
    _read_numbered_text,
    infer_repo_model_update,
    infer_repo_model_with_retry,
)
from tests.test_repo_memory_resilience import VALID_MODEL, VALID_UPDATE, SequenceLLM

BIG_FILE = "big_module.py"
BIG_FILE_LINES = 3000  # ~45k characters, longer than the old read caps
NUMBERED_ROW = re.compile(r"^\s*(\d+)\| line_(\d{4}) = \d+$", re.M)


def _big_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    repo_root.joinpath(BIG_FILE).write_text(
        "".join(f"line_{n:04d} = {n}\n" for n in range(1, BIG_FILE_LINES + 1))
    )
    return repo_root


def _infer_initial(llm: SequenceLLM, repo_root: Path) -> None:
    infer_repo_model_with_retry(
        llm=llm,
        repo_root=str(repo_root),
        repo_map={"files": [BIG_FILE], "key_files": [BIG_FILE], "entrypoints": []},
    )


def _infer_update(llm: SequenceLLM, repo_root: Path) -> None:
    infer_repo_model_update(
        llm=llm,
        repo_root=str(repo_root),
        repo_map={"files": [BIG_FILE]},
        previous_model={"summary": "Old", "sections": {}},
        diff_summary="big_module.py changed",
        changed_files=[BIG_FILE],
    )


@pytest.mark.parametrize(
    "infer, responses",
    [
        (_infer_initial, ['{"files_to_read": []}', VALID_MODEL]),
        (_infer_update, [VALID_UPDATE]),
    ],
    ids=["initial", "update"],
)
def test_large_file_reaches_the_prompt_whole_with_real_line_numbers(
    tmp_path: Path,
    infer: Callable[[SequenceLLM, Path], None],
    responses: List[Any],
) -> None:
    llm = SequenceLLM(responses)

    infer(llm, _big_repo(tmp_path))

    prompt = llm.calls[-1]["messages"][0]["content"]
    rows = [(int(shown), int(real)) for shown, real in NUMBERED_ROW.findall(prompt)]
    assert [(shown, real) for shown, real in rows if shown != real] == []
    assert [real for _, real in rows] == list(range(1, BIG_FILE_LINES + 1))
    assert "omitted" not in prompt


def test_numbers_only_the_files_real_lines(tmp_path: Path) -> None:
    path = tmp_path / "small.txt"
    path.write_text("alpha\nbeta\n")

    assert _read_numbered_text(str(path)) == "1| alpha\n2| beta"


def test_unreadable_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _read_numbered_text(str(tmp_path / "missing.txt"))


def _cites(path: str, statement: str) -> dict:
    return {
        "title": path,
        "one_liner": path,
        "claims": [{"statement": statement, "evidence": [{"path": path, "line": 1}]}],
    }


# 1500 fits one ~1.3k-character file but not two (a file larger than the
# budget on its own is withheld by name, the test below).
def test_update_windows_changed_files_across_calls_within_the_budget(tmp_path: Path) -> None:
    budget = 1500
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    names = ["a.py", "b.py", "c.py"]
    for name in names:
        repo_root.joinpath(name).write_text(f"{name} = 1\n" * 100)
    previous_model = {
        "summary": "Old",
        "sections": {
            "opt.a": _cites("a.py", "old a"),
            "opt.c": _cites("c.py", "old c"),
        },
    }
    # Each batch refreshes the section citing its file; b.py has none.
    patches = [
        {
            "summary": "S1",
            "sections": {"opt.a": _cites("a.py", "new a")},
            "removed_sections": [],
        },
        {"summary": "S2", "sections": {}, "removed_sections": []},
        {
            "summary": "S3",
            "sections": {"opt.c": _cites("c.py", "new c")},
            "removed_sections": [],
        },
    ]
    llm = SequenceLLM([json.dumps(patch) for patch in patches])

    result = infer_repo_model_update(
        llm=llm,
        repo_root=str(repo_root),
        repo_map={"files": names},
        previous_model=previous_model,
        diff_summary="a.py, b.py and c.py changed",
        changed_files=names,
        max_payload_chars=budget,
    )

    prompts = [call["messages"][0]["content"] for call in llm.calls]
    assert len(prompts) == 3
    for prompt, name in zip(prompts, names):
        payload = prompt.split("CHANGED FILE CONTENTS (with line numbers")[1]
        assert re.findall(r"=== FILE: (\S+) ===", payload) == [name]
        assert payload.count(f"{name} = 1") == 100
    # Each call patches the model the previous call produced.
    assert '"new a"' in prompts[1] and '"old a"' not in prompts[1]
    assert result["summary"] == "S3"
    assert result["sections"]["opt.a"]["claims"][0]["statement"] == "new a"
    assert result["sections"]["opt.c"]["claims"][0]["statement"] == "new c"


def test_update_lists_a_deleted_file_as_deleted(tmp_path: Path) -> None:
    # A changed file missing from a call's contents is another call's to
    # refresh, so the call that carries a deleted file has to say it is gone.
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    llm = SequenceLLM(
        [json.dumps({"summary": "S", "sections": {}, "removed_sections": ["opt.gone"]})]
    )

    infer_repo_model_update(
        llm=llm,
        repo_root=str(repo_root),
        repo_map={"files": []},
        previous_model={
            "summary": "Old",
            "sections": {"opt.gone": _cites("gone.py", "gone")},
        },
        diff_summary="gone.py | 100 ----",
        changed_files=["gone.py"],
    )

    prompt = llm.calls[0]["messages"][0]["content"]
    payload = prompt.split("CHANGED FILE CONTENTS (with line numbers")[1]
    assert "=== FILE: gone.py (deleted) ===" in payload


def test_update_withholds_binary_and_data_sized_files_by_name(tmp_path: Path) -> None:
    # The gate's campaigns commit their datasets; a 1 MB csv in a change sent
    # whole put the update past Codex's input limit on every experiment. Such
    # a file, or a binary, is listed by name and size and never shown, so the
    # call stays within its budget and the model is told what it did not see.
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    repo_root.joinpath("code.py").write_text("x = 1\n" * 10)
    repo_root.joinpath("train.csv").write_text("1,2,3\n" * 400)
    repo_root.joinpath("model.bin").write_bytes(b"\x00\x01" * 50)
    previous_model = {
        "summary": "Old",
        "sections": {
            "opt.code": _cites("code.py", "old code"),
            "opt.data": _cites("train.csv", "old data"),
        },
    }
    llm = SequenceLLM(
        [json.dumps({"summary": "S", "sections": {"opt.code": _cites("code.py", "new code")}, "removed_sections": []})]
    )

    result = infer_repo_model_update(
        llm=llm,
        repo_root=str(repo_root),
        repo_map={"files": ["code.py", "train.csv", "model.bin"]},
        previous_model=previous_model,
        diff_summary="code.py, train.csv and model.bin changed",
        changed_files=["train.csv", "model.bin", "code.py"],
        max_payload_chars=1500,
    )

    assert len(llm.calls) == 1
    payload = llm.calls[0]["messages"][0]["content"].split("CHANGED FILE CONTENTS (with line numbers")[1]
    assert re.findall(r"=== FILE: (\S+) ===", payload) == ["code.py"]
    assert "=== FILE: train.csv (2400 bytes, not shown) ===" in payload
    assert "=== FILE: model.bin (100 bytes, not shown) ===" in payload
    assert "1,2,3" not in payload and "\x00" not in payload
    assert len(payload) < 1500
    # The section citing a withheld file is kept as it is, not demanded back.
    assert result["sections"]["opt.data"]["claims"][0]["statement"] == "old data"
    assert result["sections"]["opt.code"]["claims"][0]["statement"] == "new code"


def test_initial_inference_keeps_the_updates_budget_withholding_data_and_windowing_the_rest(tmp_path: Path) -> None:
    # The baseline memory of the release gate's S1 campaign sent the planner's
    # files whole, a 1 MB dataset among them, and Codex refused the call
    # (1,400,956 characters against its 1,048,576 limit). The initial
    # inference keeps the update's budget: a binary or over-budget file is
    # listed by name and size, never shown, and the files past the first
    # budget's worth reach the model in further calls, each patching the model.
    budget = 1500
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    for name in ("a.py", "b.py"):
        repo_root.joinpath(name).write_text(f"{name} = 1\n" * 100)   # ~1.3k characters each: one per call
    repo_root.joinpath("train.csv").write_text("1,2,3\n" * 400)      # 2400 bytes: over the budget on its own
    repo_root.joinpath("model.bin").write_bytes(b"\x00\x01" * 50)
    files = ["a.py", "train.csv", "b.py", "model.bin"]
    plan = json.dumps({"files_to_read": [{"path": path, "why": "x"} for path in files]})
    llm = SequenceLLM([plan, VALID_MODEL, json.dumps({"summary": "S2", "sections": {}, "removed_sections": []})])

    result = infer_repo_model_with_retry(
        llm=llm,
        repo_root=str(repo_root),
        repo_map={"files": files, "key_files": [], "entrypoints": []},
        max_payload_chars=budget,
    )

    assert len(llm.calls) == 3, "the plan, the first budget's worth, the rest"
    first = llm.calls[1]["messages"][0]["content"].split("FILE CONTENTS (with line numbers")[1]
    assert re.findall(r"=== FILE: (\S+) ===", first) == ["a.py"]
    assert "=== FILE: train.csv (2400 bytes, not shown) ===" in first
    assert "=== FILE: model.bin (100 bytes, not shown) ===" in first
    assert "1,2,3" not in first and "\x00" not in first
    second = llm.calls[2]["messages"][0]["content"].split("CHANGED FILE CONTENTS (with line numbers")[1]
    assert re.findall(r"=== FILE: (\S+) ===", second) == ["b.py"] and "not shown" not in second
    assert result["summary"] == "S2"
