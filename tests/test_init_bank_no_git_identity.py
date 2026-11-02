import subprocess

import pytest

from kapso.learning.update_frame import UpdateFrame, init_bank


@pytest.fixture
def no_git_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("EMAIL", "")
    # Git would otherwise guess an identity from the hostname on some machines.
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "user.useConfigOnly")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "true")
    for var in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME",
                "GIT_COMMITTER_EMAIL", "GIT_CONFIG_GLOBAL"):
        monkeypatch.delenv(var, raising=False)


def test_init_bank_without_git_identity(tmp_path, no_git_identity):
    home = tmp_path / "bank.git"
    init_bank(str(home))
    head = subprocess.run(["git", "-C", str(home), "rev-parse", "main"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert head
    assert [path.name for path in tmp_path.iterdir() if path.name != "home"] == ["bank.git"]


def test_a_learner_run_commits_without_git_identity(tmp_path, no_git_identity):
    home = tmp_path / "bank.git"
    init_bank(str(home))
    run_dir = tmp_path / "run"
    (run_dir / "work").mkdir(parents=True)
    (run_dir / "work" / "headline.md").write_text("learned one thing\n")
    subprocess.run(["git", "clone", str(home), str(run_dir / "bank")], check=True)
    (run_dir / "bank" / "log.md").write_text("# Learner-run journal\n- lr-1\n")
    UpdateFrame._commit(None, run_dir, "lr-1", [])
    subject = subprocess.run(["git", "-C", str(home), "log", "-1", "--format=%s", "lr-1"],
                             capture_output=True, text=True, check=True).stdout.strip()
    assert subject == "learned one thing"


def test_failed_init_leaves_nothing_and_a_retry_succeeds(tmp_path, no_git_identity, monkeypatch):
    home = tmp_path / "bank.git"
    real = subprocess.run

    def fail_push(cmd, *a, **k):
        if "push" in cmd:
            raise subprocess.CalledProcessError(1, cmd)
        return real(cmd, *a, **k)

    monkeypatch.setattr(subprocess, "run", fail_push)
    with pytest.raises(subprocess.CalledProcessError):
        init_bank(str(home))
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setattr(subprocess, "run", real)
    init_bank(str(home))
    assert home.exists()
