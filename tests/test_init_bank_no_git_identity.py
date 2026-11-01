import subprocess

import pytest

from kapso.learning.update_frame import init_bank


@pytest.fixture
def no_git_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("EMAIL", "")
    for var in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME",
                "GIT_COMMITTER_EMAIL", "GIT_CONFIG_GLOBAL"):
        monkeypatch.delenv(var, raising=False)


def test_init_bank_without_git_identity(tmp_path, no_git_identity):
    home = tmp_path / "bank.git"
    init_bank(str(home))
    head = subprocess.run(["git", "-C", str(home), "rev-parse", "main"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert head
    assert not (tmp_path / ".seed-bank.git").exists()


def test_failed_init_leaves_no_half_made_home(tmp_path, monkeypatch):
    home = tmp_path / "bank.git"
    real = subprocess.run

    def fail_push(cmd, *a, **k):
        if "push" in cmd:
            raise subprocess.CalledProcessError(1, cmd)
        return real(cmd, *a, **k)

    monkeypatch.setattr(subprocess, "run", fail_push)
    with pytest.raises(subprocess.CalledProcessError):
        init_bank(str(home))
    assert not home.exists()
    assert not (tmp_path / ".seed-bank.git").exists()
