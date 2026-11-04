"""The `.env` precedence a campaign runs with: the shell's variables win
over every `.env`, and a later `.env` load replaces what an earlier one
set, which python-dotenv's load_dotenv cannot do in either mode."""

import os

import pytest

from kapso.core import env_file
from kapso.core.env_file import load_env_file

NAME = "KAPSO_TEST_ENV_FILE_KEY"


@pytest.fixture(autouse=True)
def fresh_process(monkeypatch):
    monkeypatch.delenv(NAME, raising=False)
    monkeypatch.setattr(env_file, "_SET_BY_ENV_FILE", {})


def _env(path, text):
    path.write_text(text)
    return str(path)


def test_a_later_env_replaces_what_an_earlier_one_set(tmp_path):
    load_env_file(_env(tmp_path / "found-here.env", f"{NAME}=here\n"))
    load_env_file(_env(tmp_path / "campaign.env", f"{NAME}=campaign\n"))
    assert os.environ[NAME] == "campaign"


def test_the_same_file_reread_after_an_edit_takes_the_new_value(tmp_path):
    path = _env(tmp_path / ".env", f"{NAME}=placeholder\n")
    load_env_file(path)
    _env(tmp_path / ".env", f"{NAME}=real\n")
    load_env_file(path)
    assert os.environ[NAME] == "real"


def test_the_shell_and_the_callers_own_code_keep_their_values(tmp_path, monkeypatch):
    monkeypatch.setenv(NAME, "shell")
    load_env_file(_env(tmp_path / ".env", f"{NAME}=file\n"))
    assert os.environ[NAME] == "shell"

    monkeypatch.delenv(NAME)
    load_env_file(str(tmp_path / ".env"))
    os.environ[NAME] = "set by the caller"
    load_env_file(_env(tmp_path / "campaign.env", f"{NAME}=campaign\n"))
    assert os.environ[NAME] == "set by the caller"


def test_no_env_and_a_name_without_a_value_load_nothing(tmp_path):
    load_env_file("")
    load_env_file(str(tmp_path / "missing.env"))
    load_env_file(_env(tmp_path / ".env", f"{NAME}\n"))
    assert NAME not in os.environ
