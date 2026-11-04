"""Loading a `.env` into the process environment, with the precedence a
campaign needs: a variable the shell set wins over every `.env`, and a
later `.env` load replaces what an earlier one set.

python-dotenv cannot express that. Its `load_dotenv` either never
overrides (so the first `.env` read wins for good) or always overrides
(so a `.env` beats the shell). Kapso reads a `.env` at import, from the
directory it runs in, before it knows whether the run is a resume; a
resume then reads the campaign's own `.env`, which the inbox prompt names
as the file a fix goes in, and that one has to win over the first. The
loads here remember the value each one set, so a later load can tell a
variable an earlier `.env` set (replaced) from one the shell or the
caller's own code set (kept).
"""

import os
from typing import Dict

from dotenv import dotenv_values

# Every variable a `.env` load put in the environment, with the value it
# put there. A variable whose current value differs was set by someone
# else since, and is theirs.
_SET_BY_ENV_FILE: Dict[str, str] = {}


def load_env_file(path: str) -> None:
    """Load the `.env` at ``path`` (an empty path or a missing file loads
    nothing). A variable set before any `.env` load, or changed since by
    anything else, keeps its value; one a `.env` load set takes the value
    from this file."""
    if not path:
        return
    for name, value in dotenv_values(path).items():
        if value is None:
            continue
        if name in os.environ and _SET_BY_ENV_FILE.get(name) != os.environ[name]:
            continue
        os.environ[name] = value
        _SET_BY_ENV_FILE[name] = value
