"""One release number (Rule 1).

kapso.__version__ sat at 0.1.0 through five releases because nothing tied
it to pyproject.toml. This makes the drift a suite failure: bump one, the
test names the other.
"""

import re
from pathlib import Path

import kapso

_PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"


def test_package_version_matches_pyproject():
    match = re.search(r'^version = "([^"]+)"$', _PYPROJECT.read_text(), re.M)
    assert match, "pyproject.toml has no version line"
    assert kapso.__version__ == match.group(1)
