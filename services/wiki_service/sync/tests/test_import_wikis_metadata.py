"""Re-importing a page must not stack another PageInfo and Category.

import_wikis.sh wrapped every file in {{PageInfo}} and [[Category]]. Once
a wiki edit was synced back, the local file already held both, so every
FORCE_REIMPORT / start.sh --fresh added one more copy of each to the
page. The sync path (transforms.add_page_metadata) already skipped tags
that were present; the importer now does the same.
"""

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "mediawiki" / "import_wikis.sh"
PAGEINFO = "{{PageInfo|type=Heuristic|title=Some_Page}}"
CATEGORY = "[[Category:Heuristics]]"

# Stands in for MediaWiki's PHP: records the text edit.php would save.
FAKE_PHP = """#!/bin/sh
case "$*" in
  *edit.php*) cat > "$CAPTURE"; echo "Saving... done" ;;
esac
"""


def find_bash4():
    """The importer uses associative arrays, which need bash 4 or newer."""
    bash = shutil.which("bash")
    if not bash:
        return None
    version = subprocess.run(
        [bash, "-c", "echo ${BASH_VERSINFO[0]}"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return bash if version.isdigit() and int(version) >= 4 else None


BASH = find_bash4()
pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash >= 4")


def import_page(tmp_path, text):
    """Run import_wikis.sh on one heuristic page; return what it saved."""
    wikis = tmp_path / "wikis"
    (wikis / "heuristics").mkdir(parents=True)
    (wikis / "heuristics" / "Some_Page.md").write_text(text)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    php = bin_dir / "php"
    php.write_text(FAKE_PHP)
    php.chmod(php.stat().st_mode | stat.S_IEXEC)

    capture = tmp_path / "saved.txt"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "WIKIS_DIR": str(wikis),
        "CAPTURE": str(capture),
    }
    command = [BASH, str(SCRIPT)]
    subprocess.run(command, env=env, check=True, capture_output=True)
    return capture.read_text()


def test_plain_page_gets_one_pageinfo_and_category(tmp_path):
    saved = import_page(tmp_path, "== Overview ==\nWritten on disk.\n")

    assert saved.startswith(PAGEINFO + "\n")
    assert saved.rstrip().endswith(CATEGORY)
    assert saved.count("{{PageInfo|") == 1
    assert saved.count(CATEGORY) == 1


def test_page_synced_back_from_wiki_is_not_wrapped_again(tmp_path):
    # What a file looks like after a wiki edit was synced to disk.
    synced = f"{PAGEINFO}\n== Overview ==\nEdited in wiki.\n\n{CATEGORY}"

    saved = import_page(tmp_path, synced)

    assert saved.count("{{PageInfo|") == 1
    assert saved.count(CATEGORY) == 1
    assert "Edited in wiki." in saved
