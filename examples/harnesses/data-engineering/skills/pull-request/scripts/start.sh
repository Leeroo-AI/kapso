# Start the deployment again: reopen every pull request of it that is closed, whether the deployment's stop
# closed it or the host did because the branch it was stacked on was deleted from the command line. A stacked
# one ends on the branch of the nearest unit under it that is not merged yet, or on the base branch (see
# reopen.py). A merged one stays merged. Run from the workspace.
set -euo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/common.sh"
if [ ! -f "$OPENED" ]; then
    echo "no $OPENED: this workspace holds no record of what the deployment opened" >&2
    exit 1
fi
python3 "$(dirname "$0")/reopen.py" "$OPENED" "$(repo_slug)" "$BASE"
