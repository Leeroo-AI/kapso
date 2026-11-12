# The pull-request deployment: one branch and one pull request per unit of the manifest, each holding only
# that unit's files as they are in the workspace. A unit that needs another unit's change to build is stacked
# on that unit's branch. Branches are made in worktrees of their own, so the workspace is never switched away
# from what was delivered. Every branch is checked on its own with the task's acceptance check before anything
# is pushed; a branch that fails alone stops the deployment with the reason. Titles and bodies are rendered
# here from the proofs, the verification report and each branch's check, never read from a file a session
# typed. Run from the workspace. The base branch is fetched, never pushed. Prints KAPSO_ENDPOINT and the
# KAPSO_VALUE_ lines the result file carries.
set -euo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/common.sh"
SCRIPTS="$(dirname "$0")"

if [ ! -f "$MANIFEST" ]; then
    echo "no $MANIFEST: write it first (the skill's step 2)" >&2
    exit 1
fi
if ! command -v gh >/dev/null 2>&1 || ! gh auth status >/dev/null 2>&1; then
    echo "the GitHub CLI (gh) is not installed or not logged in on this machine" >&2
    exit 1
fi
REPO="$(repo_slug)"
REMOTE="https://github.com/$REPO.git"
git fetch -q "$REMOTE" "$BASE"
BASE_COMMIT="$(git rev-parse FETCH_HEAD)"
gh label create "$LABEL" --repo "$REPO" --color 5319e7 --description "Opened by a Kapso deployment" >/dev/null 2>&1 || true

python3 "$SCRIPTS/open_pull_requests.py" prepare "$MANIFEST" "$PREPARED" "$BASE" "$BASE_COMMIT" "kapso/$KAPSO_DEPLOYMENT_NAME"
python3 "$SCRIPTS/verify_branches.py" "$PREPARED" "$BRANCH_CHECKS"
VERIFICATION="${KAPSO_VERIFICATION_DIR:-}/evidence_check.json"
python3 "$SCRIPTS/open_pull_requests.py" open "$PREPARED" "$OPENED" "$REPO" "$REMOTE" "$BASE" "$LABEL" "$BRANCH_CHECKS" "$VERIFICATION"

NUMBERS="$(pr_numbers | paste -sd, -)"
COUNT="$(pr_numbers | wc -l)"
if [ "$COUNT" -eq 1 ]; then
    ENDPOINT="https://github.com/$REPO/pull/$NUMBERS"
else
    ENDPOINT="https://github.com/$REPO/pulls?q=is%3Apr+label%3A$LABEL"
fi
echo "KAPSO_ENDPOINT=$ENDPOINT"
echo "KAPSO_VALUE_REPO=$REPO"
echo "KAPSO_VALUE_BASE=$BASE"
echo "KAPSO_VALUE_LABEL=$LABEL"
echo "KAPSO_VALUE_PR_NUMBERS=$NUMBERS"
