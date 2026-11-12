# Stop the deployment: close every pull request it opened that is still open, with a comment. Branches stay,
# so start.sh can reopen them; a merged pull request is left as it is.
set -euo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/common.sh"
REPO="$(repo_slug)"
for number in $(pr_numbers); do
    state="$(gh pr view "$number" --repo "$REPO" --json state --jq .state)"
    if [ "$state" = "OPEN" ]; then
        gh pr close "$number" --repo "$REPO" --comment "Closed by the deployment's stop; the branch is kept and the pull request can be reopened." >/dev/null
        echo "closed #$number"
    else
        echo "#$number is $state; left as it is"
    fi
done
