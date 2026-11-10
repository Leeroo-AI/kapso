# Exit 0 while every pull request of this deployment is open or merged.
set -euo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/common.sh"
REPO="$(repo_slug)"
for number in $(pr_numbers); do
    state="$(gh pr view "$number" --repo "$REPO" --json state --jq .state)"
    if [ "$state" != "OPEN" ] && [ "$state" != "MERGED" ]; then
        echo "pull request #$number is $state" >&2
        exit 1
    fi
done
