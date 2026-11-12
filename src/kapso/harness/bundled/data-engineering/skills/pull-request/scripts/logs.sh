# The deployment's log: each pull request's state, its checks, and the reviews and comments it has received.
set -euo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/common.sh"
REPO="$(repo_slug)"
for number in $(pr_numbers); do
    gh pr view "$number" --repo "$REPO" --json number,title,state,mergeable,reviewDecision,url \
        --jq '"#\(.number) \(.state) mergeable=\(.mergeable) review=\(.reviewDecision // "none")  \(.title)\n  \(.url)"'
    gh pr checks "$number" --repo "$REPO" 2>&1 | sed 's/^/  check: /' || true
    gh pr view "$number" --repo "$REPO" --comments --json comments,reviews \
        --jq '(.reviews[]? | "  review by \(.author.login): \(.state) \(.body)"), (.comments[]? | "  comment by \(.author.login): \(.body)")'
done
