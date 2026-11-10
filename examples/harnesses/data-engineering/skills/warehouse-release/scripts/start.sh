# Start the deployment again after a stop: the commit this deployment released before is requested, audited
# and published again. Never the base branch's tip of today: what was reviewed for this release is that commit.
set -euo pipefail
python3 "$(dirname "$0")/release.py" restart
