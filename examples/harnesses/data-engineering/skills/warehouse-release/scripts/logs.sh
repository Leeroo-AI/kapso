# The deployment's log: the platform's record of the release.
set -euo pipefail
python3 "$(dirname "$0")/release.py" logs
