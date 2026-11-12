# Exit 0 while production stands at the commit this deployment released.
set -euo pipefail
python3 "$(dirname "$0")/release.py" health
