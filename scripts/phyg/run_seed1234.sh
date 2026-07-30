#!/usr/bin/env bash
set -euo pipefail

python scripts/phyg/preflight.py
python scripts/phyg/verify_budget_and_init.py

echo "Preflight complete. Training commands are intentionally not auto-started."
echo "Run individual configs only after separate training authorization."
