#!/usr/bin/env bash
set -euo pipefail

export CUBLAS_WORKSPACE_CONFIG=:4096:8

python scripts/phyg/preflight_static.py
python scripts/phyg/preflight.py
python scripts/phyg/verify_budget_and_init.py
python scripts/phyg/verify_official_initialization.py --device cpu
python scripts/phyg/test_dataloader_rng_isolation.py
python scripts/phyg/test_amp_compat.py
python scripts/phyg/test_resume_equivalence.py
python scripts/phyg/test_experiment_logging.py
python scripts/phyg/test_real_stack_resume.py
python scripts/phyg/benchmark_100_steps.py

echo "Preflight and benchmark complete. Formal training was not started."
echo "Run individual configs only after separate training authorization."
