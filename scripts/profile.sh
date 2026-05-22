#!/usr/bin/env bash
set -euo pipefail

echo "source .venv/bin/activate && PYTHONPATH=. ncu --set full -o /tmp/profile --import-source=yes -f $@" | sudo bash
