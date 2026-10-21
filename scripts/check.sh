#!/usr/bin/env bash
# Repo gate: lint + tests.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

py="$root/.venv/Scripts/python.exe"
[ -x "$py" ] || py="$root/.venv/bin/python"
[ -x "$py" ] || py="python"

echo "== ruff"
"$py" -m ruff check .
echo "== pytest"
{ code=0; "$py" -m pytest -q || code=$?; [ "$code" -eq 0 ] || [ "$code" -eq 5 ]; }  # 5 = no tests yet
echo "ALL CHECKS PASSED"
