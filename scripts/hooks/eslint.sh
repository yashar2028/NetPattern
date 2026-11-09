#!/usr/bin/env bash
# Pre-commit hook: ESLint on the staged frontend files (paths arrive as frontend/...).
#
# Uses frontend/node_modules when it is installed on this machine, otherwise the running frontend
# container (node_modules only exists inside it). With neither it skips; CI runs ESLint anyway.
set -euo pipefail

files=("${@#frontend/}")

if [[ -x frontend/node_modules/.bin/eslint ]]; then
  cd frontend
  exec node_modules/.bin/eslint "${files[@]}"
fi

if docker compose ps --status running --services 2>/dev/null | grep -qx frontend; then
  exec docker compose exec -T frontend npx eslint "${files[@]}"
fi

echo "eslint: skipped (no frontend/node_modules and the frontend container is not running); CI runs it"
