#!/usr/bin/env bash
# Start NetPattern in GitHub Codespaces with one command.
#
#   ./codespaces_run.sh              build and start, make the ports public, follow the logs
#   ./codespaces_run.sh <args>       passed to `docker_run.sh up` (e.g. --no-follow)
#
# Differences from docker_run.sh: the browser reaches the app through the forwarded
# https://<codespace>-<port>.<domain> addresses, so the frontend is pointed at the forwarded API
# address and the API allows the forwarded frontend origin. Shell variables take precedence
# over .env in docker compose, so .env stays as it is.
set -euo pipefail

cd "$(dirname "$0")"

[[ -n "${CODESPACE_NAME:-}" ]] || {
  echo "error: not in a codespace; use ./docker_run.sh" >&2
  exit 1
}

# A setting from the shell, else from .env, else the default.
setting() {
  local name="$1" default="$2" value="${!1:-}"
  if [[ -z "$value" && -f .env ]]; then
    value="$(grep -E "^${name}=" .env | tail -n 1 | cut -d= -f2- | tr -d '\r' || true)"
  fi
  echo "${value:-$default}"
}

backend_port="$(setting BACKEND_PORT 8000)"
frontend_port="$(setting FRONTEND_PORT 5173)"
domain="${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:-app.github.dev}"

export NETPATTERN_API_URL="https://${CODESPACE_NAME}-${backend_port}.${domain}"
export NETPATTERN_FRONTEND_URL="https://${CODESPACE_NAME}-${frontend_port}.${domain}"
export VITE_API_BASE_URL="$NETPATTERN_API_URL"
export CORS_ORIGINS="$NETPATTERN_FRONTEND_URL"

# The first sandbox environment puts about 10 GB (PyTorch) into the Nix store.
free_gb="$(df -Pk / | awk 'NR == 2 { print int($4 / 1048576) }')"
if [[ "$free_gb" -lt 20 ]]; then
  echo "warning: ${free_gb} GB free; the first training run needs about 10 GB for PyTorch" >&2
fi

follow=1
args=()
for arg in "$@"; do
  if [[ "$arg" == --no-follow ]]; then follow=0; else args+=("$arg"); fi
done

./docker_run.sh up --cpu --no-follow ${args[@]+"${args[@]}"}

# Public ports: the frontend calls the API from the browser without a GitHub login.
gh codespace ports visibility "${backend_port}:public" "${frontend_port}:public" -c "$CODESPACE_NAME" \
  || echo "warning: could not make the ports public; set it in the Ports tab" >&2

# bwrap needs user namespaces, which the codespace's Docker may not grant. Without them sandbox
# jobs would fail, so the worker falls back to running without isolation (development only).
if ! docker compose exec -T worker bwrap --unshare-all --die-with-parent --ro-bind / / true \
  >/dev/null 2>&1; then
  echo "warning: bwrap does not work in this codespace; worker restarted with SANDBOX_ISOLATION=none" >&2
  SANDBOX_ISOLATION=none docker compose up -d --no-deps worker
fi

if [[ "$follow" == 1 ]]; then
  echo "following the logs (Ctrl+C stops following; the stack keeps running)"
  exec docker compose logs -f
fi
