#!/usr/bin/env bash
# Start NetPattern with one command: db, backend (API), worker and frontend.
#
#   ./docker_run.sh                  build and start, wait for the API, print the addresses,
#                                    then follow the logs (Ctrl+C stops following, not the stack)
#   ./docker_run.sh up --gpu | --cpu force the GPU override on or off (default: detected)
#   ./docker_run.sh up --no-follow   return once the stack is up
#   ./docker_run.sh <compose args>   anything else goes to docker compose with the same files,
#                                    e.g. down, ps, logs worker, restart worker
#
# The first run creates .env from .env.example with random secrets.
# codespaces_run.sh sets NETPATTERN_FRONTEND_URL / NETPATTERN_API_URL to the forwarded addresses.
set -euo pipefail

cd "$(dirname "$0")"

die() {
  echo "error: $*" >&2
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

random_secret() {
  openssl rand -hex 32 2>/dev/null || od -An -tx1 -N32 /dev/urandom | tr -d ' \n'
}

ensure_env() {
  if [[ -f .env ]]; then
    if grep -qE '^(POSTGRES_PASSWORD|AUTH_JWT_SECRET)=change-me' .env; then
      echo "warning: .env still has a change-me secret; fine locally, never in production" >&2
    fi
    return
  fi
  [[ -f .env.example ]] || die ".env.example is missing"
  sed -e "s|^POSTGRES_PASSWORD=.*|POSTGRES_PASSWORD=$(random_secret)|" \
    -e "s|^AUTH_JWT_SECRET=.*|AUTH_JWT_SECRET=$(random_secret)|" \
    .env.example >.env
  echo "created .env with a random POSTGRES_PASSWORD and AUTH_JWT_SECRET"
  # PostgreSQL keeps the password it was created with.
  if docker volume inspect netpattern_postgres_data >/dev/null 2>&1; then
    echo "warning: the database volume already exists with its old password. Put that password" >&2
    echo "         in .env, or remove the volume (docker compose down -v deletes all data)." >&2
  fi
}

has_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q GPU
}

wait_for_api() {
  local url="$1"
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl not found: not waiting for the API"
    return
  fi
  printf "waiting for the API"
  for _ in $(seq 1 90); do
    if curl -fsS "$url/health" >/dev/null 2>&1; then
      echo " ok"
      return
    fi
    printf "."
    sleep 2
  done
  echo
  echo "warning: the API did not answer within 3 minutes; see ./docker_run.sh logs backend" >&2
}

command -v docker >/dev/null 2>&1 || die "Docker is not installed"
docker info >/dev/null 2>&1 || die "Docker is not running. Start Docker Desktop (or the service) and retry."

action="${1:-up}"
[[ $# -gt 0 ]] && shift

gpu=auto
follow=1
up_args=()
if [[ "$action" == "up" ]]; then
  for arg in "$@"; do
    case "$arg" in
      --gpu) gpu=yes ;;
      --cpu) gpu=no ;;
      --no-follow) follow=0 ;;
      *) up_args+=("$arg") ;;
    esac
  done
fi
if [[ "$gpu" == auto ]]; then
  if has_gpu; then gpu=yes; else gpu=no; fi
fi

compose=(docker compose -f docker-compose.yml)
[[ "$gpu" == yes ]] && compose+=(-f docker-compose.gpu.yml)

if [[ "$action" != "up" ]]; then
  exec "${compose[@]}" "$action" "$@"
fi

ensure_env
[[ "$gpu" == yes ]] && echo "NVIDIA GPU: using docker-compose.gpu.yml"

"${compose[@]}" up --build -d ${up_args[@]+"${up_args[@]}"}

backend_port="$(setting BACKEND_PORT 8000)"
frontend_port="$(setting FRONTEND_PORT 5173)"
postgres_port="$(setting POSTGRES_PORT 5434)"
api_url="${NETPATTERN_API_URL:-http://localhost:${backend_port}}"
frontend_url="${NETPATTERN_FRONTEND_URL:-http://localhost:${frontend_port}}"

wait_for_api "http://localhost:${backend_port}"

cat <<EOF

NetPattern is running
  Frontend:    ${frontend_url}   (the header shows "backend: ok"; npm install may take a minute)
  API:         ${api_url}/health
  API docs:    ${api_url}/docs
  PostgreSQL:  localhost:${postgres_port}
  Stop:        ./docker_run.sh down

EOF

if [[ "$follow" == 1 ]]; then
  echo "following the logs (Ctrl+C stops following; the stack keeps running)"
  exec "${compose[@]}" logs -f
fi
