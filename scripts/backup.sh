#!/usr/bin/env bash
# Back up what cannot be rebuilt: the database and the storage volume's user data.
#
#   scripts/backup.sh [target-dir]        (default: ./backups/<timestamp>)
#
# Environments (/data/envs), engine caches and the Nix store are rebuilt from the pinned flakes,
# so they are not included. Engine snapshots are small and are what pins old sandboxes: included.
set -euo pipefail

target="${1:-backups/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$target"
compose=(docker compose -f docker-compose.yml)
[[ -f docker-compose.prod.yml && "${NETPATTERN_PROD:-1}" == "1" ]] && compose+=(-f docker-compose.prod.yml)

echo "database → $target/database.dump"
"${compose[@]}" exec -T db sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" --format=custom' \
  > "$target/database.dump"

echo "storage → $target/storage.tar.gz (datasets, models, runs, engine snapshots)"
"${compose[@]}" exec -T backend tar -C /data -czf - \
  --exclude=./cache --exclude=./envs --exclude=./sessions --exclude=./inference \
  . > "$target/storage.tar.gz"

echo "done: $(du -sh "$target" | cut -f1) in $target"
echo "restore: see docs/deploy.md#restore"
