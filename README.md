# NetPattern

NetPattern is a platform for developing CNNs for imaging tasks visually, then using, tracking and
improving them. Users bring their images (PNG, JPEG, DICOM, NIfTI), build a network on a canvas,
train it on the hardware tier they choose, evaluate it, and call it from their own software
through a REST API. Every model lives in its own Nix-pinned sandbox, so it behaves the same way
in the future as it does today.

The blueprint, decisions and roadmap are in [PLAN.md](PLAN.md).

## Stack

- Frontend: React (JavaScript), Vite, React Router, Axios.
- Backend: FastAPI, SQLAlchemy async, Alembic, PostgreSQL.
- Worker: the same backend image, running sandbox processes in Nix-pinned environments.
- Engine (`backend/engine`): the PyTorch training/serving engine. It runs **only** inside sandbox
  environments built by `backend/flake.nix`.

## Repository layout

`frontend/` contains the Vite app.
`backend/` contains the API (`app/`), the worker (`app/workers`), the sandbox boundary
(`app/sandbox`), migrations, the engine package (`engine/`) and the Nix flake (`flake.nix`).
`docker-compose.yml` runs everything: db, backend, worker, frontend (three images).

## Run

Everything runs in Docker; nothing runs directly on Windows.

```bash
cp .env.example .env              # once; then set POSTGRES_PASSWORD
docker compose up --build
```

- Frontend: http://localhost:5173 (the header shows `backend: ok`)
- API: http://localhost:8000/health, docs at http://localhost:8000/docs
- PostgreSQL: `localhost:5434`

On a machine with an NVIDIA GPU:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

## Checks

Run these while the stack is up:

```bash
# Backend tests (including the real-database health check)
docker compose exec backend pytest

# Build the pinned CPU sandbox environment and run one engine job inside it.
# The first run downloads several GB (PyTorch); afterwards it's cached in the nix_store volume.
docker compose exec worker python -m app.sandbox.smoke

# Engine tests inside the same pinned packages
docker compose exec worker nix develop .#engine-dev --command pytest engine/tests

# Frontend lint
docker compose exec frontend npm run lint
```

The first `nix build` creates `backend/flake.lock`. Commit it: it is the pin.

## Development notes

Install the pre-commit hooks (black, isort, flake8) with:

```bash
pre-commit install
```

Create a migration:

```bash
docker compose exec backend alembic revision --autogenerate -m "describe change"
```

If you run Nix from a git checkout outside Docker, Nix only sees files tracked by git. Run
`git add` on new files before `nix build`.
