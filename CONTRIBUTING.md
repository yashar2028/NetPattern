# Contributing

## Start

```bash
cp .env.example .env            # set POSTGRES_PASSWORD and AUTH_JWT_SECRET
docker compose up --build       # db, backend (8000), worker, frontend (5173)
```

Everything runs in containers; the code is bind-mounted, so the backend and frontend reload on
save. Restart the worker after changing `backend/app/workers` or `backend/app/sandbox`
(`docker compose restart worker`).

## Read first

- [docs/architecture.md](docs/architecture.md): components, flows, data model, conventions
- [docs/extending.md](docs/extending.md): where to add layers, tasks, formats, job kinds, routes, pages
- [docs/scaling.md](docs/scaling.md) and [docs/deploy.md](docs/deploy.md): machines and production
- The planning document `PLAN.md` is kept locally by the project lead (it is not in git); decisions
  that matter for the code are in these docs.

## Rules of thumb

- Keep changes small and focused; match the style around you (black/isort, line length 100;
  ESLint for the frontend).
- Engine changes bump the engine version (see extending.md). Never break old pinned sandboxes:
  new spec fields need defaults.
- Database changes need an Alembic migration (autogenerate, then review).
- Every feature gets a test: engine tests in `backend/engine/tests` (run in the pinned dev shell),
  API tests in `backend/tests` (mark PostgreSQL tests with `pytest.mark.integration`).
- User data never leaves its sandbox: new mounts must stay read-only and limited to the job's own
  folders (`app/sandbox/isolation.py`, `app/workers/handlers.py`).

## Checks

```bash
docker compose exec backend sh -c "black --check . && isort --check-only . && flake8 --max-line-length 100 --extend-ignore E203,W503,E704 app tests migrations engine && pytest"
docker compose exec worker sh -c "cd engine && nix develop .#engine-dev --command pytest"
docker compose exec frontend sh -c "npm run lint && npm run build"
```

CI (`.github/workflows/lint.yml`) runs the linters, the unit tests and the frontend build.
