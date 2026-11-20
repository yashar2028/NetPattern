# Contributing

## Start

```bash
./docker_run.sh                 # db, backend (8000), worker, frontend (5173); creates .env once
./codespaces_run.sh             # the same in GitHub Codespaces
```

Everything runs in containers; the code is bind-mounted, so the backend and frontend reload on
save. Restart the worker after changing `backend/app/workers` or `backend/app/sandbox`
(`./docker_run.sh restart worker`).

## Workflow

- Pick or create an issue on the project board; branch from `dev` (`feature/<short-name>`).
- Open a pull request into `dev` and fill in the template. CI must pass; Copilot reviews it.
- `dev` is merged into `main` through a pull request when it is releasable. `main` is always
  deployable.

## Frontend design baseline

The UI imported from `feature/frontend_base_changes` (`e2950bb`) is the design baseline.
Do not overwrite or redesign it in unrelated changes or conflict resolutions. If a change
needs to alter the baseline, call it out in the pull request and get explicit approval.
Otherwise, match the baseline and focus on screens and components it did not change.

The baseline covers the shared theme (`frontend/src/design.css`, `frontend/src/index.css`),
the app shell (`App.jsx`, `AppLayout.jsx`, `PublicLayout.jsx`), the Home, Auth, Sandboxes,
Sandbox, Datasets, Models, Deployments, API Keys and Teams pages, and the sandbox creation
flow in `frontend/src/features/pipeline/createProject.js`. Shared style changes can affect
these screens even when their page files are untouched.

For approved UI work, use the app's dropdown-first layout: keep primary actions, pipeline
Task settings, and canvases visible; put secondary forms, choices, and histories behind labeled
disclosures. Keep buttons, cards, and short forms sized to their content. Use bounded, keyboard-
scrollable tables where users need to compare records, not a dropdown that hides the records.

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

CI (`.github/workflows/ci.yml`) runs on every push, on any branch: the linters, the unit tests,
the integration tests against PostgreSQL (with coverage), the frontend lint and build, a check of
every compose file combination and a build of the Docker images. A pull request shows the results
of its newest commit; `main` only accepts the merge when `Backend`, `Frontend` and `Docker` pass.

After CI passes on `main`, `.github/workflows/deploy.yml` deploys to the production server. It is
switched off (the job shows as skipped) until the repository variable `DEPLOY_ENABLED` is `true`;
see [docs/deploy.md](docs/deploy.md#deploy-from-github).

Pre-commit (`pre-commit install`) runs black, isort, flake8, ESLint and gitleaks before each
commit. ESLint uses `frontend/node_modules` if installed, otherwise the running frontend
container; with neither it is skipped and left to CI.
