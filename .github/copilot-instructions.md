# NetPattern: notes for Copilot (code review)

NetPattern is a platform for building, training and deploying CNNs for images. FastAPI +
SQLAlchemy (async) + Alembic + PostgreSQL backend, a worker that runs jobs in Nix-pinned,
bubblewrap-isolated sandboxes, a PyTorch engine (`backend/engine`) and a React (JavaScript, Vite)
frontend. Everything runs in Docker.

When reviewing, check these project rules first:

- **Sandbox boundary.** User data and user code never leave their sandbox. New mounts in
  `app/sandbox/isolation.py` or `app/workers/handlers.py` must be read-only and limited to the
  job's own folders. Flag anything that widens access, enables network inside a sandbox, or runs
  user code outside it.
- **Database changes** need an Alembic migration in `backend/migrations/versions`. Flag model
  changes without one, and migrations that drop or rewrite data without saying so.
- **Engine compatibility.** Old sandboxes keep their pinned engine. Engine changes bump the engine
  version; new spec fields need defaults so old specs still load.
- **Secrets** come from settings/environment, never from code or committed files. Settings for
  secrets have no defaults.
- **Auth and access.** Every API route that touches a resource checks ownership or team access
  (`app/services/access.py`). API keys respect their scopes and restrictions.
- **Tests.** Every feature or fix has a test. Tests that need PostgreSQL are marked
  `pytest.mark.integration`.
- **Style** is enforced by black, isort, flake8 (line length 100) and ESLint; do not comment on
  formatting those tools already handle.

Prefer few, concrete comments about correctness, security and missing tests over style remarks.
