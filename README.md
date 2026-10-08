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
- Worker: the same backend image. It claims jobs from a PostgreSQL queue and runs them in
  Nix-pinned environments, isolated with bubblewrap.
- Engine (`backend/engine`): the PyTorch engine (datasets, models, layer graphs, training,
  metrics). It runs **only** inside sandbox environments built from `backend/engine/flake.nix`.

## Repository layout

- `frontend/`: the Vite app.
- `backend/app/`: the API, the worker (`workers/`), the sandbox boundary (`sandbox/`).
- `backend/engine/`: the engine package and its flake (environment templates, capability packs).
- `backend/examples/`: ready-to-run job requests.
- `docker-compose.yml`: db, backend, worker, frontend (three images).

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

## Use the app

Open http://localhost:5173 and create an account (or sign in).

1. **Datasets**: create a dataset and upload a zip of your image folder. Each upload is a new
   version; "Profile" checks class counts, image sizes and problems.
2. **Sandboxes**: create a sandbox, then a pipeline inside it. A new pipeline starts as a
   pretrained ResNet-18 chain: Dataset → Transforms → Model → Trainer → Evaluator.
3. Click a node to edit it. In the Dataset node pick your dataset version. In the Model node
   choose a zoo model, or "Build custom layers instead" to open the layer editor, which shows each
   layer's output shape and problems as you edit.
4. **Run** saves the pipeline as a new version and starts a training run (or a one-batch sanity
   check) on CPU or GPU. The run page shows progress live, then the charts, per-epoch table and
   evaluation (confusion matrix, per-class scores).

The first time a sandbox opens in the editor its environment starts, which can take a minute.

## Use the API

Interactive docs: http://localhost:8000/docs (OpenAPI at `/v1/openapi.json`).

1. `POST /v1/auth/register` (or `/v1/auth/login`) returns `access_token`. Send it as
   `Authorization: Bearer …`, or create an API key with `POST /v1/api-keys` and use that instead.
2. `POST /v1/sandboxes` creates a sandbox.
3. `POST /v1/datasets`, then upload a zip of your dataset folder with
   `POST /v1/datasets/{id}/versions` (multipart field `file`).
4. `POST /v1/sandboxes/{id}/pipelines` with a `spec`. In dataset nodes use
   `"dataset_version_id": "dsv_…"` (plus optional `"subpath"`) instead of `root`.
5. `POST /v1/sandboxes/{id}/runs` with `{"pipeline_version_id": "plv_…"}`.
6. Follow it with `GET /v1/runs/{id}/events/stream` (Server-Sent Events), then read
   `GET /v1/runs/{id}`, `GET /v1/runs` and `GET /v1/usage`.

## Train from the worker CLI (no account needed; for development)

```bash
# Test data: an Oxford-IIIT Pet subset (classification, segmentation masks, head boxes)
docker compose exec worker python -m app.datasets.samples oxford-pets
# …and synthetic NIfTI volumes, written by a sandbox job
docker compose exec worker python -m app.datasets.samples synthetic-nifti

# Submit a job and follow its live events
docker compose exec worker python -m app.workers.cli submit examples/train_resnet18_pets.json --wait
docker compose exec worker python -m app.workers.cli submit examples/train_unet_pets_segmentation.json --wait
docker compose exec worker python -m app.workers.cli submit examples/train_fasterrcnn_pets_heads.json --wait
docker compose exec worker python -m app.workers.cli submit examples/train_nifti_slices_2p5d.json --wait
docker compose exec worker python -m app.workers.cli submit examples/train_custom_cnn_synthetic.json --wait

# Other job kinds: sanity (overfit one batch), profile (dataset statistics)
docker compose exec worker python -m app.workers.cli submit examples/sanity_custom_cnn_synthetic.json --wait
docker compose exec worker python -m app.workers.cli submit examples/profile_pets.json --wait

# Manage jobs
docker compose exec worker python -m app.workers.cli list
docker compose exec worker python -m app.workers.cli status <job_id> --full
docker compose exec worker python -m app.workers.cli cancel <job_id>

# Editor features through a session process (shape inference, params, FLOPs, per-node issues)
docker compose exec worker python -m app.workers.cli session validate_architecture \
    --params-file examples/validate_graph_session.json
```

Each run writes `request.json`, `result.json`, `stderr.log` and `checkpoints/` to
`/data/runs/<job_id>` in the storage volume. Every event is also stored in `job_events`.

## Environments

- **Templates:** `pytorch-cpu` and `pytorch-cuda12`.
- **Capability packs** are added automatically when a pipeline needs them: `medical` for DICOM
  and NIfTI data, `detection` for detection metrics.
- Each environment gets its own generated `flake.nix` + `flake.lock` under `/data/envs/<id>`.

```bash
docker compose exec worker python -m app.workers.cli env --packs medical   # build or show one
docker compose exec worker python -m app.workers.cli verify-env            # locks re-evaluate identically
docker compose exec worker python -m app.sandbox.smoke                     # one engine job, no queue
```

Isolation is set by `SANDBOX_ISOLATION` in `.env`:
- `bwrap` (the default): each sandbox runs as `nobody` in its own namespaces, with no network,
  the Nix store and datasets read-only, and write access only to its run folder.
- `none`: for debugging the engine only.

## Checks

```bash
docker compose exec backend pytest                                   # API, queue, sandbox runner
docker compose exec worker sh -c "cd engine && nix develop .#engine-dev --command pytest"
docker compose exec frontend npm run lint
```

`backend/engine/flake.lock` pins nixpkgs for every environment. Commit it.

## Development notes

Install the pre-commit hooks (black, isort, flake8) with:

```bash
pre-commit install
```

Create a migration:

```bash
docker compose exec backend alembic revision --autogenerate -m "describe change"
```

The worker does not auto-reload. After changing worker or sandbox code, run
`docker compose restart worker`.

If you run Nix from a git checkout outside Docker, Nix only sees files tracked by git. Run
`git add` on new files before `nix build`.
