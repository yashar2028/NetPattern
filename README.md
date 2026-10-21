# NetPattern

NetPattern is a platform for developing CNNs for imaging tasks visually, then using, tracking and
improving them. Users bring their images (PNG, JPEG, DICOM, NIfTI), build a network on a canvas,
train it on the hardware tier they choose, evaluate it, and call it from their own software
through a REST API. Every model lives in its own Nix-pinned sandbox, so it behaves the same way
in the future as it does today.

Documentation for contributors: [architecture](docs/architecture.md), [extending](docs/extending.md),
[deploying on a VM](docs/deploy.md), [scaling to more machines](docs/scaling.md) and
[CONTRIBUTING.md](CONTRIBUTING.md).

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
- `sdk/python/`: a small Python client for deployed models.
- `docs/`: architecture, extending, deploy, scaling.
- `docker-compose.yml`: db, backend, worker, frontend. Overrides: `.gpu.yml` (NVIDIA), `.prod.yml`
  (production on a VM), `.worker-node.yml` (an extra worker machine).
- `scripts/backup.sh`: database + storage backup.

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
4. **Run** saves the pipeline as a new version and starts a training run, a one-batch sanity
   check or a learning-rate finder on the hardware tier you pick. The run page shows progress
   live, then charts, the per-epoch table, evaluation (confusion matrix, per-class scores) and the
   misclassified images. "Report" opens a printable report; "Reproduce locally" downloads a bundle
   that rebuilds the exact environment with Nix.
5. **Sweep** (pipeline page) runs the pipeline over a grid of values, or as k-fold
   cross-validation; select runs on the sandbox page and **Compare** them.
6. **Register as model** (run page), then **Deploy** it on the Models page. The deployment page has
   a playground (with Grad-CAM explanations), copy-paste API snippets and the request log.
   **Export** gives weights, TorchScript and ONNX.
7. **API keys**: create an inference key (optionally for one deployment) to call models from your
   software. **Teams** share sandboxes and datasets with colleagues.

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
   `GET /v1/runs/{id}`, `GET /v1/runs` and `GET /v1/usage` (compute units = tier rate × seconds,
   see `GET /v1/hardware-tiers`).
7. `POST /v1/models` with `{"run_id": …, "name": …}`, then `POST /v1/deployments` with
   `{"model_version_id": …, "name": …, "mode": "serverless" | "dedicated"}`.
8. Call it with an inference key (`POST /v1/api-keys` with `"scope": "inference"`):

   ```bash
   curl -X POST http://localhost:8000/v1/deployments/<dep_id>/predict \
     -H "Authorization: Bearer $NETPATTERN_API_KEY" -F "files=@image.png"
   ```

   or from Python with the SDK in `sdk/python` (`Client().predict("dep_…", ["image.png"])`).

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
  and NIfTI data, `detection` for detection metrics, `export` for ONNX export.
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

## Worker capacity

One worker runs `WORKER_MAX_JOBS` jobs at once (default 1). Each job gets a bubble the size of its
hardware tier (`cpu`: 2 cores, `cpu-large`: 8 cores, `gpu`: 1 GPU + 4 cores; configurable), pinned
to its own cores and GPUs. See [docs/deploy.md](docs/deploy.md#worker-capacity-bubbles).

## Production

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

Port 80 serves the app with the API on the same origin. See [docs/deploy.md](docs/deploy.md).

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
