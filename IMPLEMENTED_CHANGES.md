# Dataset recovery, retention and batch inference


## Scope and verification

This document describes the modified and untracked application files present in the
working tree relative to HEAD when this document was created. It is a branch change
summary, not a claim that the platform is production-ready.

Implemented Jira scope: **A1, A2, A3, A6, B1 and B2**, plus a detection sanity-test
fix, loss-logging cleanup and a local-development Compose configuration change.

Validation reported from the user's Docker environment on 2026-10-14:

- Backend: **60 tests passed**.
- Engine after the detection fix: **77 tests passed**, with no warnings reported.
- Frontend: `npm run lint` completed without reported lint issues.
- A browser-driven Oxford Pets profile completed: 240 images, six classes with
  40 images each, and a 168/36/36 train/validation/test split.

These are recorded results from that environment, not a fresh test run on
2026-10-16. Profiling success does not verify model accuracy or Bubblewrap isolation.

## A1 — Recoverable dataset uploads

- Dataset versions now have an upload `status`: `uploading`, `ready` or `failed`.
  Retention adds the additional terminal status `purged`.
- A version row reserves its number before extraction. Failed/interrupted uploads
  retain their numbers rather than blocking subsequent uploads.
- Concurrent number-allocation conflicts are retried up to three times.
- An old folder occupying a newly reserved number is removed before extraction.
- Extraction errors mark the reserved version failed and clean its staging folder.
- Normal version listings, profiling and training selection use ready versions.
- The worker sweeps stale uploading rows, old temporary archives, staging folders
  and version folders without a live ready/uploading row.
- The existing archive safety checks remain in place.

An interrupted upload is not necessarily failed immediately after restart: it must
be older than `UPLOAD_STALE_SECONDS` before the sweep treats it as stale.

## A2 — Dataset soft deletion, restoration and retention

- Datasets and individual versions now have nullable `deleted_at` timestamps.
- Dataset deletion marks the parent deleted without immediately removing files.
- Individual versions can be deleted/restored separately.
- Responses expose `deleted_at` and a calculated `purge_after` timestamp.
- Live dataset listings exclude deleted datasets. Deleted datasets have a separate
  list filter; deleted versions can be included in a dataset detail response.
- Deleted parents/versions cannot be selected for new training or prediction
  inputs. Profiling deleted data is rejected, and uploads to deleted parents are
  rejected.
- Restoring a parent does not restore versions that were individually deleted.
  Restore the parent before restoring one of its versions.
- After retention, the worker deletes an expired parent and its version rows and
  folder. For an individually expired version, it removes the files but retains a
  `purged` row so the version number is not reused within that dataset.
- Purge skips datasets/versions referenced by queued or running jobs.
- Uploaded prediction inputs also have age-based retention; active prediction jobs
  protect their input folders. Completed prediction reruns need those files and
  stop working after the inputs are removed.

Cleanup runs at worker startup and through the existing recovery path. The nominal
interval is 60 seconds **when the worker reaches its idle queue branch**; it is not
an independent scheduled service and can be delayed by long jobs or a busy queue.

## A3 — Job-to-dataset references and rerun checks

- Jobs record a JSONB `dataset_version_ids` list with a GIN index.
- API training, profile and prediction jobs populate the list with the versions
  they read. Prediction jobs reference their input dataset, not the training data.
- Dataset/version deletion is rejected with HTTP 409 `dataset_in_use` when a queued
  or running job references it.
- Reruns check retained version records and their live/deleted state, dataset
  folders, uploaded prediction inputs and source checkpoints as applicable.
- Missing/deleted inputs produce HTTP 409 `inputs_missing` with an explanation.
- Restoring retained data allows the affected rerun to be submitted again.
- A folder-existence check also covers older jobs that lack recorded version IDs.

## A6 — Training recovery after interruption

- A worker retry (`attempts > 1`) sends a resume flag to the training runtime.
- `last.pt` now stores optimizer state plus resumable training state:
  - LR schedule progress, mixed-precision GradScaler state and epoch completion;
  - best metric/epoch, history and early-stopping patience;
  - Python, NumPy, CPU Torch, available CUDA and training-loader RNG state.
- A completed epoch checkpoint resumes at the next epoch.
- A checkpoint saved during graceful cancellation marks the epoch incomplete; a
  retry repeats that epoch using the saved partial weights/state. It is not exact
  batch-level continuation or a rollback to the previous epoch boundary.
- Missing checkpoints start fresh. Resume-loading errors emit a warning and take
  the fresh-start control path; see the limitations below.
- A resumed event reports the restart epoch and completed epoch count.
- Explicit API reruns create new jobs and do not automatically resume the old run.

## B1 — Checkpoint downloads

- Authenticated owners can download `best.pt` or `last.pt` from their run.
- Downloads use a streamed file response with an attachment filename.
- Only the two supported checkpoint names and validated run references are used;
  missing checkpoints and other users' runs are not exposed.

## B2 — Asynchronous batch prediction

- A completed training run can start a new `predict` job.
- Requests accept either uploaded images or an owned, live dataset version with an
  optional subfolder. Labels are not required for these input images.
- The request records the source run, checkpoint choice, original pipeline and
  exact training `environment_id`.
- The worker prepares read-only access to the source checkpoint folder and input
  folder. It does not deliberately mount the training dataset for prediction.
  These filesystem restrictions are enforced only with Bubblewrap enabled.
- The engine rebuilds the architecture, loads checkpoint weights with
  `weights_only=True`, restores task metadata and applies evaluation transforms.
- Regression restores the saved target mean/std to return original-scale values.
- Supported inputs include natural images and the engine's DICOM/NIfTI readers;
  volumes expand into per-slice predictions using the pipeline's slicing settings.
- Prediction progress events and cancellation use the existing job mechanisms.
- Selected image-reading/preprocessing failures become per-image error entries;
  indexing and model-forward failures can still fail the entire job.

| Task | Output per image/slice |
| --- | --- |
| Single-label classification | Predicted class, score and up to five ranked classes |
| Multi-label classification | Labels above the task threshold and scores for all classes |
| Regression | Named values in the training targets' original scale |
| Semantic segmentation | Class-index PNG mask and per-class pixel fractions |
| Bounding-box detection | Class, confidence and XYXY box coordinates |

Masks are resized to original input dimensions. Detection boxes are scaled from
transformed image dimensions; this is not a general inverse for crops/rotations.

### Prediction limits

- Uploads: at most 100 files and 200 MiB combined by default.
- Engine: at most 1,000 images/slices per prediction job; truncation adds a warning.
- Engine batch size is clamped to 1–64.
- Detection minimum score defaults to 0.5. Multi-label thresholds use the saved
  pipeline task specification instead.
- Structured results are stored inline in the job result; masks are run artifacts.

## API additions and changes

All paths below require authentication and enforce resource ownership.

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/v1/datasets?deleted=true` | List deleted parents instead of live datasets |
| GET | `/v1/datasets/{id}?include_deleted=true` | Include individually deleted, ready versions |
| DELETE | `/v1/datasets/{id}` | Changed from immediate deletion to soft deletion |
| POST | `/v1/datasets/{id}/restore` | Restore a retained parent |
| DELETE | `/v1/datasets/{id}/versions/{number}` | Soft-delete one version |
| POST | `/v1/datasets/{id}/versions/{number}/restore` | Restore one retained version |
| POST | `/v1/runs/{id}/rerun` | Now checks required inputs before enqueueing |
| GET | `/v1/runs/{id}/checkpoints/{best\|last}` | Download checkpoint |
| POST | `/v1/runs/{id}/predict` | Start an asynchronous prediction run |
| GET | `/v1/runs/{id}/predictions/{name}` | Download a generated class-index PNG mask |

Prediction submission uses multipart form data:

- `files`: image uploads, **or** `dataset_version_id`, not both;
- `subpath`: optional folder within the selected version;
- `checkpoint`: `best` (default) or `last`;
- `hardware_tier`: `cpu` (default) or `gpu`;
- `score_threshold`: 0–1, default 0.5, for detection outputs.

The response is HTTP 202 with a run. Existing run detail and event endpoints provide
results/progress. There is no new frontend prediction interface or live-serving API.

## Detection sanity test and loss logging

- Sanity runs now use `build_optimizer()` with the configured optimizer settings
  and model head groups rather than hardcoded Adam.
- The randomly initialized detector's test explicitly uses Adam at `0.0001`
  instead of the previous `0.002`; its 100 steps and pass threshold are unchanged.
- A regression test checks that sanity runs honor optimizer selection/settings.
- Sanity and training loss logging use detached scalar values, removing the
  reported requires-grad tensor-to-scalar warning.

## Database migrations

Apply in order after revision `20261008_0003`:

1. [0004](backend/migrations/versions/20261014_0004_upload_status_job_datasets.py):
   adds version status (existing rows default to ready), job version-ID lists
   (existing jobs default to an empty list), and the GIN index.
2. [0005](backend/migrations/versions/20261014_0005_dataset_soft_delete.py):
   adds nullable deletion timestamps to datasets and versions.

Existing jobs are **not backfilled** with dataset-version IDs. Migration downgrade
removes the new metadata; it cannot recover files already removed by retention.
The existing backend startup command applies Alembic upgrades automatically.

## Settings and local deployment change

| Setting | Default in Python settings | Purpose |
| --- | --- | --- |
| `UPLOAD_STALE_SECONDS` | 7200 | Age threshold for interrupted-upload cleanup |
| `DATASET_RETENTION_DAYS` | 30 | Retention after dataset/version deletion |
| `PREDICT_INPUTS_RETENTION_DAYS` | 30 | Age-based retention of prediction uploads |
| `PREDICT_MAX_FILES` | 100 | Prediction upload file limit |
| `PREDICT_MAX_UPLOAD_BYTES` | 209715200 | Prediction upload byte limit |

These new settings must reach the relevant backend/worker process to override
defaults. They have not been wired as individual variables in the Compose file;
adding them only to Compose's host-side `.env` does not automatically forward them.

The worker's Compose isolation fallback changed from `bwrap` to `none`:
`${SANDBOX_ISOLATION:-none}`. An explicit environment value still overrides it.
The Python default was already `none` and was not changed by this branch.

**Security warning:** `none` bypasses Bubblewrap, removing its per-job filesystem,
namespace and network isolation. This is a trusted local-development workaround
for the demonstrated Docker seccomp/namespace restriction, not a production fix.
No privileged mode, extra capability or seccomp-unconfined override was added by
these changes. The old Compose comment claiming default-seccomp compatibility is
still present and is inaccurate on the tested Docker Desktop installation.

Recreate the worker after environment/Compose changes; restarting alone does not
replace the container's environment. Do not use `docker compose down -v` to update
the app: that deletes persistent volumes containing accounts, datasets and runs.

## Complete changed-file inventory

### Modified files

- [README.md](README.md): checkpoint/prediction usage and recovery/retention notes.
- [docker-compose.yml](docker-compose.yml): local isolation fallback changed to none.
- [datasets.py](backend/app/api/datasets.py): upload reservation, deletion/restoration,
  list filters, version guards and profiling links.
- [runs.py](backend/app/api/runs.py): checkpoint/prediction routes and rerun checks.
- [config.py](backend/app/core/config.py): upload, retention and prediction limits.
- [dataset.py](backend/app/models/dataset.py): status constants and deletion fields.
- [job.py](backend/app/models/job.py): dataset-version references and GIN index.
- [api.py](backend/app/schemas/api.py): status/deletion/retention response fields.
- [runs.py](backend/app/services/runs.py): resolution, ownership, references and
  input/checkpoint validation helpers.
- [storage.py](backend/app/services/storage.py): prediction-input root and path guards.
- [uploads.py](backend/app/services/uploads.py): staging helper and stale-upload sweep.
- [handlers.py](backend/app/workers/handlers.py): prediction preparation and resume flag.
- [main.py](backend/app/workers/main.py): cleanup hooks and logging in recovery.
- [queue.py](backend/app/workers/queue.py): enqueue dataset-version references.
- [pipeline.py](backend/engine/src/netpattern_engine/pipeline.py): train resume flag,
  configured sanity optimizer and detached logging.
- [runtime.py](backend/engine/src/netpattern_engine/runtime.py): predict dispatch and
  training resume request handling.
- [base.py](backend/engine/src/netpattern_engine/tasks/base.py): restore task state.
- [classification.py](backend/engine/src/netpattern_engine/tasks/classification.py):
  restore regression normalization statistics.
- [checkpoint.py](backend/engine/src/netpattern_engine/training/checkpoint.py): save
  additional resume metadata.
- [optim.py](backend/engine/src/netpattern_engine/training/optim.py): schedule persistence.
- [seed.py](backend/engine/src/netpattern_engine/training/seed.py): RNG capture/restore.
- [trainer.py](backend/engine/src/netpattern_engine/training/trainer.py): resume/save
  training state and detached logging.
- [test_training.py](backend/engine/tests/test_training.py): resume equivalence,
  missing-checkpoint fallback, optimizer selection and detection stability tests.
- [test_api.py](backend/tests/test_api.py): upload recovery, retention/restoration,
  active-job guards, reruns, checkpoint access and prediction-submission tests.
- [test_sandbox.py](backend/tests/test_sandbox.py): retry flags and prediction path/
  sandbox mount preparation tests.

### New files

- [retention.py](backend/app/services/retention.py): deletion and prediction-input purge.
- [inference.py](backend/engine/src/netpattern_engine/inference.py): batch inference.
- [test_inference.py](backend/engine/tests/test_inference.py): classification,
  regression, segmentation, unreadable-image, runtime and task-mismatch tests.
- [migration 0004](backend/migrations/versions/20261014_0004_upload_status_job_datasets.py).
- [migration 0005](backend/migrations/versions/20261014_0005_dataset_soft_delete.py).
- [.vscode/tasks.json](.vscode/tasks.json): optional editor task for the focused
  detection sanity test. This is a development convenience, not runtime behavior.
- [IMPLEMENTED_CHANGES.md](IMPLEMENTED_CHANGES.md): this branch change summary.

No frontend application source files changed. Account registration and the Oxford
Pets downloader were pre-existing features, not new implementations in this branch.
Local `.env` values and Docker-volume data are not part of this change inventory
and must not be committed as credentials or dataset artifacts.

## Known limitations and follow-up work

- Passing tests do not establish production isolation, load capacity or model quality.
- Legacy jobs with empty version-ID lists are not protected by the new reference
  guard; checking a retained folder alone cannot establish its soft-deletion state.
- Reference checks, enqueue/delete operations and filesystem cleanup are not one
  transaction; concurrency hardening remains necessary.
- Purge commits database changes before best-effort file removal. Upload sweeping
  can later remove stale version folders, but this is not transactional cleanup or
  a guarantee that all empty parent folders/filesystem errors are reconciled.
- `purge_after` is a calculated deadline, not strictly enforced by restore routes:
  restoration can still succeed after it until the worker actually purges the item.
- Dataset detail can return an owned deleted parent; normal listings hide it.
- Existing pinned engine snapshots do not automatically acquire new prediction or
  resume code. Old environments need an explicit compatibility/upgrade strategy.
- Resume-loading failure may happen after weights/optimizer have already been
  partly loaded; the fallback does not rebuild a fully fresh trainer atomically.
- Retrying an incomplete epoch is not exact batch-by-batch recovery. Previously
  emitted events may remain even when work is repeated.
- Detection prediction output is implemented but has no end-to-end detection
  inference regression test. Multi-label inference also has no dedicated test.
- Output geometry only reverses scaling, not arbitrary spatial evaluation transforms.
- Inference rebuilds models with materialization and can prefetch pretrained weights
  before loading the trained checkpoint; checkpoint-only reconstruction is not yet
  independent of those original weight sources/cache.
- No checksum/integrity verification, backups, model registry, live model serving,
  ONNX export, S3 backend or multiple-dataset pipelines was implemented here.
- Soft deletion/restoration and predictions are API-only additions; no corresponding
  frontend controls were added.
- Synthetic-data generation/CAD integration remains deferred.
- Shared Compose isolation remains to be resolved: retain `bwrap` as the shared
  fallback and set `SANDBOX_ISOLATION=none` explicitly in local `.env` for trusted
  development, rather than distributing `none` as the deployment default.