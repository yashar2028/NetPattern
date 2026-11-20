# NetPattern — outstanding work

Updated: 2026-10-16

This is a revisit-able engineering backlog, not a claim that every suggested feature
is required for the initial product. IDs A1–I1 are conversation placeholders, not
actual Jira issue keys. No Jira tickets are created or updated by this document.

Implementation details: [IMPLEMENTED_CHANGES.md](IMPLEMENTED_CHANGES.md).

## Current baseline

- Implemented scope: **A1, A2, A3, A6, B1 and B2**.
- Implementation branch: `feat/dataset-recovery-and-batch-inference`.
- Pushed commit: `549be9c` — dataset recovery, retention and batch inference.
- At the last confirmed Git operation, the branch was pushed but not merged into
  `main`. Check the current pull-request status before planning follow-up changes.
- Reported validation: 60 backend tests, 77 engine tests and frontend lint passed.
  These historical results do not replace tests after subsequent changes.
- Dataset profiling has also succeeded through the browser with Oxford Pets.
- Current isolation workaround disables Bubblewrap through the shared Compose
  fallback. This is not a production-ready security configuration.

## Status and priority guide

- **Immediate:** correctness/security follow-ups to the implemented functionality.
- **High:** remaining reliability work and explicitly requested platform foundations.
- **Product decision:** useful proposals, not mandatory initial-release commitments.
- **Conditional:** verify the requirement/environment before implementing.
- Checkboxes indicate outstanding work; completed original tickets are summarized
  separately so they are not accidentally implemented twice.

## 1. Immediate follow-ups

### SEC1 — Resolve sandbox deployment and secure defaults

- [ ] Restore `bwrap` as the shared Compose fallback; keep any `none` override
  explicit and local to trusted development.
- [ ] Correct the Compose comment claiming compatibility with Docker's default
  seccomp profile on the tested machine.
- [ ] Decide the supported production execution model: a tested, narrowly scoped
  seccomp profile for nested Bubblewrap, or a separately deployed Linux worker.
  The latter requires deployment/storage/network changes.
- [ ] Test the complete Bubblewrap policy, not only simple user-namespace creation:
  filesystem visibility, read-only data, write boundaries, network isolation,
  process isolation and termination behavior.
- [ ] Add a fail-closed isolation preflight with an actionable diagnostic. Do not
  silently downgrade to `none` when isolation fails.

Evidence: namespace creation failed under the tested Docker policy and succeeded
with seccomp disabled. Do not fix this by adding `privileged: true`, broadly granting
`SYS_ADMIN`, or adopting `seccomp=unconfined` as the production solution.

### REL1 — Backfill dataset references for existing jobs

- [ ] Resolve older stored dataset paths to version IDs where possible.
- [ ] Record unresolved legacy references explicitly and handle them conservatively.
- [ ] Test deletion protection for jobs created before migration 0004 and decide how
  development/CLI jobs should participate in protection.

Existing jobs default to an empty ID list. Folder existence is checked for reruns,
but cannot by itself detect soft deletion or protect the data from deletion.

### REL2 — Serialize deletion, enqueue, restoration and purge

- [ ] Design a common locking/transaction strategy for reference checks and writes.
- [ ] Cover concurrent delete-versus-submit, purge-versus-restore and upload-versus-
  delete cases, including multi-worker execution.
- [ ] Add concurrency regression tests rather than relying only on sequential API tests.

### REL3 — Make cleanup failure recoverable and scheduling explicit

- [ ] Reconcile DB records and filesystem artifacts when purge/removal fails.
- [ ] Surface cleanup errors instead of relying only on best-effort deletion.
- [ ] Run maintenance independently of a worker reaching the idle queue branch, or
  document and monitor its maximum delay under sustained workload.
- [ ] Test interruption after DB commit but before file removal, and retry cleanup.

Cleanup currently runs at startup and nominally every 60 seconds only when the worker
reaches its idle queue branch. Long jobs/a continuously occupied queue can delay it.

### REL4 — Clarify and enforce retention semantics

- [ ] Decide whether restore is allowed after `purge_after` but before physical purge.
  Enforce the decision consistently in routes, response fields and documentation.
- [ ] Review the visibility of deleted parents in direct dataset detail responses.
- [ ] Test retained, expired, purged and partially cleaned data states.

### REL5 — Make failed resume loading atomic

- [ ] Validate/load resume state without leaving a partially restored trainer.
- [ ] On fallback, rebuild a genuinely fresh model, optimizer, scheduler and RNG state,
  or fail clearly instead of claiming a fresh start.
- [ ] Test corrupt, incomplete and incompatible checkpoint state.
- [ ] Document incomplete-epoch replay and repeated historical events. Exact batch-level
  recovery is a separate, optional enhancement, not currently implemented.

### INF1 — Close prediction coverage gaps

- [ ] Add end-to-end detection inference tests covering labels, thresholds and box geometry.
- [ ] Add dedicated multi-label inference tests.
- [ ] Extend cancellation, input-limit and ownership tests for prediction artifacts.
- [ ] Verify volume input/indexing failures separately from per-image preprocessing errors.

The passing detection sanity test covers training, not detection prediction.

### INF2 — Correct or restrict spatial evaluation transforms

- [ ] Track inverse spatial mappings for masks/boxes, or reject unsupported crops,
  flips and rotations during prediction.
- [ ] Test predictions against known coordinates on non-square images.

Current output mapping reverses resizing only; arbitrary transforms can misplace outputs.

### INF3 — Handle pinned-engine feature compatibility

- [ ] Detect whether a recorded environment supports prediction and resume capabilities.
- [ ] Give an explicit compatibility error for unsupported older snapshots.
- [ ] Provide a deliberate upgrade/new-run path without silently changing a historical
  run's environment or claiming it is still identical.

### INF4 — Reconstruct from checkpoints without fetching original weights

- [ ] Build the correct architecture without downloading/loading pretrained weights
  that will immediately be overwritten by the checkpoint.
- [ ] Preserve architecture layout, normalization and model-input metadata.
- [ ] Test checkpoint inference with empty caches and no network.

### CFG1 — Wire configuration overrides into deployment

- [ ] Explicitly forward the new upload, retention and prediction limits to the relevant
  Compose services, or choose another documented settings-injection mechanism.
- [ ] Validate positive/allowed limit values and test effective settings.

Adding a value only to Compose's host-side `.env` does not forward it automatically
unless the Compose service uses it.

## 2. Remaining original Jira backlog

### Epic A — Data recovery and integrity

#### A4 — Sandbox/pipeline soft deletion and run-artifact cleanup — High

- [ ] Add deletion/restoration and retention for sandboxes and pipelines.
- [ ] Preserve historical run/pipeline references until intentional purge.
- [ ] Remove run folders consistently with their records; avoid orphaned artifacts.
- [ ] Test active-job protection, restoration and purge failure recovery.

Dependencies: align with A2 retention and REL2/REL3. Dataset retention alone does not
implement sandbox or pipeline retention.

#### A5 — Dataset and checkpoint checksums — High

- [ ] Hash uploaded files while streaming and persist an immutable manifest/version hash.
- [ ] Hash checkpoints and verify integrity before reuse/download as appropriate.
- [ ] Provide an explicit policy/backfill for existing data without hashes.
- [ ] Test corruption and missing manifest entries.

Unblocks trustworthy worker caching, migrations and exact duplicate detection.

#### A7 — Backups and tested restoration — High, conditional verification

- [ ] Confirm whether backups already exist outside this repository.
- [ ] Back up PostgreSQL and durable datasets/run artifacts to a separate location.
- [ ] Define retention, recovery-point/recovery-time objectives and encryption/access rules.
- [ ] Restore into a clean environment and verify records against their files.

Soft deletion is not a backup. Do not close this task merely because volumes persist.

### Epic B — Using trained models

#### B3 — Versioned model registry — High

- [ ] Register immutable model versions tied to source run, checkpoint, metrics,
  pipeline/preprocessing and pinned environment.
- [ ] Add owned list/detail/register operations and model lifecycle rules.

Depends on existing B1/B2 and integrity/compatibility follow-ups.

#### B4 — Live prediction service — Product decision

- [ ] Design reusable model processes and routing rather than starting a batch job per call.
- [ ] Implement authenticated prediction, bounded concurrency, rate limits, idle shutdown,
  cancellation/timeouts and usage recording.
- [ ] Test isolation and worker resource exhaustion.

Depends on B3 and SEC1. Editor sessions are a possible starting point, not a complete
production inference-serving architecture.

#### B5 — Model export — Product decision

- [ ] Choose supported export targets/model families; evaluate ONNX and current PyTorch
  export support instead of promising every model or format upfront.
- [ ] Bundle input/preprocessing/class metadata and compare exported-runtime outputs
  against PyTorch within documented tolerances.

Depends on B3. Detection exports require separate compatibility validation.

#### B6 — Frontend model and artifact workflows — High/product scope decision

- [ ] Add model/version browsing and registration once B3 exists.
- [ ] Add checkpoint download and batch-prediction submission/results UI for existing B1/B2.
- [ ] Add image previews, segmentation masks and detection overlays.
- [ ] Add live API examples/export controls only when B4/B5 are available.

#### B7 — External model import — Decision spike

- [ ] Decide whether user-provided weights/models are in scope.
- [ ] Define allowed architectures/formats and safe resource-bounded validation.
- [ ] Assess ONNX/safetensors/weights-only checkpoints; do not permit arbitrary pickle
  execution or assume any model format is intrinsically risk-free.

### Epic C — Storage abstraction and S3 — High, guided by E1

- [ ] **C1:** Introduce a storage interface with a local implementation while preserving
  explicit local paths required by sandbox execution.
- [ ] **C2:** Add an S3-compatible implementation, development service/provider setup,
  credential handling and tested encryption/access settings. Select the provider first.
- [ ] **C3:** Stage datasets in a checksum-verified worker-local cache; add safe eviction,
  tenant separation and protection for files used by active jobs. Depends on C2/A5.
- [ ] **C4:** Persist checkpoints/results/logs in object storage; add authenticated or
  short-lived artifact downloads and crash-safe publication. Depends on C2.
- [ ] **C5:** Migrate existing volume data only if needed, with manifests, verification,
  idempotent retries and rollback planning. Depends on C2/A5.

### Epic D — Multiple datasets per pipeline — High, explicitly requested

- [ ] **D1:** Extend pipeline specification/edge validation for multiple sources, with
  task compatibility, split roles, tags, class mapping and optional sampling weights.
- [ ] **D2:** Combine source indexes/loaders, harmonize labels/channels and implement
  reproducible sampling/splitting. Preserve groups across sources, not merely within
  each source. Depends on D1.
- [ ] **D3:** Add per-source and combined profiling/evaluation. Define weighted versus
  unweighted overall metrics. Depends on D2.
- [ ] **D4:** Detect duplicate content and cross-split leakage; define reject/warn rules.
  Depends on D2/A5. Exact hashes do not detect every near-duplicate image.
- [ ] **D5:** Support multiple source nodes and settings on the canvas. Depends on D1/D2.

This is origin-independent dataset composition, not a commitment to synthetic generation.

### Epic E — Data ingestion — Research first

- [ ] **E1 — High, spike:** Compare private object storage and import/archive repositories
  on access, region, limits, versioning, licences and costs. Treat Kaggle/Hugging Face
  as potential sources and Zenodo as potential archival/publishing infrastructure;
  verify current terms and capabilities before selection. DVC/lakeFS are optional.
- [ ] **E2 — Medium:** Choose resumable/multipart uploads and secure additional archive
  formats, including tar only with traversal/link/decompression-bomb protections.
  Add UI progress/resume. An S3 design depends on C2; tus need not.
- [ ] **E3 — Medium:** Store third-party credentials securely, with encryption/key
  management, redaction, revocation and tenant-bound access.
- [ ] **E4 — Medium:** Implement selected, version-pinned Kaggle/Hugging Face/Zenodo
  connectors as import jobs; persist source provenance/licences. Depends on E1 and
  E3 for authenticated/private imports.
- [ ] **E5 — Optional:** Generic URL import with DNS/redirect-aware SSRF defenses,
  scheme restrictions, egress policy, timeouts and resource limits.
- [ ] **E6 — Optional:** Detect likely dataset formats and suggest settings; ask users
  to confirm ambiguous layouts rather than silently assigning labels.
- [ ] **E7 — Medium:** Record licence/provenance and surface usage restrictions.
  Verify actual dataset/version terms rather than hardcoding blanket provider licences.

### Epic F — Access and governance — Prioritize with intended customers

- [ ] **F1 — Partial:** Extend the existing top-level listing (maximum 50 entries) with
  paginated subfolder browsing and safe thumbnail generation.
- [ ] **F2:** Add dataset-version download/export with ownership checks; signed URLs
  depend on C2, but local authenticated downloads need not.
- [ ] **F3 — Product decision:** Organisations, memberships, roles and shared resources;
  migrate ownership checks and test tenant separation.
- [ ] **F4:** Storage quotas, atomic usage/reservation accounting and actionable upload errors.
- [ ] **F5:** Audit logging, encryption/access policy and lawful erasure workflows.
  Prefixes alone are not tenant isolation; access controls must enforce boundaries.
  GDPR workflows require a policy for backups, legal retention and data minimization,
  not merely immediate deletion of live files.
- [ ] **F6 — Existing API/UI gap:** Add dataset/version delete, trash and restore controls
  with retention dates, purge status and active-job error handling.

### Epic G — Additional AI capabilities — Product decisions, not initial requirements

- [ ] **G1:** Evaluate anomaly detection on normal-only manufacturing images; benchmark
  candidate methods and libraries before adding a new fit/evaluation path.
- [ ] **G2:** High-resolution tiling, annotation-aware crops and prediction stitching/
  box merging. Test small defects and boundary effects.
- [ ] **G3:** Instance segmentation: task schema, COCO masks, targets, heads, mask metrics
  and inference/UI support. Removing the Mask R-CNN filter alone is insufficient.
- [ ] **G4:** Budgeted hyperparameter search with child-job tracking, cancellation,
  reproducibility and comparisons; avoid repeated tuning on the final test set.
- [ ] **G5 — Verification only first:** Test frozen pretrained/DINO-style backbones
  through existing timm and freeze-patch support before creating a new feature.

### Epic H — Platform scalability — Demand-driven

- [ ] **H1:** Multi-machine workers: storage availability, environment materialization,
  editor-session routing/ownership and worker discovery. Depends on C3/C4; object
  storage alone does not make sessions or Nix environments distributed.
- [ ] **H2 — Later:** Multi-GPU DDP with resource reservation, process lifecycle and
  checkpoint/recovery tests. First establish a demonstrated workload requirement.
- [ ] **H3:** Per-user/organisation concurrency limits and fair scheduling, with atomic
  claims and resource-aware admission.
- [ ] **H4 — Verify first:** Enforce global/per-worker/per-user session limits. Idle
  timeout already exists; do not recreate it as a new feature.
- [ ] **H5 — Business decision:** Define pricing and usage semantics before implementing
  rated usage, billing integrations or invoices.

## 3. Completed original scope — do not duplicate

| ID | Implemented functionality | Remaining caveat |
| --- | --- | --- |
| A1 | Upload reservation/status and stale-upload cleanup | Concurrency/reconciliation follow-ups above |
| A2 | Dataset/version delete, restore and retention | Deadline semantics, cleanup scheduling and UI |
| A3 | Recorded references, active-job guard and rerun validation | Legacy backfill and atomicity |
| A6 | Resumable training checkpoints and retry flag | Atomic fallback and incomplete-epoch behavior |
| B1 | Owned best/last checkpoint downloads | Integrity checks pending A5 |
| B2 | Async batch prediction and mask downloads | Coverage, geometry and old-engine compatibility |

Use the team's review/merge criteria to close these tickets. A pushed implementation
is not automatically a merged release or proof that all follow-up risks are resolved.

## 4. Deferred and dropped scope

- **Deferred:** Christian's CAD/rendering/generative synthetic-data proposal.
- **Deferred:** CVAT/Label Studio integration, SAM-assisted labelling and active learning.
- **Deferred:** 3D CNN support, training large foundation models and keypoint detection.
- **Dropped I1:** Do not add the previously referenced plan document merely because it
  is absent; it is explicitly ignored by the repository.

These may be revisited through explicit product decisions, not silently folded into
the initial storage/recovery implementation.

## 5. Suggested order when revisiting

1. Resolve SEC1 and the immediate correctness gaps before any shared/public deployment.
2. Finish A4/A5/A7 reliability work; coordinate REL2/REL3 with it.
3. Complete E1 storage decisions, then C1–C4 and D1–D5 as separate deliverable tracks.
4. Add model registry and practical frontend access to existing predictions/artifacts.
5. Select import, team/governance and AI-expansion features based on validated user needs.

For each selected task, add an owner, acceptance criteria, dependencies and actual
Jira key. Recheck the current code and branch status before implementing: this is a
dated backlog snapshot, not an automatically synchronized issue tracker.