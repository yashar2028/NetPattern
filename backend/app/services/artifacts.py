"""Things a run can hand out: a "reproduce locally" bundle and a printable report (Phase 6)."""

from __future__ import annotations

import base64
import html
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any

from app.models.job import Environment, Job

# A dataset root inside the storage volume: /data/datasets/users/<user>/<dataset>/v<n>[/sub]
DATASET_ROOT = re.compile(r"^.*/datasets/users/[^/]+/[^/]+/v\d+")
REPORT_ERROR_IMAGES = 12
REPORT_CSS = """
body { font-family: system-ui, sans-serif; margin: 32px auto; max-width: 960px; padding: 0 16px;
  color: #161a23; }
table { border-collapse: collapse; margin: 8px 0 24px; font-size: 14px; }
th, td { border: 1px solid #e2e5ee; padding: 4px 10px; text-align: left; }
td { font-variant-numeric: tabular-nums; }
.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 12px; }
figure { margin: 0; font-size: 12px; }
img { width: 100%; aspect-ratio: 1; object-fit: contain; background: #f6f7fb; }
@media print { body { margin: 0; } }
"""


def _local_request(request: dict[str, Any]) -> dict[str, Any]:
    """The engine request with dataset roots pointing at ./data instead of the server volume."""
    request = json.loads(json.dumps(request))
    for node in request.get("pipeline", {}).get("nodes", []):
        params = node.get("params", {})
        if node.get("type") == "dataset" and isinstance(params.get("root"), str):
            params["root"] = DATASET_ROOT.sub("./data", params["root"])
    return request


def reproduce_bundle(job: Job, environment: Environment) -> bytes:
    match = re.search(r'path:([^"]+)"', environment.flake_nix)
    snapshot = Path(match.group(1)) if match else None
    if snapshot is None or not snapshot.is_dir():
        raise FileNotFoundError(f"engine snapshot {snapshot} is missing")
    request_file = Path(job.run_dir or "") / "request.json"
    request = json.loads(request_file.read_text(encoding="utf-8")) if request_file.exists() else {}
    flake = environment.flake_nix.replace(f"path:{snapshot}", "path:./engine")
    readme = f"""# Reproduce run {job.id}

The exact environment of this run: the same nixpkgs revision, Python, PyTorch and NetPattern
engine ({environment.engine_version}), pinned by `flake.nix` and `engine/flake.lock`.

Needs Nix with flakes on {environment.system}. Extract this archive with `unzip` (it keeps
file permissions, which are part of the environment hash).

1. Put the dataset this run used into `./data` (the extracted zip of the dataset version).
   Synthetic datasets need nothing.
2. Build the environment (the first build downloads PyTorch; later ones are instant):

       nix build .#default

   The store path of `./result` should end in the environment hash `{environment.env_hash}`.
3. Run the job:

       ./result/bin/netpattern-runtime job --request request.json --output result.json

   Events stream to the terminal; `result.json` has the metrics. Training is seeded, so the
   numbers match the platform run on the same hardware type.
"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("flake.nix", flake)
        archive.writestr("request.json", json.dumps(_local_request(request), indent=2))
        archive.writestr("README.md", readme)
        for path in sorted(snapshot.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, f"engine/{path.relative_to(snapshot).as_posix()}")
    return buffer.getvalue()


# --------------------------------------------------------------------------- report


def _table(header: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in header)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(_fmt(cell))}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return "—" if value is None else str(value)


def run_report(job: Job, environment: Environment | None) -> str:
    result = job.result or {}
    training = result.get("training") or {}
    environment_info = result.get("environment") or {}
    sections = []

    facts = [
        ["Run", job.id],
        ["Kind", job.kind],
        ["Status", job.status.value],
        ["Started", job.started_at],
        ["Finished", job.finished_at],
        ["Hardware tier", job.hardware_tier],
        ["Engine", environment_info.get("engine")],
        ["PyTorch", environment_info.get("torch")],
        ["Environment hash", environment.env_hash if environment else None],
        [
            "Samples (train / val / test)",
            " / ".join(
                str((result.get("dataset") or {}).get("splits", {}).get(s, 0))
                for s in ("train", "val", "test")
            ),
        ],
        ["Parameters", (result.get("model") or {}).get("params")],
        ["Best epoch", f"{training.get('best_epoch')} of {training.get('epochs_run')}"],
    ]
    sections.append(_table(["", ""], facts))

    history = training.get("history") or []
    if history:
        columns = sorted({key for epoch in history for key in epoch["metrics"]})
        rows = [[e["epoch"], *(e["metrics"].get(c) for c in columns)] for e in history]
        sections.append("<h2>Training</h2>" + _table(["epoch", *columns], rows))

    for split, evaluation in (result.get("evaluation") or {}).items():
        part = [f"<h2>Evaluation: {html.escape(split)}</h2>"]
        part.append(_table(["metric", "value"], [[k, v] for k, v in evaluation["metrics"].items()]))
        details = evaluation.get("details") or {}
        matrix = details.get("confusion_matrix")
        if matrix:
            labels = matrix["labels"]
            rows = [[labels[i], *row] for i, row in enumerate(matrix["matrix"])]
            part.append("<h3>Confusion matrix (rows: true, columns: predicted)</h3>")
            part.append(_table(["", *labels], rows))
        per_class = details.get("per_class") or []
        if per_class:
            keys = [k for k in per_class[0] if k != "class"]
            part.append("<h3>Per class</h3>")
            part.append(
                _table(["class", *keys], [[r["class"], *(r[k] for k in keys)] for r in per_class])
            )
        sections.append("".join(part))

    errors = result.get("errors") or {}
    if errors.get("samples"):
        figures = []
        for sample in errors["samples"][:REPORT_ERROR_IMAGES]:
            path = Path(job.run_dir or "") / sample["image"]
            if not path.is_file():
                continue
            data = base64.b64encode(path.read_bytes()).decode()
            caption = f"true {sample['true']}, predicted {sample['predicted']} ({sample['score']})"
            figures.append(
                f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(caption)}">'
                f"<figcaption>{html.escape(caption)}</figcaption></figure>"
            )
        sections.append(
            f"<h2>Misclassified {html.escape(errors['split'])} samples "
            f"({errors['total_errors']} of {errors['evaluated']})</h2>"
            f'<div class="gallery">{"".join(figures)}</div>'
        )

    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Run report {html.escape(job.id)}</title><style>{REPORT_CSS}</style></head><body>"
        "<h1>NetPattern run report</h1>"
        "<p>Generated from the stored run record; numbers are the platform's own evaluation.</p>"
        + "".join(sections)
        + "</body></html>"
    )
