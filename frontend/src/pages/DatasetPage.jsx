import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { BarChart3, Upload } from "lucide-react";

import { errorMessage } from "../api/client";
import { getDataset, profileVersion, uploadVersion } from "../api/datasetsApi";
import { TASK_TYPES } from "../constants";
import { formatBytes, formatDate } from "../utils/format";

const FORMATS = ["image_folder", "csv", "segmentation_folders", "coco", "voc", "yolo"];

export default function DatasetPage() {
  const { datasetId } = useParams();
  const navigate = useNavigate();
  const [dataset, setDataset] = useState(null);
  const [progress, setProgress] = useState(null);
  const [profile, setProfile] = useState({ version: "", task: TASK_TYPES[0].value, format: "image_folder", subpath: "" });
  const [error, setError] = useState("");

  const load = useCallback(
    () => getDataset(datasetId).then(setDataset).catch((caught) => setError(errorMessage(caught))),
    [datasetId]
  );
  useEffect(() => {
    load();
  }, [load]);

  const upload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setError("");
    setProgress(0);
    try {
      await uploadVersion(datasetId, file, setProgress);
      await load();
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setProgress(null);
      event.target.value = "";
    }
  };

  const runProfile = async (event) => {
    event.preventDefault();
    try {
      const run = await profileVersion(datasetId, profile.version, {
        task: { type: profile.task },
        dataset: { format: profile.format },
        subpath: profile.subpath || null,
      });
      navigate(`/runs/${run.id}`);
    } catch (caught) {
      setError(errorMessage(caught));
    }
  };

  if (!dataset) return <p className="muted page-message">{error || "Loading…"}</p>;
  const latest = dataset.versions.at(-1);

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="muted small">
            <Link to="/datasets">Datasets</Link> /
          </p>
          <h1>{dataset.name}</h1>
          <p className="muted small">
            id <code>{dataset.id}</code>
          </p>
        </div>
        <label className="button primary file-button">
          <Upload size={16} aria-hidden="true" /> Upload zip
          <input type="file" accept=".zip,application/zip" onChange={upload} hidden />
        </label>
      </header>

      {progress !== null && (
        <div className="panel">
          <p>{progress < 1 ? `Uploading… ${Math.round(progress * 100)}%` : "Extracting the archive…"}</p>
          <progress max="1" value={progress} />
        </div>
      )}
      {error && <p className="error" role="alert">{error}</p>}

      <section className="section">
        <h2>Versions</h2>
        {dataset.versions.length === 0 ? (
          <p className="muted">No versions yet. Upload a zip of your dataset folder.</p>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>Version</th>
                <th>Files</th>
                <th>Size</th>
                <th>Top level</th>
                <th>Uploaded</th>
                <th>Id (for pipelines)</th>
              </tr>
            </thead>
            <tbody>
              {dataset.versions.map((version) => (
                <tr key={version.id}>
                  <td>v{version.number}</td>
                  <td>{version.files}</td>
                  <td>{formatBytes(version.size_bytes)}</td>
                  <td className="small">{version.top_level.join("  ")}</td>
                  <td>{formatDate(version.created_at)}</td>
                  <td>
                    <code>{version.id}</code>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      {latest && (
        <section className="section">
          <h2>Profile a version</h2>
          <p className="muted">Class counts, image sizes, intensities and warnings, computed in a sandbox.</p>
          <form className="panel inline-form" onSubmit={runProfile}>
            <label className="field">
              <span>Version</span>
              <select required value={profile.version} onChange={(e) => setProfile({ ...profile, version: e.target.value })}>
                <option value="">choose…</option>
                {dataset.versions.map((version) => (
                  <option key={version.id} value={version.number}>
                    v{version.number}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Task</span>
              <select value={profile.task} onChange={(e) => setProfile({ ...profile, task: e.target.value })}>
                {TASK_TYPES.map((task) => (
                  <option key={task.value} value={task.value}>
                    {task.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Format</span>
              <select value={profile.format} onChange={(e) => setProfile({ ...profile, format: e.target.value })}>
                {FORMATS.map((format) => (
                  <option key={format}>{format}</option>
                ))}
              </select>
            </label>
            <label className="field grow">
              <span>Folder inside the version (optional)</span>
              <input value={profile.subpath} placeholder="e.g. classification" onChange={(e) => setProfile({ ...profile, subpath: e.target.value })} />
            </label>
            <button className="button" type="submit">
              <BarChart3 size={16} aria-hidden="true" /> Profile
            </button>
          </form>
        </section>
      )}
    </div>
  );
}
