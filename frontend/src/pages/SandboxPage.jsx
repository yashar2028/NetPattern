import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { GitCompare, Plus, RefreshCw, Workflow } from "lucide-react";

import { errorMessage } from "../api/client";
import { listRuns } from "../api/runsApi";
import { createPipeline, getSandbox, listPipelines, upgradeSandbox } from "../api/sandboxesApi";
import StatusBadge from "../components/StatusBadge";
import { TEMPLATES } from "../features/pipeline/defaults";
import { formatDate } from "../utils/format";

export default function SandboxPage() {
  const { sandboxId } = useParams();
  const navigate = useNavigate();
  const [sandbox, setSandbox] = useState(null);
  const [pipelines, setPipelines] = useState([]);
  const [runs, setRuns] = useState([]);
  const [name, setName] = useState("");
  const [template, setTemplate] = useState(TEMPLATES[0].id);
  const [error, setError] = useState("");
  const [picked, setPicked] = useState([]);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    getSandbox(sandboxId).then(setSandbox).catch((caught) => setError(errorMessage(caught)));
    listPipelines(sandboxId).then(setPipelines).catch(() => {});
    listRuns({ sandbox_id: sandboxId }).then(setRuns).catch(() => {});
  }, [sandboxId]);

  const newPipeline = async (event) => {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const spec = TEMPLATES.find((item) => item.id === template).spec();
      const pipeline = await createPipeline(sandboxId, { name, spec });
      navigate(`/sandboxes/${sandboxId}/pipelines/${pipeline.id}`);
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setBusy(false);
    }
  };

  const upgrade = async () => {
    if (!window.confirm("Use the newest engine for the next runs? Earlier runs keep their own versions.")) return;
    try {
      setSandbox(await upgradeSandbox(sandboxId));
    } catch (caught) {
      setError(errorMessage(caught));
    }
  };

  if (error && !sandbox) return <p className="error page-message" role="alert">{error}</p>;
  if (!sandbox) return <p className="muted page-message">Loading…</p>;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="muted small">
            <Link to="/sandboxes">Sandboxes</Link> /
          </p>
          <h1>{sandbox.name}</h1>
          <p className="muted">
            {sandbox.template} ·{" "}
            {sandbox.engine_version ? (
              <>
                engine {sandbox.engine_version}, pinned{" "}
                <code title={sandbox.engine_snapshot}>{sandbox.engine_snapshot.slice(0, 12)}</code>
              </>
            ) : (
              "engine is pinned on first use"
            )}
          </p>
        </div>
        {sandbox.engine_version && (
          <button type="button" className="button ghost" onClick={upgrade}>
            <RefreshCw size={16} aria-hidden="true" /> Upgrade engine
          </button>
        )}
      </header>
      {error && <p className="error" role="alert">{error}</p>}

      <section className="section">
        <h2>Pipelines</h2>
        <details className="compact-disclosure">
        <summary>New pipeline</summary>
        <form className="panel inline-form" onSubmit={newPipeline}>
          <label className="field grow">
            <span>New pipeline</span>
            <input required placeholder="e.g. resnet18 fine-tune" value={name} onChange={(e) => setName(e.target.value)} />
          </label>
          <label className="field">
            <span>Start from</span>
            <select value={template} onChange={(e) => setTemplate(e.target.value)}>
              {TEMPLATES.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.label}
                </option>
              ))}
            </select>
          </label>
          <button className="button primary" type="submit" disabled={busy}>
            <Plus size={16} aria-hidden="true" /> {busy ? "Creating…" : "Create"}
          </button>
        </form>
        </details>
        {pipelines.length === 0 && <p className="muted small">Create a pipeline to choose a starting model and open the editor.</p>}
        <div className="card-grid">
          {pipelines.map((pipeline) => (
            <Link key={pipeline.id} className="card card-link" to={`/sandboxes/${sandboxId}/pipelines/${pipeline.id}`}>
              <Workflow className="card-icon" size={20} aria-hidden="true" />
              <h3>{pipeline.name}</h3>
              <p className="muted small">
                {pipeline.latest_version ? `version ${pipeline.latest_version}` : "not saved yet"} · {formatDate(pipeline.created_at)}
              </p>
            </Link>
          ))}
        </div>
      </section>

      <section className="section">
        <div className="page-header">
          <h2>Runs</h2>
          {runs.length > 1 && (
            <button
              type="button"
              className="button"
              disabled={picked.length < 2}
              onClick={() => navigate(`/compare?runs=${picked.join(",")}`)}
            >
              <GitCompare size={16} aria-hidden="true" /> Compare {picked.length > 1 ? picked.length : ""} runs
            </button>
          )}
        </div>
        {runs.length === 0 ? (
          <p className="muted">No runs yet. Open a pipeline and press Run.</p>
        ) : (
          <div className="table-scroll compact-list" role="region" aria-label="Recent sandbox runs" tabIndex={0}>
          <table className="table">
            <thead>
              <tr>
                <th>
                  <span className="visually-hidden">Compare</span>
                </th>
                <th>Run</th>
                <th>Kind</th>
                <th>Status</th>
                <th>Tier</th>
                <th>Started</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr key={run.id}>
                  <td>
                    <input
                      type="checkbox"
                      aria-label={`compare ${run.id}`}
                      checked={picked.includes(run.id)}
                      onChange={(event) =>
                        setPicked(event.target.checked ? [...picked, run.id] : picked.filter((id) => id !== run.id))
                      }
                    />
                  </td>
                  <td>
                    <Link to={`/runs/${run.id}`}>
                      <code>{run.id.slice(0, 16)}</code>
                    </Link>
                  </td>
                  <td>{run.kind}</td>
                  <td>
                    <StatusBadge status={run.status} />
                  </td>
                  <td>{run.hardware_tier}</td>
                  <td>{formatDate(run.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          </div>
        )}
      </section>
    </div>
  );
}
