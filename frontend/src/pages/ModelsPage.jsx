import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Download, Rocket } from "lucide-react";

import { errorMessage } from "../api/client";
import { createDeployment, exportVersion, listModels } from "../api/modelsApi";
import { formatDate, formatMetric } from "../utils/format";

function DeployForm({ version, modelName }) {
  const navigate = useNavigate();
  const [form, setForm] = useState({ name: `${modelName}-v${version.number}`, mode: "serverless" });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const deploy = async (event) => {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const deployment = await createDeployment({ model_version_id: version.id, ...form });
      navigate(`/deployments/${deployment.id}`);
    } catch (caught) {
      setError(errorMessage(caught));
      setBusy(false);
    }
  };

  return (
    <form className="inline-form compact" onSubmit={deploy}>
      <input aria-label="Deployment name" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} required />
      <select aria-label="Mode" value={form.mode} onChange={(e) => setForm({ ...form, mode: e.target.value })}>
        <option value="serverless">serverless (starts on demand)</option>
        <option value="dedicated">dedicated (always loaded)</option>
      </select>
      <button className="button primary" type="submit" disabled={busy}>
        <Rocket size={16} aria-hidden="true" /> {busy ? "Starting…" : "Deploy"}
      </button>
      {error && <span className="error small">{error}</span>}
    </form>
  );
}

export default function ModelsPage() {
  const navigate = useNavigate();
  const [models, setModels] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    listModels().then(setModels).catch((caught) => setError(errorMessage(caught)));
  }, []);

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Models</h1>
          <p className="muted">
            Trained runs frozen as versions. Register one from a completed run&apos;s page, then deploy it.
          </p>
        </div>
      </header>
      {error && <p className="error" role="alert">{error}</p>}
      {models?.length === 0 && <p className="muted">No models yet. Open a completed training run and press “Register as model”.</p>}
      {models?.map((model) => (
        <section key={model.id} className="section panel">
          <h2>{model.name}</h2>
          <p className="muted small">
            <code>{model.id}</code> · created {formatDate(model.created_at)}
          </p>
          <div className="table-scroll">
            <table className="table">
              <thead>
                <tr>
                  <th>Version</th>
                  <th>Task</th>
                  <th>Test metrics</th>
                  <th>Run</th>
                  <th>Deploy</th>
                  <th>Export</th>
                </tr>
              </thead>
              <tbody>
                {model.versions.map((version) => {
                  const metrics = version.metrics.test || version.metrics.val || {};
                  return (
                    <tr key={version.id}>
                      <td>v{version.number}</td>
                      <td>{version.task.type}</td>
                      <td className="small">
                        {Object.entries(metrics)
                          .slice(0, 3)
                          .map(([name, value]) => `${name} ${formatMetric(value)}`)
                          .join(" · ")}
                      </td>
                      <td>{version.run_id ? <Link to={`/runs/${version.run_id}`}>run</Link> : "—"}</td>
                      <td>
                        <DeployForm version={version} modelName={model.name} />
                      </td>
                      <td>
                        <button
                          type="button"
                          className="button ghost"
                          title="Weights, TorchScript and ONNX, made in the model's own environment"
                          onClick={async () => {
                            try {
                              const run = await exportVersion(model.id, version.number);
                              navigate(`/runs/${run.id}`);
                            } catch (caught) {
                              setError(errorMessage(caught));
                            }
                          }}
                        >
                          <Download size={16} aria-hidden="true" /> Export
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      ))}
    </div>
  );
}
