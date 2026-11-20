import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Pause, Play, Trash2, Upload } from "lucide-react";

import { errorMessage } from "../api/client";
import {
  deleteDeployment,
  deploymentRequests,
  deploymentSchema,
  getDeployment,
  predict,
  setDeploymentState,
} from "../api/modelsApi";
import LineChart from "../components/LineChart";
import PredictionView from "../components/PredictionView";
import StatusBadge from "../components/StatusBadge";
import { API_BASE_URL } from "../constants";
import { formatDate, formatMetric } from "../utils/format";

function snippets(deploymentId) {
  const url = `${API_BASE_URL}/v1/deployments/${deploymentId}/predict`;
  return {
    curl: `curl -X POST ${url} \\
  -H "Authorization: Bearer $NETPATTERN_API_KEY" \\
  -F "files=@image.png" \\
  -F 'options={"top_k": 3}'`,
    python: `import os, requests

response = requests.post(
    "${url}",
    headers={"Authorization": f"Bearer {os.environ['NETPATTERN_API_KEY']}"},
    files=[("files", open("image.png", "rb"))],
)
print(response.json()["outputs"])`,
    sdk: `from netpattern import Client  # pip install ./sdk/python

client = Client(api_key=os.environ["NETPATTERN_API_KEY"], base_url="${API_BASE_URL}")
print(client.predict("${deploymentId}", ["image.png"]))`,
  };
}

function Playground({ deployment, onCalled }) {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [explain, setExplain] = useState(false);
  const canExplain = deployment.taskType?.startsWith("classification");

  const run = async (event) => {
    const chosen = event.target.files?.[0];
    event.target.value = "";
    if (!chosen) return;
    setFile(chosen);
    setResult(null);
    setError("");
    setBusy(true);
    try {
      setResult(await predict(deployment.id, [chosen], { top_k: 3, explain: explain && canExplain }));
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setBusy(false);
      onCalled();
    }
  };

  return (
    <section className="section">
      <h2>Try it</h2>
      <div className="panel stack">
        <label className="button file-button">
          <Upload size={16} aria-hidden="true" /> {busy ? "Predicting…" : "Choose an image"}
          <input type="file" accept="image/*,.dcm" onChange={run} hidden disabled={deployment.status !== "active"} />
        </label>
        {canExplain && (
          <label className="field checkbox">
            <input type="checkbox" checked={explain} onChange={(event) => setExplain(event.target.checked)} />
            <span>Explain (Grad-CAM heatmap of what the model looked at)</span>
          </label>
        )}
        {deployment.status !== "active" && <p className="muted small">The deployment must be active.</p>}
        {error && <p className="error">{error}</p>}
        {result && (
          <>
            <p className="muted small">
              {result.model_version} · model {formatMetric(result.timing_ms.inference)} ms · total{" "}
              {formatMetric(result.timing_ms.total)} ms
            </p>
            <PredictionView task={result.task} output={result.outputs[0]} file={file} />
          </>
        )}
      </div>
    </section>
  );
}

export default function DeploymentPage() {
  const { deploymentId } = useParams();
  const navigate = useNavigate();
  const [deployment, setDeployment] = useState(null);
  const [schema, setSchema] = useState(null);
  const [requests, setRequests] = useState([]);
  const [tab, setTab] = useState("curl");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const loadRequests = useCallback(
    () => deploymentRequests(deploymentId).then(setRequests).catch(() => {}),
    [deploymentId]
  );
  useEffect(() => {
    getDeployment(deploymentId).then(setDeployment).catch((caught) => setError(errorMessage(caught)));
    deploymentSchema(deploymentId).then(setSchema).catch(() => {});
    loadRequests();
  }, [deploymentId, loadRequests]);

  const latency = useMemo(
    () => [
      {
        name: "latency (ms)",
        data: [...requests].reverse().map((record, index) => [index + 1, record.latency_ms]),
      },
    ],
    [requests]
  );

  const act = async (action) => {
    setBusy(true);
    setError("");
    try {
      if (action === "delete") {
        if (!window.confirm("Delete this deployment? Its API keys stop working.")) return;
        await deleteDeployment(deploymentId);
        navigate("/deployments");
        return;
      }
      setDeployment(await setDeploymentState(deploymentId, action));
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setBusy(false);
    }
  };

  if (!deployment) return <p className={`page-message ${error ? "error" : "muted"}`}>{error || "Loading…"}</p>;
  const code = snippets(deployment.id);

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="muted small">
            <Link to="/deployments">Deployments</Link> /
          </p>
          <h1>
            {deployment.name} <StatusBadge status={deployment.status} />
          </h1>
          <p className="muted small">
            <code>{deployment.id}</code> · {deployment.mode} · {deployment.hardware_tier}
            {schema ? ` · ${schema.model_version} · ${schema.task}` : ""} · created {formatDate(deployment.created_at)}
          </p>
        </div>
        <div className="toolbar">
          {deployment.status === "active" ? (
            <button type="button" className="button ghost" disabled={busy} onClick={() => act("pause")}>
              <Pause size={16} aria-hidden="true" /> Pause
            </button>
          ) : (
            <button type="button" className="button" disabled={busy} onClick={() => act("resume")}>
              <Play size={16} aria-hidden="true" /> {busy ? "Starting…" : "Start"}
            </button>
          )}
          <button type="button" className="button ghost" disabled={busy} onClick={() => act("delete")}>
            <Trash2 size={16} aria-hidden="true" /> Delete
          </button>
        </div>
      </header>

      {error && <p className="error" role="alert">{error}</p>}
      {deployment.error && <div className="error-box">{deployment.error}</div>}
      {deployment.parity && (
        <p className={deployment.parity.passed === false ? "error" : "muted"}>
          Parity check:{" "}
          {deployment.parity.passed === null
            ? "no samples stored with this model"
            : `${deployment.parity.passed ? "passed" : "failed"} on ${deployment.parity.samples} samples (largest difference ${formatMetric(deployment.parity.max_difference)})`}
        </p>
      )}

      <Playground deployment={{ ...deployment, taskType: schema?.task }} onCalled={loadRequests} />

      <section className="section">
        <h2>Use via API</h2>
        <p className="muted">
          Create an inference key for this deployment on the <Link to="/api-keys">API keys</Link> page.
        </p>
        <div className="toolbar" role="tablist">
          {["curl", "python", "sdk"].map((name) => (
            <button
              key={name}
              type="button"
              role="tab"
              aria-selected={tab === name}
              className={`button ${tab === name ? "primary" : "ghost"}`}
              onClick={() => setTab(name)}
            >
              {name === "sdk" ? "Python SDK" : name}
            </button>
          ))}
        </div>
        <pre className="code-block">{code[tab]}</pre>
        {schema && (
          <details>
            <summary>Schema: inputs, outputs and limits</summary>
            <pre className="code-block">{JSON.stringify(schema, null, 2)}</pre>
          </details>
        )}
      </section>

      <section className="section">
        <h2>Requests</h2>
        {requests.length === 0 ? (
          <p className="muted">No calls yet.</p>
        ) : (
          <>
            <figure className="panel chart-panel">
              <figcaption>Latency of the last {requests.length === 1 ? "call" : `${requests.length} calls`}</figcaption>
              <LineChart series={latency} xName="call" label="Latency per call in milliseconds" height={200} />
            </figure>
            <div className="table-scroll">
              <table className="table compact">
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Status</th>
                    <th className="number">Images</th>
                    <th className="number">Latency (ms)</th>
                    <th className="number">Model time (ms)</th>
                    <th>Key</th>
                  </tr>
                </thead>
                <tbody>
                  {requests.map((record) => (
                    <tr key={record.id}>
                      <td>{formatDate(record.created_at)}</td>
                      <td className={record.status_code >= 400 ? "error" : undefined} title={record.error || ""}>
                        {record.status_code}
                      </td>
                      <td className="number">{record.images}</td>
                      <td className="number">{formatMetric(record.latency_ms)}</td>
                      <td className="number">{formatMetric(record.compute_ms)}</td>
                      <td className="small">{record.api_key_id ? <code>{record.api_key_id.slice(0, 12)}</code> : "browser"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </section>
    </div>
  );
}
