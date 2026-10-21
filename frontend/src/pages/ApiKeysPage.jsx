import { useCallback, useEffect, useState } from "react";
import { KeyRound } from "lucide-react";

import { createApiKey, listApiKeys, revokeApiKey } from "../api/authApi";
import { errorMessage } from "../api/client";
import { listDeployments } from "../api/modelsApi";
import { formatDate } from "../utils/format";

export default function ApiKeysPage() {
  const [keys, setKeys] = useState([]);
  const [deployments, setDeployments] = useState([]);
  const [form, setForm] = useState({ name: "", scope: "inference", deployment_id: "" });
  const [created, setCreated] = useState(null);
  const [error, setError] = useState("");

  const load = useCallback(() => listApiKeys().then(setKeys).catch((caught) => setError(errorMessage(caught))), []);
  useEffect(() => {
    load();
    listDeployments().then(setDeployments).catch(() => {});
  }, [load]);

  const create = async (event) => {
    event.preventDefault();
    setError("");
    try {
      const payload = { name: form.name, scope: form.scope };
      if (form.scope === "inference" && form.deployment_id) payload.deployment_id = form.deployment_id;
      setCreated(await createApiKey(payload));
      setForm({ ...form, name: "" });
      load();
    } catch (caught) {
      setError(errorMessage(caught));
    }
  };

  const revoke = async (key) => {
    if (!window.confirm(`Revoke “${key.name}”? Programs using it stop working.`)) return;
    await revokeApiKey(key.id);
    load();
  };

  const deploymentName = (id) => deployments.find((d) => d.id === id)?.name || id;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>API keys</h1>
          <p className="muted">
            Inference keys may only call deployments (schema and predict); use them in production. Full keys act as
            you on the whole API.
          </p>
        </div>
      </header>
      <form className="panel inline-form" onSubmit={create}>
        <label className="field grow">
          <span>Name</span>
          <input required value={form.name} placeholder="e.g. factory line 3" onChange={(e) => setForm({ ...form, name: e.target.value })} />
        </label>
        <label className="field">
          <span>Scope</span>
          <select value={form.scope} onChange={(e) => setForm({ ...form, scope: e.target.value })}>
            <option value="inference">inference</option>
            <option value="all">all (full API)</option>
          </select>
        </label>
        {form.scope === "inference" && (
          <label className="field">
            <span>Deployment</span>
            <select value={form.deployment_id} onChange={(e) => setForm({ ...form, deployment_id: e.target.value })}>
              <option value="">any of mine</option>
              {deployments.map((deployment) => (
                <option key={deployment.id} value={deployment.id}>
                  {deployment.name}
                </option>
              ))}
            </select>
          </label>
        )}
        <button className="button primary" type="submit">
          <KeyRound size={16} aria-hidden="true" /> Create key
        </button>
      </form>
      {error && <p className="error" role="alert">{error}</p>}
      {created && (
        <div className="panel" role="status">
          <p>
            Copy this key now; it is not shown again: <code className="secret">{created.key}</code>
          </p>
        </div>
      )}
      {keys.length > 0 && (
        <table className="table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Scope</th>
              <th>Starts with</th>
              <th>Created</th>
              <th>Last used</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {keys.map((key) => (
              <tr key={key.id} className={key.revoked_at ? "muted" : undefined}>
                <td>{key.name}</td>
                <td>
                  {key.scope}
                  {key.deployment_id ? ` · ${deploymentName(key.deployment_id)}` : ""}
                </td>
                <td>
                  <code>{key.lookup}…</code>
                </td>
                <td>{formatDate(key.created_at)}</td>
                <td>{formatDate(key.last_used_at)}</td>
                <td>
                  {key.revoked_at ? (
                    "revoked"
                  ) : (
                    <button type="button" className="link-button" onClick={() => revoke(key)}>
                      Revoke
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
