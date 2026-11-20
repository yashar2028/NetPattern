import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Boxes, Plus } from "lucide-react";

import { errorMessage } from "../api/client";
import { createSandbox, environmentTemplates, getUsage, listSandboxes } from "../api/sandboxesApi";
import { listTeams } from "../api/teamsApi";
import { formatCount, formatDate, formatSeconds } from "../utils/format";

export default function SandboxesPage() {
  const navigate = useNavigate();
  const [sandboxes, setSandboxes] = useState(null);
  const [templates, setTemplates] = useState(["pytorch-cpu"]);
  const [usage, setUsage] = useState(null);
  const [form, setForm] = useState({ name: "", template: "pytorch-cpu", team_id: "" });
  const [teams, setTeams] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    listSandboxes().then(setSandboxes).catch((caught) => setError(errorMessage(caught)));
    environmentTemplates().then((catalog) => setTemplates(catalog.templates)).catch(() => {});
    getUsage().then(setUsage).catch(() => {});
    listTeams().then(setTeams).catch(() => {});
  }, []);

  const create = async (event) => {
    event.preventDefault();
    try {
      const sandbox = await createSandbox({ ...form, team_id: form.team_id || null });
      navigate(`/sandboxes/${sandbox.id}`);
    } catch (caught) {
      setError(errorMessage(caught));
    }
  };

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Sandboxes</h1>
          <p className="muted">One sandbox per CNN you build. Its software versions stay pinned.</p>
        </div>
        {usage && (
          <div className="stat" title="Compute units = tier rate × seconds of finished runs and dedicated deployments">
            <span className="stat-value">{formatCount(Math.round(usage.total_units))} units</span>
            <span className="stat-label">
              {formatSeconds(usage.total_seconds)} of compute · {usage.jobs} runs · {usage.inference_calls} API calls
            </span>
          </div>
        )}
      </header>

      <form className="panel inline-form" onSubmit={create}>
        <label className="field grow">
          <span>New sandbox</span>
          <input
            required
            placeholder="e.g. chest x-ray classifier"
            value={form.name}
            onChange={(event) => setForm({ ...form, name: event.target.value })}
          />
        </label>
        <label className="field">
          <span>Environment</span>
          <select value={form.template} onChange={(event) => setForm({ ...form, template: event.target.value })}>
            {templates.map((template) => (
              <option key={template} value={template}>
                {template}
              </option>
            ))}
          </select>
        </label>
        {teams.length > 0 && (
          <label className="field">
            <span>Share with</span>
            <select value={form.team_id} onChange={(event) => setForm({ ...form, team_id: event.target.value })}>
              <option value="">only me</option>
              {teams.map((team) => (
                <option key={team.id} value={team.id}>
                  {team.name}
                </option>
              ))}
            </select>
          </label>
        )}
        <button className="button primary" type="submit">
          <Plus size={16} aria-hidden="true" /> Create
        </button>
      </form>

      {error && <p className="error" role="alert">{error}</p>}
      {sandboxes === null && !error && <p className="muted">Loading…</p>}
      {sandboxes?.length === 0 && <p className="muted">No sandboxes yet. Create your first one above.</p>}

      <div className="card-grid">
        {sandboxes?.map((sandbox) => (
          <Link key={sandbox.id} to={`/sandboxes/${sandbox.id}`} className="card card-link">
            <Boxes className="card-icon" size={20} aria-hidden="true" />
            <h2>{sandbox.name}</h2>
            <p>{sandbox.template}</p>
            <p className="muted small">
              {sandbox.engine_version ? `engine ${sandbox.engine_version} (pinned)` : "engine pinned on first use"}
              {" · "}created {formatDate(sandbox.created_at)}
              {sandbox.team_id ? " · shared with a team" : ""}
            </p>
          </Link>
        ))}
      </div>
    </div>
  );
}
