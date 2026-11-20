import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ArrowRight, Boxes, Database, Plus, Workflow } from "lucide-react";

import { errorMessage } from "../api/client";
import { createPipeline, createSandbox, environmentTemplates, getUsage, listSandboxes } from "../api/sandboxesApi";
import { listTeams } from "../api/teamsApi";
import { createProject } from "../features/pipeline/createProject";
import { TEMPLATES } from "../features/pipeline/defaults";
import { formatCount, formatDate, formatSeconds } from "../utils/format";

export default function SandboxesPage() {
  const navigate = useNavigate();
  const [sandboxes, setSandboxes] = useState(null);
  const [templates, setTemplates] = useState(["pytorch-cpu"]);
  const [usage, setUsage] = useState(null);
  const [form, setForm] = useState({ name: "", template: "pytorch-cpu", team_id: "" });
  const [pipelineName, setPipelineName] = useState("");
  const [pipelineTemplate, setPipelineTemplate] = useState(TEMPLATES[0].id);
  const [createdSandbox, setCreatedSandbox] = useState(null);
  const [busy, setBusy] = useState(false);
  const [teams, setTeams] = useState([]);
  const [error, setError] = useState("");
  const [warnings, setWarnings] = useState([]);

  useEffect(() => {
    listSandboxes().then(setSandboxes).catch((caught) => setError(errorMessage(caught)));
    environmentTemplates().then((catalog) => setTemplates(catalog.templates)).catch((caught) =>
      setWarnings((current) => [...current, `Could not load environments: ${errorMessage(caught)}`])
    );
    getUsage().then(setUsage).catch((caught) =>
      setWarnings((current) => [...current, `Could not load usage: ${errorMessage(caught)}`])
    );
    listTeams().then(setTeams).catch((caught) =>
      setWarnings((current) => [...current, `Could not load teams: ${errorMessage(caught)}`])
    );
  }, []);

  const create = async (event) => {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const { sandbox, pipeline } = await createProject(
        { form, pipelineName, pipelineTemplate, existingSandbox: createdSandbox },
        { createSandbox, createPipeline },
        (sandbox) => {
        setCreatedSandbox(sandbox);
        setSandboxes((current) => current ? [sandbox, ...current] : [sandbox]);
        }
      );
      navigate(`/sandboxes/${sandbox.id}/pipelines/${pipeline.id}`);
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setBusy(false);
    }
  };

  const createOnly = async () => {
    if (!createdSandbox && !form.name.trim()) {
      setError("Enter a sandbox name to continue.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const sandbox = createdSandbox || await createSandbox({ ...form, team_id: form.team_id || null });
      navigate(`/sandboxes/${sandbox.id}`);
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Your workspace</p>
          <h1>Build something new</h1>
          <p className="muted">Create a sandbox and your first pipeline, then open the visual editor.</p>
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

      <section className="journey" aria-label="Your workflow">
        <div><span>01</span><strong>Set up</strong><small>Name your workspace</small></div>
        <div><span>02</span><strong>Add data</strong><small>Upload your images</small></div>
        <div><span>03</span><strong>Build &amp; train</strong><small>Configure your pipeline</small></div>
        <div><span>04</span><strong>Deploy</strong><small>Put your model to work</small></div>
      </section>

      <form className="panel creation-form" onSubmit={create}>
        <div className="creation-heading">
          <div className="feature-icon"><Plus size={20} aria-hidden="true" /></div>
          <div><h2>Start a new project</h2><p className="muted">A sandbox keeps your pipeline and its software environment together.</p></div>
        </div>
        <div className="form-grid">
        <label className="field">
          <span>Sandbox name</span>
          <input
            required
            placeholder="e.g. chest x-ray classifier"
            value={form.name}
            disabled={Boolean(createdSandbox) || busy}
            onChange={(event) => setForm({ ...form, name: event.target.value })}
          />
        </label>
        <label className="field">
          <span>Environment</span>
          <select disabled={Boolean(createdSandbox) || busy} value={form.template} onChange={(event) => setForm({ ...form, template: event.target.value })}>
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
            <select disabled={Boolean(createdSandbox) || busy} value={form.team_id} onChange={(event) => setForm({ ...form, team_id: event.target.value })}>
              <option value="">only me</option>
              {teams.map((team) => (
                <option key={team.id} value={team.id}>
                  {team.name}
                </option>
              ))}
            </select>
          </label>
        )}
        <label className="field">
          <span>Pipeline name</span>
          <input required disabled={busy} placeholder="e.g. first experiment" value={pipelineName} onChange={(event) => setPipelineName(event.target.value)} />
        </label>
        <label className="field">
          <span>Starting point</span>
          <select disabled={busy} value={pipelineTemplate} onChange={(event) => setPipelineTemplate(event.target.value)}>
            {TEMPLATES.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
          </select>
        </label>
        </div>
        <p className="muted small">A pipeline describes your data, model and training steps. You can change the starting point in the editor.</p>
        {createdSandbox && error && <p className="small">Your sandbox was created. Retry creating the pipeline, or <Link to={`/sandboxes/${createdSandbox.id}`}>open the sandbox</Link>.</p>}
        <div className="creation-actions">
        <button className="button primary" type="submit" disabled={busy}>
          {busy ? "Creating…" : createdSandbox ? "Retry pipeline" : "Create and open editor"} <ArrowRight size={16} aria-hidden="true" />
        </button>
        <button className="button ghost" type="button" onClick={createOnly} disabled={busy}>
          {createdSandbox ? "Open sandbox" : "Create sandbox only"}
        </button>
        </div>
      </form>

      {error && <p className="error" role="alert">{error}</p>}
      {warnings.map((warning) => <p className="error small" role="alert" key={warning}>{warning}</p>)}
      {sandboxes === null && !error && <p className="muted">Loading…</p>}
      <div className="section-heading">
        <div><h2>Your sandboxes</h2><p className="muted small">Return to a project or compare its runs.</p></div>
        <Link className="button" to="/datasets"><Database size={16} aria-hidden="true" /> Manage datasets</Link>
      </div>
      {sandboxes?.length === 0 && <div className="empty-state"><Workflow size={24} aria-hidden="true" /><strong>Your projects will appear here</strong><p>Start with the form above, or <Link to="/datasets">add a dataset</Link> first.</p></div>}

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
