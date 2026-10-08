import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { ReactFlowProvider } from "@xyflow/react";
import { Loader2, Play, Save } from "lucide-react";
import "@xyflow/react/dist/style.css";

import { errorIssues, errorMessage } from "../api/client";
import { listDatasets } from "../api/datasetsApi";
import { startRun } from "../api/runsApi";
import { getPipeline, getVersion, listVersions, saveVersion, sessionCall } from "../api/sandboxesApi";
import SchemaForm from "../components/SchemaForm";
import LayerEditor from "../features/pipeline/LayerEditor";
import { DatasetForm, EvaluatorForm, ModelForm, TransformsForm } from "../features/pipeline/NodeForms";
import PipelineCanvas, { PipelinePalette } from "../features/pipeline/PipelineCanvas";
import { NODE_BY_TYPE, defaultSpec } from "../features/pipeline/defaults";

function NodePanel({ node, spec, schema, datasets, sandboxId, onParams, onOpenLayers }) {
  const props = { params: node.params || {}, onChange: onParams, schemas: schema.schemas };
  let form;
  switch (node.type) {
    case "dataset":
      form = <DatasetForm {...props} datasets={datasets} />;
      break;
    case "transforms":
      form = <TransformsForm {...props} capabilities={schema.capabilities} />;
      break;
    case "model":
      form = <ModelForm {...props} sandboxId={sandboxId} taskType={spec.task.type} onOpenLayers={onOpenLayers} />;
      break;
    case "trainer":
      form = <SchemaForm schema={schema.schemas.training} value={props.params} onChange={onParams} />;
      break;
    case "evaluator":
      form = <EvaluatorForm {...props} />;
      break;
    default:
      form = null;
  }
  return (
    <aside className="side-panel" aria-label={`${node.type} settings`}>
      <h3>{NODE_BY_TYPE[node.type]?.label}</h3>
      {form}
    </aside>
  );
}

export default function PipelinePage() {
  const { sandboxId, pipelineId } = useParams();
  const navigate = useNavigate();
  const [pipeline, setPipeline] = useState(null);
  const [versions, setVersions] = useState([]);
  const [version, setVersion] = useState(null);
  const [spec, setSpecState] = useState(null);
  const [dirty, setDirty] = useState(false);
  const [schema, setSchema] = useState(null);
  const [schemaError, setSchemaError] = useState("");
  const [datasets, setDatasets] = useState([]);
  const [selected, setSelected] = useState(null);
  const [layersOpen, setLayersOpen] = useState(false);
  const [runOptions, setRunOptions] = useState({ kind: "train", hardware_tier: "cpu" });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const setSpec = useCallback((update) => {
    setSpecState((current) => (typeof update === "function" ? update(current) : update));
    setDirty(true);
  }, []);

  const loadVersion = useCallback(
    async (number) => {
      const loaded = await getVersion(pipelineId, number);
      setVersion(loaded);
      setSpecState(loaded.spec);
      setDirty(false);
      setSelected(null);
    },
    [pipelineId]
  );

  useEffect(() => {
    (async () => {
      try {
        const found = await getPipeline(pipelineId);
        setPipeline(found);
        setVersions(await listVersions(pipelineId));
        if (found.latest_version) await loadVersion(found.latest_version);
        else setSpecState(defaultSpec());
      } catch (caught) {
        setError({ message: errorMessage(caught) });
      }
    })();
    listDatasets().then(setDatasets).catch(() => {});
  }, [pipelineId, loadVersion]);

  // The forms come from the sandbox's own engine; the first call may start the environment.
  useEffect(() => {
    sessionCall(sandboxId, "schema").then(setSchema).catch((caught) => setSchemaError(errorMessage(caught)));
  }, [sandboxId]);

  const save = async () => {
    const saved = await saveVersion(pipelineId, spec);
    setVersion(saved);
    setVersions(await listVersions(pipelineId));
    setDirty(false);
    return saved;
  };

  const act = async (action) => {
    setBusy(true);
    setError(null);
    try {
      await action();
    } catch (caught) {
      setError({ message: errorMessage(caught), issues: errorIssues(caught) });
    } finally {
      setBusy(false);
    }
  };

  const run = () =>
    act(async () => {
      const target = dirty || !version ? await save() : version;
      const started = await startRun(sandboxId, { pipeline_version_id: target.id, ...runOptions });
      navigate(`/runs/${started.id}`);
    });

  const switchVersion = (number) => {
    if (dirty && !window.confirm("Discard unsaved changes?")) return;
    act(() => loadVersion(Number(number)));
  };

  if (!pipeline || !spec) {
    return <p className={`page-message ${error ? "error" : "muted"}`}>{error?.message || "Loading…"}</p>;
  }

  const selectedNode = spec.nodes.find((node) => node.id === selected);
  const modelNode = spec.nodes.find((node) => node.type === "model");
  const architecture = modelNode?.params?.architecture;
  const setNodeParams = (id, params) =>
    setSpec((current) => ({
      ...current,
      nodes: current.nodes.map((node) => (node.id === id ? { ...node, params } : node)),
    }));

  return (
    <div className="pipeline-page">
      <header className="page-header">
        <div>
          <p className="muted small">
            <Link to="/sandboxes">Sandboxes</Link> / <Link to={`/sandboxes/${sandboxId}`}>sandbox</Link> /
          </p>
          <h1>{pipeline.name}</h1>
        </div>
        <div className="toolbar">
          <label className="field inline">
            <span>Version</span>
            <select value={version?.number || ""} onChange={(event) => switchVersion(event.target.value)}>
              {!version && <option value="">unsaved</option>}
              {versions.map((item) => (
                <option key={item.id} value={item.number}>
                  v{item.number}
                </option>
              ))}
            </select>
          </label>
          {dirty && <span className="badge badge-queued">unsaved changes</span>}
          <button type="button" className="button" disabled={busy || !dirty} onClick={() => act(save)}>
            <Save size={16} aria-hidden="true" /> Save
          </button>
          <select
            aria-label="Run kind"
            value={runOptions.kind}
            onChange={(event) => setRunOptions({ ...runOptions, kind: event.target.value })}
          >
            <option value="train">train</option>
            <option value="sanity">sanity check (one batch)</option>
          </select>
          <select
            aria-label="Hardware"
            value={runOptions.hardware_tier}
            onChange={(event) => setRunOptions({ ...runOptions, hardware_tier: event.target.value })}
          >
            <option value="cpu">CPU</option>
            <option value="gpu">GPU</option>
          </select>
          <button type="button" className="button primary" disabled={busy} onClick={run}>
            <Play size={16} aria-hidden="true" /> Run
          </button>
        </div>
      </header>

      {error && (
        <div className="error-box" role="alert">
          <p>{error.message}</p>
          {error.issues?.length > 0 && (
            <ul>
              {error.issues.map((issue, index) => (
                <li key={index}>
                  {[issue.node_id, issue.field].filter(Boolean).join(".")} {issue.message}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {!schema ? (
        <div className="panel">
          {schemaError ? (
            <p className="error">The sandbox's engine did not answer: {schemaError}</p>
          ) : (
            <p>
              <Loader2 size={16} className="spin inline-icon" aria-hidden="true" /> Starting this sandbox's engine… the
              first start can take a minute.
            </p>
          )}
        </div>
      ) : (
        <>
          <div className="panel task-bar">
            <SchemaForm
              key={version?.id || "new"}
              schema={schema.schemas.task}
              value={spec.task}
              onChange={(task) => setSpec((current) => ({ ...current, task }))}
            />
          </div>
          <ReactFlowProvider>
            <div className="workspace two-column">
              <div className="canvas-column">
                <PipelinePalette spec={spec} />
                <PipelineCanvas spec={spec} onSpecChange={setSpec} selected={selected} onSelect={setSelected} />
              </div>
              {selectedNode ? (
                <NodePanel
                  key={`${version?.id}-${selectedNode.id}`}
                  node={selectedNode}
                  spec={spec}
                  schema={schema}
                  datasets={datasets}
                  sandboxId={sandboxId}
                  onParams={(params) => setNodeParams(selectedNode.id, params)}
                  onOpenLayers={() => setLayersOpen(true)}
                />
              ) : (
                <aside className="side-panel muted">
                  <p>Select a node to edit its settings.</p>
                  <p className="small">Data flows Dataset → Transforms → Model → Trainer → Evaluator.</p>
                </aside>
              )}
            </div>
          </ReactFlowProvider>
        </>
      )}

      {layersOpen && schema && architecture?.kind === "graph" && (
        <LayerEditor
          sandboxId={sandboxId}
          task={spec.task}
          architecture={architecture}
          positions={spec.ui?.layers || {}}
          ops={schema.capabilities.graph_ops}
          onChange={(nextArchitecture, positions) =>
            setSpec((current) => ({
              ...current,
              nodes: current.nodes.map((node) =>
                node.id === modelNode.id ? { ...node, params: { ...node.params, architecture: nextArchitecture } } : node
              ),
              ui: { ...current.ui, layers: positions },
            }))
          }
          onClose={() => setLayersOpen(false)}
        />
      )}
    </div>
  );
}
