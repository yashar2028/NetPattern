import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { ReactFlowProvider } from "@xyflow/react";
import { Loader2, Play, Repeat, Save, SlidersHorizontal } from "lucide-react";
import "@xyflow/react/dist/style.css";

import { errorIssues, errorMessage } from "../api/client";
import { listDatasets } from "../api/datasetsApi";
import { startRun } from "../api/runsApi";
import {
  getPipeline,
  getSandbox,
  getVersion,
  hardwareTiers,
  listVersions,
  saveVersion,
  sessionCall,
  startSweep,
} from "../api/sandboxesApi";
import SchemaForm from "../components/SchemaForm";
import GraphItemPanel from "../features/pipeline/GraphItemPanel";
import LayerEditor from "../features/pipeline/LayerEditor";
import ModelLayers from "../features/pipeline/ModelLayers";
import { DatasetForm, EvaluatorForm, ModelForm, TransformsForm } from "../features/pipeline/NodeForms";
import SweepPanel from "../features/pipeline/SweepPanel";
import PipelineCanvas from "../features/pipeline/PipelineCanvas";
import { NODE_BY_TYPE, defaultSpec } from "../features/pipeline/defaults";
import { blocksRun } from "../features/pipeline/layerGraph";
import useModelValidation from "../features/pipeline/useModelValidation";
import "../features/pipeline/editor.css";

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
  return form;
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
  const [schemaFailure, setSchemaFailure] = useState(null);
  const [schemaAttempt, setSchemaAttempt] = useState(0);
  const [datasets, setDatasets] = useState([]);
  const [tiers, setTiers] = useState([{ name: "cpu", cpus: 2, gpus: 0, rate: 1 }]);
  const [selected, setSelected] = useState(null);
  const [layersOpen, setLayersOpen] = useState(false);
  const [initialIssue, setInitialIssue] = useState(null);
  const [panel, setPanel] = useState(null);
  const [runOptions, setRunOptions] = useState({ kind: "train", hardware_tier: "cpu" });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const headingRef = useRef(null);
  const workspace = useRef(null);
  const layerReturnFocus = useRef(null);
  const runOptionsButton = useRef(null);

  const activatePanel = (kind, id, anchor) => {
    if (kind === "node") setSelected(id);
    setPanel({ kind, id, anchor });
  };
  const dismissPanel = (restore) => {
    setPanel(null);
    if (restore) {
      const anchor = panel?.anchor;
      requestAnimationFrame(() => (anchor?.isConnected ? anchor : headingRef.current)?.focus({ preventScroll: true }));
    }
  };

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
      setPanel(null);
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

  // GPU tiers need a CUDA environment; the server marks the ones this sandbox cannot use.
  useEffect(() => {
    getSandbox(sandboxId)
      .then((sandbox) => hardwareTiers(sandbox.template))
      .then(setTiers)
      .catch(() => {});
  }, [sandboxId]);

  // The forms come from the sandbox's own engine; the first call may start the environment.
  useEffect(() => {
    let active = true;
    sessionCall(sandboxId, "schema")
      .then((result) => active && setSchema({ sandboxId, result }))
      .catch((caught) => active && setSchemaFailure({ sandboxId, message: errorMessage(caught) }));
    return () => { active = false; };
  }, [sandboxId, schemaAttempt]);

  const engineSchema = schema?.sandboxId === sandboxId ? schema.result : null;
  const schemaError = schemaFailure?.sandboxId === sandboxId ? schemaFailure.message : "";
  const retryEngine = () => {
    setSchemaFailure(null);
    setSchemaAttempt((attempt) => attempt + 1);
  };
  const modelNode = spec?.nodes.find((node) => node.type === "model");
  const architecture = modelNode?.params?.architecture;
  const custom = architecture?.kind === "graph";
  const taskType = spec?.task.type;
  const knownOutputs = taskType === "regression" ? spec?.task.targets?.length : spec?.task.classes?.length;
  const validation = useModelValidation(sandboxId, architecture, taskType, knownOutputs, Boolean(engineSchema && custom));
  const issues = validation.analysis?.issues || [];
  const confirmedIssues = issues.filter((issue) => blocksRun(issue, validation.analysis.output_shape, taskType, knownOutputs));
  const datasetNode = spec?.nodes.find((node) => node.type === "dataset");
  const needsDataset = !datasetNode || (datasetNode.params?.format !== "synthetic" && !datasetNode.params?.dataset_version_id);
  const modelBlock = custom && (!architecture.nodes?.length || confirmedIssues.length > 0);
  const runBlocked = !engineSchema || validation.checking || modelBlock || !modelNode || needsDataset;
  const openLayers = (issue = null) => {
    const id = modelNode?.id;
    layerReturnFocus.current = (id && workspace.current?.querySelector(`.react-flow__node[data-id="${CSS.escape(id)}"]`))
      || headingRef.current;
    setInitialIssue(issue);
    setLayersOpen(true);
  };

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

  const runnable = () => {
    if (runBlocked) throw new Error("Resolve the pipeline readiness checks before running.");
  };

  const run = () =>
    act(async () => {
      runnable();
      const target = dirty || !version ? await save() : version;
      const started = await startRun(sandboxId, { pipeline_version_id: target.id, ...runOptions });
      navigate(`/runs/${started.id}`);
    });

  const sweep = (options) =>
    act(async () => {
      runnable();
      const target = dirty || !version ? await save() : version;
      const started = await startSweep(sandboxId, { pipeline_version_id: target.id, ...options });
      navigate(`/compare?runs=${started.runs.map((run) => run.id).join(",")}`);
    });

  const switchVersion = (number) => {
    if (dirty && !window.confirm("Discard unsaved changes?")) return;
    act(() => loadVersion(Number(number)));
  };

  if (!pipeline || !spec) {
    return <p className={`page-message ${error ? "error" : "muted"}`}>{error?.message || "Loading…"}</p>;
  }

  const panelNode = panel?.kind === "node" && spec.nodes.find((node) => node.id === panel.id);
  const panelEdge = panel?.kind === "edge" && spec.edges.find(([source, target]) => `${source}->${target}` === panel.id);
  const stepLabel = (id) => NODE_BY_TYPE[spec.nodes.find((node) => node.id === id)?.type]?.label || id;
  const warningsByNode = new Map();
  if (engineSchema && needsDataset && datasetNode) {
    warningsByNode.set(datasetNode.id, [{
      message: "Choose a dataset version or use synthetic data.",
      action: "Open dataset settings",
      select: datasetNode.id,
    }]);
  }
  if (engineSchema && modelNode && custom) {
    const modelWarnings = issues.map((issue) => ({
      message: `${issue.node_id || "Model"}${issue.field ? ` · ${issue.field}` : ""}: ${issue.message}${
        confirmedIssues.includes(issue) ? "" : " (Preview only; training uses the task and selected dataset.)"}`,
      action: "Open layer editor",
      issue,
    }));
    if (!architecture.nodes?.length && !modelWarnings.length) {
      modelWarnings.push({ message: "The model has no layers.", action: "Add model layers", issue: null });
    }
    warningsByNode.set(modelNode.id, modelWarnings);
  }
  const setNodeParams = (id, params) =>
    setSpec((current) => ({
      ...current,
      nodes: current.nodes.map((node) => (node.id === id ? { ...node, params } : node)),
    }));

  return (
    <div className="pipeline-page">
      <header className="page-header graph-header">
        <div>
          <p className="muted small">
            <Link to="/sandboxes">Sandboxes</Link> / <Link to={`/sandboxes/${sandboxId}`}>sandbox</Link> /
          </p>
          <h1 ref={headingRef} tabIndex={-1}>{pipeline.name}</h1>
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
          <span className={`badge${dirty ? " badge-queued" : ""}`} role="status">{dirty ? "Unsaved changes" : version ? "Pipeline saved" : "No saved version"}</span>
          <button type="button" className="button" disabled={busy || !dirty} onClick={() => act(save)}>
            <Save size={16} aria-hidden="true" /> Save
          </button>
          <button type="button" className="button primary" disabled={busy || runBlocked} aria-describedby="pipeline-readiness" onClick={run}>
            <Play size={16} aria-hidden="true" /> Run
          </button>
          <button type="button" className="button" ref={runOptionsButton}
            aria-expanded={panel?.kind === "run" || panel?.kind === "sweep"}
            onClick={(event) => activatePanel("run", null, event.currentTarget)}>
            <SlidersHorizontal size={16} aria-hidden="true" /> Run options
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

      <div className="editor-readiness">
        <div className="editor-status-row">
          <div id="pipeline-readiness" className="editor-status-copy small" role="status">
            <strong className={schemaError && !engineSchema ? "error" : ""}>
              {engineSchema ? "Engine ready." : schemaError ? "Engine startup failed." : (
                <><Loader2 size={16} className="spin inline-icon" aria-hidden="true" /> Starting the sandbox engine…</>
              )}
            </strong>
            <span className="muted">
              {!engineSchema ? schemaError || "You can view the pipeline while it starts." :
                modelBlock ? "Can’t run: check the warnings on the Model node." :
                validation.checking ? "Checking model…" :
                !modelNode ? "Can’t run: add a Model step." :
                needsDataset ? (datasetNode ? "Can’t run: check the Dataset node." : "Can’t run: add a Dataset step.") :
                validation.error ? "Model precheck is unavailable; training will validate the model." :
                "Ready to run. Training will check the selected dataset."}
            </span>
          </div>
          {!engineSchema && schemaError && (
            <button type="button" className="button" onClick={retryEngine}>Retry engine startup</button>
          )}
          {engineSchema && validation.error && (
            <button type="button" className="button" onClick={validation.retry}>Retry model precheck</button>
          )}
        </div>
      </div>
      {engineSchema && (
        <section className="task-bar pipeline-task-settings" aria-labelledby="pipeline-task-heading">
          <h2 id="pipeline-task-heading">Task settings</h2>
          <SchemaForm
            key={version?.id || "new"}
            schema={engineSchema.schemas.task}
            value={spec.task}
            onChange={(task) => setSpec((current) => ({ ...current, task }))}
          />
        </section>
      )}
      <ReactFlowProvider>
        <PipelineCanvas spec={spec} onSpecChange={setSpec} selected={selected} onSelect={setSelected}
          workspace={workspace} panel={panel} onActivate={activatePanel} onDismiss={dismissPanel} fallback={headingRef}
          warningsByNode={warningsByNode} onWarning={(warning) => {
            if (warning.select) {
              const anchor = workspace.current.querySelector(`.react-flow__node[data-id="${CSS.escape(warning.select)}"]`);
              activatePanel("node", warning.select, anchor);
            } else {
              openLayers(warning.issue);
            }
          }} />
      </ReactFlowProvider>
      {panelNode && (
        <GraphItemPanel key={panelNode.id} id="pipeline-node-settings" anchor={panel.anchor} workspace={workspace}
          fallback={headingRef} title={`${stepLabel(panelNode.id)} settings`} onDismiss={dismissPanel}>
          {engineSchema ? (
            <NodePanel node={panelNode} spec={spec} schema={engineSchema} datasets={datasets} sandboxId={sandboxId}
              onParams={(params) => setNodeParams(panelNode.id, params)} onOpenLayers={() => openLayers()} />
          ) : <p className="muted small">Step settings are available when the engine is ready.</p>}
        </GraphItemPanel>
      )}
      {panelEdge && (
        <GraphItemPanel key={panel.id} id="pipeline-connection-settings" anchor={panel.anchor} workspace={workspace}
          fallback={headingRef} title="Connection settings" onDismiss={dismissPanel}>
          <p>{panelEdge.map(stepLabel).join(" → ")}</p>
          <button type="button" className="button ghost" onClick={() => {
            setSpec((current) => ({ ...current, edges: current.edges.filter(([s, t]) => s !== panelEdge[0] || t !== panelEdge[1]) }));
            dismissPanel(true);
          }}>Disconnect</button>
        </GraphItemPanel>
      )}
      {panel?.kind === "run" && (
        <GraphItemPanel id="pipeline-run-settings" anchor={panel.anchor} workspace={workspace}
          fallback={runOptionsButton} title="Run options" onDismiss={dismissPanel}>
          <label className="field">Run kind
            <select value={runOptions.kind} onChange={(event) => setRunOptions({ ...runOptions, kind: event.target.value })}>
              <option value="train">train</option>
              <option value="sanity">sanity check (one batch)</option>
              <option value="lr_find">learning-rate finder</option>
            </select>
          </label>
          <label className="field">Hardware
            <select value={runOptions.hardware_tier} onChange={(event) => setRunOptions({ ...runOptions, hardware_tier: event.target.value })}>
              {tiers.map((tier) => (
                <option
                  key={tier.name}
                  value={tier.name}
                  title={tier.reason || tier.description}
                  disabled={tier.available === false}
                >
                  {tier.name} · {tier.cpus} cores{tier.gpus ? ` + ${tier.gpus} GPU` : ""} · {tier.rate} units/s
                  {tier.available === false ? " · needs a CUDA environment" : ""}
                </option>
              ))}
            </select>
          </label>
          <button type="button" className="button" disabled={busy || runBlocked}
            onClick={() => activatePanel("sweep", null, panel.anchor)}>
            <Repeat size={16} aria-hidden="true" /> Configure sweep
          </button>
        </GraphItemPanel>
      )}
      {panel?.kind === "sweep" && !runBlocked && (
        <GraphItemPanel id="pipeline-sweep-settings" anchor={panel.anchor} workspace={workspace}
          fallback={runOptionsButton} title="Sweep" onDismiss={dismissPanel}>
          <SweepPanel tiers={tiers} onStart={sweep} />
        </GraphItemPanel>
      )}

      {layersOpen && engineSchema && architecture?.kind === "graph" && (
        <LayerEditor
          sandboxId={sandboxId}
          task={spec.task}
          architecture={architecture}
          positions={spec.ui?.layers || {}}
          ops={engineSchema.capabilities.graph_ops}
          validation={validation}
          knownOutputs={knownOutputs}
          initialIssue={initialIssue}
          fallbackFocus={layerReturnFocus}
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
      {layersOpen && architecture?.kind === "pretrained" && (
        <ModelLayers
          sandboxId={sandboxId}
          task={spec.task}
          architecture={architecture}
          fallbackFocus={layerReturnFocus}
          onClose={() => setLayersOpen(false)}
        />
      )}
    </div>
  );
}
