import { useEffect, useState } from "react";
import { Layers, Plus, Trash2 } from "lucide-react";

import { sessionCall } from "../../api/sandboxesApi";
import SchemaForm, { JsonField } from "../../components/SchemaForm";

const FREEZE_BACKBONE = { op: "freeze", targets: ["@backbone"] };

// --------------------------------------------------------------------------- dataset

export function DatasetForm({ params, onChange, schemas, datasets }) {
  const synthetic = params.format === "synthetic";
  const keep = (next) =>
    next.format === "synthetic"
      ? next
      : { ...next, dataset_version_id: params.dataset_version_id, subpath: params.subpath };
  return (
    <>
      {!synthetic && (
        <>
          <label className="field">
            <span>
              Dataset version<abbr title="required"> *</abbr>
            </span>
            <select
              value={params.dataset_version_id || ""}
              onChange={(event) => onChange({ ...params, dataset_version_id: event.target.value || undefined })}
            >
              <option value="">choose…</option>
              {datasets.flatMap((dataset) =>
                dataset.versions.map((version) => (
                  <option key={version.id} value={version.id}>
                    {dataset.name} · v{version.number} ({version.files} files)
                  </option>
                ))
              )}
            </select>
          </label>
          <label className="field">
            <span>Folder inside the version (optional)</span>
            <input
              value={params.subpath || ""}
              placeholder="e.g. classification"
              onChange={(event) => onChange({ ...params, subpath: event.target.value || undefined })}
            />
          </label>
        </>
      )}
      <SchemaForm schema={schemas.dataset} value={params} onChange={(next) => onChange(keep(next))} hidden={["root"]} />
    </>
  );
}

// --------------------------------------------------------------------------- transforms

function TransformList({ title, ops, palette, onChange }) {
  const [choice, setChoice] = useState(palette[0]?.op || "");
  return (
    <fieldset className="fieldset">
      <legend>{title}</legend>
      {ops.map((op, index) => {
        const definition = palette.find((entry) => entry.op === op.op);
        return (
          <div key={`${op.op}-${index}`} className="list-item">
            <div className="list-item-header">
              <strong>{op.op}</strong>
              <button
                type="button"
                className="icon-button"
                aria-label={`remove ${op.op}`}
                onClick={() => onChange(ops.filter((_, i) => i !== index))}
              >
                <Trash2 size={14} aria-hidden="true" />
              </button>
            </div>
            {definition && (
              <SchemaForm
                schema={definition.params}
                value={op.params || {}}
                onChange={(params) => onChange(ops.map((item, i) => (i === index ? { ...item, params } : item)))}
              />
            )}
          </div>
        );
      })}
      <div className="inline-form compact">
        <select value={choice} onChange={(event) => setChoice(event.target.value)}>
          {palette.map((entry) => (
            <option key={entry.op} value={entry.op}>
              {entry.op}
            </option>
          ))}
        </select>
        <button type="button" className="button" onClick={() => onChange([...ops, { op: choice, params: {} }])}>
          <Plus size={14} aria-hidden="true" /> Add
        </button>
      </div>
    </fieldset>
  );
}

export function TransformsForm({ params, onChange, schemas, capabilities }) {
  const palette = capabilities.transform_ops || [];
  return (
    <>
      <SchemaForm schema={schemas.transforms} value={params} onChange={onChange} hidden={["train", "eval"]} />
      <TransformList
        title="Training augmentations"
        ops={params.train || []}
        palette={palette}
        onChange={(train) => onChange({ ...params, train: train.length ? train : undefined })}
      />
      <TransformList
        title="Evaluation transforms"
        ops={params.eval || []}
        palette={palette}
        onChange={(evalOps) => onChange({ ...params, eval: evalOps.length ? evalOps : undefined })}
      />
    </>
  );
}

// --------------------------------------------------------------------------- model

function useZoo(sandboxId, query, task, source) {
  const [models, setModels] = useState([]);
  useEffect(() => {
    if (!source) return undefined;
    let active = true;
    const timer = window.setTimeout(() => {
      sessionCall(sandboxId, "list_zoo", { query: query || undefined, task, source, limit: 40 })
        .then((result) => active && setModels(result))
        .catch(() => active && setModels([]));
    }, 300);
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [sandboxId, query, task, source]);
  return models;
}

export function ModelForm({ params, onChange, sandboxId, taskType, onOpenLayers }) {
  const architecture = params.architecture || {};
  const setArchitecture = (next) => onChange({ ...params, architecture: next });

  if (architecture.kind === "graph") {
    return (
      <div className="stack">
        <p>
          Custom layer graph: <strong>{architecture.nodes?.length || 0} layers</strong>, input{" "}
          {architecture.input?.shape?.join(" × ")}.
        </p>
        <button type="button" className="button primary" onClick={onOpenLayers}>
          <Layers size={16} aria-hidden="true" /> Open layer editor
        </button>
        <button
          type="button"
          className="link-button"
          onClick={() =>
            window.confirm("Replace the custom layers with a zoo model?") &&
            setArchitecture({ kind: "pretrained", base: { source: "torchvision", name: "resnet18", weights: "DEFAULT" } })
          }
        >
          Use a zoo model instead
        </button>
      </div>
    );
  }
  return (
    <ZooModelForm
      architecture={architecture}
      onChange={setArchitecture}
      sandboxId={sandboxId}
      taskType={taskType}
      onCustom={() => {
        setArchitecture({ kind: "graph", input: { shape: [3, 64, 64] }, nodes: [], edges: [] });
        onOpenLayers();
      }}
    />
  );
}

function ZooModelForm({ architecture, onChange, sandboxId, taskType, onCustom }) {
  const base = architecture.base || { source: "torchvision", name: "", weights: "DEFAULT" };
  const patches = architecture.patches || [];
  const frozen = patches.some((p) => p.op === "freeze" && p.targets?.includes("@backbone"));
  const otherPatches = patches.filter((p) => !(p.op === "freeze" && p.targets?.length === 1 && p.targets[0] === "@backbone"));
  const zooTask = base.source === "netpattern" ? undefined : taskType;
  const models = useZoo(sandboxId, base.name, zooTask, base.source);
  const encoders = useZoo(sandboxId, base.encoder, undefined, base.source === "netpattern" ? "timm" : null);
  const selected = models.find((model) => model.name === base.name);

  const setBase = (changes) => onChange({ ...architecture, kind: "pretrained", base: { ...base, ...changes } });
  const setPatches = (freeze, rest) => {
    const next = [...(freeze ? [FREEZE_BACKBONE] : []), ...rest];
    onChange({ ...architecture, kind: "pretrained", base, patches: next.length ? next : undefined });
  };

  return (
    <div className="stack">
      <label className="field">
        <span>Source</span>
        <select value={base.source} onChange={(event) => setBase({ source: event.target.value, name: event.target.value === "netpattern" ? "unet" : "" })}>
          <option value="torchvision">torchvision</option>
          <option value="timm">timm</option>
          <option value="netpattern">netpattern (U-Net)</option>
        </select>
      </label>
      <label className="field">
        <span>Model</span>
        <input list="zoo-models" value={base.name} placeholder="search, e.g. resnet" onChange={(event) => setBase({ name: event.target.value })} />
        <datalist id="zoo-models">
          {models.map((model) => (
            <option key={model.name} value={model.name} />
          ))}
        </datalist>
      </label>
      {selected && (
        <p className="muted small">
          {selected.params ? `${(selected.params / 1e6).toFixed(1)} M parameters` : ""}
          {selected.gflops ? ` · ${selected.gflops.toFixed(1)} GFLOPs` : ""}
          {selected.metrics?.["acc@1"] ? ` · ${selected.trained_on} top-1 ${selected.metrics["acc@1"]}%` : ""}
          {selected.license ? ` · license ${selected.license}` : ""}
        </p>
      )}
      {base.source === "netpattern" && (
        <label className="field">
          <span>Encoder (timm model)</span>
          <input list="zoo-encoders" value={base.encoder || ""} placeholder="resnet34" onChange={(event) => setBase({ encoder: event.target.value || undefined })} />
          <datalist id="zoo-encoders">
            {encoders.map((model) => (
              <option key={model.name} value={model.name} />
            ))}
          </datalist>
        </label>
      )}
      <label className="field">
        <span>Weights</span>
        <select value={base.weights ?? ""} onChange={(event) => setBase({ weights: event.target.value || null })}>
          <option value="DEFAULT">pretrained (default weights)</option>
          {(selected?.weights || []).filter((w) => w !== "DEFAULT").map((weights) => (
            <option key={weights} value={weights}>
              {weights}
            </option>
          ))}
          <option value="">blank (random initialization)</option>
        </select>
      </label>
      <label className="field checkbox">
        <input type="checkbox" checked={frozen} onChange={(event) => setPatches(event.target.checked, otherPatches)} />
        <span>Freeze the backbone (train only the new head)</span>
      </label>
      <details>
        <summary>Advanced</summary>
        <JsonField
          label={<span>options (e.g. {"{"}"min_size": 320{"}"} for detection)</span>}
          value={base.options}
          onChange={(options) => setBase({ options })}
        />
        <JsonField
          label={<span>patches (replace / insert_after / remove)</span>}
          value={otherPatches.length ? otherPatches : undefined}
          onChange={(rest) => setPatches(frozen, Array.isArray(rest) ? rest : [])}
        />
      </details>
      {taskType !== "detection.bbox" && (
        <button type="button" className="link-button" onClick={onCustom}>
          Build custom layers instead
        </button>
      )}
    </div>
  );
}

// --------------------------------------------------------------------------- evaluator

export function EvaluatorForm({ params, onChange }) {
  const splits = params.splits || ["val", "test"];
  const toggle = (split) => {
    const next = splits.includes(split) ? splits.filter((s) => s !== split) : [...splits, split];
    onChange({ ...params, splits: next });
  };
  return (
    <div className="stack">
      <p className="muted small">The best checkpoint is evaluated on these splits after training.</p>
      {["val", "test"].map((split) => (
        <label key={split} className="field checkbox">
          <input type="checkbox" checked={splits.includes(split)} onChange={() => toggle(split)} />
          <span>{split}</span>
        </label>
      ))}
    </div>
  );
}
