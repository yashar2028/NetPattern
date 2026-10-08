import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Background,
  Controls,
  Handle,
  Position,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
} from "@xyflow/react";
import { AlertTriangle, X } from "lucide-react";

import { errorIssues, errorMessage } from "../../api/client";
import { sessionCall } from "../../api/sandboxesApi";
import SchemaForm from "../../components/SchemaForm";
import { formatCount } from "../../utils/format";

const DRAG_TYPE = "application/netpattern-layer";
const OUTPUT_KEYS = ["out_features", "out_channels"];
const VALIDATE_DELAY_MS = 500;

function LayerNodeView({ data, selected }) {
  const { label, op, shape, params, issues, isInput } = data;
  return (
    <div className={`flow-node layer-node${selected ? " selected" : ""}${issues.length ? " invalid" : ""}`}>
      {!isInput && <Handle type="target" position={Position.Top} />}
      <div className="flow-node-title">
        {op}
        {issues.length > 0 && <AlertTriangle size={14} aria-label="has problems" className="issue-icon" />}
      </div>
      <div className="flow-node-detail">
        {label !== op && <span>{label} · </span>}
        {shape ? shape.join(" × ") : "shape unknown"}
        {params ? ` · ${formatCount(params)} params` : ""}
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

const nodeTypes = { layer: LayerNodeView };

function OpPalette({ ops, onAdd }) {
  const [query, setQuery] = useState("");
  const groups = useMemo(() => {
    const matching = ops.filter((op) => op.op.toLowerCase().includes(query.toLowerCase()));
    return Object.entries(Object.groupBy(matching, (op) => op.category));
  }, [ops, query]);
  return (
    <aside className="palette" aria-label="Layers">
      <input type="search" placeholder="search layers" value={query} onChange={(event) => setQuery(event.target.value)} />
      <p className="muted small">Drag onto the canvas or double-click to add.</p>
      {groups.map(([category, members]) => (
        <div key={category}>
          <h3>{category}</h3>
          {members.map((op) => (
            <div
              key={op.op}
              className="palette-item"
              draggable
              title={op.description}
              onDragStart={(event) => event.dataTransfer.setData(DRAG_TYPE, op.op)}
              onDoubleClick={() => onAdd(op.op)}
            >
              {op.op}
            </div>
          ))}
        </div>
      ))}
    </aside>
  );
}

function ParamsPanel({ node, op, issues, onChange, onRemove }) {
  const symbolic = OUTPUT_KEYS.filter((key) => op.params.properties?.[key]);
  const toggled = symbolic.filter((key) => node.params[key] === "$num_outputs");
  return (
    <aside className="side-panel" aria-label="Layer settings">
      <h3>{node.op}</h3>
      <p className="muted small">
        id <code>{node.id}</code> · {op.description}
      </p>
      {issues.map((issue, index) => (
        <p key={index} className="error small">
          {issue.field ? `${issue.field}: ` : ""}
          {issue.message}
        </p>
      ))}
      {symbolic.map((key) => (
        <label key={key} className="field checkbox">
          <input
            type="checkbox"
            checked={toggled.includes(key)}
            onChange={(event) =>
              onChange({ ...node.params, [key]: event.target.checked ? "$num_outputs" : undefined })
            }
          />
          <span>{key.replaceAll("_", " ")} = number of classes/targets</span>
        </label>
      ))}
      <SchemaForm
        key={node.id}
        schema={op.params}
        value={node.params}
        hidden={toggled}
        onChange={(params) => onChange(params)}
      />
      <button type="button" className="button ghost" onClick={onRemove}>
        Remove layer
      </button>
    </aside>
  );
}

function Editor({ sandboxId, task, architecture, positions, ops, onChange, onClose }) {
  const { screenToFlowPosition } = useReactFlow();
  const [selected, setSelected] = useState(null);
  const [selectedEdge, setSelectedEdge] = useState(null);
  const [outputs, setOutputs] = useState(task.classes?.length || task.targets?.length || "");
  const [analysis, setAnalysis] = useState(null);
  const [checking, setChecking] = useState(false);
  const opsByName = useMemo(() => Object.fromEntries(ops.map((op) => [op.op, op])), [ops]);

  // Debounced shape inference in the sandbox's engine session.
  useEffect(() => {
    let active = true;
    const timer = window.setTimeout(async () => {
      setChecking(true);
      try {
        const result = await sessionCall(sandboxId, "validate_architecture", {
          architecture,
          task: { type: task.type },
          num_outputs: outputs ? Number(outputs) : undefined,
        });
        if (active) setAnalysis(result);
      } catch (caught) {
        const issues = errorIssues(caught);
        if (active) setAnalysis({ ok: false, issues: issues.length ? issues : [{ message: errorMessage(caught) }] });
      } finally {
        if (active) setChecking(false);
      }
    }, VALIDATE_DELAY_MS);
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [sandboxId, architecture, task.type, outputs]);

  const issuesFor = useCallback(
    (id) => (analysis?.issues || []).filter((issue) => issue.node_id === id),
    [analysis]
  );

  const nodes = useMemo(() => {
    const input = {
      id: "input",
      type: "layer",
      position: positions.input || { x: 0, y: 0 },
      deletable: false,
      selected: selected === "input",
      data: { label: "input", op: "Input", shape: architecture.input.shape, issues: [], isInput: true },
    };
    return [
      input,
      ...architecture.nodes.map((node, index) => ({
        id: node.id,
        type: "layer",
        position: positions[node.id] || { x: 0, y: (index + 1) * 100 },
        selected: selected === node.id,
        data: {
          label: node.id,
          op: node.op,
          shape: analysis?.nodes?.[node.id]?.shape,
          params: analysis?.nodes?.[node.id]?.params,
          issues: issuesFor(node.id),
        },
      })),
    ];
  }, [architecture, positions, selected, analysis, issuesFor]);

  const edges = useMemo(
    () =>
      architecture.edges.map(([source, target]) => {
        const id = `${source}->${target}`;
        return { id, source, target, selected: id === selectedEdge };
      }),
    [architecture.edges, selectedEdge]
  );

  const setArchitecture = (changes) => onChange({ ...architecture, ...changes }, positions);

  const removeNodes = (ids) => {
    const nextPositions = { ...positions };
    ids.forEach((id) => delete nextPositions[id]);
    onChange(
      {
        ...architecture,
        nodes: architecture.nodes.filter((node) => !ids.includes(node.id)),
        edges: architecture.edges.filter(([s, t]) => !ids.includes(s) && !ids.includes(t)),
        output: ids.includes(architecture.output) ? undefined : architecture.output,
      },
      nextPositions
    );
    if (ids.includes(selected)) setSelected(null);
  };

  const addNode = (opName, position) => {
    const prefix = opName.toLowerCase();
    let number = 1;
    while (architecture.nodes.some((node) => node.id === `${prefix}${number}`)) number += 1;
    const id = `${prefix}${number}`;
    const last = architecture.nodes.at(-1);
    const where = position || { x: 0, y: (positions[last?.id]?.y ?? 0) + 100 };
    onChange(
      { ...architecture, nodes: [...architecture.nodes, { id, op: opName, params: {} }] },
      { ...positions, [id]: where }
    );
    setSelected(id);
  };

  const onNodesChange = (changes) => {
    const selection = changes.find((c) => c.type === "select" && c.selected);
    if (selection) setSelected(selection.id);
    const removed = changes.filter((c) => c.type === "remove").map((c) => c.id);
    if (removed.length) return removeNodes(removed);
    const moves = changes.filter((c) => c.type === "position" && c.position);
    if (moves.length) {
      const next = { ...positions };
      moves.forEach((move) => (next[move.id] = move.position));
      onChange(architecture, next);
    }
  };

  const onEdgesChange = (changes) => {
    const selection = changes.find((c) => c.type === "select");
    if (selection) setSelectedEdge(selection.selected ? selection.id : null);
    const removed = changes.filter((c) => c.type === "remove").map((c) => c.id);
    if (removed.length) {
      setArchitecture({ edges: architecture.edges.filter(([s, t]) => !removed.includes(`${s}->${t}`)) });
    }
  };

  const onConnect = ({ source, target }) => {
    if (target === "input" || architecture.edges.some(([s, t]) => s === source && t === target)) return;
    setArchitecture({ edges: [...architecture.edges, [source, target]] });
  };

  const onDrop = (event) => {
    event.preventDefault();
    const op = event.dataTransfer.getData(DRAG_TYPE);
    if (op) addNode(op, screenToFlowPosition({ x: event.clientX, y: event.clientY }));
  };

  const selectedNode = architecture.nodes.find((node) => node.id === selected);
  const generalIssues = (analysis?.issues || []).filter((issue) => !issue.node_id);
  const [channels, height, width] = architecture.input.shape;
  const setInput = (index, value) => {
    const shape = [...architecture.input.shape];
    shape[index] = Number(value) || 1;
    setArchitecture({ input: { shape } });
  };

  return (
    <div className="layer-editor" role="dialog" aria-label="Layer editor">
      <header className="layer-toolbar">
        <h2>Layer editor</h2>
        <label className="field inline">
          <span>Input C × H × W</span>
          <span className="shape-inputs">
            <input type="number" min="1" value={channels} onChange={(e) => setInput(0, e.target.value)} aria-label="channels" />
            <input type="number" min="1" value={height} onChange={(e) => setInput(1, e.target.value)} aria-label="height" />
            <input type="number" min="1" value={width} onChange={(e) => setInput(2, e.target.value)} aria-label="width" />
          </span>
        </label>
        <label className="field inline">
          <span title="Used only for this preview; runs take the count from the dataset">Outputs (preview)</span>
          <input type="number" min="1" value={outputs} placeholder="2" onChange={(e) => setOutputs(e.target.value)} />
        </label>
        <div className="layer-summary" aria-live="polite">
          {checking ? (
            <span className="muted">checking…</span>
          ) : analysis?.ok ? (
            <span>
              output {analysis.output_shape?.join(" × ")} · {formatCount(analysis.params)} params
              {analysis.flops ? ` · ${formatCount(analysis.flops)} FLOPs` : ""}
            </span>
          ) : analysis ? (
            <span className="error">{analysis.issues?.length || 0} problem(s)</span>
          ) : null}
        </div>
        <button type="button" className="button primary" onClick={onClose}>
          <X size={16} aria-hidden="true" /> Done
        </button>
      </header>
      {generalIssues.length > 0 && (
        <ul className="issue-list" role="alert">
          {generalIssues.map((issue, index) => (
            <li key={index}>
              {issue.field ? <code>{issue.field}</code> : null} {issue.message}
            </li>
          ))}
        </ul>
      )}
      <div className="workspace">
        <OpPalette ops={ops} onAdd={(op) => addNode(op)} />
        <div className="canvas" onDragOver={(event) => event.preventDefault()} onDrop={onDrop}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onPaneClick={() => setSelected(null)}
            fitView
            colorMode="system"
          >
            <Background gap={20} />
            <Controls showInteractive={false} />
          </ReactFlow>
        </div>
        {selectedNode && opsByName[selectedNode.op] ? (
          <ParamsPanel
            node={selectedNode}
            op={opsByName[selectedNode.op]}
            issues={issuesFor(selectedNode.id)}
            onChange={(params) =>
              setArchitecture({
                nodes: architecture.nodes.map((node) => (node.id === selectedNode.id ? { ...node, params } : node)),
              })
            }
            onRemove={() => removeNodes([selectedNode.id])}
          />
        ) : (
          <aside className="side-panel muted">
            <p>Select a layer to edit it. Connect layers from the bottom handle to the top handle.</p>
            <p className="small">Sizes coming into a layer (in_channels, in_features) are worked out for you.</p>
          </aside>
        )}
      </div>
    </div>
  );
}

/** Full-screen editor for a custom layer graph (PLAN §6.2). */
export default function LayerEditor(props) {
  return (
    <ReactFlowProvider>
      <Editor {...props} />
    </ReactFlowProvider>
  );
}
