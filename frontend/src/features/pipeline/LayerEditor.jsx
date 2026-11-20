import { useCallback, useMemo, useRef, useState } from "react";
import {
  Background,
  BaseEdge,
  Controls,
  EdgeToolbar,
  getBezierPath,
  Handle,
  MarkerType,
  Position,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
} from "@xyflow/react";
import { ArrowDown, Crosshair, Plus, Square, X } from "lucide-react";

import SchemaForm from "../../components/SchemaForm";
import { formatCount } from "../../utils/format";
import EditorDialog from "./EditorDialog";
import GraphItemPanel from "./GraphItemPanel";
import { arrangeLayers, connectionProblem, filterLayerOptions, insertLayerOnEdge } from "./layerGraph";
import NodeWarnings from "./NodeWarnings";
import useCanvasLayout, { activateCanvasItem } from "./useCanvasLayout";
import useModelValidation from "./useModelValidation";

const DRAG_TYPE = "application/netpattern-layer";
const OUTPUT_KEYS = ["out_features", "out_channels"];
const EDGE_STYLE = { markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14 } };

function LayerNodeView({ data, selected }) {
  const { label, op, shape, params, warnings, onWarning, isInput, isOutput, tone } = data;
  const Icon = isInput ? ArrowDown : Square;
  return (
    <div data-tone={isInput ? "dataset" : isOutput ? "evaluator" : tone}
      className={`flow-node canvas-card layer-node${selected ? " selected" : ""}${warnings.length ? " invalid" : ""}`}>
      {!isInput && <Handle type="target" position={Position.Top} />}
      <Icon className="canvas-card-icon" size={16} strokeWidth={1.6} aria-hidden="true" />
      <div className="canvas-card-body">
        <div className="flow-node-title">{op}</div>
        <div className="flow-node-detail">
          {label !== op && !isInput && <span>{label} · </span>}
          {shape ? shape.join(" × ") : "shape unknown"}
          {params ? ` · ${formatCount(params)} params` : ""}
        </div>
        {isOutput && <span className="badge canvas-output">Output</span>}
      </div>
      <NodeWarnings label={isInput ? "Input" : `${op} (${label})`} warnings={warnings} onWarning={onWarning} />
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

const nodeTypes = { layer: LayerNodeView };

function LayerEdge({ id, source, target, data, markerEnd, style, ...coordinates }) {
  const [path, x, y] = getBezierPath(coordinates);
  return (
    <>
      <BaseEdge id={id} path={path} markerEnd={markerEnd} style={style} />
      <EdgeToolbar edgeId={id} x={x} y={y} isVisible className="nopan nodrag">
        <button
          type="button"
          className="layer-edge-add"
          aria-label={`Add layer between ${source} and ${target}`}
          aria-pressed={data.inserting}
          aria-controls="layer-picker"
          onClick={(event) => {
            event.stopPropagation();
            data.onAdd(id, event.currentTarget);
          }}
        >
          <Plus size={14} strokeWidth={1.6} aria-hidden="true" />
        </button>
      </EdgeToolbar>
    </>
  );
}

const edgeTypes = { layer: LayerEdge };

function OpPalette({ ops, edges, labelFor, insertionEdge, searchRef, onAdd, onCancelInsertion }) {
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("");
  const insertionIndex = edges.findIndex(([source, target]) => insertionEdge === `${source}->${target}`);
  const missingConnection = Boolean(insertionEdge && insertionIndex === -1);
  const categories = [...new Set(ops.map((op) => op.category))];
  const matching = filterLayerOptions(ops, query, category);
  const groups = Object.entries(Object.groupBy(matching, (op) => op.category));
  return (
    <aside id="layer-picker" className="layer-library" aria-labelledby="layer-library-title"
      onKeyDown={(event) => {
        if (event.key === "Escape" && insertionEdge) {
          event.preventDefault();
          event.stopPropagation();
          onCancelInsertion();
        }
      }}>
      <div className="layer-library-controls">
        <h3 id="layer-library-title">Add layer</h3>
        {insertionEdge && (
          <div className="layer-insertion-context">
            <p className="small" role="status">{missingConnection
              ? "That connection no longer exists."
              : `Insert between ${edges[insertionIndex].map(labelFor).join(" → ")}.`}</p>
            <button type="button" className="icon-button" aria-label="Cancel insertion" title="Cancel insertion" onClick={onCancelInsertion}>
              <X size={14} strokeWidth={1.6} aria-hidden="true" />
            </button>
          </div>
        )}
        <input ref={searchRef} type="search" aria-label="Search layers" placeholder="Search layers" value={query}
          onChange={(event) => setQuery(event.target.value)} />
        <select aria-label="Layer category" value={category} onChange={(event) => setCategory(event.target.value)}>
          <option value="">All categories</option>
          {categories.map((name) => <option key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</option>)}
        </select>
      </div>
      <div className="graph-picker layer-library-results">
        <p className="muted small" role="status">{matching.length} layer option{matching.length === 1 ? "" : "s"}</p>
        {!matching.length && (
          <div className="stack">
            <p>No layers match these filters.</p>
            <button type="button" className="button" onClick={() => { setQuery(""); setCategory(""); }}>Clear filters</button>
          </div>
        )}
        {groups.map(([group, members]) => (
          <div key={group} className="layer-option-group">
            <h4>{group}</h4>
            {members.map((op) => (
              <button type="button"
                key={op.op}
                className="palette-item"
                disabled={missingConnection}
                draggable={!insertionEdge}
                aria-label={`Add ${op.op}`}
                aria-description={op.description}
                onDragStart={(event) => event.dataTransfer.setData(DRAG_TYPE, op.op)}
                onClick={() => onAdd(op.op, undefined, insertionIndex)}>
                <strong>{op.op}</strong>
                <span className="muted small">{op.description}</span>
              </button>
            ))}
          </div>
        ))}
      </div>
    </aside>
  );
}

function LayerConnections({ source, nodes, edges, labelFor, onChange }) {
  const [target, setTarget] = useState("");
  const ids = nodes.map((node) => node.id);
  const from = source;
  const to = ids.includes(target) ? target : "";
  const connected = edges.some(([s, t]) => s === from && t === to);
  const problem = from && to ? connectionProblem(edges, from, to) : "";

  return (
    <details className="layer-connections">
      <summary>Connect from this layer</summary>
      <div className="stack">
        <label className="field">To layer
          <select value={to} onChange={(event) => setTarget(event.target.value)}>
            <option value="">Choose a destination</option>
            {nodes.filter(({ id }) => id !== "input" && id !== source).map(({ id }) => <option key={id} value={id}>{labelFor(id)}</option>)}
          </select>
        </label>
        <div className="toolbar">
          <button type="button" className="button" disabled={!from || !to || Boolean(problem)} onClick={() => onChange([...edges, [from, to]])}>Connect layers</button>
          <button type="button" className="button ghost" disabled={!connected} onClick={() => onChange(edges.filter(([s, t]) => s !== from || t !== to))}>Disconnect layers</button>
        </div>
        {problem && <p className="small muted" role="status">{problem}</p>}
      </div>
    </details>
  );
}

function ParamsPanel({ node, op, onChange, onRemove }) {
  const symbolic = OUTPUT_KEYS.filter((key) => op.params.properties?.[key]);
  const toggled = symbolic.filter((key) => node.params[key] === "$num_outputs");
  return (
    <div className="stack">
      <p className="muted small">
        id <code>{node.id}</code> · {op.description}
      </p>
      {symbolic.map((key) => (
        <label key={key} className="field checkbox">
          <input
            type="checkbox"
            data-output-field={key}
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
    </div>
  );
}

function Editor({ sandboxId, task, architecture, positions, ops, validation, knownOutputs, initialIssue, fallbackFocus, onChange, onClose }) {
  const { measurements, trackMeasurements, keepVisible, screenToFlowPosition } = useCanvasLayout();
  const { fitView } = useReactFlow();
  const [selected, setSelected] = useState(initialIssue?.node_id ?? null);
  const [selectedEdge, setSelectedEdge] = useState(null);
  const [outputs, setOutputs] = useState("");
  const [connectionError, setConnectionError] = useState("");
  const [layoutStatus, setLayoutStatus] = useState("");
  const [insertionEdge, setInsertionEdge] = useState(null);
  const [item, setItem] = useState(() => initialIssue ? {
    kind: "node", id: architecture.nodes.some(({ id }) => id === initialIssue.node_id) ? initialIssue.node_id : "input",
  } : null);
  const [focusedIssue, setFocusedIssue] = useState(() => initialIssue ? { ...initialIssue } : null);
  const inspector = useRef(null);
  const graphHeading = useRef(null);
  const inputOptions = useRef(null);
  const outputControl = useRef(null);
  const graph = useRef(null);
  const layerSearch = useRef(null);
  const insertionTrigger = useRef(null);
  const sinks = architecture.nodes.filter(({ id }) => !architecture.edges.some(([source]) => source === id));
  const outputId = architecture.output ?? (sinks.length === 1 ? sinks[0].id : null);
  const opsByName = useMemo(() => Object.fromEntries(ops.map((op) => [op.op, op])), [ops]);

  const previewOutputs = Number(outputs);
  const preview = useModelValidation(sandboxId, architecture, task.type, previewOutputs, previewOutputs > 0);
  const { analysis, checking, error, retry } = previewOutputs > 0 ? preview : validation;

  const focusIssueField = useCallback(() => {
    if (!focusedIssue) return false;
    let target;
    if (focusedIssue.node_id === "input" || focusedIssue.field?.includes("input.shape")) {
      const axis = Number(focusedIssue.field?.match(/shape\.(\d)$/)?.[1] ?? 0);
      target = inputOptions.current?.querySelectorAll("input")[axis];
    } else if (focusedIssue.field === "output") {
      target = outputControl.current;
    } else if (focusedIssue.node_id) {
      const key = focusedIssue.field?.replace(/^params\./, "").split(".")[0];
      const contents = inspector.current;
      if (!contents) return false;
      const field = Array.from(contents.querySelectorAll("[data-schema-field]"))
        .find((element) => element.dataset.schemaField === key);
      const symbolic = Array.from(contents.querySelectorAll("[data-output-field]"))
        .find((element) => element.dataset.outputField === key);
      target = field?.querySelector("input, select, textarea") || symbolic;
    }
    target?.focus();
    return Boolean(target);
  }, [focusedIssue]);

  const nodes = useMemo(() => {
    const warnings = new Map();
    const layerIds = new Set(architecture.nodes.map(({ id }) => id));
    for (const issue of analysis?.issues || []) {
      const owner = layerIds.has(issue.node_id) ? issue.node_id : issue.field === "output" && layerIds.has(outputId) ? outputId : "input";
      if (!warnings.has(owner)) warnings.set(owner, []);
      warnings.get(owner).push({
        message: `${issue.field ? `${issue.field}: ` : ""}${issue.message}`,
        action: owner === "input" && !issue.node_id && !issue.field?.includes("input.shape") && issue.field !== "output" ? undefined : "Go to setting",
        issue,
      });
    }
    const onWarning = ({ issue }) => {
      const id = layerIds.has(issue.node_id) ? issue.node_id : (issue.field === "output" && outputId) || "input";
      setSelected(id);
      setSelectedEdge(null);
      setInsertionEdge(null);
      setItem({ kind: "node", id, anchor: graph.current?.querySelector(`.react-flow__node[data-id="${CSS.escape(id)}"]`) });
      setFocusedIssue({ ...issue });
    };
    const input = {
      id: "input",
      ariaLabel: "Input. Enter to open settings.",
      type: "layer",
      position: positions.input || { x: 0, y: 0 },
      measured: measurements.get("input"),
      deletable: false,
      selected: selected === "input",
      data: { label: "input", op: "Input", shape: architecture.input.shape, warnings: warnings.get("input") || [], onWarning, isInput: true },
    };
    return [
      input,
      ...architecture.nodes.map((node, index) => ({
        id: node.id,
        ariaLabel: `${node.op} (${node.id}). Enter to open settings.`,
        type: "layer",
        position: positions[node.id] || { x: 0, y: (index + 1) * 100 },
        measured: measurements.get(node.id),
        selected: selected === node.id,
        data: {
          label: node.id,
          op: node.op,
          tone: opsByName[node.op]?.category,
          isOutput: outputId === node.id,
          shape: analysis?.nodes?.[node.id]?.shape,
          params: analysis?.nodes?.[node.id]?.params,
          warnings: warnings.get(node.id) || [],
          onWarning,
        },
      })),
    ];
  }, [architecture, positions, selected, analysis, measurements, outputId, opsByName]);

  const chooseInsertion = useCallback((edge, trigger) => {
    insertionTrigger.current = trigger;
    setItem(null);
    setInsertionEdge(edge);
    layerSearch.current?.focus({ preventScroll: true });
  }, []);

  const cancelInsertion = () => {
    setInsertionEdge(null);
    const trigger = insertionTrigger.current;
    (trigger?.isConnected ? trigger : layerSearch.current)?.focus({ preventScroll: true });
  };

  const edges = useMemo(
    () =>
      architecture.edges.map(([source, target]) => {
        const id = `${source}->${target}`;
        return {
          id, source, target, type: "layer", selected: id === selectedEdge,
          data: { onAdd: chooseInsertion, inserting: insertionEdge === id },
        };
      }),
    [architecture.edges, selectedEdge, chooseInsertion, insertionEdge]
  );

  const setArchitecture = (changes) => onChange({ ...architecture, ...changes }, positions);

  const openItem = (kind, id, anchor) => {
    setInsertionEdge(null);
    setSelected(kind === "node" ? id : null);
    setSelectedEdge(kind === "edge" ? id : null);
    setFocusedIssue(null);
    setItem({ kind, id, anchor });
  };
  const closeItem = (restore) => {
    setItem(null);
    if (restore) {
      const anchor = item?.anchor;
      requestAnimationFrame(() => (anchor?.isConnected ? anchor : layerSearch.current)?.focus({ preventScroll: true }));
    }
  };

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
    if (item?.kind === "node" && ids.includes(item.id)) closeItem(true);
  };

  const addNode = (opName, position, insertAt = -1) => {
    if (!opsByName[opName]) return;
    const prefix = opName.toLowerCase();
    let number = 1;
    while (architecture.nodes.some((node) => node.id === `${prefix}${number}`)) number += 1;
    const id = `${prefix}${number}`;
    const last = nodes.at(-1)?.position;
    const edge = architecture.edges[insertAt];
    const target = edge && nodes.find((node) => node.id === edge[1])?.position;
    const where = position || (target ? { x: target.x + 180, y: target.y - 50 } : {
      x: last?.x ?? 0, y: (last?.y ?? 0) + 100,
    });
    onChange(
      {
        ...architecture,
        nodes: [...architecture.nodes, { id, op: opName, params: {} }],
        edges: edge ? insertLayerOnEdge(architecture.edges, insertAt, id) : architecture.edges,
      },
      { ...positions, [id]: where }
    );
    setSelected(id);
  };

  const onNodesChange = (changes) => {
    trackMeasurements(changes);
    const selection = changes.find((c) => c.type === "select" && c.selected);
    if (selection) setSelected(selection.id);
    const removed = changes.filter((c) => c.type === "remove").map((c) => c.id);
    if (removed.length) return removeNodes(removed);
    const moves = changes.filter((c) => c.type === "position" && c.position);
    if (moves.length) {
      const next = { ...positions };
      moves.forEach((move) => (next[move.id] = keepVisible(move.position, move.id)));
      onChange(architecture, next);
    }
  };

  const onEdgesChange = (changes) => {
    const selection = changes.find((c) => c.type === "select");
    if (selection) setSelectedEdge(selection.selected ? selection.id : null);
    const removed = changes.filter((c) => c.type === "remove").map((c) => c.id);
    if (removed.length) {
      setArchitecture({ edges: architecture.edges.filter(([s, t]) => !removed.includes(`${s}->${t}`)) });
      if (item?.kind === "edge" && removed.includes(item.id)) closeItem(true);
    }
  };

  const onConnect = ({ source, target }) => {
    const problem = connectionProblem(architecture.edges, source, target);
    setConnectionError(problem);
    if (problem) return;
    setArchitecture({ edges: [...architecture.edges, [source, target]] });
  };

  const onDrop = (event) => {
    event.preventDefault();
    const op = event.dataTransfer.getData(DRAG_TYPE);
    if (op) {
      addNode(op, keepVisible(screenToFlowPosition({ x: event.clientX, y: event.clientY })));
      setInsertionEdge(null);
    }
  };

  const itemNode = item?.kind === "node" && architecture.nodes.find((node) => node.id === item.id);
  const itemEdge = item?.kind === "edge" && edges.find((edge) => edge.id === item.id);
  const isInput = item?.kind === "node" && item.id === "input";
  const labelFor = (id) => id === "input" ? "Input" : `${id} (${architecture.nodes.find((node) => node.id === id)?.op ?? "unknown"})`;
  const moveItem = (dx, dy) => {
    const position = nodes.find((node) => node.id === item?.id)?.position;
    if (position) {
      onChange(architecture, { ...positions, [item.id]: keepVisible({ x: position.x + dx, y: position.y + dy }, item.id) });
    }
  };
  const autoArrange = async () => {
    const layout = arrangeLayers(nodes, architecture.edges);
    if (nodes.some(({ id, position }) => position.x !== layout.positions[id].x || position.y !== layout.positions[id].y)) {
      onChange(architecture, layout.positions);
    }
    const fitted = await fitView({ padding: 0.18, maxZoom: 1 });
    setLayoutStatus(!fitted ? "Layout arranged. Use Fit View to bring it into view." :
      layout.unresolved ? "Layout arranged. Layers affected by a loop are on the bottom row." : "Layout arranged.");
  };
  const jumpToSelected = async () => {
    if (!selected) return;
    const fitted = await fitView({ nodes: [{ id: selected }], padding: 0.25, maxZoom: 1.15 });
    setLayoutStatus(fitted ? `Showing ${labelFor(selected)}.` : "Could not bring the selected layer into view. Try Fit View.");
  };
  const [channels, height, width] = architecture.input.shape;
  const setInput = (index, value) => {
    const shape = [...architecture.input.shape];
    shape[index] = Number(value) || 1;
    setArchitecture({ input: { shape } });
  };

  return (
    <EditorDialog titleId="layer-editor-title" className="layer-workbench" fallbackFocus={fallbackFocus} onClose={onClose}>
      <header className="layer-toolbar graph-header">
        <h2 id="layer-editor-title" tabIndex={-1}>Layer editor</h2>
        <div className="layer-summary" role="status">
          {checking ? (
            <span className="muted">Checking model…</span>
          ) : error ? (
            <span className="error">Model validation unavailable: {error}</span>
          ) : analysis?.ok ? (
            <span>
              Preview ready · output {analysis.output_shape?.join(" × ")} · {formatCount(analysis.params)} params
              {analysis.flops ? ` · ${formatCount(analysis.flops)} FLOPs` : ""}
            </span>
          ) : analysis ? (
            <span className="visually-hidden">Model warnings are shown on the graph nodes.</span>
          ) : null}
        </div>
        {error && <button type="button" className="button" onClick={retry}>Retry model validation</button>}
        <button type="button" className="button primary" onClick={onClose}>Done</button>
      </header>
      <section className="layer-workspace" aria-labelledby="graph-heading">
          <h3 id="graph-heading" className="visually-hidden" tabIndex={-1} ref={graphHeading}>Layer graph</h3>
          <OpPalette ops={ops} edges={architecture.edges} labelFor={labelFor} insertionEdge={insertionEdge}
            searchRef={layerSearch} onCancelInsertion={cancelInsertion} onAdd={(...args) => {
              addNode(...args);
              if (insertionEdge) layerSearch.current?.focus({ preventScroll: true });
              setInsertionEdge(null);
            }} />
          <div className="canvas graph-surface" ref={graph} onDragOver={(event) => event.preventDefault()} onDrop={onDrop}
            onKeyDownCapture={(event) => activateCanvasItem(event, openItem)}>
            <div className="graph-tools">
              <button type="button" className="button" onClick={autoArrange} aria-describedby="arrange-help">Auto-arrange</button>
              {selected && (
                <button type="button" className="button graph-icon-button" aria-label="Jump to selected layer"
                  title="Jump to selected layer" onClick={jumpToSelected}><Crosshair size={16} strokeWidth={1.6} aria-hidden="true" /></button>
              )}
            </div>
            <p id="arrange-help" className="visually-hidden">Auto-arrange replaces the current layout. It does not change connections or layer settings.</p>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            edgeTypes={edgeTypes}
            nodeTypes={nodeTypes}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={(event, node) => {
              if (!event.target.closest(".react-flow__handle, .node-warnings")) openItem("node", node.id, event.currentTarget);
            }}
            onEdgeClick={(event, edge) => openItem("edge", edge.id, event.currentTarget)}
            onNodeDragStart={() => setItem(null)}
            onPaneClick={() => { setItem(null); setInsertionEdge(null); setSelected(null); setSelectedEdge(null); }}
            autoPanOnNodeDrag={false}
            minZoom={0.05}
            fitView
            fitViewOptions={{ padding: 0.16, maxZoom: 1 }}
            defaultEdgeOptions={EDGE_STYLE}
            defaultMarkerColor="var(--graph-edge)"
            colorMode="light"
            aria-label="Model layer graph"
          >
            <Background gap={20} size={1.4} color="var(--graph-dot)" />
            <Controls showInteractive={false} />
          </ReactFlow>
          <p className={`graph-feedback small${layoutStatus || connectionError ? "" : " visually-hidden"}`}
            role="status">{connectionError || layoutStatus}</p>
          <div className="graph-caption small" aria-hidden="true">Model graph · {architecture.nodes.length} layers</div>
          </div>
        {(itemNode || isInput || itemEdge) && (
          <GraphItemPanel key={`${item.kind}:${item.id}`} anchor={item.anchor} fallback={layerSearch} workspace={graph}
            title={itemEdge ? "Connection settings" : `${isInput ? "Input" : itemNode.op} settings`}
            onDismiss={closeItem} focusIssueField={focusIssueField}>
            {itemEdge ? (
              <>
                <p>{labelFor(itemEdge.source)} → {labelFor(itemEdge.target)}</p>
                <button type="button" className="button" onClick={() => chooseInsertion(itemEdge.id, item.anchor)}>
                  Insert layer here
                </button>
                <button type="button" className="button ghost" onClick={() => {
                  setArchitecture({ edges: architecture.edges.filter(([s, t]) => s !== itemEdge.source || t !== itemEdge.target) });
                  setSelectedEdge(null);
                  closeItem(true);
                }}>Disconnect</button>
              </>
            ) : (
              <div className="stack" ref={inspector}>
                {isInput && (
                  <fieldset className="fieldset" ref={inputOptions}>
                    <legend>Input shape</legend>
                    {[["Channels", channels], ["Height", height], ["Width", width]].map(([axis, value], index) => (
                      <label key={axis} className="field">{axis}
                        <input type="number" min="1" value={value} onChange={(event) => setInput(index, event.target.value)} />
                      </label>
                    ))}
                  </fieldset>
                )}
                {(isInput || item.id === outputId) && (
                  <label className="field">Outputs (preview)
                    <input type="number" min="1" value={outputs} aria-describedby="preview-output-help"
                      placeholder={String(knownOutputs || (task.type === "regression" ? 1 : 2))}
                      onChange={(event) => setOutputs(event.target.value)} />
                    <small id="preview-output-help">Preview only. {knownOutputs
                      ? `This task specifies ${knownOutputs} outputs.`
                      : "Training uses the task and dataset to determine the output count."}</small>
                  </label>
                )}
                {itemNode && opsByName[itemNode.op] && (
                  <ParamsPanel node={itemNode} op={opsByName[itemNode.op]}
                    onChange={(params) => setArchitecture({
                      nodes: architecture.nodes.map((node) => node.id === item.id ? { ...node, params } : node),
                    })}
                    onRemove={() => removeNodes([item.id])} />
                )}
                <LayerConnections source={item.id} nodes={nodes} edges={architecture.edges} labelFor={labelFor}
                  onChange={(edges) => { setConnectionError(""); setArchitecture({ edges }); }} />
                <details className="compact-disclosure">
                  <summary>Move {isInput ? "input" : "layer"}</summary>
                  <div className="toolbar">
                    {[["Left", -32, 0], ["Up", 0, -32], ["Down", 0, 32], ["Right", 32, 0]].map(([label, dx, dy]) => (
                      <button key={label} type="button" className="button" onClick={() => moveItem(dx, dy)}>Move {label.toLowerCase()}</button>
                    ))}
                  </div>
                </details>
                {isInput ? <p tabIndex={-1} ref={outputControl} className="small muted">To choose the model output, open a layer and select Set as output.</p> : (
                  <button type="button" className="button" ref={outputControl} aria-pressed={item.id === outputId}
                    onClick={() => setArchitecture({ output: item.id })}>Set as output</button>
                )}
              </div>
            )}
          </GraphItemPanel>
        )}
      </section>
    </EditorDialog>
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
