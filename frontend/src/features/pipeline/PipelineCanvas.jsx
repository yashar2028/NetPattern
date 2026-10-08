import { useCallback, useMemo, useState } from "react";
import { Background, Controls, Handle, Position, ReactFlow, useReactFlow } from "@xyflow/react";
import { Brain, Database, FlaskConical, Gauge, Wand2 } from "lucide-react";

import { NODE_BY_TYPE, NODE_TYPES, defaultParams } from "./defaults";

const ICONS = { dataset: Database, transforms: Wand2, model: Brain, trainer: Gauge, evaluator: FlaskConical };
const DRAG_TYPE = "application/netpattern-node";

function summary(node) {
  const params = node.params || {};
  switch (node.type) {
    case "dataset":
      return params.format === "synthetic"
        ? "synthetic data"
        : `${params.format}${params.dataset_version_id ? "" : " · pick a dataset"}`;
    case "transforms":
      return `${params.preset || "from_weights"} · ${(params.train || []).length} augmentations`;
    case "model": {
      const arch = params.architecture || {};
      return arch.kind === "graph" ? `custom layers (${arch.nodes?.length || 0})` : arch.base?.name || "choose a model";
    }
    case "trainer":
      return `${params.epochs ?? 10} epochs · batch ${params.batch_size ?? 32}`;
    case "evaluator":
      return (params.splits || ["val", "test"]).join(" + ");
    default:
      return "";
  }
}

function PipelineNodeView({ data, selected }) {
  const Icon = ICONS[data.node.type];
  return (
    <div className={`flow-node${selected ? " selected" : ""}`}>
      {data.node.type !== "dataset" && <Handle type="target" position={Position.Left} />}
      <div className="flow-node-title">
        <Icon size={16} aria-hidden="true" /> {NODE_BY_TYPE[data.node.type]?.label}
      </div>
      <div className="flow-node-detail">{summary(data.node)}</div>
      {data.node.type !== "evaluator" && <Handle type="source" position={Position.Right} />}
    </div>
  );
}

const nodeTypes = { pipeline: PipelineNodeView };

export function PipelinePalette({ spec }) {
  const present = new Set(spec.nodes.map((node) => node.type));
  return (
    <aside className="palette palette-row" aria-label="Pipeline nodes">
      <span className="muted small">Drag a node onto the canvas, then connect left to right:</span>
      {NODE_TYPES.map((node) => {
        const Icon = ICONS[node.type];
        const used = present.has(node.type);
        return (
          <div
            key={node.type}
            className={`palette-item${used ? " used" : ""}`}
            draggable={!used}
            onDragStart={(event) => event.dataTransfer.setData(DRAG_TYPE, node.type)}
            title={used ? "already on the canvas" : "drag onto the canvas"}
          >
            <Icon size={16} aria-hidden="true" /> {node.label}
          </div>
        );
      })}
    </aside>
  );
}

export default function PipelineCanvas({ spec, onSpecChange, selected, onSelect }) {
  const { screenToFlowPosition } = useReactFlow();
  const [selectedEdge, setSelectedEdge] = useState(null);
  const positions = spec.ui?.pipeline;

  const nodes = useMemo(
    () =>
      spec.nodes.map((node, index) => ({
        id: node.id,
        type: "pipeline",
        position: positions?.[node.id] || { x: index * 200, y: 80 },
        data: { node },
        selected: node.id === selected,
      })),
    [spec.nodes, positions, selected]
  );
  const edges = useMemo(
    () =>
      spec.edges.map(([source, target]) => {
        const id = `${source}->${target}`;
        return { id, source, target, selected: id === selectedEdge };
      }),
    [spec.edges, selectedEdge]
  );

  // Only moves and deletions change the spec; selection and size measurements do not.
  const onNodesChange = useCallback(
    (changes) => {
      const selection = changes.find((c) => c.type === "select" && c.selected);
      if (selection) onSelect(selection.id);
      const removed = changes.filter((c) => c.type === "remove").map((c) => c.id);
      const moves = changes.filter((c) => c.type === "position" && c.position);
      if (!removed.length && !moves.length) return;
      onSpecChange((current) => {
        const positions = { ...(current.ui?.pipeline || {}) };
        moves.forEach((move) => (positions[move.id] = move.position));
        removed.forEach((id) => delete positions[id]);
        return {
          ...current,
          nodes: current.nodes.filter((node) => !removed.includes(node.id)),
          edges: current.edges.filter(([s, t]) => !removed.includes(s) && !removed.includes(t)),
          ui: { ...current.ui, pipeline: positions },
        };
      });
      if (removed.includes(selected)) onSelect(null);
    },
    [onSpecChange, onSelect, selected]
  );

  const onEdgesChange = useCallback(
    (changes) => {
      const selection = changes.find((c) => c.type === "select");
      if (selection) setSelectedEdge(selection.selected ? selection.id : null);
      const removed = changes.filter((c) => c.type === "remove").map((c) => c.id);
      if (!removed.length) return;
      onSpecChange((current) => ({
        ...current,
        edges: current.edges.filter(([s, t]) => !removed.includes(`${s}->${t}`)),
      }));
    },
    [onSpecChange]
  );

  const onConnect = useCallback(
    ({ source, target }) =>
      onSpecChange((current) =>
        current.edges.some(([s, t]) => s === source && t === target)
          ? current
          : { ...current, edges: [...current.edges, [source, target]] }
      ),
    [onSpecChange]
  );

  const onDrop = useCallback(
    (event) => {
      event.preventDefault();
      const type = event.dataTransfer.getData(DRAG_TYPE);
      if (!type || spec.nodes.some((node) => node.type === type)) return;
      const id = NODE_BY_TYPE[type].id;
      const position = screenToFlowPosition({ x: event.clientX, y: event.clientY });
      onSpecChange((current) => ({
        ...current,
        nodes: [...current.nodes, { id, type, params: defaultParams(type) }],
        ui: { ...current.ui, pipeline: { ...(current.ui?.pipeline || {}), [id]: position } },
      }));
      onSelect(id);
    },
    [spec.nodes, screenToFlowPosition, onSpecChange, onSelect]
  );

  return (
    <div className="canvas" onDragOver={(event) => event.preventDefault()} onDrop={onDrop}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onPaneClick={() => onSelect(null)}
        fitView
        fitViewOptions={{ padding: 0.05, maxZoom: 1 }}
        colorMode="system"
      >
        <Background gap={20} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
