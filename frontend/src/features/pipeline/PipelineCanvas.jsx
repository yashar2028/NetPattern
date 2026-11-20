import { useCallback, useMemo, useState } from "react";
import { Background, Controls, Handle, MarkerType, Position, ReactFlow } from "@xyflow/react";
import { Box, Check, Database, Play, Plus, SlidersHorizontal } from "lucide-react";

import { NODE_BY_TYPE, NODE_TYPES, defaultParams } from "./defaults";
import GraphItemPanel from "./GraphItemPanel";
import NodeWarnings from "./NodeWarnings";
import useCanvasLayout, { activateCanvasItem } from "./useCanvasLayout";

const ICONS = { dataset: Database, transforms: SlidersHorizontal, model: Box, trainer: Play, evaluator: Check };
const DRAG_TYPE = "application/netpattern-node";
const EDGE_STYLE = { markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14 } };

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
  const label = NODE_BY_TYPE[data.node.type]?.label;
  return (
    <div data-tone={data.node.type} className={`flow-node canvas-card${selected ? " selected" : ""}${data.warnings.length ? " invalid" : ""}`}>
      {data.node.type !== "dataset" && <Handle type="target" position={Position.Left} />}
      <Icon className="canvas-card-icon" size={16} strokeWidth={1.6} aria-hidden="true" />
      <div className="canvas-card-body">
        <div className="flow-node-title">{label}</div>
        <div className="flow-node-detail">{summary(data.node)}</div>
      </div>
      <NodeWarnings label={label} warnings={data.warnings} onWarning={data.onWarning} />
      {data.node.type !== "evaluator" && <Handle type="source" position={Position.Right} />}
    </div>
  );
}

const nodeTypes = { pipeline: PipelineNodeView };

export default function PipelineCanvas({
  spec, onSpecChange, selected, onSelect, warningsByNode, onWarning,
  workspace, panel, onActivate, onDismiss, fallback,
}) {
  const { measurements, trackMeasurements, keepVisible, screenToFlowPosition } = useCanvasLayout();
  const [selectedEdge, setSelectedEdge] = useState(null);
  const positions = spec.ui?.pipeline;
  const missing = NODE_TYPES.filter(({ type }) => !spec.nodes.some((node) => node.type === type));

  const nodes = useMemo(
    () =>
      spec.nodes.map((node, index) => ({
        id: node.id,
        type: "pipeline",
        ariaLabel: `${NODE_BY_TYPE[node.type].label}. Enter to open settings.`,
        position: positions?.[node.id] || { x: index * 200, y: 80 },
        measured: measurements.get(node.id),
        data: { node, warnings: warningsByNode.get(node.id) || [], onWarning },
        selected: node.id === selected,
      })),
    [spec.nodes, positions, selected, measurements, warningsByNode, onWarning]
  );
  const edges = useMemo(
    () =>
      spec.edges.map(([source, target]) => {
        const id = `${source}->${target}`;
        return { id, source, target, selected: id === selectedEdge || (panel?.kind === "edge" && id === panel.id) };
      }),
    [spec.edges, selectedEdge, panel]
  );

  const onNodesChange = (changes) => {
    trackMeasurements(changes);
    const selection = changes.find((c) => c.type === "select" && c.selected);
    if (selection) onSelect(selection.id);
    const removed = changes.filter((c) => c.type === "remove").map((c) => c.id);
    const moves = changes.filter((c) => c.type === "position" && c.position);
    if (!removed.length && !moves.length) return;
    const movedPositions = Object.fromEntries(moves.map((move) => [move.id, keepVisible(move.position, move.id)]));
    onSpecChange((current) => {
      const positions = { ...current.ui?.pipeline, ...movedPositions };
      removed.forEach((id) => delete positions[id]);
      return {
        ...current,
        nodes: removed.length ? current.nodes.filter((node) => !removed.includes(node.id)) : current.nodes,
        edges: removed.length ? current.edges.filter(([s, t]) => !removed.includes(s) && !removed.includes(t)) : current.edges,
        ui: { ...current.ui, pipeline: positions },
      };
    });
    if (removed.includes(selected)) onSelect(null);
  };

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

  const addStep = (type, point) => {
    const definition = missing.find((node) => node.type === type);
    if (!definition || !point) return;
    const { id } = definition;
    const position = keepVisible(screenToFlowPosition(point));
    onSpecChange((current) => current.nodes.some((node) => node.type === type) ? current : ({
      ...current,
      nodes: [...current.nodes, { id, type, params: defaultParams(type) }],
      ui: { ...current.ui, pipeline: { ...(current.ui?.pipeline || {}), [id]: position } },
    }));
    onSelect(id);
    onDismiss(!point.fromDrag);
  };
  const onDrop = (event) => {
    event.preventDefault();
    addStep(event.dataTransfer.getData(DRAG_TYPE), { x: event.clientX, y: event.clientY, fromDrag: true });
  };

  return (
    <div className="canvas graph-surface" ref={workspace} onDragOver={(event) => event.preventDefault()} onDrop={onDrop}
      onKeyDownCapture={(event) => activateCanvasItem(event, onActivate)}>
      <div className="graph-tools">
        {missing.length > 0 && (
          <button type="button" className="button" aria-expanded={panel?.kind === "add"}
            onClick={(event) => onActivate("add", null, event.currentTarget)}>
            <Plus size={16} aria-hidden="true" /> Add step
          </button>
        )}
      </div>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={(event, node) => {
          if (!event.target.closest(".react-flow__handle, .node-warnings")) onActivate("node", node.id, event.currentTarget);
        }}
        onEdgeClick={(event, edge) => onActivate("edge", edge.id, event.currentTarget)}
        onNodeDragStart={() => onDismiss(false)}
        onPaneClick={() => { onDismiss(false); onSelect(null); setSelectedEdge(null); }}
        autoPanOnNodeDrag={false}
        minZoom={0.05}
        fitView
        fitViewOptions={{ padding: 0.12, maxZoom: 1 }}
        defaultEdgeOptions={EDGE_STYLE}
        defaultMarkerColor="var(--graph-edge)"
        colorMode="light"
        aria-label="Pipeline graph"
      >
        <Background gap={20} size={1.4} color="var(--graph-dot)" />
        <Controls showInteractive={false} />
      </ReactFlow>
      <div className="graph-caption small" aria-hidden="true">Pipeline · {spec.nodes.length} steps</div>
      {panel?.kind === "add" && missing.length > 0 && (
        <GraphItemPanel id="pipeline-step-picker" anchor={panel.anchor} workspace={workspace} fallback={fallback}
          title="Add a step" closeLabel="Close picker" onDismiss={onDismiss}>
          <p className="muted small">Choose a step or drag it to the canvas.</p>
          <div className="graph-picker">
            {missing.map(({ type, label }) => {
              const Icon = ICONS[type];
              return (
                <button key={type} type="button" className="palette-item" draggable
                  onDragStart={(event) => event.dataTransfer.setData(DRAG_TYPE, type)}
                  onClick={() => {
                    const box = workspace.current.getBoundingClientRect();
                    addStep(type, { x: box.left + box.width * 0.35, y: box.top + box.height * 0.62 });
                  }}>
                  <Icon size={16} aria-hidden="true" /> {label}
                </button>
              );
            })}
          </div>
        </GraphItemPanel>
      )}
    </div>
  );
}
