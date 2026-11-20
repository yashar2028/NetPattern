import { useState } from "react";
import { useReactFlow, useStoreApi } from "@xyflow/react";

const EDGE_PADDING = 8;

export function canvasPanelPlacement(workspaceTop, width, height) {
  const top = Math.max(EDGE_PADDING, Math.min(workspaceTop + EDGE_PADDING, height - 220));
  return {
    top,
    maxWidth: width - 2 * EDGE_PADDING,
    maxHeight: height - top - EDGE_PADDING,
  };
}

export function activateCanvasItem(event, activate) {
  if (!["Enter", " "].includes(event.key)) return;
  const el = event.target;
  const kind = el.matches(".react-flow__node") ? "node" : el.matches(".react-flow__edge") ? "edge" : null;
  if (!kind) return;
  event.preventDefault();
  event.stopPropagation();
  activate(kind, el.dataset.id, el);
}

export function keepNodeInView(position, start, end, size) {
  const clamp = (value, min, max) => Math.min(Math.max(value, min), Math.max(min, max));
  return {
    x: clamp(position.x, start.x, end.x - (size?.width ?? 0)),
    y: clamp(position.y, start.y, end.y - (size?.height ?? 0)),
  };
}

export default function useCanvasLayout() {
  const { screenToFlowPosition } = useReactFlow();
  const store = useStoreApi();
  const [measurements, setMeasurements] = useState(() => new Map());

  const trackMeasurements = (changes) => {
    const updates = changes.filter(
      (change) => change.type === "remove" || (change.type === "dimensions" && change.dimensions)
    );
    if (!updates.length) return;
    // Controlled React Flow nodes need their measured sizes carried across position updates.
    setMeasurements((current) => {
      const next = new Map(current);
      for (const change of updates) {
        if (change.type === "remove") next.delete(change.id);
        else next.set(change.id, change.dimensions);
      }
      return next;
    });
  };

  const keepVisible = (position, id) => {
    const rect = store.getState().domNode?.getBoundingClientRect();
    if (!rect) return position;
    const padX = Math.min(EDGE_PADDING, rect.width / 2);
    const padY = Math.min(EDGE_PADDING, rect.height / 2);
    const start = screenToFlowPosition({ x: rect.left + padX, y: rect.top + padY });
    const end = screenToFlowPosition({ x: rect.right - padX, y: rect.bottom - padY });
    return keepNodeInView(position, start, end, measurements.get(id));
  };

  return { measurements, trackMeasurements, keepVisible, screenToFlowPosition };
}
