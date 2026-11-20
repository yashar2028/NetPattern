export function connectionProblem(edges, source, target) {
  if (!source || !target) return "Choose both layers.";
  if (target === "input") return "The model input cannot receive a connection.";
  if (edges.some(([from, to]) => from === source && to === target)) return "These layers are already connected.";
  const visited = new Set();
  const pending = [target];
  while (pending.length) {
    const next = pending.pop();
    if (next === source) return "This connection would create a loop.";
    if (visited.has(next)) continue;
    visited.add(next);
    for (const [from, to] of edges) {
      if (from === next) pending.push(to);
    }
  }
  return "";
}

export function insertLayerOnEdge(edges, index, id) {
  return edges.flatMap(([source, target], i) => i === index ? [[source, id], [id, target]] : [[source, target]]);
}

export function arrangeLayers(nodes, edges) {
  const layers = new Map(nodes.map((node) => [node.id, { ...node, incoming: 0, depth: 0, next: [] }]));
  for (const [source, target] of edges) {
    const from = layers.get(source);
    const to = layers.get(target);
    if (!from || !to) continue;
    from.next.push(to);
    to.incoming += 1;
  }

  const queue = [...layers.values()].filter((layer) => layer.incoming === 0);
  for (const layer of queue) {
    for (const child of layer.next) {
      child.depth = Math.max(child.depth, layer.depth + 1);
      if (--child.incoming === 0) queue.push(child);
    }
  }

  const rows = new Map();
  const unresolved = [];
  for (const layer of layers.values()) {
    if (layer.incoming) unresolved.push(layer);
    else {
      if (!rows.has(layer.depth)) rows.set(layer.depth, []);
      rows.get(layer.depth).push(layer);
    }
  }

  const ordered = [...rows.keys()].sort((a, b) => a - b).map((depth) => rows.get(depth));
  // Keep cyclic drafts visible even when those nodes cannot be ranked.
  if (unresolved.length) ordered.push(unresolved);
  const width = (node) => Math.max(node.measured?.width ?? 0, 240);
  const height = (node) => Math.max(node.measured?.height ?? 0, 64);
  const gap = 48;
  let y = 0;
  const positions = Object.fromEntries(ordered.flatMap((row) => {
    const span = row.reduce((total, layer) => total + width(layer), 0) + gap * (row.length - 1);
    let x = -span / 2;
    const entries = row.map((layer) => {
      const position = { x, y };
      x += width(layer) + gap;
      return [layer.id, position];
    });
    y += Math.max(...row.map(height)) + gap;
    return entries;
  }));
  return { positions, unresolved: unresolved.length };
}

export function filterLayerOptions(ops, query, category) {
  const terms = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
  return ops.filter((op) => {
    if (category && op.category !== category) return false;
    const searchable = `${op.op} ${op.category} ${op.description || ""}`.toLowerCase();
    return terms.every((term) => searchable.includes(term));
  });
}

export function blocksRun(issue, outputShape, taskType, knownOutputs) {
  if (knownOutputs || !outputShape) return true;
  // The engine compares to a placeholder count if the dataset's classes are not yet known.
  if (taskType === "segmentation.semantic") {
    return outputShape.length !== 3 || !/^a segmentation model must output \d+ x H x W/.test(issue.message);
  }
  return outputShape.length !== 1 || !/^the model must output a vector of \d+ values/.test(issue.message);
}
