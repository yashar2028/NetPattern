/** Pipeline node types in chain order (PLAN §9). */
export const NODE_TYPES = [
  { type: "dataset", id: "data", label: "Dataset", required: true },
  { type: "transforms", id: "tf", label: "Transforms", required: false },
  { type: "model", id: "model", label: "Model", required: true },
  { type: "trainer", id: "train", label: "Trainer", required: true },
  { type: "evaluator", id: "eval", label: "Evaluator", required: false },
];

export const NODE_BY_TYPE = Object.fromEntries(NODE_TYPES.map((node) => [node.type, node]));

export function defaultParams(type) {
  switch (type) {
    case "dataset":
      return { format: "image_folder" };
    case "transforms":
      return { preset: "from_weights" };
    case "model":
      return {
        architecture: {
          kind: "pretrained",
          base: { source: "torchvision", name: "resnet18", weights: "DEFAULT" },
        },
      };
    case "trainer":
      return { epochs: 5, batch_size: 16, num_workers: 2 };
    case "evaluator":
      return { splits: ["val", "test"] };
    default:
      return {};
  }
}

/** A ready-to-edit pipeline: a pretrained ResNet-18 on an image-folder dataset. */
export function defaultSpec() {
  const nodes = NODE_TYPES.map((node) => ({ id: node.id, type: node.type, params: defaultParams(node.type) }));
  const edges = NODE_TYPES.slice(1).map((node, index) => [NODE_TYPES[index].id, node.id]);
  const positions = Object.fromEntries(NODE_TYPES.map((node, index) => [node.id, { x: index * 200, y: 80 }]));
  return {
    task: { type: "classification.single_label" },
    nodes,
    edges,
    ui: { pipeline: positions },
  };
}
