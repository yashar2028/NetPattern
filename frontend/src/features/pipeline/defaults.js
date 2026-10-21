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

function withChanges(spec, changes) {
  return {
    ...spec,
    task: changes.task || spec.task,
    nodes: spec.nodes.map((node) => (changes[node.type] ? { ...node, params: changes[node.type] } : node)),
  };
}

const SMALL_CNN = {
  kind: "graph",
  input: { shape: [3, 64, 64] },
  nodes: [
    { id: "conv1", op: "Conv2d", params: { out_channels: 32, kernel_size: 3, padding: 1 } },
    { id: "bn1", op: "BatchNorm2d", params: {} },
    { id: "relu1", op: "ReLU", params: {} },
    { id: "pool1", op: "MaxPool2d", params: { kernel_size: 2 } },
    { id: "conv2", op: "Conv2d", params: { out_channels: 64, kernel_size: 3, padding: 1 } },
    { id: "relu2", op: "ReLU", params: {} },
    { id: "gap", op: "GlobalAvgPool", params: {} },
    { id: "fc", op: "Linear", params: { out_features: "$num_outputs" } },
  ],
  edges: [
    ["input", "conv1"],
    ["conv1", "bn1"],
    ["bn1", "relu1"],
    ["relu1", "pool1"],
    ["pool1", "conv2"],
    ["conv2", "relu2"],
    ["relu2", "gap"],
    ["gap", "fc"],
  ],
};

/** Starting points for "New pipeline"; everything stays editable on the canvas. */
export const TEMPLATES = [
  { id: "resnet18", label: "Classification: fine-tune a pretrained ResNet-18", spec: defaultSpec },
  {
    id: "small-cnn",
    label: "Classification: a small CNN of your own (layer editor)",
    spec: () =>
      withChanges(defaultSpec(), {
        transforms: { preset: "basic", image_size: 64 },
        model: { architecture: SMALL_CNN },
        trainer: { epochs: 15, batch_size: 32, num_workers: 2 },
      }),
  },
  {
    id: "unet",
    label: "Segmentation: U-Net with a pretrained ResNet-34 encoder",
    spec: () =>
      withChanges(defaultSpec(), {
        task: { type: "segmentation.semantic" },
        dataset: { format: "segmentation_folders" },
        transforms: { preset: "basic", image_size: 256 },
        model: { architecture: { kind: "pretrained", base: { source: "netpattern", name: "unet", encoder: "resnet34", weights: "DEFAULT" } } },
        trainer: { epochs: 20, batch_size: 8, num_workers: 2 },
      }),
  },
  {
    id: "detection",
    label: "Object detection: Faster R-CNN (MobileNet, 320 px)",
    spec: () =>
      withChanges(defaultSpec(), {
        task: { type: "detection.bbox" },
        dataset: { format: "voc" },
        transforms: { preset: "basic" },
        model: {
          architecture: {
            kind: "pretrained",
            base: { source: "torchvision", name: "fasterrcnn_mobilenet_v3_large_320_fpn", weights: "DEFAULT", options: { min_size: 320 } },
          },
        },
        trainer: { epochs: 10, batch_size: 4, num_workers: 2 },
      }),
  },
];
