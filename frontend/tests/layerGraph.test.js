import assert from "node:assert/strict";
import test from "node:test";

import { arrangeLayers, blocksRun, connectionProblem, filterLayerOptions, insertLayerOnEdge } from "../src/features/pipeline/layerGraph.js";

test("inserting a layer replaces only the chosen connection", () => {
  const edges = [["input", "conv"], ["conv", "pool"], ["conv", "skip"]];
  assert.deepEqual(insertLayerOnEdge(edges, 1, "relu"), [
    ["input", "conv"], ["conv", "relu"], ["relu", "pool"], ["conv", "skip"],
  ]);
});

test("connection controls reject loops, duplicate edges, and edges into input", () => {
  const edges = [["input", "conv"], ["conv", "relu"], ["relu", "pool"]];
  assert.match(connectionProblem(edges, "pool", "conv"), /loop/);
  assert.match(connectionProblem(edges, "relu", "pool"), /already connected/);
  assert.match(connectionProblem(edges, "conv", "input"), /cannot receive/);
  assert.equal(connectionProblem(edges, "pool", "output"), "");
});

test("an unknown dataset class count does not block a valid-rank preview", () => {
  const issue = { message: "the model must output a vector of 2 values, got [4]" };
  assert.equal(blocksRun(issue, [4], "classification.single_label", undefined), false);
  assert.equal(blocksRun(issue, [4], "classification.single_label", 2), true);
  assert.equal(blocksRun(issue, [4, 8, 8], "classification.single_label", undefined), true);
  assert.equal(blocksRun({ message: "not reachable from the input" }, undefined, "classification.single_label", undefined), true);
});

test("arranging keeps parallel branches apart and places their merge below both", () => {
  const nodes = [
    { id: "input" },
    { id: "left", measured: { width: 300, height: 130 } },
    { id: "right" },
    { id: "merge" },
  ];
  const edges = [["input", "left"], ["input", "right"], ["left", "merge"], ["right", "merge"]];
  const { positions, unresolved } = arrangeLayers(nodes, edges);
  assert.equal(positions.left.y, positions.right.y);
  assert.ok(positions.left.x + nodes[1].measured.width < positions.right.x);
  assert.ok(positions.merge.y > positions.left.y + nodes[1].measured.height);
  assert.equal(unresolved, 0);
});

test("arranging makes cyclic drafts visible without changing their connections", () => {
  const nodes = [{ id: "input" }, { id: "a" }, { id: "b" }];
  const edges = [["input", "a"], ["a", "b"], ["b", "a"]];
  const { positions, unresolved } = arrangeLayers(nodes, edges);
  assert.equal(unresolved, 2);
  assert.ok(positions.a.y > positions.input.y && positions.a.x !== positions.b.x);
  assert.deepEqual(edges, [["input", "a"], ["a", "b"], ["b", "a"]]);
});

test("layer search matches descriptions and respects the selected category", () => {
  const conv = { op: "Conv2d", category: "convolution", description: "2D convolution" };
  const block = { op: "BasicBlock", category: "block", description: "ResNet basic residual block" };
  const ops = [conv, block];
  assert.deepEqual(filterLayerOptions(ops, "  residual RESNET ", "block"), [block]);
  assert.deepEqual(filterLayerOptions(ops, "residual", "convolution"), []);
  assert.deepEqual(filterLayerOptions(ops, "conv2d", ""), [conv]);
});
