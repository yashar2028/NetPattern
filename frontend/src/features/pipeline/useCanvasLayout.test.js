import assert from "node:assert/strict";
import test from "node:test";

import { canvasPanelPlacement, keepNodeInView } from "./useCanvasLayout.js";

test("a tall editor header leaves a scrollable panel within a narrow viewport", () => {
  const { top, maxWidth, maxHeight } = canvasPanelPlacement(510, 320, 640);
  assert.equal(maxWidth, 304);
  assert.ok(top < 510 && maxHeight >= 200);
  assert.ok(top + maxHeight < 640);
});

const start = { x: -40, y: 20 };
const end = { x: 360, y: 220 };
const size = { width: 150, height: 45 };

test("keeps the whole node visible when dragged beyond either corner", () => {
  assert.deepEqual(keepNodeInView({ x: -100, y: -50 }, start, end, size), start);
  assert.deepEqual(keepNodeInView({ x: 400, y: 250 }, start, end, size), { x: 210, y: 175 });
});

test("does not change positions that already fit", () => {
  const position = { x: 80, y: 85 };
  assert.deepEqual(keepNodeInView(position, start, end, size), position);
});

test("keeps the leading edge reachable when the node is wider than the viewport", () => {
  assert.deepEqual(keepNodeInView({ x: 400, y: 85 }, start, end, { width: 500, height: 45 }), {
    x: start.x,
    y: 85,
  });
});
