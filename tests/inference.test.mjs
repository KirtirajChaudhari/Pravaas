import test from "node:test";
import assert from "node:assert/strict";
import { iou, nms, decodeYOLO } from "../dist/inference.js";

test("IoU handles overlap and disjoint boxes", () => {
  assert.equal(
    iou({ x: 0, y: 0, w: 10, h: 10 }, { x: 20, y: 20, w: 10, h: 10 }),
    0,
  );
  assert.equal(
    iou({ x: 0, y: 0, w: 10, h: 10 }, { x: 0, y: 0, w: 10, h: 10 }),
    1,
  );
  assert.ok(
    Math.abs(
      iou({ x: 0, y: 0, w: 10, h: 10 }, { x: 5, y: 0, w: 10, h: 10 }) - 1 / 3,
    ) < 1e-8,
  );
});
test("NMS suppresses only overlapping boxes of the same class", () => {
  const boxes = [
    { x: 0, y: 0, w: 10, h: 10, score: 0.8, classId: 0 },
    { x: 1, y: 1, w: 10, h: 10, score: 0.9, classId: 0 },
    { x: 0, y: 0, w: 10, h: 10, score: 0.7, classId: 1 },
  ];
  assert.deepEqual(
    nms(boxes).map((x) => x.score),
    [0.9, 0.7],
  );
  assert.equal(boxes.length, 3);
});
test("YOLO decoding reverses letterboxing, clips and filters scores", () => {
  // Two candidates, two classes. Image 200x100 is letterboxed to 416 square.
  const data = new Float32Array([
    208, 208, 208, 208, 208, 208, 104, 104, 0.9, 0.1, 0.2, 0.2,
  ]);
  const boxes = decodeYOLO(data, [1, 6, 2], ["rail", "obstacle"], 0.4, {
    scale: 2.08,
    px: 0,
    py: 104,
    width: 200,
    height: 100,
  });
  assert.equal(boxes.length, 1);
  assert.equal(boxes[0].classId, 0);
  assert.ok(Math.abs(boxes[0].x - 50) < 1e-6);
  assert.ok(Math.abs(boxes[0].y - 25) < 1e-6);
  assert.ok(Math.abs(boxes[0].w - 100) < 1e-6);
  assert.ok(Math.abs(boxes[0].h - 50) < 1e-6);
});
test("Malformed model output fails instead of fabricating findings", () => {
  assert.throws(
    () => decodeYOLO(new Float32Array(10), [1, 5, 2], ["a", "b"], 0.4, {}),
    /shape/,
  );
});
