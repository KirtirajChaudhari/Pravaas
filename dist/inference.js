// Pure geometry is exported for reproducible Node tests.
export function iou(a, b) {
  const w = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const h = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  return (w * h) / (a.w * a.h + b.w * b.h - w * h || 1);
}
export function nms(boxes, threshold = 0.45) {
  const remaining = [...boxes].sort((a, b) => b.score - a.score),
    chosen = [];
  while (remaining.length && chosen.length < 100) {
    const box = remaining.shift();
    chosen.push(box);
    for (let i = remaining.length - 1; i >= 0; i--)
      if (
        remaining[i].classId === box.classId &&
        iou(box, remaining[i]) > threshold
      )
        remaining.splice(i, 1);
  }
  return chosen;
}
export function decodeYOLO(data, dims, classes, confidence, geometry) {
  const [, channels, count] = dims;
  if (channels !== classes.length + 4)
    throw new Error("Unexpected detector output shape.");
  const boxes = [],
    { scale, px, py, width, height } = geometry;
  for (let i = 0; i < count; i++) {
    let score = -Infinity,
      classId = 0;
    for (let c = 0; c < classes.length; c++)
      if (data[(c + 4) * count + i] > score) {
        score = data[(c + 4) * count + i];
        classId = c;
      }
    if (score < confidence) continue;
    const cx = data[i],
      cy = data[count + i],
      bw = data[2 * count + i],
      bh = data[3 * count + i];
    const x = Math.max(0, Math.min(width, (cx - bw / 2 - px) / scale));
    const y = Math.max(0, Math.min(height, (cy - bh / 2 - py) / scale));
    const x2 = Math.max(0, Math.min(width, (cx + bw / 2 - px) / scale));
    const y2 = Math.max(0, Math.min(height, (cy + bh / 2 - py) / scale));
    if (x2 > x && y2 > y)
      boxes.push({
        x,
        y,
        w: x2 - x,
        h: y2 - y,
        score,
        classId,
        label: classes[classId],
      });
  }
  return nms(boxes);
}

let runtime;
const sessions = new Map();
export async function inspect(image, model, confidence, onStatus = () => {}) {
  if (!runtime) {
    runtime = await import("./vendor/ort.wasm.min.mjs");
    runtime.env.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;
    runtime.env.wasm.numThreads = 1;
  }
  if (!sessions.has(model.id)) {
    onStatus("Loading model…");
    // Bound memory: keep only the active model's WASM session.
    for (const s of sessions.values()) await s.release();
    sessions.clear();
    sessions.set(
      model.id,
      await runtime.InferenceSession.create(model.path, {
        executionProviders: ["wasm"],
        externalData: model.externalData || [],
      }),
    );
  }
  onStatus("Analyzing image…");
  const size = model.input_size,
    canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const width = image.naturalWidth,
    height = image.naturalHeight;
  const scale = Math.min(size / width, size / height);
  const dw = Math.round(width * scale),
    dh = Math.round(height * scale);
  const px = Math.round((size - dw) / 2 - 0.1),
    py = Math.round((size - dh) / 2 - 0.1);
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, size, size);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  if (model.task === "detect") ctx.drawImage(image, px, py, dw, dh);
  else ctx.drawImage(image, 0, 0, size, size);
  const rgba = ctx.getImageData(0, 0, size, size).data,
    data = new Float32Array(size * size * 3),
    plane = size * size;
  for (let i = 0; i < plane; i++)
    for (let c = 0; c < 3; c++) data[c * plane + i] = rgba[i * 4 + c] / 255;
  const session = sessions.get(model.id),
    start = performance.now();
  const output = await session.run({
    [session.inputNames[0]]: new runtime.Tensor("float32", data, [
      1,
      3,
      size,
      size,
    ]),
  });
  const tensor = output[session.outputNames[0]],
    ms = performance.now() - start;
  if (model.task === "detect")
    return {
      task: "detect",
      boxes: decodeYOLO(tensor.data, tensor.dims, model.classes, confidence, {
        scale,
        px,
        py,
        width,
        height,
      }),
      ms,
      width,
      height,
      confidence,
    };
  const probabilities = Array.from(tensor.data),
    classId = probabilities.indexOf(Math.max(...probabilities));
  return {
    task: "classify",
    probabilities,
    classId,
    label: model.classes[classId],
    score: probabilities[classId],
    ms,
    width,
    height,
    confidence,
  };
}
