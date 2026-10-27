import fs from "node:fs";
import path from "node:path";
const registry = JSON.parse(fs.readFileSync("dist/data/registry.json", "utf8"));
for (const model of registry.models) {
  for (const p of [
    model.path,
    ...(model.externalData || []).map((f) => f.data),
  ]) {
    if (!fs.existsSync(path.join("dist", p)))
      throw new Error(`Missing model asset: ${p}`);
    if (fs.statSync(path.join("dist", p)).size > 25 * 1024 * 1024)
      throw new Error(`Static asset exceeds 25 MiB: ${p}`);
  }
}
for (const sample of registry.samples)
  if (!fs.existsSync(path.join("dist", sample.path)))
    throw new Error(`Missing sample: ${sample.path}`);
const experiments = JSON.parse(
  fs.readFileSync("dist/data/experiments.json", "utf8"),
);
if (experiments.models.length !== 3)
  throw new Error("Expected three measured classification experiments.");
console.log(
  `Validated static application: ${registry.models.length} models, ${registry.samples.length} samples, ${experiments.models.length} experiments. Output: dist/`,
);
