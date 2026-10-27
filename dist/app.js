import { inspect } from "./inference.js";

const paths = {
  scan: "M4 9V5a1 1 0 0 1 1-1h4m6 0h4a1 1 0 0 1 1 1v4m0 6v4a1 1 0 0 1-1 1h-4m-6 0H5a1 1 0 0 1-1-1v-4M8 8h8v8H8z",
  chart: "M4 3v17h17M8 15v-4m5 4V7m5 8V4",
  database:
    "M20 6c0 2-4 3-8 3S4 8 4 6s4-3 8-3 8 1 8 3Zm-16 0v6c0 2 4 3 8 3s8-1 8-3V6M4 12v6c0 2 4 3 8 3s8-1 8-3v-6",
  history: "M3 11a9 9 0 1 1 2 7M3 4v7h7m2-4v5l3 2",
  code: "m8 7-5 5 5 5m8-10 5 5-5 5m-3-13-2 16",
  help: "M9 9a3 3 0 1 1 5 2c-2 1-2 2-2 3m0 3h.01M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0Z",
  upload: "M12 16V3m-5 5 5-5 5 5M4 15v5a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-5",
  download: "M12 3v13m-5-5 5 5 5-5M4 16v4h16v-4",
  layers: "m12 3 10 5-10 5L2 8l10-5Zm-10 9 10 5 10-5M2 16l10 5 10-5",
  shield: "m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6l8-3Zm-4 9 3 3 5-6",
  expand: "M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5",
  image: "M3 3h18v18H3zM3 17l5-5 4 4 3-3 6 6M7 7h.01",
  sliders: "M4 6h7m4 0h5M4 12h2m4 0h10M4 18h10m4 0h2M11 3v6M6 9v6m8 0v6",
  play: "m8 4 12 8-12 8V4Z",
  focus:
    "M7 3H3v4m14-4h4v4M3 17v4h4m14-4v4h-4M16 12a4 4 0 1 1-8 0 4 4 0 0 1 8 0Z",
  info: "M12 11v6m0-10h.01M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0Z",
};
function icons(root = document) {
  root
    .querySelectorAll("[data-icon]")
    .forEach(
      (e) =>
        (e.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="${paths[e.dataset.icon] || paths.scan}"/></svg>`),
    );
}
icons();
const $ = (id) => document.getElementById(id);
const esc = (s) =>
  String(s ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const pct = (n) => (100 * n).toFixed(1) + "%";
const state = {
  registry: null,
  experiments: null,
  model: null,
  sample: null,
  imageReady: false,
  busy: false,
  result: null,
  overlay: true,
  imageToken: 0,
  objectURL: null,
  history: [],
};
try {
  state.history = JSON.parse(localStorage.getItem("pravaas-history") || "[]");
  if (!Array.isArray(state.history)) state.history = [];
} catch {
  state.history = [];
}
function toast(message) {
  $("toast").textContent = message;
  $("toast").hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => ($("toast").hidden = true), 4200);
}
function download(data, name, type = "application/json") {
  const b = new Blob(
    [type === "application/json" ? JSON.stringify(data, null, 2) : data],
    { type },
  );
  const u = URL.createObjectURL(b);
  const a = document.createElement("a");
  a.href = u;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(u), 1000);
}
function clearResult() {
  state.result = null;
  $("results").innerHTML =
    '<div class="empty-result"><span data-icon="focus"></span><strong>Ready when you are</strong><p>Run an inspection to see model findings and confidence scores.</p></div>';
  $("result-count").textContent = "—";
  $("export-button").disabled = true;
  $("image-state").textContent = "READY TO INSPECT";
  $("inference-status").textContent = "Model loads on first run.";
  $("inference-status").classList.remove("error-text");
  draw();
  icons($("results"));
}
function setModels() {
  const mode = $("task-select").value;
  const available = state.registry.models.filter((m) => m.mode === mode);
  $("model-select").innerHTML = available
    .map((m) => `<option value="${esc(m.id)}">${esc(m.name)}</option>`)
    .join("");
  if (mode === "classification") $("model-select").value = "finetuned";
  updateModel();
  renderSamples();
}
function updateModel() {
  state.model = state.registry.models.find(
    (m) => m.id === $("model-select").value,
  );
  $("model-description").textContent = state.model.description;
  const classification = state.model.task === "classify";
  $("confidence").disabled = classification;
  $("confidence-value").textContent = classification
    ? "All scores"
    : $("confidence").value + "%";
  clearResult();
}
function renderSamples() {
  const mode = $("task-select").value;
  const samples = [...state.registry.samples].sort(
    (a, b) => (b.mode === mode) - (a.mode === mode),
  );
  $("sample-grid").innerHTML = samples
    .map(
      (s) =>
        `<button class="sample-card ${state.sample?.id === s.id ? "active" : ""}" data-sample="${esc(s.id)}" aria-label="Inspect ${esc(s.title)}" ${state.busy ? "disabled" : ""}><img src="${esc(s.path)}" alt="${esc(s.title)}" loading="lazy"><div><div><strong>${esc(s.title)}</strong><small>${esc(s.subtitle)}</small></div><span class="sample-arrow">↗</span></div></button>`,
    )
    .join("");
  $("sample-grid")
    .querySelectorAll("[data-sample]")
    .forEach((b) => (b.onclick = () => selectSample(b.dataset.sample)));
}
async function selectSample(id) {
  if (state.busy) return;
  const s = state.registry.samples.find((s) => s.id === id);
  state.sample = s;
  if ($("task-select").value !== s.mode) {
    $("task-select").value = s.mode;
    setModels();
  }
  await loadImage(s.path, s.title, s.source);
  renderSamples();
}
async function loadImage(url, name, source) {
  const token = ++state.imageToken;
  state.imageReady = false;
  $("run-button").disabled = true;
  clearResult();
  const im = $("inspection-image");
  try {
    await new Promise((resolve, reject) => {
      im.onload = resolve;
      im.onerror = () => reject(new Error("This image could not be decoded."));
      im.src = url;
    });
    if (token !== state.imageToken) return;
    if (im.naturalWidth * im.naturalHeight > 40_000_000)
      throw new Error("Please resize this image below 40 megapixels.");
    state.imageReady = true;
    $("image-name").textContent = name;
    $("image-dimensions").textContent =
      `${im.naturalWidth} × ${im.naturalHeight} px`;
    $("sample-source").textContent = source;
    im.alt = name;
    $("run-button").disabled = false;
    draw();
  } catch (e) {
    toast(e.message);
    $("inference-status").textContent = e.message;
    $("image-state").textContent = "IMAGE UNAVAILABLE";
  }
}
async function upload(file) {
  if (!file || state.busy) return;
  if (!["image/jpeg", "image/png", "image/webp"].includes(file.type))
    return toast("Choose a JPG, PNG, or WebP image.");
  if (file.size > 20 * 1024 * 1024)
    return toast("Please choose an image smaller than 20 MB.");
  if (state.objectURL) URL.revokeObjectURL(state.objectURL);
  state.objectURL = URL.createObjectURL(file);
  state.sample = null;
  await loadImage(state.objectURL, file.name, "Your image · local only");
  renderSamples();
}
function draw() {
  const canvas = $("overlay"),
    im = $("inspection-image");
  canvas.width = im.naturalWidth || 1;
  canvas.height = im.naturalHeight || 1;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!state.overlay || !state.result || state.result.task !== "detect") return;
  const scale = canvas.width / 750;
  ctx.lineWidth = Math.max(2, 2.2 * scale);
  ctx.font = `600 ${Math.max(12, 13 * scale)}px system-ui`;
  state.result.boxes.forEach((b, i) => {
    const color = ["#74e0b5", "#ffca79", "#8fb3ff", "#e6a2f8"][b.classId % 4];
    ctx.strokeStyle = color;
    ctx.strokeRect(b.x, b.y, b.w, b.h);
    const text = `${b.label}  ${Math.round(b.score * 100)}%`,
      h = Math.max(23, 25 * scale),
      w = ctx.measureText(text).width + 14 * scale;
    const y = Math.max(0, b.y - h);
    ctx.fillStyle = color;
    ctx.fillRect(b.x, y, Math.min(w, canvas.width - b.x), h);
    ctx.fillStyle = "#162e2b";
    ctx.fillText(text, b.x + 7 * scale, y + h * 0.7);
  });
}
function renderResult() {
  const r = state.result;
  let content = "";
  if (r.task === "detect") {
    $("result-count").textContent = r.boxes.length;
    if (!r.boxes.length)
      content =
        '<div class="empty-result"><span data-icon="scan"></span><strong>No detections at this threshold</strong><p>This is not evidence that the track is safe. Try a lower threshold or another image.</p></div>';
    else content = r.boxes.map((b) => finding(b.label, b.score)).join("");
  } else {
    $("result-count").textContent = "2 classes";
    content = r.probabilities
      .map((p, i) => finding(state.model.classes[i], p))
      .join("");
  }
  content += `<p class="result-meta">${(r.ms / 1000).toFixed(2)} s inference · ${esc(state.model.name)}<br>${r.task === "detect" ? "Boxes indicate model predictions, not verified hazards." : "Image-level prediction. No defect location is inferred."}</p>`;
  $("results").innerHTML = content;
  icons($("results"));
  $("export-button").disabled = false;
  $("image-state").textContent = "INSPECTION COMPLETE";
  draw();
}
function finding(label, score) {
  return `<div class="finding"><div class="finding-row"><span class="finding-name"><i></i>${esc(label)}</span><span class="finding-score">${pct(score)}</span></div><div class="score-track"><span style="width:${score * 100}%"></span></div></div>`;
}
async function run() {
  if (state.busy || !state.imageReady) return;
  state.busy = true;
  clearResult();
  $("run-button").disabled = true;
  $("task-select").disabled = true;
  $("model-select").disabled = true;
  $("confidence").disabled = true;
  document.body.classList.add("busy");
  document.querySelector(".scan-line").hidden = false;
  $("image-state").textContent = "ANALYZING";
  renderSamples();
  try {
    state.result = await inspect(
      $("inspection-image"),
      state.model,
      Number($("confidence").value) / 100,
      (message) => {
        $("run-text").textContent = message;
        $("inference-status").textContent = message;
      },
    );
    renderResult();
    $("inference-status").textContent = "Complete · processed on your device";
    const report = {
      id: crypto.randomUUID(),
      created_at: new Date().toISOString(),
      image: $("image-name").textContent,
      sample: state.sample?.id || null,
      model: state.model.id,
      model_name: state.model.name,
      model_origin: state.model.origin,
      model_sha256: state.model.sha256 || state.model.sha256_source,
      source: state.model.source || state.registry.classification_source,
      note: "Research output; human review required. Confidence is not a safety probability.",
      ...state.result,
    };
    state.result.report = report;
    state.history.unshift(report);
    state.history = state.history.slice(0, 100);
    try {
      localStorage.setItem("pravaas-history", JSON.stringify(state.history));
    } catch {
      toast("Inspection complete. Browser history storage is unavailable.");
    }
  } catch (e) {
    console.error(e);
    $("inference-status").textContent =
      "Model could not run. Check your connection and try again.";
    $("inference-status").classList.add("error-text");
    $("image-state").textContent = "INSPECTION FAILED";
    toast("Inference failed. No findings were generated.");
  } finally {
    state.busy = false;
    document.body.classList.remove("busy");
    document.querySelector(".scan-line").hidden = true;
    $("run-text").textContent = "Run inspection";
    $("run-button").disabled = !state.imageReady;
    $("task-select").disabled = false;
    $("model-select").disabled = false;
    $("confidence").disabled = state.model.task === "classify";
    renderSamples();
  }
}
function heading(eyebrow, title, description, button = "") {
  return `<div class="page-heading"><div><div class="eyebrow"><span></span>${eyebrow}</div><h1>${title}</h1><p>${description}</p></div>${button}</div>`;
}
function renderLab() {
  const e = state.experiments;
  if (!e) {
    $("page-lab").innerHTML = heading(
      "MODEL LAB",
      "Experiments unavailable.",
      "The experiment report could not be loaded.",
    );
    return;
  }
  const models = e.models;
  $("page-lab").innerHTML =
    heading(
      "MEASURE. COMPARE. UNDERSTAND.",
      "Evidence behind every model.",
      "CNN and transfer learning, evaluated on the same held-out images.",
      '<button id="download-experiments" class="button secondary"><span data-icon="download"></span> Export experiments</button>',
    ) +
    `<div class="summary-grid"><div class="summary-card"><div class="summary-icon blue" data-icon="image"></div><div><span>Held-out test images</span><strong>${e.dataset.counts.test}<small>Never used for checkpoint selection</small></strong></div></div><div class="summary-card"><div class="summary-icon purple" data-icon="layers"></div><div><span>Training strategies</span><strong>03<small>Scratch · Frozen · Fine-tuned</small></strong></div></div><div class="summary-card"><div class="summary-icon green" data-icon="chart"></div><div><span>Reproducible experiment</span><strong class="text-value">Seed ${e.seed}<small>${e.input_size} × ${e.input_size} input · ${esc(e.hardware.device.toUpperCase())}</small></strong></div></div></div>
 <div class="data-panel"><h2>Classification benchmark</h2><p>Precision, recall, and F1 treat <strong>Defective</strong> as the positive class. These are measured test results, not detector mAP.</p><div class="table-scroll"><table><thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>Train time</th><th>Parameters</th></tr></thead><tbody>${models.map((m) => `<tr><td><strong>${esc(m.name)}</strong><small>${m.epochs} epochs · checkpoint ${m.best_epoch}</small></td><td>${pct(m.accuracy)}</td><td>${pct(m.precision)}</td><td>${pct(m.recall)}</td><td>${pct(m.f1)}</td><td>${m.training_seconds.toFixed(1)} s</td><td>${(m.parameters / 1e6).toFixed(2)} M<small>${(m.trainable_parameters / 1e6).toFixed(2)} M trainable</small></td></tr>`).join("")}</tbody></table></div></div>
 <div class="section-line"><h2>Inside the experiment</h2><span class="muted">Confusion matrices & validation curves</span></div><div class="lab-grid">${models.map((m) => `<article class="data-panel"><h3>${esc(m.name)}</h3><p>${esc(m.strategy)}</p><div class="confusion"><div></div><span class="axis">Pred. defective</span><span class="axis">Pred. non-def.</span><span class="axis">Actual<br>def.</span><div class="cell">${m.confusion_matrix[0][0]}</div><div class="cell off">${m.confusion_matrix[0][1]}</div><span class="axis">Actual<br>non-def.</span><div class="cell off">${m.confusion_matrix[1][0]}</div><div class="cell">${m.confusion_matrix[1][1]}</div></div><div class="cm-caption">Rows: actual · Columns: predicted</div>${curve(m.history)}<div class="chart-caption"><span>Validation accuracy</span><span>Epochs 1–${m.epochs}</span></div></article>`).join("")}</div>
 <div class="data-panel"><h2>What this experiment establishes</h2><p>${esc(e.dataset.split_policy)}. ${e.dataset.duplicates_removed} exact duplicate images removed. ${esc(e.test_policy)} ${esc(e.dataset.limitation)}</p><p>The dataset is small, and images from the same capture session may share backgrounds. The benchmark is an academic comparison; it does not establish performance across routes, weather, cameras, or unseen defect types. Fine-tuning time excludes its earlier frozen-feature training stage.</p><p>Latency in the JSON report is batched CPU evaluation, not browser latency. The inspection studio measures browser inference separately, excluding initial model download.</p></div>
 <div class="data-panel"><h2>Inherited detector checkpoints</h2><div class="table-scroll"><table><thead><tr><th>Model</th><th>Verified label map</th><th>Evaluation status</th></tr></thead><tbody>${state.registry.models
   .filter((m) => m.task === "detect")
   .map(
     (m) =>
       `<tr><td>${esc(m.name)}<small>Reference weights, exported to ONNX</small></td><td>${esc(m.classes.join(", "))}</td><td>Independent mAP not established<small>Original training/test overlap is unknown</small></td></tr>`,
   )
   .join(
     "",
   )}</tbody></table></div><p>The rail detector's single label is “railway”. Its boxes localize rail regions; they do not identify a crack or a defective rail. The obstacle checkpoint exposes six classes; vehicle is absent from its label map.</p></div>
 <div class="data-panel"><h2>Reproduce the lab</h2><p>Training scripts, dataset manifest, selected weights, and exact experiment results are included in the source project.</p><pre class="code-block">python ml/prepare.py data/kaggle.zip --output data/classification\npython ml/train.py --data data/classification --output artifacts \\\n  --epochs 12 --finetune-epochs 5 --threads 2\npython ml/publish_artifacts.py --artifacts artifacts</pre></div>`;
  $("download-experiments").onclick = () =>
    download(e, "pravaas-experiments.json");
  icons($("page-lab"));
}
function curve(history) {
  const coords = history
    .map(
      (r, i) =>
        `${10 + (i * 260) / Math.max(1, history.length - 1)},${75 - r.val_accuracy * 60}`,
    )
    .join(" ");
  return `<svg class="chart" viewBox="0 0 280 90" role="img" aria-label="Validation accuracy by epoch"><path d="M10 15H270M10 45H270M10 75H270" stroke="#edf0f6" stroke-width="1"/><polyline points="${coords}" fill="none" stroke="#6f8fe5" stroke-width="2.5" stroke-linejoin="round"/><text x="10" y="87" fill="#9aa5b8" font-size="7">0%</text><text x="10" y="10" fill="#9aa5b8" font-size="7">100%</text></svg>`;
}
function renderDatasets() {
  const d = state.experiments?.dataset;
  const reference = state.registry.reference_dataset;
  $("page-datasets").innerHTML =
    heading(
      "KNOW YOUR DATA.",
      "Traceable from the first image.",
      "Dataset sources, label maps, and the limits that matter.",
    ) +
    `<article class="data-panel dataset-card"><div><div class="dataset-title"><span data-icon="database"></span><h2>Railway Track Fault Detection</h2><span class="tag">TRAINED IN THIS BUILD</span></div><p>Salman Ibne Eunus · Kaggle · version 2. Binary image classification: defective and non-defective railway track.</p><div class="pills"><span class="pill">${d ? Object.values(d.counts).reduce((a, b) => a + b, 0) : "—"} retained images</span><span class="pill">${d?.counts.train ?? "—"} train</span><span class="pill">${d?.counts.val ?? "—"} validation</span><span class="pill">${d?.counts.test ?? "—"} test</span><span class="pill">${d?.duplicates_removed ?? "—"} exact duplicates removed</span></div><p>Used for all three classification experiments. ${esc(d?.split_policy || "")}.</p><p>License shown by Kaggle: “Data files © Original Authors”. Attribution is retained; this rebuild does not grant additional rights to the images.</p><a href="${esc(state.registry.classification_source)}" target="_blank" rel="noreferrer" class="data-link">View the requested Kaggle dataset ↗</a></div><button id="download-manifest" class="button secondary"><span data-icon="download"></span> Data manifest</button></article>
 <article class="data-panel"><div class="dataset-title"><span data-icon="layers"></span><h2>Rail localization reference dataset</h2><span class="tag">REFERENCE DATA</span></div><p>The Rail-defect-detection repository includes ${reference.images} annotated images, including ${reference.roboflow_named_images} filenames carrying Roboflow export markers. Its README links to the Roboflow rail-defect project. The checkpoint label map contains one class: <strong>railway</strong>.</p><div class="pills"><span class="pill">YOLO bounding boxes</span><span class="pill">1 checkpoint class</span><span class="pill">Roboflow lineage partially documented</span></div><p>The supplied Roboflow search link is a discovery page, not a pinned dataset/version. Direct Universe access returned HTTP 403 in this environment. No fresh Roboflow export, license, or independent train/test split was verified. The bundled reference data supports demonstration; its lineage is not represented as a newly downloaded Roboflow release.</p><a href="https://universe.roboflow.com/defect-rexb3/rail-defect" target="_blank" rel="noreferrer" class="data-link">Reference-linked Roboflow project ↗</a> &nbsp; <a href="https://universe.roboflow.com/search?q=railway%20track" target="_blank" rel="noreferrer" class="data-link">Explore railway datasets ↗</a></article>
 <article class="data-panel"><div class="dataset-title"><span data-icon="scan"></span><h2>Pravaas obstacle annotations</h2><span class="tag">ORIGINAL PROJECT</span></div><p>The original project contains image annotations for railway obstacles. The deployed checkpoint recognizes Animal, Debris, Human, Object, Stone, and Tree. A later annotation archive adds vehicle, but that does not add the class to the existing model.</p><p>Checkpoint metrics and training membership could not be independently reconstructed. Included samples are demonstrations, not held-out performance evidence.</p><a href="https://github.com/KirtirajChaudhari/Pravaas" target="_blank" rel="noreferrer" class="data-link">View original project ↗</a></article>
 <div class="data-panel"><h2>A reproducible path to a stronger model</h2><p>Pin a licensed Roboflow project and version, audit the class definitions, and group near-duplicate frames or capture sessions before splitting. Use the included YOLO training command with that export, then report held-out per-class precision, recall, mAP50, and mAP50–95. Keep image-level classification and bounding-box detection evaluations separate.</p><pre class="code-block">python ml/train_detector.py --data path/to/data.yaml \\\n  --model yolo11n.pt --epochs 50 --device 0</pre></div>`;
  $("download-manifest").onclick = async () => {
    try {
      download(
        await fetchJSON("data/manifest.json"),
        "pravaas-dataset-manifest.json",
      );
    } catch {
      toast("Dataset manifest could not be downloaded.");
    }
  };
  icons($("page-datasets"));
}
function renderHistory() {
  const rows = state.history;
  $("page-history").innerHTML =
    heading(
      "YOUR RESEARCH TRAIL.",
      "Every inspection, in context.",
      "Stored in this browser. Reports contain results and filenames; images are not saved.",
      `<div class="history-actions"><button id="history-csv" class="button secondary" ${rows.length ? "" : "disabled"}><span data-icon="download"></span> Export CSV</button><button id="history-clear" class="button secondary" ${rows.length ? "" : "disabled"}>Clear history</button></div>`,
    ) +
    `<div class="data-panel">${rows.length ? `<div class="table-scroll"><table><thead><tr><th>Image</th><th>Model</th><th>Result</th><th>Time</th><th>Report</th></tr></thead><tbody>${rows.map((r) => `<tr><td><strong>${esc(r.image)}</strong><small>${esc(new Date(r.created_at).toLocaleString())}</small></td><td>${esc(r.model_name)}</td><td>${r.task === "detect" ? `${r.boxes.length} detections` : esc(r.label) + " · " + pct(r.score)}</td><td>${(r.ms / 1000).toFixed(2)} s</td><td><button class="button secondary" data-report="${esc(r.id)}">JSON ↓</button></td></tr>`).join("")}</tbody></table></div>` : '<div class="empty-history"><span data-icon="history"></span><h2>Your first inspection starts here.</h2><p>Run a sample or upload an image. Your results will appear here.</p><a href="#inspect" class="button primary">Open inspection studio →</a></div>'}</div>`;
  $("history-clear").onclick = () => {
    state.history = [];
    try {
      localStorage.removeItem("pravaas-history");
    } catch {}
    renderHistory();
    toast("Inspection history cleared from this browser.");
  };
  $("history-csv").onclick = () => {
    const cell = (v) =>
      '"' +
      String(v ?? "")
        .replace(/^[=+@-]/, "'" + "$&")
        .replaceAll('"', '""') +
      '"';
    const cols = [
      "created_at",
      "image",
      "model_name",
      "task",
      "result",
      "inference_ms",
    ];
    const lines = rows.map((r) =>
      [
        r.created_at,
        r.image,
        r.model_name,
        r.task,
        r.task === "detect"
          ? r.boxes.map((b) => `${b.label}:${pct(b.score)}`).join("; ")
          : `${r.label}:${pct(r.score)}`,
        r.ms.toFixed(2),
      ]
        .map(cell)
        .join(","),
    );
    download(
      [cols.join(","), ...lines].join("\r\n"),
      "pravaas-inspections.csv",
      "text/csv;charset=utf-8",
    );
  };
  $("page-history")
    .querySelectorAll("[data-report]")
    .forEach(
      (b) =>
        (b.onclick = () =>
          download(
            rows.find((r) => r.id === b.dataset.report),
            "pravaas-inspection.json",
          )),
    );
  icons($("page-history"));
}
function route() {
  const name = location.hash.slice(1) || "inspect";
  const page = ["inspect", "lab", "datasets", "history"].includes(name)
    ? name
    : "inspect";
  document
    .querySelectorAll(".page")
    .forEach((p) => (p.hidden = p.id !== `page-${page}`));
  document.querySelectorAll(".nav-link").forEach((a) => {
    const active = a.dataset.page === page;
    a.classList.toggle("active", active);
    if (active) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  $("breadcrumb").textContent = {
    inspect: "Inspection",
    lab: "Model lab",
    datasets: "Datasets",
    history: "Inspection history",
  }[page];
  if (state.registry) {
    if (page === "lab") renderLab();
    if (page === "datasets") renderDatasets();
    if (page === "history") renderHistory();
  }
  window.scrollTo(0, 0);
}
async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Could not load ${url}`);
  return r.json();
}
$("upload-top").onclick = () => {
  if (!state.busy) $("file-input").click();
};
$("file-input").onchange = (e) => {
  upload(e.target.files[0]);
  e.target.value = "";
};
$("task-select").onchange = setModels;
$("model-select").onchange = updateModel;
$("confidence").oninput = () => {
  $("confidence-value").textContent = $("confidence").value + "%";
  clearResult();
};
$("run-button").onclick = run;
$("export-button").onclick = () =>
  state.result?.report &&
  download(state.result.report, "pravaas-inspection.json");
$("overlay-button").onclick = () => {
  state.overlay = !state.overlay;
  $("overlay-button").setAttribute("aria-pressed", String(state.overlay));
  draw();
};
$("expand-button").onclick = () => {
  if (!state.imageReady) return;
  const c = $("expanded-canvas"),
    im = $("inspection-image");
  c.width = im.naturalWidth;
  c.height = im.naturalHeight;
  const ctx = c.getContext("2d");
  ctx.drawImage(im, 0, 0);
  if (state.overlay) ctx.drawImage($("overlay"), 0, 0);
  $("image-dialog").showModal();
};
$("help-button").onclick = () => $("help-dialog").showModal();
document
  .querySelectorAll(".dialog-close")
  .forEach((b) => (b.onclick = () => b.closest("dialog").close()));
$("help-lab").onclick = () => $("help-dialog").close();
const zone = $("drop-zone");
["dragenter", "dragover"].forEach((name) =>
  zone.addEventListener(name, (e) => {
    e.preventDefault();
    if (!state.busy) zone.classList.add("dragover");
  }),
);
["dragleave", "drop"].forEach((name) =>
  zone.addEventListener(name, (e) => {
    e.preventDefault();
    zone.classList.remove("dragover");
  }),
);
zone.addEventListener("drop", (e) => upload(e.dataTransfer.files[0]));
window.addEventListener("hashchange", route);
document.addEventListener("keydown", (e) => {
  if (
    e.key === "Enter" &&
    e.target === document.body &&
    !document.querySelector("dialog[open]") &&
    !$("page-inspect").hidden
  )
    run();
});
route();
try {
  [state.registry, state.experiments] = await Promise.all([
    fetchJSON("data/registry.json"),
    fetchJSON("data/experiments.json").catch(() => null),
  ]);
  $("model-count").innerHTML =
    `${String(state.registry.models.length).padStart(2, "0")} <small>Real ONNX inference</small>`;
  setModels();
  await selectSample(state.registry.samples[0].id);
  route();
} catch (e) {
  console.error(e);
  $("inference-status").textContent =
    "Workspace assets could not load. Reload to retry.";
  $("inference-status").classList.add("error-text");
  toast("Workspace assets could not load. Check your connection.");
}
