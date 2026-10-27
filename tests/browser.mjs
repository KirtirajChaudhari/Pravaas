// Run against npm run dev: node tests/browser.mjs
// Uses system Chromium if CHROMIUM_PATH is set; otherwise Playwright's browser.
import { chromium } from "playwright";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
const browser = await chromium.launch({
  headless: true,
  ...(process.env.CHROMIUM_PATH
    ? { executablePath: process.env.CHROMIUM_PATH }
    : {}),
  args: ["--no-sandbox"],
});
const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } }),
  errors = [];
page.on("pageerror", (e) => errors.push(e.message));
const base = process.env.BASE_URL || "http://localhost:4173";
try {
  await page.goto(base, { waitUntil: "networkidle" });
  await page.waitForFunction(
    () => !document.querySelector("#run-button").disabled,
  );
  async function run() {
    await page.click("#run-button");
    await page.waitForFunction(
      () => !document.querySelector("#run-button").disabled,
      {},
      { timeout: 180000 },
    );
    assert.match(
      await page.locator("#inference-status").innerText(),
      /Complete/,
    );
  }
  await run();
  assert.match(await page.locator("#results").innerText(), /Stone/);
  let [download] = await Promise.all([
    page.waitForEvent("download"),
    page.click("#export-button"),
  ]);
  let report = JSON.parse(await fs.readFile(await download.path(), "utf8"));
  assert.equal(report.task, "detect");
  assert.ok(report.boxes.length > 0);
  assert.equal(report.model, "obstacle-yolo");
  await page.click("#overlay-button");
  assert.equal(
    await page.locator("#overlay-button").getAttribute("aria-pressed"),
    "false",
  );
  await page.click("#overlay-button");
  await page.click("#expand-button");
  assert.equal(
    await page.locator("#image-dialog").evaluate((e) => e.open),
    true,
  );
  await page.locator("#image-dialog .dialog-close").click();
  await page.click('[data-sample="rail-joint"]');
  await run();
  assert.match(await page.locator("#results").innerText(), /railway/i);
  await page.click('[data-sample="fault-sample"]');
  for (const id of ["cnn", "mobilenet", "finetuned"]) {
    await page.selectOption("#model-select", id);
    await run();
    assert.match(await page.locator("#results").innerText(), /Defective/);
    assert.match(await page.locator("#results").innerText(), /Non-defective/i);
  }
  await page.setInputFiles("#file-input", "dist/samples/rail-joint.jpg");
  await page.waitForFunction(
    () =>
      document.querySelector("#image-name").textContent === "rail-joint.jpg",
  );
  assert.match(await page.locator("#sample-source").innerText(), /local only/);
  await page.setInputFiles("#file-input", {
    name: "bad.txt",
    mimeType: "text/plain",
    buffer: Buffer.from("not an image"),
  });
  assert.match(await page.locator("#toast").innerText(), /JPG/);
  await page.click('[data-page="history"]');
  assert.equal(await page.locator("#page-history tbody tr").count(), 5);
  [download] = await Promise.all([
    page.waitForEvent("download"),
    page.click("#history-csv"),
  ]);
  assert.match(
    await fs.readFile(await download.path(), "utf8"),
    /inference_ms/,
  );
  await page.reload();
  await page.waitForSelector("#page-history tbody tr");
  assert.equal(await page.locator("#page-history tbody tr").count(), 5);
  await page.click('[data-page="lab"]');
  assert.equal(await page.locator(".confusion .cell").count(), 12);
  assert.match(await page.locator("#page-lab").innerText(), /22/);
  await page.screenshot({ path: "/tmp/pravaas-lab.png", fullPage: true });
  await page.click('[data-page="datasets"]');
  assert.match(await page.locator("#page-datasets").innerText(), /403/);
  await page.click('[data-page="inspect"]');
  await page.screenshot({ path: "/tmp/pravaas-desktop.png", fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
    false,
  );
  assert.equal(await page.locator(".sidebar-bottom").isVisible(), false);
  await page.screenshot({ path: "/tmp/pravaas-mobile.png", fullPage: true });
  await page.click('[data-page="history"]');
  await page.click("#history-clear");
  assert.match(
    await page.locator("#page-history").innerText(),
    /first inspection/,
  );
  assert.deepEqual(errors, []);
  console.log(
    "PASS: all five models, reports, uploads, overlays, navigation, history persistence, and mobile layout.",
  );
} finally {
  await browser.close();
}
