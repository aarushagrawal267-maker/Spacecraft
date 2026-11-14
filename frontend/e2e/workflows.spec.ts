import { expect, test, type Page } from "@playwright/test";

// These workflows run against the real SMAP/MSL benchmark on an isolated database.
// Everything they check is produced by the backend during the test (no fixtures).

async function noPageErrors(page: Page) {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  return errors;
}

test.describe.configure({ mode: "serial" });

test("mission overview shows figures from the installed dataset", async ({ page }) => {
  const errors = await noPageErrors(page);
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Spacecraft telemetry intelligence" })).toBeVisible();
  await expect(page.getByText("Soil Moisture Active Passive")).toBeVisible();
  // 54 SMAP + 27 MSL channels, read from the database
  await expect(page.getByText("54", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("27", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("Benchmark loaded · SMAP + MSL")).toBeVisible();
  // fresh database: no evaluated model yet, and the UI says so instead of inventing numbers
  await expect(page.getByText("No models evaluated yet")).toBeVisible();
  expect(errors).toEqual([]);
});

test("telemetry explorer shows a channel with its reference labels", async ({ page }) => {
  await page.goto("/telemetry?chan=T-9");
  await expect(page.getByRole("heading", { name: "Channel T-9" })).toBeVisible();
  await expect(page.getByText("1,096 values")).toBeVisible();
  await expect(page.getByText("Expert-labelled anomaly sequences")).toBeVisible();
  const zoom = page.getByRole("button", { name: "Zoom" }).first();
  await zoom.click();
  await expect(page.getByRole("button", { name: "Reset zoom" })).toBeVisible();
  await page.getByRole("radio", { name: "Train" }).click();
  await expect(page.getByText(/Training split · used to fit models/)).toBeVisible();
});

test("dataset explorer lists all channels and discloses the P-2 merge", async ({ page }) => {
  await page.goto("/dataset");
  await expect(page.getByText("81 channels")).toBeVisible();
  await expect(page.getByText(/merges them into one stream with 104 sequences/)).toBeVisible();
  await expect(page.getByText(/channel identifiers are anonymised/i)).toBeVisible();
});

test("train a quick LSTM in the laboratory, then inspect, re-run inference and replay it", async ({ page }) => {
  await page.goto("/lab?chan=T-9");
  await expect(page.getByRole("heading", { name: "Train and evaluate on real telemetry" })).toBeVisible();
  await page.getByRole("combobox", { name: "Method" }).selectOption("lstm_forecaster");
  await page.getByRole("radio", { name: "Quick" }).click();
  await page.getByRole("combobox", { name: "Channel" }).selectOption("T-9");
  const epochs = page.locator("label", { hasText: /^epochs/ }).locator("input");
  await epochs.fill("3");
  await page.getByRole("button", { name: "Start experiment" }).click();
  await expect(page.getByText(/Experiment #\d+ queued/)).toBeVisible();

  // Real training: epoch losses stream in, then the experiment completes.
  await expect(page.getByText("Training loss (MSE)")).toBeVisible({ timeout: 120_000 });
  await expect(page.getByText("Evaluation of this experiment")).toBeVisible({ timeout: 240_000 });

  // Detection centre with the stored result
  await page.getByRole("button", { name: /Inspect T-9 in Detection Centre/ }).click();
  await expect(page.getByText("Stored evaluation result")).toBeVisible();
  await expect(page.getByText(/expert label/).first()).toBeVisible();

  // Live inference executes the saved model again
  await page.getByRole("button", { name: "Run inference" }).click();
  await expect(page.getByText("Inference complete")).toBeVisible({ timeout: 60_000 });
  await expect(page.getByText("Live inference result")).toBeVisible();

  // Historical replay of the same run
  await page.goto("/monitor");
  await expect(page.getByText(/not a live connection to an/)).toBeVisible();
  await page.getByRole("radio", { name: "80 pts/s" }).click();
  await page.getByRole("button", { name: "Start monitoring" }).click();
  await expect(page.getByText(/revealed [1-9][\d,]* \/ 1,096/)).toBeVisible();
  await page.getByRole("button", { name: "Pause" }).click();
  await expect(page.getByRole("button", { name: "Resume" })).toBeVisible();
  const pausedAt = await page.getByText(/revealed [\d,]+ \/ 1,096/).textContent();
  await page.waitForTimeout(800);
  expect(await page.getByText(/revealed [\d,]+ \/ 1,096/).textContent()).toBe(pausedAt); // really paused
  await page.getByRole("radio", { name: "1000 pts/s" }).click();
  await page.getByRole("button", { name: "Resume" }).click();
  await expect(page.getByText("End of the historical test stream reached.")).toBeVisible({ timeout: 60_000 });
  await expect(page.getByText(/revealed 1,096 \/ 1,096/)).toBeVisible();
  await page.getByRole("button", { name: "Reset" }).click();
  await expect(page.getByText(/revealed 0 \/ 1,096/)).toBeVisible();
});

test("experiment history records the run and offers exports", async ({ page, request }) => {
  await page.goto("/experiments");
  const row = page.locator("tr", { hasText: "LSTM forecaster · quick" }).first();
  await expect(row).toBeVisible();
  await expect(row.getByText("Completed")).toBeVisible();
  await row.getByRole("link").first().click();
  await expect(page.getByText("Reproducibility")).toBeVisible();
  await expect(page.getByText("Training history")).toBeHidden();
  const id = page.url().split("/").pop();
  const res = await request.get(`/api/export/experiments/${id}.json`);
  expect(res.ok()).toBeTruthy();
  const body = await res.json();
  expect(body.method).toBe("lstm_forecaster");
  expect(body.runs[0].chan_id).toBe("T-9");
});

test("evaluation and investigation pages work on the new result", async ({ page }) => {
  await page.goto("/evaluation");
  await expect(page.getByText("Detection metrics by method")).toBeVisible();
  await expect(page.getByText("Point-wise confusion matrix")).toBeVisible();
  await page.getByRole("radio", { name: "Point-wise" }).click();
  await expect(page.getByText(/No point adjustment is applied/)).toBeVisible();
  await page.goto("/investigate");
  await expect(page.getByRole("heading", { name: "Detected intervals" })).toBeVisible();
  await page.goto("/docs");
  await expect(page.getByRole("heading", { name: "Evaluation methodology" })).toBeVisible();
});
