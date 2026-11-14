import { defineConfig } from "@playwright/test";
import path from "node:path";
import os from "node:os";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));

// End-to-end tests run against an isolated stack: a backend on :8001 with a
// throwaway database (real dataset, jobs enabled) and the built UI on :5174.
const tmp = path.join(os.tmpdir(), `heliograph-e2e-${process.pid}`);
const isWin = process.platform === "win32";
const py = path.resolve(here, "..", "backend", ".venv", isWin ? "Scripts/python.exe" : "bin/python");

export default defineConfig({
  testDir: "./e2e",
  timeout: 300_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: { baseURL: "http://127.0.0.1:5174", viewport: { width: 1440, height: 1000 }, trace: "off" },
  webServer: [
    {
      command: `"${py}" -m uvicorn app.main:app --host 127.0.0.1 --port 8001`,
      cwd: path.resolve(here, "..", "backend"),
      url: "http://127.0.0.1:8001/api/health",
      timeout: 120_000,
      reuseExistingServer: false,
      env: {
        HELIOGRAPH_DB_PATH: path.join(tmp, "e2e.db"),
        HELIOGRAPH_ARTIFACTS_DIR: path.join(tmp, "artifacts"),
        HELIOGRAPH_REFERENCE_DIR: path.join(tmp, "no-reference"),
      },
    },
    {
      command: "npx vite preview --port 5174 --strictPort --host 127.0.0.1",
      url: "http://127.0.0.1:5174",
      timeout: 120_000,
      reuseExistingServer: false,
      env: { HELIOGRAPH_API_PORT: "8001" },
    },
  ],
});
