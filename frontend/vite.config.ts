import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The UI always calls relative /api/... URLs; Vite proxies them to FastAPI on :8000
// in both `vite dev` and `vite preview` (used by start.ps1).
const proxy = {
  "/api": { target: "http://127.0.0.1:8000", changeOrigin: false },
};

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173, strictPort: true, proxy },
  preview: { port: 5173, strictPort: true, proxy },
  build: { chunkSizeWarningLimit: 6000, sourcemap: false },
});
