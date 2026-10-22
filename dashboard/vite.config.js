import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteSingleFile } from "vite-plugin-singlefile";
import path from "node:path";

// The exported system state is inlined at build time, producing one self-contained HTML file.
const data = process.env.DASHBOARD_DATA || path.resolve(__dirname, "../state/dashboard/data.json");

export default defineConfig({
  plugins: [react(), viteSingleFile()],
  resolve: { alias: { "@data": data } },
  build: { outDir: "dist", assetsInlineLimit: 100000000, chunkSizeWarningLimit: 5000 },
});
