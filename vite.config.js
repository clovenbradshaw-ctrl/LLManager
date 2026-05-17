import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Relative base + docs/ output so GitHub Pages can serve it
// directly via "Deploy from a branch" (main / docs).
export default defineConfig({
  base: "./",
  plugins: [react()],
  build: { outDir: "docs" },
  // sqlite-wasm and transformers.js ship their own WASM/worker assets and
  // do not pre-bundle cleanly — let Vite serve them as-is.
  optimizeDeps: { exclude: ["@sqlite.org/sqlite-wasm", "@xenova/transformers"] },
  worker: { format: "es" },
});
