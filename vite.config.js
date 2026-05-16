import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Relative base + docs/ output so GitHub Pages can serve it
// directly via "Deploy from a branch" (main / docs).
export default defineConfig({
  base: "./",
  plugins: [react()],
  build: { outDir: "docs" },
});
