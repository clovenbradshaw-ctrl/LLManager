import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Served from https://<user>.github.io/llmanager/
export default defineConfig({
  base: "/llmanager/",
  plugins: [react()],
});
