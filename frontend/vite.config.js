import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// base "./" lets the built site run from any folder or static host (GitHub Pages, Netlify, ...).
export default defineConfig({
  base: "./",
  plugins: [react()],
  test: {
    environment: "node", // component tests switch to jsdom with a comment at the top of the file
    setupFiles: ["./src/test-setup.js"],
  },
});
