import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig(({ command }) => ({
  base: command === "build" ? "/pymab/" : "/",
  plugins: [react()],
  publicDir: ".generated/public",
  server: { port: 5173 },
  preview: { port: 4173 },
}));
