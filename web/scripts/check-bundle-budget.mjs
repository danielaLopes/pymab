import { gzip } from "node:zlib";
import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";

const gzipAsync = promisify(gzip);
const webRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const assetsRoot = path.join(webRoot, "dist", "assets");
const entries = (await readdir(assetsRoot)).filter(
  (file) => file.endsWith(".js") && !file.includes("worker"),
);
const sizes = await Promise.all(
  entries.map(async (file) => ({
    file,
    bytes: (await gzipAsync(await readFile(path.join(assetsRoot, file)))).byteLength,
  })),
);
const total = sizes.reduce((sum, item) => sum + item.bytes, 0);
const budget = 350 * 1024;
console.log(`Main application JavaScript: ${(total / 1024).toFixed(1)} KiB gzip`);
if (total > budget) {
  throw new Error(
    `Main JavaScript exceeds the 350 KiB gzip budget by ${((total - budget) / 1024).toFixed(1)} KiB`,
  );
}
