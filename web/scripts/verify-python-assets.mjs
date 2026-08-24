import { execFile } from "node:child_process";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";

import { fileSha256, safeJoin } from "./asset-utils.mjs";

const execute = promisify(execFile);
const webRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const publicRoot = path.join(webRoot, ".generated", "public");
const manifest = JSON.parse(await readFile(path.join(publicRoot, "runtime-manifest.json"), "utf8"));

for (const [relative, expected] of Object.entries(manifest.assets)) {
  const actual = await fileSha256(safeJoin(publicRoot, relative));
  if (actual !== expected) throw new Error(`Generated asset hash mismatch: ${relative}`);
}

const wheel = safeJoin(publicRoot, manifest.pymabFilename);
const check = [
  "import sys",
  `sys.path.insert(0, ${JSON.stringify(wheel)})`,
  "import pymab",
  `assert pymab.__version__ == ${JSON.stringify(manifest.pymabVersion)}`,
].join("; ");
await execute("uv", ["run", "--no-project", "--with", "numpy", "python", "-I", "-c", check]);
