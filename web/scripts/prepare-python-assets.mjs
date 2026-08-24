import { execFile } from "node:child_process";
import { cp, mkdir, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";

import {
  ensureVerifiedDownload,
  fileSha256,
  makeManifest,
  resolveDependencies,
  safeJoin,
} from "./asset-utils.mjs";

const execute = promisify(execFile);
const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const webRoot = path.resolve(scriptDirectory, "..");
const repositoryRoot = path.resolve(webRoot, "..");
const generatedRoot = path.join(webRoot, ".generated");
const publicRoot = safeJoin(generatedRoot, "public");
const cacheRoot = safeJoin(generatedRoot, "cache");
const wheelBuildRoot = safeJoin(generatedRoot, "wheels");
const pyodideSource = path.join(webRoot, "node_modules", "pyodide");
const pyodidePublic = safeJoin(publicRoot, "pyodide");
const clean = process.argv.includes("--clean");

if (clean) await rm(generatedRoot, { recursive: true, force: true });
await mkdir(cacheRoot, { recursive: true });
await mkdir(publicRoot, { recursive: true });
await rm(wheelBuildRoot, { recursive: true, force: true });
await mkdir(wheelBuildRoot, { recursive: true });

await execute("uv", ["build", "--wheel", "--out-dir", wheelBuildRoot], { cwd: repositoryRoot });
const wheels = (await readdir(wheelBuildRoot)).filter((file) =>
  /^pymab-.*-py3-none-any\.whl$/.test(file),
);
if (wheels.length !== 1)
  throw new Error(`Expected exactly one PyMAB wheel, found ${wheels.length}`);
const pymabFilename = wheels[0];
const versionMatch = /^pymab-([^-]+)-/.exec(pymabFilename);
if (!versionMatch) throw new Error(`Cannot parse PyMAB version from ${pymabFilename}`);

await rm(pyodidePublic, { recursive: true, force: true });
await mkdir(pyodidePublic, { recursive: true });
const runtimeFiles = [
  "pyodide.mjs",
  "pyodide.asm.mjs",
  "pyodide.asm.wasm",
  "python_stdlib.zip",
  "pyodide-lock.json",
];
for (const file of runtimeFiles) {
  const source = path.join(pyodideSource, file);
  if (!(await stat(source)).isFile()) throw new Error(`Missing Pyodide runtime file: ${file}`);
  await cp(source, safeJoin(pyodidePublic, file));
}

const packageJson = JSON.parse(await readFile(path.join(pyodideSource, "package.json"), "utf8"));
const lock = JSON.parse(await readFile(path.join(pyodideSource, "pyodide-lock.json"), "utf8"));
if (packageJson.version !== "314.0.5") throw new Error(`Unexpected Pyodide ${packageJson.version}`);
const dependencyAssets = [];
for (const dependency of resolveDependencies(lock, ["numpy"])) {
  const cacheFile = safeJoin(cacheRoot, dependency.file_name);
  const url = `https://cdn.jsdelivr.net/pyodide/v${packageJson.version}/full/${dependency.file_name}`;
  await ensureVerifiedDownload({
    url,
    destination: cacheFile,
    expectedHash: dependency.sha256,
  });
  await cp(cacheFile, safeJoin(pyodidePublic, dependency.file_name));
  dependencyAssets.push({
    name: dependency.name,
    file: `pyodide/${dependency.file_name}`,
    sha256: dependency.sha256,
  });
}

const wheelsPublic = safeJoin(publicRoot, "wheels");
await mkdir(wheelsPublic, { recursive: true });
await cp(path.join(wheelBuildRoot, pymabFilename), safeJoin(wheelsPublic, pymabFilename));
const bridgeRelative = "python/pymab-demo-bridge.zip";
const bridgeDestination = safeJoin(publicRoot, bridgeRelative);
await execute("python3", [
  path.join(scriptDirectory, "package_bridge.py"),
  path.join(webRoot, "python", "pymab_demo"),
  bridgeDestination,
]);

const runtimeAssets = await Promise.all(
  runtimeFiles.map(async (file) => ({
    name: file,
    file: `pyodide/${file}`,
    sha256: await fileSha256(path.join(pyodidePublic, file)),
  })),
);
const commit = (await execute("git", ["rev-parse", "HEAD"], { cwd: repositoryRoot })).stdout.trim();
const pymabAsset = {
  name: "pymab",
  file: `wheels/${pymabFilename}`,
  version: versionMatch[1],
  sha256: await fileSha256(path.join(wheelsPublic, pymabFilename)),
};
const bridgeAsset = {
  name: "bridge",
  file: bridgeRelative,
  sha256: await fileSha256(bridgeDestination),
};
const manifest = makeManifest({
  pyodideVersion: packageJson.version,
  lock,
  pymabWheel: pymabAsset,
  bridge: bridgeAsset,
  commit,
  assets: [...runtimeAssets, ...dependencyAssets],
});
await writeFile(
  safeJoin(publicRoot, "runtime-manifest.json"),
  `${JSON.stringify(manifest, null, 2)}\n`,
);
await execute("node", [path.join(scriptDirectory, "verify-python-assets.mjs")], { cwd: webRoot });
console.log(`Prepared PyMAB ${pymabAsset.version} for Pyodide ${packageJson.version}`);
