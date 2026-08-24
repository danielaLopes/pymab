import { createHash } from "node:crypto";
import { readFile, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";

export function sha256(buffer) {
  return createHash("sha256").update(buffer).digest("hex");
}

export async function fileSha256(file) {
  return sha256(await readFile(file));
}

export async function ensureVerifiedDownload({ url, destination, expectedHash, fetcher = fetch }) {
  try {
    if ((await fileSha256(destination)) === expectedHash) return "cache";
    await rm(destination, { force: true });
  } catch (error) {
    if (error?.code !== "ENOENT") throw error;
  }
  const partial = `${destination}.part`;
  await rm(partial, { force: true });
  const response = await fetcher(url);
  if (!response.ok) throw new Error(`Download failed (${response.status}): ${url}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  const actualHash = sha256(bytes);
  if (actualHash !== expectedHash) {
    throw new Error(`SHA-256 mismatch for ${path.basename(destination)}`);
  }
  await writeFile(partial, bytes, { flag: "wx" });
  await rename(partial, destination);
  return "download";
}

export function resolveDependencies(lock, roots) {
  const packages = lock.packages ?? {};
  const resolved = new Map();
  const pending = [...roots];
  while (pending.length > 0) {
    const name = pending.shift();
    if (resolved.has(name)) continue;
    const metadata = packages[name];
    if (!metadata) throw new Error(`Package ${name} is missing from pyodide-lock.json`);
    if (!metadata.file_name || !metadata.sha256) {
      throw new Error(`Package ${name} is missing a filename or SHA-256`);
    }
    resolved.set(name, metadata);
    pending.push(...(metadata.depends ?? []));
  }
  return [...resolved.entries()]
    .map(([name, metadata]) => ({ name, ...metadata }))
    .sort((left, right) => left.name.localeCompare(right.name));
}

export function safeJoin(root, ...parts) {
  const absoluteRoot = path.resolve(root);
  const candidate = path.resolve(absoluteRoot, ...parts);
  if (candidate !== absoluteRoot && !candidate.startsWith(`${absoluteRoot}${path.sep}`)) {
    throw new Error(`Refusing path outside generated directory: ${candidate}`);
  }
  return candidate;
}

export function makeManifest({ pyodideVersion, lock, pymabWheel, bridge, commit, assets }) {
  return {
    schemaVersion: 1,
    pyodideVersion,
    pythonVersion: lock.info?.python ?? "unknown",
    numpyFilename: assets.find((asset) => asset.name === "numpy")?.file ?? "",
    pymabFilename: pymabWheel.file,
    pymabVersion: pymabWheel.version,
    bridgeFilename: bridge.file,
    sourceCommit: commit,
    assets: Object.fromEntries(
      [...assets, pymabWheel, bridge]
        .map((asset) => [asset.file, asset.sha256])
        .sort(([left], [right]) => left.localeCompare(right)),
    ),
  };
}
