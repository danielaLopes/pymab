/// <reference lib="webworker" />

interface RuntimeManifest {
  pymabFilename: string;
  pymabVersion: string;
  bridgeFilename: string;
  numpyFilename: string;
  assets: Record<string, string>;
}

interface PyodideRuntime {
  FS: { writeFile(path: string, data: Uint8Array): void; unlink(path: string): void };
  loadPackage(name: string): Promise<void>;
  runPython(code: string): unknown;
  runPythonAsync(code: string): Promise<unknown>;
  setStdout(options: { batched(value: string): void }): void;
  setStderr(options: { batched(value: string): void }): void;
}

interface PyodideModule {
  loadPyodide(options: { indexURL: string }): Promise<PyodideRuntime>;
}

const scope = self as DedicatedWorkerGlobalScope;
const baseUrl = new URL(import.meta.env.BASE_URL, scope.location.origin);
const outputLimit = 64 * 1024;
let runtime: PyodideRuntime;
let manifest: RuntimeManifest;

async function fetchVerified(relative: string): Promise<Uint8Array> {
  const expected = manifest.assets[relative];
  if (!expected) throw new Error(`No manifest hash for ${relative}`);
  const response = await fetch(new URL(relative, baseUrl));
  if (!response.ok) throw new Error(`Could not load ${relative}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  const digest = [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
  if (digest !== expected) throw new Error(`Integrity check failed for ${relative}`);
  return bytes;
}

async function boot() {
  scope.postMessage({ type: "progress", message: "Opening a clean Python runtime…" });
  const response = await fetch(new URL("runtime-manifest.json", baseUrl));
  if (!response.ok) throw new Error("Could not load the runtime manifest");
  manifest = (await response.json()) as RuntimeManifest;
  const moduleUrl = new URL("pyodide/pyodide.mjs", baseUrl).href;
  const module = (await import(/* @vite-ignore */ moduleUrl)) as PyodideModule;
  runtime = await module.loadPyodide({ indexURL: new URL("pyodide/", baseUrl).href });
  scope.postMessage({ type: "progress", message: "Loading NumPy and PyMAB…" });
  await fetchVerified(manifest.numpyFilename);
  await runtime.loadPackage("numpy");
  const wheel = manifest.pymabFilename;
  const bytes = await fetchVerified(wheel);
  const temporary = `/tmp/${wheel.split("/").at(-1)}`;
  runtime.FS.writeFile(temporary, bytes);
  runtime.runPython(`
import site, zipfile
with zipfile.ZipFile(${JSON.stringify(temporary)}) as archive:
    archive.extractall(site.getsitepackages()[0])
`);
  runtime.FS.unlink(temporary);
  const version = String(runtime.runPython("import pymab; pymab.__version__"));
  if (version !== manifest.pymabVersion) throw new Error(`Unexpected PyMAB version ${version}`);
  scope.postMessage({ type: "ready" });
}

scope.addEventListener("message", (event: MessageEvent<{ type: "run"; code: string }>) => {
  if (event.data.type !== "run") return;
  void (async () => {
    let stdout = "";
    let stderr = "";
    let truncated = false;
    const append = (target: "stdout" | "stderr", chunk: string) => {
      const used = stdout.length + stderr.length;
      const room = Math.max(0, outputLimit - used);
      const completeChunk = `${chunk}\n`;
      if (completeChunk.length > room) truncated = true;
      const bounded = completeChunk.slice(0, room);
      if (target === "stdout") stdout += bounded;
      else stderr += bounded;
    };
    runtime.setStdout({ batched: (value) => append("stdout", value) });
    runtime.setStderr({ batched: (value) => append("stderr", value) });
    try {
      await runtime.runPythonAsync(event.data.code);
      scope.postMessage({
        type: "result",
        result: { status: "success", stdout, stderr, truncated },
      });
    } catch (error) {
      const raw = error instanceof Error ? error.message : String(error);
      const sanitized = raw.replaceAll(/\/tmp\/[\w./-]+/g, "<python>");
      scope.postMessage({
        type: "result",
        result: {
          status: sanitized.includes("SyntaxError") ? "syntax" : "runtime",
          stdout,
          stderr,
          truncated,
          error: sanitized,
        },
      });
    }
  })();
});

void boot().catch((error: unknown) => {
  scope.postMessage({
    type: "error",
    message: error instanceof Error ? error.message : String(error),
  });
});

export {};
