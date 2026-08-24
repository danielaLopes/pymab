/// <reference lib="webworker" />

import { requestSchema } from "./protocol";

interface RuntimeManifest {
  pymabFilename: string;
  pymabVersion: string;
  bridgeFilename: string;
  sourceCommit: string;
  numpyFilename: string;
  assets: Record<string, string>;
}

interface PyodideRuntime {
  FS: { writeFile(path: string, data: Uint8Array): void; unlink(path: string): void };
  loadPackage(name: string): Promise<void>;
  runPython(code: string): unknown;
  runPythonAsync(code: string): Promise<unknown>;
}

interface PyodideModule {
  loadPyodide(options: { indexURL: string }): Promise<PyodideRuntime>;
}

const scope = self as DedicatedWorkerGlobalScope;
const baseUrl = new URL(import.meta.env.BASE_URL, scope.location.origin);
let runtime: PyodideRuntime | null = null;
let manifest: RuntimeManifest | null = null;

function progress(stage: "runtime" | "numpy" | "pymab" | "lesson", message: string) {
  scope.postMessage({ type: "progress", stage, message });
}

async function fetchVerified(relative: string): Promise<Uint8Array> {
  if (!manifest) throw new Error("Runtime manifest is unavailable");
  const expected = manifest.assets[relative];
  if (!expected) throw new Error(`No manifest hash for ${relative}`);
  const response = await fetch(new URL(relative, baseUrl));
  if (!response.ok) throw new Error(`Could not load ${relative}: ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  const digest = [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
  if (digest !== expected) throw new Error(`Integrity check failed for ${relative}`);
  return bytes;
}

async function boot(): Promise<void> {
  if (runtime) return;
  progress("runtime", "Opening the Python runtime");
  const manifestResponse = await fetch(new URL("runtime-manifest.json", baseUrl));
  if (!manifestResponse.ok) throw new Error("Could not load runtime-manifest.json");
  manifest = (await manifestResponse.json()) as RuntimeManifest;
  const moduleUrl = new URL("pyodide/pyodide.mjs", baseUrl).href;
  const pyodideModule = (await import(/* @vite-ignore */ moduleUrl)) as PyodideModule;
  runtime = await pyodideModule.loadPyodide({ indexURL: new URL("pyodide/", baseUrl).href });
  progress("numpy", "Loading NumPy");
  await fetchVerified(manifest.numpyFilename);
  await runtime.loadPackage("numpy");
  progress("pymab", "Installing the checked-out PyMAB wheel");
  const artifacts = [manifest.pymabFilename, manifest.bridgeFilename];
  for (const relative of artifacts) {
    const bytes = await fetchVerified(relative);
    const temporary = `/tmp/${relative.split("/").at(-1) ?? "artifact.zip"}`;
    runtime.FS.writeFile(temporary, bytes);
    runtime.runPython(`
import site, zipfile
with zipfile.ZipFile(${JSON.stringify(temporary)}) as archive:
    archive.extractall(site.getsitepackages()[0])
`);
    runtime.FS.unlink(temporary);
  }
  const importedVersion = String(runtime.runPython("import pymab; pymab.__version__"));
  if (importedVersion !== manifest.pymabVersion) {
    throw new Error(`Expected PyMAB ${manifest.pymabVersion}, imported ${importedVersion}`);
  }
  runtime.runPython("from pymab_demo.entrypoint import dispatch_json");
  progress("lesson", "PyMAB is ready");
}

scope.addEventListener("message", (event: MessageEvent<unknown>) => {
  void (async () => {
    const request = requestSchema.parse(event.data);
    await boot();
    if (!runtime || !manifest) throw new Error("Runtime boot did not complete");
    const withCommit =
      request.type === "initialize" || request.type === "startLesson"
        ? { ...request, sourceCommit: manifest.sourceCommit }
        : request;
    const requestJson = JSON.stringify(withCommit);
    const responseJson = await runtime.runPythonAsync(
      `dispatch_json(${JSON.stringify(requestJson)})`,
    );
    scope.postMessage(JSON.parse(String(responseJson)));
  })().catch((error: unknown) => {
    const requestId =
      typeof event.data === "object" && event.data !== null && "requestId" in event.data
        ? String(event.data.requestId)
        : "";
    scope.postMessage({
      type: "error",
      requestId,
      error: {
        code: "BOOT_FAILED",
        message: error instanceof Error ? error.message : String(error),
        recoverable: true,
      },
    });
  });
});

export {};
