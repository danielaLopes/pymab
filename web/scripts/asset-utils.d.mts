export interface LockPackage {
  file_name: string;
  sha256: string;
  depends?: string[];
}

export interface PyodideLock {
  info?: { python?: string };
  packages?: Record<string, LockPackage>;
}

export interface Asset {
  name: string;
  file: string;
  sha256: string;
  version?: string;
}

export function sha256(buffer: Uint8Array): string;
export function fileSha256(file: string): Promise<string>;
export function ensureVerifiedDownload(input: {
  url: string;
  destination: string;
  expectedHash: string;
  fetcher?: typeof fetch;
}): Promise<"cache" | "download">;
export function resolveDependencies(
  lock: PyodideLock,
  roots: string[],
): Array<LockPackage & { name: string }>;
export function safeJoin(root: string, ...parts: string[]): string;
export function makeManifest(input: {
  pyodideVersion: string;
  lock: PyodideLock;
  pymabWheel: Asset & { version: string };
  bridge: Asset;
  commit: string;
  assets: Asset[];
}): {
  pythonVersion: string;
  pymabVersion: string;
  numpyFilename: string;
  sourceCommit: string;
  assets: Record<string, string>;
};
