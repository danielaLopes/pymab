export interface BrowserSupport {
  supported: boolean;
  reason?: string;
}

export function detectBrowserSupport(): BrowserSupport {
  if (typeof WebAssembly !== "object") {
    return { supported: false, reason: "This browser does not provide WebAssembly." };
  }
  if (typeof Worker !== "function") {
    return { supported: false, reason: "This browser does not provide Web Workers." };
  }
  return { supported: true };
}
