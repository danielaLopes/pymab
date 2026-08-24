export interface LabResult {
  status: "success" | "syntax" | "runtime" | "timeout" | "stopped";
  stdout: string;
  stderr: string;
  truncated: boolean;
  error?: string;
}

interface LabWorkerMessage {
  type: "ready" | "progress" | "result" | "error";
  message?: string;
  result?: LabResult;
}

export class LabClient {
  private worker: Worker | null = null;
  private ready: Promise<void> | null = null;
  private pending: ((result: LabResult) => void) | null = null;
  private timer: number | null = null;
  private progressListener: ((message: string) => void) | null = null;
  private readyReject: ((error: Error) => void) | null = null;

  onProgress(listener: (message: string) => void): void {
    this.progressListener = listener;
  }

  async run(code: string): Promise<LabResult> {
    if (this.pending) throw new Error("Python code is already running");
    await this.ensureReady();
    return new Promise((resolve) => {
      this.pending = resolve;
      this.timer = window.setTimeout(() => {
        this.finish({
          status: "timeout",
          stdout: "",
          stderr: "Execution stopped after 5 seconds.",
          truncated: false,
        });
        this.destroy();
      }, 5000);
      this.worker?.postMessage({ type: "run", code });
    });
  }

  stop(): void {
    if (this.pending) {
      this.finish({
        status: "stopped",
        stdout: "",
        stderr: "Execution stopped by you.",
        truncated: false,
      });
    } else {
      this.readyReject?.(new Error("Execution stopped by you."));
    }
    this.destroy();
  }

  dispose(): void {
    this.stop();
  }

  private ensureReady(): Promise<void> {
    if (this.ready) return this.ready;
    this.worker = new Worker(new URL("./lab.worker.ts", import.meta.url), { type: "module" });
    this.ready = new Promise((resolve, reject) => {
      this.readyReject = reject;
      this.worker?.addEventListener("message", (event: MessageEvent<LabWorkerMessage>) => {
        if (event.data.type === "ready") {
          this.readyReject = null;
          resolve();
        }
        if (event.data.type === "progress")
          this.progressListener?.(event.data.message ?? "Loading Python…");
        if (event.data.type === "result" && event.data.result) this.finish(event.data.result);
        if (event.data.type === "error") {
          const error = new Error(event.data.message ?? "Python Lab failed to start");
          if (this.pending)
            this.finish({
              status: "runtime",
              stdout: "",
              stderr: error.message,
              truncated: false,
              error: error.message,
            });
          else {
            this.readyReject = null;
            reject(error);
          }
        }
      });
      this.worker?.addEventListener("error", (event) =>
        reject(new Error(event.message || "Python Lab worker crashed")),
      );
    });
    return this.ready;
  }

  private finish(result: LabResult): void {
    if (this.timer !== null) window.clearTimeout(this.timer);
    this.timer = null;
    const resolve = this.pending;
    this.pending = null;
    resolve?.(result);
  }

  private destroy(): void {
    this.worker?.terminate();
    this.worker = null;
    this.ready = null;
    this.readyReject = null;
  }
}
