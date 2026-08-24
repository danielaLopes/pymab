import { progressSchema, responseSchema } from "./protocol";
import type { LessonRequest, LessonResponse, RuntimeProgress } from "./protocol";

type WorkerFactory = () => Worker;
type ProgressListener = (progress: RuntimeProgress) => void;

interface PendingRequest {
  requestId: string;
  sessionId?: string;
  resolve: (response: LessonResponse) => void;
  reject: (error: Error) => void;
}

export class WorkerClient {
  private worker: Worker;
  private pending: PendingRequest | null = null;
  private progressListeners = new Set<ProgressListener>();
  private disposed = false;

  constructor(
    private readonly workerFactory: WorkerFactory = () =>
      new Worker(new URL("./lesson.worker.ts", import.meta.url), { type: "module" }),
  ) {
    this.worker = this.workerFactory();
    this.bindWorker();
  }

  onProgress(listener: ProgressListener): () => void {
    this.progressListeners.add(listener);
    return () => this.progressListeners.delete(listener);
  }

  initialize(sourceCommit?: string): Promise<LessonResponse> {
    return this.send({
      type: "initialize",
      requestId: this.id(),
      ...(sourceCommit ? { sourceCommit } : {}),
    });
  }

  send(request: LessonRequest): Promise<LessonResponse> {
    if (this.disposed) return Promise.reject(new Error("Worker client is disposed"));
    if (this.pending) return Promise.reject(new Error("A lesson command is already running"));
    return new Promise((resolve, reject) => {
      this.pending = {
        requestId: request.requestId,
        ...(request.type === "initialize" ? {} : { sessionId: request.sessionId }),
        resolve,
        reject,
      };
      this.worker.postMessage(request);
    });
  }

  restart(): void {
    this.pending?.reject(new Error("Lesson worker restarted"));
    this.pending = null;
    this.worker.terminate();
    this.worker = this.workerFactory();
    this.disposed = false;
    this.bindWorker();
  }

  dispose(): void {
    this.pending?.reject(new Error("Lesson worker disposed"));
    this.pending = null;
    this.worker.terminate();
    this.disposed = true;
  }

  static requestId(): string {
    return typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  private id(): string {
    return WorkerClient.requestId();
  }

  private bindWorker(): void {
    this.worker.addEventListener("message", (event: MessageEvent<unknown>) => {
      const progress = progressSchema.safeParse(event.data);
      if (progress.success) {
        this.progressListeners.forEach((listener) => listener(progress.data));
        return;
      }
      const parsed = responseSchema.safeParse(event.data);
      if (!parsed.success) {
        this.fail(new Error(`Malformed worker response: ${parsed.error.message}`));
        return;
      }
      if (!this.pending || parsed.data.requestId !== this.pending.requestId) return;
      if (
        "sessionId" in parsed.data &&
        this.pending.sessionId &&
        parsed.data.sessionId !== this.pending.sessionId
      ) {
        this.fail(new Error("Stale lesson session response"));
        return;
      }
      const pending = this.pending;
      this.pending = null;
      pending.resolve(parsed.data);
    });
    this.worker.addEventListener("error", (event) => {
      this.fail(new Error(event.message || "Lesson worker crashed"));
    });
  }

  private fail(error: Error): void {
    this.pending?.reject(error);
    this.pending = null;
  }
}
