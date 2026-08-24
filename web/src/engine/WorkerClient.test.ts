import { WorkerClient } from "./WorkerClient";

class FakeWorker extends EventTarget {
  messages: unknown[] = [];
  terminated = false;
  postMessage(message: unknown) {
    this.messages.push(message);
  }
  terminate() {
    this.terminated = true;
  }
  respond(data: unknown) {
    this.dispatchEvent(new MessageEvent("message", { data }));
  }
  crash(message = "worker exploded") {
    this.dispatchEvent(new ErrorEvent("error", { message }));
  }
}

describe("WorkerClient", () => {
  it("correlates responses and reports progress", async () => {
    const worker = new FakeWorker();
    const client = new WorkerClient(() => worker as unknown as Worker);
    const stages: string[] = [];
    client.onProgress((progress) => stages.push(progress.stage));
    const promise = client.send({ type: "initialize", requestId: "one" });
    worker.respond({ type: "progress", stage: "runtime", message: "boot" });
    worker.respond({
      type: "ready",
      requestId: "one",
      packageVersion: "2.0.0",
      sourceCommit: "abc",
    });
    await expect(promise).resolves.toMatchObject({ type: "ready" });
    expect(stages).toEqual(["runtime"]);
  });

  it("rejects concurrent commands and stale sessions", async () => {
    const worker = new FakeWorker();
    const client = new WorkerClient(() => worker as unknown as Worker);
    const first = client.send({ type: "step", requestId: "one", sessionId: "session-a" });
    await expect(
      client.send({ type: "reset", requestId: "two", sessionId: "session-a" }),
    ).rejects.toThrow(/already running/);
    worker.respond({ type: "disposed", requestId: "one", sessionId: "session-b" });
    await expect(first).rejects.toThrow(/Stale/);
  });

  it("terminates and rejects pending work on disposal", async () => {
    const worker = new FakeWorker();
    const client = new WorkerClient(() => worker as unknown as Worker);
    const pending = client.send({ type: "initialize", requestId: "one" });
    client.dispose();
    await expect(pending).rejects.toThrow(/disposed/);
    expect(worker.terminated).toBe(true);
  });

  it("rejects malformed responses instead of leaking unvalidated state", async () => {
    const worker = new FakeWorker();
    const client = new WorkerClient(() => worker as unknown as Worker);
    const pending = client.send({ type: "initialize", requestId: "one" });

    worker.respond({ type: "ready", requestId: "one", packageVersion: 2 });

    await expect(pending).rejects.toThrow(/Malformed worker response/);
  });

  it("ignores unrelated request IDs and resolves only the matching response", async () => {
    const worker = new FakeWorker();
    const client = new WorkerClient(() => worker as unknown as Worker);
    const pending = client.send({ type: "initialize", requestId: "one" });

    worker.respond({
      type: "ready",
      requestId: "another-request",
      packageVersion: "2.0.0",
      sourceCommit: "abc",
    });
    worker.respond({
      type: "ready",
      requestId: "one",
      packageVersion: "2.0.0",
      sourceCommit: "abc",
    });

    await expect(pending).resolves.toMatchObject({ requestId: "one" });
  });

  it("reports crashes and can recreate a clean worker", async () => {
    const first = new FakeWorker();
    const restarted = new FakeWorker();
    const workers = [first, restarted];
    const client = new WorkerClient(() => workers.shift() as unknown as Worker);
    const pending = client.send({ type: "initialize", requestId: "one" });

    first.crash();
    await expect(pending).rejects.toThrow(/worker exploded/);

    client.restart();
    expect(first.terminated).toBe(true);
    const ready = client.send({ type: "initialize", requestId: "two" });
    restarted.respond({
      type: "ready",
      requestId: "two",
      packageVersion: "2.0.0",
      sourceCommit: "abc",
    });
    await expect(ready).resolves.toMatchObject({ requestId: "two" });
  });
});
