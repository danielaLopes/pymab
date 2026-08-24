import "@testing-library/jest-dom/vitest";

class TestWorker extends EventTarget {
  postMessage(): void {}
  terminate(): void {}
}

Object.defineProperty(globalThis, "Worker", { value: TestWorker, configurable: true });

afterEach(() => {
  window.localStorage?.clear();
  window.location.hash = "";
});
