import "@testing-library/jest-dom/vitest";

class TestWorker extends EventTarget {
  postMessage(): void {}
  terminate(): void {}
}

Object.defineProperty(globalThis, "Worker", { value: TestWorker, configurable: true });

class TestResizeObserver {
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
}

Object.defineProperty(globalThis, "ResizeObserver", {
  value: TestResizeObserver,
  configurable: true,
});

afterEach(() => {
  window.localStorage?.clear();
  window.location.hash = "";
});
