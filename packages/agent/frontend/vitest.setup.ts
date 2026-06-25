import "@testing-library/jest-dom/vitest";

// jsdom does not implement ResizeObserver, but several rendered components
// (ai-elements Conversation → use-stick-to-bottom, @tanstack/react-virtual)
// require it at mount. Provide a no-op stub so component tests can render.
/* eslint-disable @typescript-eslint/no-empty-function */
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
  takeRecords() {
    return [];
  }
}
/* eslint-enable @typescript-eslint/no-empty-function */

if (typeof globalThis.ResizeObserver === "undefined") {
  globalThis.ResizeObserver = ResizeObserverStub as unknown as typeof ResizeObserver;
}

// jsdom lacks layout; react-virtual / stick-to-bottom read scroll/element
// sizes. ensure offsetHeight etc. default to 0 (already the jsdom default) —
// no further polyfill needed for the perf/memo tests which don't assert layout.

