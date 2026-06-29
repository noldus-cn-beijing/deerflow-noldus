import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// @testing-library/react 在每个测试间自动清理 DOM（vitest 不像 jest 默认带 auto-cleanup）。
// 不加的话同一文件多 test 的 render 会堆叠在 document.body，后一个测试会查到前一个测试的
// 残留组件（gallery 测试因 per_subject 子区 DOM 堆叠被前一个用例的旧渲染污染而误判）。
afterEach(() => {
  cleanup();
});

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

