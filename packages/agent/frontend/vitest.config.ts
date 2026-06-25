import path from "node:path";

import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  test: {
    // jest-dom matchers (toBeInTheDocument, …) for the component tests added
    // by the Phase0#7 runtime-performance work. Component tests opt into the
    // jsdom environment via a per-file `// @vitest-environment jsdom` docblock
    // so the pure-logic tests (utils.test.ts, stream-error.test.ts, …) keep
    // running in the default Node environment unchanged.
    setupFiles: ["./vitest.setup.ts"],
    // These files use node:test (run with `node --test`) instead of vitest,
    // so vitest must not try to discover them as suites.
    exclude: [
      "**/node_modules/**",
      "**/.next/**",
      "src/core/api/stream-mode.test.ts",
      "src/core/uploads/file-validation.test.mjs",
      "src/core/uploads/prompt-input-files.test.mjs",
    ],
  },
});
