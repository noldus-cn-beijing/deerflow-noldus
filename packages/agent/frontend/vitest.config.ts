import path from "node:path";

import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  test: {
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
