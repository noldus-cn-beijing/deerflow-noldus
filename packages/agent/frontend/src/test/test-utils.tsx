// @vitest-environment jsdom
import { render } from "@testing-library/react";
import type { ReactNode } from "react";

import { I18nProvider } from "@/core/i18n/context";
import { SubtasksProvider } from "@/core/tasks/context";

/**
 * Minimal provider wrapper for component tests.
 *
 * Many workspace components read from the i18n + subtask contexts at render
 * time (MessageList calls useI18n + useUpdateSubtask). This helper mounts the
 * real providers with stub initial values so tests can render in isolation
 * without standing up the full Next.js app/router.
 */
export function renderWithProviders(node: ReactNode) {
  return render(node, {
    wrapper: ({ children }) => (
      <I18nProvider initialLocale="en-US">
        <SubtasksProvider>{children}</SubtasksProvider>
      </I18nProvider>
    ),
  });
}

// Re-export so test files need only one import.
export { cleanup, render, screen } from "@testing-library/react";
export { default as userEvent } from "@testing-library/user-event";
