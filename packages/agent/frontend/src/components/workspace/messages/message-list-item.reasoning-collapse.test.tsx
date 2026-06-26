// @vitest-environment jsdom
import type { Message } from "@langchain/langgraph-sdk";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";

import { I18nProvider } from "@/core/i18n/context";

/**
 * chat-render-jank-on-open fix (Fix 3, 2026-06-26): historical reasoning
 * blocks (a thread that is NOT streaming) must mount COLLAPSED so opening a
 * heavy thread doesn't trigger N synchronous Radix Collapsible
 * `useLayoutEffect` height measurements (2.5% of the open-jank CPU profile).
 * The in-flight streaming message keeps its live-open behavior.
 *
 * Deterministic signal: we stub `MarkdownContent` so the reasoning body becomes
 * a stable DOM node carrying a data marker. ReasoningPanel renders that body
 * only inside `{showThinking && <ChainOfThoughtContent>…}` — so the marker is
 * present exactly when the panel is expanded at mount.
 */

vi.mock("@/components/workspace/messages/markdown-content", () => ({
  MarkdownContent: ({ content }: { content: string }) => (
    <div data-testid="md">{content}</div>
  ),
}));
// Stub the registry Collapsible wrappers so we don't depend on Radix layout
// effects in jsdom. We only need them to surface the content they receive —
// note ChainOfThoughtStep takes its payload via the `label` prop (not children),
// so the passthrough must render both to be a faithful stub.
vi.mock("@/components/ai-elements/chain-of-thought", async () => {
  const passthrough = ({ children }: { children?: ReactNode }) => (
    <>{children}</>
  );
  const stepPassthrough = ({
    label,
    children,
  }: {
    label?: ReactNode;
    children?: ReactNode;
  }) => (
    <>
      {label}
      {children}
    </>
  );
  return {
    ChainOfThought: passthrough,
    ChainOfThoughtContent: passthrough,
    ChainOfThoughtStep: stepPassthrough,
    ChainOfThoughtSearchResult: passthrough,
    ChainOfThoughtSearchResults: passthrough,
  };
});
vi.mock("@/components/ai-elements/message", () => ({
  Message: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  MessageContent: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  MessageResponse: ({ children }: { children: ReactNode }) => <>{children}</>,
  MessageToolbar: () => null,
}));
vi.mock("next/navigation", () => ({
  useParams: () => ({ thread_id: "t1" }),
  // eslint-disable-next-line @typescript-eslint/no-empty-function
  useRouter: () => ({ push: () => {}, replace: () => {} }),
}));
vi.mock("@/components/feedback/feedback-buttons", () => ({
  FeedbackButtons: () => null,
}));
vi.mock("@/components/workspace/copy-button", () => ({
  CopyButton: () => null,
}));

const { MessageListItem } = await import(
  "@/components/workspace/messages/message-list-item"
);

const REASONING_MARKER = "REASONING-BODY-MARKER";

function makeAssistantMessage(): Message {
  return {
    id: "ai-1",
    type: "ai",
    content: "final answer",
    // ReasoningPanel renders when reasoning content is present. We pack the
    // marker into reasoning_content; extractReasoningContentFromMessage reads
    // it off additional_kwargs.reasoning_content (see core/messages/utils).
    additional_kwargs: { reasoning_content: REASONING_MARKER },
  } as unknown as Message;
}

function renderWith(message: Message, isLoading: boolean) {
  // Dynamic import keeps the mock registry above intact per test.
  return import("@testing-library/react").then(({ render }) =>
    render(
      <I18nProvider initialLocale="en-US">
        <MessageListItem
          message={message}
          threadId="t1"
          isLoading={isLoading}
          messageRunIds={new Map()}
        />
      </I18nProvider>,
    ),
  );
}

describe("ReasoningPanel — historical collapse (Fix 3, 2026-06-26)", () => {
  it("mounts the reasoning body COLLAPSED for a historical (non-streaming) message", async () => {
    const { queryByText } = await renderWith(makeAssistantMessage(), false);
    // Body not mounted at open → no layout-effect measurement cost.
    expect(queryByText(REASONING_MARKER)).toBeNull();
  });

  it("mounts the reasoning body OPEN for the in-flight (streaming) message", async () => {
    const { getByText } = await renderWith(makeAssistantMessage(), true);
    // Live thinking trace stays visible while streaming (纪律: don't touch
    // the in-flight display).
    expect(getByText(REASONING_MARKER)).toBeDefined();
  });
});
