// @vitest-environment jsdom
import type { BaseStream } from "@langchain/langgraph-sdk/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type * as UtilsModule from "@/core/messages/utils";
import type { AgentThreadState } from "@/core/threads";
import { renderWithProviders } from "@/test/test-utils";

// Spy on the grouping function. The Phase0#7 perf work wraps the
// `groupMessages(messages, mapper)` call in `useMemo([deferredMessages])` so
// that an unrelated re-render (e.g. a parent state change that does NOT alter
// the messages array identity) must NOT re-run the O(n) grouping pass.
//
// We mock the module so we can count calls. The real implementation is still
// exercised by src/core/messages/utils.test.ts (pure-logic, Node env).
vi.mock("@/core/messages/utils", async () => {
  const actual = await vi.importActual<typeof UtilsModule>(
    "@/core/messages/utils",
  );
  return {
    ...actual,
    groupMessages: vi.fn(actual.groupMessages),
  };
});

const { groupMessages } = await import("@/core/messages/utils");

// Stub the heavy child components so MessageList renders in isolation without
// pulling in markdown / artifact / chain-of-thought trees. We only care that
// the grouping pass is memoized, not that children render.
vi.mock("@/components/workspace/messages/message-list-item", () => ({
  MessageListItem: () => <div data-testid="mli" />,
}));
vi.mock("@/components/workspace/messages/message-group", () => ({
  MessageGroup: () => <div data-testid="mg" />,
}));
vi.mock("@/components/workspace/messages/subtask-card", () => ({
  SubtaskCard: () => <div data-testid="sc" />,
}));
vi.mock("@/components/workspace/messages/markdown-content", () => ({
  MarkdownContent: () => <div data-testid="md" />,
}));
vi.mock("@/components/workspace/messages/clarification-options", () => ({
  ClarificationOptions: () => null,
}));
vi.mock("@/components/workspace/messages/quality-warning-banner", () => ({
  QualityWarningBanner: () => null,
}));
vi.mock("@/components/workspace/streaming-indicator", () => ({
  StreamingIndicator: () => null,
}));
vi.mock("@/components/workspace/artifacts/inline-artifact-summary", () => ({
  InlineArtifactSummary: () => null,
}));

const { MessageList } = await import("@/components/workspace/messages/message-list");

function makeThread(overrides: Partial<BaseStream<AgentThreadState>> = {}) {
  return {
    messages: [],
    values: {},
    isLoading: false,
    isThreadLoading: false,
    error: null,
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    stop: async () => {},
    ...overrides,
  } as unknown as BaseStream<AgentThreadState>;
}

describe("MessageList — streaming throttle (Phase0#7 Step 1)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("does NOT re-run groupMessages on an unrelated re-render with the same messages identity", async () => {
    const messages = [
      { id: "h1", type: "human", content: "hi" },
      {
        id: "a1",
        type: "ai",
        content: "hello",
      },
    ] as BaseStream<AgentThreadState>["messages"];

    const thread = makeThread({ messages });

    const view = renderWithProviders(
      <MessageList threadId="t1" thread={thread} paddingBottom={0} />,
    );

    const callsAfterFirstRender = (groupMessages as unknown as ReturnType<typeof vi.fn>).mock.calls.length;
    expect(callsAfterFirstRender).toBeGreaterThanOrEqual(1);

    // Force an unrelated re-render: same thread object identity is fine, but
    // React only re-renders on state/prop change. Re-render with a NEW parent
    // wrapper state but the SAME messages array reference.
    view.rerender(
      <MessageList threadId="t1" thread={thread} paddingBottom={0} />,
    );
    view.rerender(
      <MessageList threadId="t1" thread={thread} paddingBottom={0} />,
    );

    const callsAfterRerender = (groupMessages as unknown as ReturnType<typeof vi.fn>).mock.calls.length;
    // Memoized: grouping must not run again when messages identity is unchanged.
    expect(callsAfterRerender).toBe(callsAfterFirstRender);
  });

  it("re-runs groupMessages when the messages array identity actually changes", async () => {
    const messages = [
      { id: "h1", type: "human", content: "hi" },
    ] as BaseStream<AgentThreadState>["messages"];

    const view = renderWithProviders(
      <MessageList
        threadId="t1"
        thread={makeThread({ messages })}
        paddingBottom={0}
      />,
    );
    const callsBefore = (groupMessages as unknown as ReturnType<typeof vi.fn>).mock.calls.length;

    // New messages array identity (streaming appended a new message).
    const grown = [...messages, { id: "a1", type: "ai", content: "hi back" }] as BaseStream<AgentThreadState>["messages"];
    view.rerender(
      <MessageList
        threadId="t1"
        thread={makeThread({ messages: grown })}
        paddingBottom={0}
      />,
    );

    const callsAfter = (groupMessages as unknown as ReturnType<typeof vi.fn>).mock.calls.length;
    expect(callsAfter).toBeGreaterThan(callsBefore);
  });
});
