// @vitest-environment jsdom
import type { Message } from "@langchain/langgraph-sdk";
import { useState, type ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";

import { I18nProvider } from "@/core/i18n/context";

/**
 * Phase0#7 Step 2 — hot message components are wrapped in React.memo so an
 * unrelated parent re-render (identical props) does not re-execute their
 * subtree. Deterministic form of the spec's "Profiler 验：流式时历史消息组
 * 重渲染次数 = 0".
 *
 * Counting approach: we stub a leaf that MessageListItem always renders
 * (AIElementMessage) with a render-counter. If MessageListItem is memo'd and
 * its props are identical, the parent re-render bails out and the leaf does
 * NOT re-execute. (A <Profiler> on the parent fires on every parent commit
 * regardless of child bail-out, so it is not a reliable signal here.)
 */

let aiMessageRenders = 0;
vi.mock("@/components/ai-elements/message", () => ({
  Message: ({ children }: { children: ReactNode }) => {
    aiMessageRenders++;
    return <div data-testid="ai-msg">{children}</div>;
  },
  MessageContent: ({ children }: { children: ReactNode }) => (
    <div data-testid="ai-content">{children}</div>
  ),
  MessageResponse: ({ children }: { children: ReactNode }) => (
    <>{children}</>
  ),
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

function CountingHarness({ child }: { child: ReactNode }) {
  const [, setTick] = useState(0);
  return (
    <I18nProvider initialLocale="en-US">
      <button type="button" onClick={() => setTick((n) => n + 1)}>
        bump
      </button>
      {child}
    </I18nProvider>
  );
}

function makeHumanMessage(): Message {
  return {
    id: "msg-1",
    type: "human",
    content: "hello world",
  } as unknown as Message;
}

describe("MessageListItem — memo boundary (Phase0#7 Step 2)", () => {
  it("does not re-render its subtree on a parent re-render with identical props", async () => {
    const { fireEvent, render } = await import("@testing-library/react");
    aiMessageRenders = 0;

    const message = makeHumanMessage();
    const messageRunIds = new Map<string, string>();

    const { getByText } = render(
      <CountingHarness
        child={
          <MessageListItem
            message={message}
            threadId="t1"
            isLoading={false}
            messageRunIds={messageRunIds}
          />
        }
      />,
    );
    const rendersAfterMount = aiMessageRenders;
    expect(rendersAfterMount).toBeGreaterThanOrEqual(1);

    // Unrelated parent re-render (state tick) with identical child props.
    fireEvent.click(getByText("bump"));
    fireEvent.click(getByText("bump"));

    // Memoized: the leaf under MessageListItem must not have re-rendered.
    expect(aiMessageRenders).toBe(rendersAfterMount);
  });
});
