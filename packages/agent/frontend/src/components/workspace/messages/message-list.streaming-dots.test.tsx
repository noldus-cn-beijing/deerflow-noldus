// @vitest-environment jsdom
//
// spec 2026-06-30-clarification-awaiting-streaming-dots-fix §四 TDD.
//
// Bug: when `ask_clarification` interrupts the run (DecisionCard rendered,
// agent paused awaiting the user), `thread.isLoading` is still true on the
// frontend (SSE not torn down), so the trailing `StreamingIndicator` (the
// three `animate-bouncing` dots) kept bouncing — implying the agent was still
// working when it was actually waiting for a decision.
//
// Fix (message-list.tsx trailing): gate the StreamingIndicator on
// `!lastClarificationIsAwaiting(messages)` (reusing the existing pure helper).
//
// These tests render the REAL StreamingIndicator (NOT mocked) and assert on
// its `animate-bouncing` dots, so they fail if the guard is absent or neutered
// (vacuous-guard self-check per spec §四).
import type { BaseStream } from "@langchain/langgraph-sdk/react";
import { describe, expect, it, vi } from "vitest";

import type { AgentThreadState } from "@/core/threads";
import { renderWithProviders } from "@/test/test-utils";

// Stub the heavy child components so MessageList renders in isolation. The
// StreamingIndicator is intentionally LEFT REAL — the dots it renders are the
// thing under test.
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
vi.mock("@/components/workspace/messages/decision-card", () => ({
  DecisionCard: () => <div data-testid="decision-card" />,
}));
vi.mock("@/components/workspace/messages/clarification-options", () => ({
  ClarificationOptions: () => null,
}));
vi.mock("@/components/workspace/messages/quality-warning-banner", () => ({
  QualityWarningBanner: () => null,
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

/** An AI message that issued an ask_clarification tool_call (still unanswered). */
function aiWithClarify(id: string, toolCallId: string) {
  return {
    id,
    type: "ai",
    content: "",
    tool_calls: [
      {
        id: toolCallId,
        name: "ask_clarification",
        args: { clarification_type: "ambiguous_requirement", options: ["a", "b"] },
      },
    ],
  };
}

/** The ToolMessage result the backend returns for an ask_clarification call. */
function clarifyTool(id: string, toolCallId: string) {
  return {
    id,
    type: "tool",
    name: "ask_clarification",
    tool_call_id: toolCallId,
    content: "请确认",
  };
}

function human(id: string, content = "ans") {
  return { id, type: "human", content };
}

/** A normal streaming AI message (no ask_clarification) with content. */
function aiStreaming(id: string) {
  return { id, type: "ai", content: "正在分析…" };
}

describe("MessageList — trailing StreamingIndicator vs awaiting-clarification (spec 2026-06-30)", () => {
  it("hides the bouncing dots while a clarification is awaiting (even though isLoading=true)", () => {
    // Stream ends on an unanswered ask_clarification; run is interrupted but
    // thread.isLoading is still true on the frontend (the bug condition).
    const messages = [
      human("h0", "帮我分析"),
      aiWithClarify("a1", "tc1"),
      clarifyTool("t1", "tc1"),
    ] as BaseStream<AgentThreadState>["messages"];

    renderWithProviders(
      <MessageList
        threadId="t1"
        thread={makeThread({ messages, isLoading: true })}
        paddingBottom={0}
        // eslint-disable-next-line @typescript-eslint/no-empty-function
        onSelectClarificationOption={() => {}}
      />,
    );

    // The three animate-bouncing dots must NOT be present.
    const dots = document.querySelectorAll(".animate-bouncing");
    expect(dots.length).toBe(0);
  });

  it("still shows the bouncing dots during normal streaming (last AI msg is not a clarification)", () => {
    const messages = [
      human("h0", "帮我分析"),
      aiStreaming("a1"),
    ] as BaseStream<AgentThreadState>["messages"];

    renderWithProviders(
      <MessageList
        threadId="t1"
        thread={makeThread({ messages, isLoading: true })}
        paddingBottom={0}
      />,
    );

    const dots = document.querySelectorAll(".animate-bouncing");
    // StreamingIndicator renders exactly 3 bouncing dots.
    expect(dots.length).toBe(3);
  });

  it("shows the bouncing dots again once the user has answered (run resumed, isLoading=true)", () => {
    // The clarification was answered by a human message; the run resumed, so
    // it is no longer awaiting → the indicator must come back.
    const messages = [
      human("h0", "帮我分析"),
      aiWithClarify("a1", "tc1"),
      clarifyTool("t1", "tc1"),
      human("h1", "a"),
    ] as BaseStream<AgentThreadState>["messages"];

    renderWithProviders(
      <MessageList
        threadId="t1"
        thread={makeThread({ messages, isLoading: true })}
        paddingBottom={0}
        // eslint-disable-next-line @typescript-eslint/no-empty-function
        onSelectClarificationOption={() => {}}
      />,
    );

    const dots = document.querySelectorAll(".animate-bouncing");
    expect(dots.length).toBe(3);
  });
});
