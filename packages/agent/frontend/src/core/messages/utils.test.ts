import type { Message } from "@langchain/langgraph-sdk";
import { describe, expect, it } from "vitest";

import {
  extractContentFromMessage,
  extractReasoningContentFromMessage,
  groupMessages,
  hasContent,
  hasReasoning,
} from "./utils";

function makeAIMsg(
  id: string,
  opts: {
    content?: string;
    reasoning?: string;
    toolCalls?: Array<{ name: string; args: Record<string, unknown> }>;
  } = {},
): Message {
  return {
    type: "ai",
    id,
    content: opts.content ?? "",
    additional_kwargs:
      opts.reasoning != null ? { reasoning_content: opts.reasoning } : {},
    tool_calls: opts.toolCalls,
  } as Message;
}

function groupTypes(messages: Message[]): string[] {
  return groupMessages(messages, (g) => g.type);
}

describe("groupMessages", () => {
  it("AI message: reasoning + content + no tool_calls → 1 group, type 'assistant'", () => {
    const msg = makeAIMsg("1", {
      content: "Hello",
      reasoning: "Let me think...",
    });
    const result = groupTypes([msg]);
    expect(result).toEqual(["assistant"]);
  });

  it("AI message: reasoning + tool_calls → 1 group, type 'assistant:processing'", () => {
    const msg = makeAIMsg("1", {
      content: "Using a tool...",
      reasoning: "I need to call a tool",
      toolCalls: [{ name: "search", args: {} }],
    });
    const result = groupTypes([msg]);
    expect(result).toEqual(["assistant:processing"]);
  });

  it("AI message: reasoning only → 1 group, type 'assistant:processing'", () => {
    const msg = makeAIMsg("1", {
      reasoning: "Let me think...",
    });
    const result = groupTypes([msg]);
    expect(result).toEqual(["assistant:processing"]);
  });

  it("AI message: content only → 1 group, type 'assistant'", () => {
    const msg = makeAIMsg("1", {
      content: "Hello",
    });
    const result = groupTypes([msg]);
    expect(result).toEqual(["assistant"]);
  });

  it("AI message: tool_calls only → 1 group, type 'assistant:processing'", () => {
    const msg = makeAIMsg("1", {
      toolCalls: [{ name: "search", args: {} }],
    });
    const result = groupTypes([msg]);
    expect(result).toEqual(["assistant:processing"]);
  });

  it("2 consecutive AI messages: 1st reasoning+tool_calls, 2nd reasoning+content → 2 groups", () => {
    const msg1 = makeAIMsg("1", {
      content: "Using a tool...",
      reasoning: "I need to call a tool",
      toolCalls: [{ name: "search", args: {} }],
    });
    const msg2 = makeAIMsg("2", {
      content: "Here is the answer",
      reasoning: "Now I can answer",
    });
    const result = groupTypes([msg1, msg2]);
    expect(result).toEqual(["assistant:processing", "assistant"]);
  });
});

describe("groupMessages — streaming continuity (no flicker)", () => {
  // Streaming order from server: reasoning chunks → content chunks → tool_calls
  // chunks (if any). Each chunk re-classifies the same message. Without
  // streaming-aware classification, the SAME message id jumps between groups
  // (reasoning → assistant:processing, then +content+no tool_calls →
  // assistant, then +tool_calls → assistant:processing), unmounting and
  // remounting the React tree on every chunk = visible "flicker / reload".
  //
  // Fix: while a message is the last in the array AND isStreaming=true, keep
  // it pinned to processing instead of promoting to the final "assistant"
  // group. After the stream ends, re-classification picks the final group.

  it("isStreaming=true: last AI msg with reasoning+content (no tools yet) stays in processing", () => {
    const msg = makeAIMsg("1", {
      content: "partial answer streaming...",
      reasoning: "deciding next action",
    });
    const result = groupMessages([msg], (g) => g.type);
    expect(result).toEqual(["assistant:processing"]);
  });

  it("isStreaming=false: same message classifies as assistant (no pinning)", () => {
    const msg = makeAIMsg("1", {
      content: "complete answer",
      reasoning: "decided",
    });
    const result = groupTypes([msg]);
    expect(result).toEqual(["assistant"]);
  });

  it("isStreaming=true: non-last messages classify normally (pinning ONLY on last)", () => {
    const msg1 = makeAIMsg("1", {
      content: "first turn final",
      reasoning: "thought 1",
    });
    const msg2 = makeAIMsg("2", {
      content: "second turn streaming",
      reasoning: "thought 2",
    });
    const result = groupMessages([msg1, msg2], (g) => g.type);
    expect(result).toEqual(["assistant", "assistant:processing"]);
  });
});

describe("inline <think> handling — streaming TTFT", () => {
  it("closed <think>...</think> in string content: reasoning extracted, content cleaned", () => {
    const msg = makeAIMsg("1", {
      content: "<think>I will analyze</think>Here is the answer",
    });
    expect(extractReasoningContentFromMessage(msg)).toBe("I will analyze");
    expect(extractContentFromMessage(msg)).toBe("Here is the answer");
  });

  it("multiple closed <think> blocks: all reasoning concatenated", () => {
    const msg = makeAIMsg("1", {
      content: "<think>first</think>visible<think>second</think> answer",
    });
    expect(extractReasoningContentFromMessage(msg)).toBe("first\n\nsecond");
    expect(extractContentFromMessage(msg)).toBe("visible answer");
  });

  it("unclosed <think> at end (mid-stream): reasoning streams live, content empty", () => {
    const msg = makeAIMsg("1", {
      content: "<think>I am still thinking about",
    });
    expect(extractReasoningContentFromMessage(msg)).toBe("I am still thinking about");
    expect(extractContentFromMessage(msg)).toBe("");
  });

  it("closed <think> + open trailing <think> (mid-stream second block): both reasonings shown", () => {
    const msg = makeAIMsg("1", {
      content: "<think>first done</think>partial answer<think>second thought streaming",
    });
    expect(extractReasoningContentFromMessage(msg)).toBe(
      "first done\n\nsecond thought streaming",
    );
    expect(extractContentFromMessage(msg)).toBe("partial answer");
  });

  it("content without any <think>: returned as-is, no reasoning", () => {
    const msg = makeAIMsg("1", {
      content: "plain answer with no thinking",
    });
    expect(extractReasoningContentFromMessage(msg)).toBeNull();
    expect(extractContentFromMessage(msg)).toBe("plain answer with no thinking");
  });

  it("unclosed <think> with empty body (just <think> typed): does not crash, no reasoning yet", () => {
    const msg = makeAIMsg("1", {
      content: "<think>",
    });
    // Open tag with nothing after it — reasoning is empty string, treat as null
    expect(extractReasoningContentFromMessage(msg)).toBeNull();
    expect(extractContentFromMessage(msg)).toBe("");
  });

  it("preamble before an unclosed <think> stays in content", () => {
    const msg = makeAIMsg("1", {
      content: "Here is part of the answer.<think>but wait, let me reconsider",
    });
    expect(extractContentFromMessage(msg)).toBe("Here is part of the answer.");
    expect(extractReasoningContentFromMessage(msg)).toBe(
      "but wait, let me reconsider",
    );
  });

  it("hasReasoning recognises an unclosed <think> tag mid-stream", () => {
    expect(hasReasoning(makeAIMsg("1", { content: "<think>thinking in progress" }))).toBe(true);
  });

  it("hasContent excludes an unclosed <think> tail when no preamble exists", () => {
    expect(hasContent(makeAIMsg("1", { content: "<think>thinking in progress" }))).toBe(false);
  });

  it("hasContent stays true when preamble precedes an unclosed <think>", () => {
    expect(hasContent(makeAIMsg("1", { content: "preamble<think>still thinking" }))).toBe(true);
  });

  it("a literal <think> inside markdown inline code is not treated as reasoning", () => {
    const msg = makeAIMsg("1", {
      content: "Use `<think>` markers to delimit reasoning sections.",
    });
    expect(extractContentFromMessage(msg)).toBe(
      "Use `<think>` markers to delimit reasoning sections.",
    );
    expect(extractReasoningContentFromMessage(msg)).toBeNull();
    expect(hasReasoning(msg)).toBe(false);
  });

  it("a backtick-prefixed <think> mid-stream is not split into reasoning", () => {
    const msg = makeAIMsg("1", {
      content: "Documentation: `<think>",
    });
    expect(extractContentFromMessage(msg)).toBe(
      "Documentation: `<think>",
    );
    expect(extractReasoningContentFromMessage(msg)).toBeNull();
  });
});
