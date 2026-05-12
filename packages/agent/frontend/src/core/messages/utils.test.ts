import { describe, expect, it } from "vitest";
import type { Message } from "@langchain/langgraph-sdk";
import { groupMessages } from "./utils";

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
