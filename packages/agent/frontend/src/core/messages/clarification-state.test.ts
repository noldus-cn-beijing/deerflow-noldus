import type { Message } from "@langchain/langgraph-sdk";
import { describe, expect, it } from "vitest";

import {
  answeredOptionIndex,
  isClarificationAnswered,
  lastClarificationIsAwaiting,
  normalizeClarificationType,
} from "./clarification-state";

function makeAIWithClarify(
  id: string,
  toolCallId: string,
  clarificationType = "ambiguous_requirement",
): Message {
  return {
    type: "ai",
    id,
    content: "",
    tool_calls: [
      {
        id: toolCallId,
        name: "ask_clarification",
        args: { clarification_type: clarificationType, options: [] },
      },
    ],
  } as Message;
}

function makeHuman(id: string): Message {
  return { type: "human", id, content: "ans" } as Message;
}

function makeTool(id: string, toolCallId: string): Message {
  return {
    type: "tool",
    id,
    name: "ask_clarification",
    tool_call_id: toolCallId,
    content: "...",
  } as Message;
}

describe("isClarificationAnswered (spec#5 §3.1/§3.4)", () => {
  it("returns false when no human message follows the ask_clarification", () => {
    const msgs = [makeAIWithClarify("a1", "tc1"), makeTool("t1", "tc1")];
    expect(isClarificationAnswered(msgs, "tc1")).toBe(false);
  });

  it("returns true once a human message arrives after the clarification", () => {
    const msgs = [
      makeAIWithClarify("a1", "tc1"),
      makeTool("t1", "tc1"),
      makeHuman("h1"),
    ];
    expect(isClarificationAnswered(msgs, "tc1")).toBe(true);
  });

  it("treats a human message that came BEFORE the clarification as unanswered", () => {
    const msgs = [
      makeHuman("h0"),
      makeAIWithClarify("a1", "tc1"),
      makeTool("t1", "tc1"),
    ];
    expect(isClarificationAnswered(msgs, "tc1")).toBe(false);
  });

  it("returns false when the toolCallId is unknown", () => {
    const msgs = [makeAIWithClarify("a1", "tc1")];
    expect(isClarificationAnswered(msgs, "does-not-exist")).toBe(false);
    expect(isClarificationAnswered(msgs, undefined)).toBe(false);
  });
});

describe("lastClarificationIsAwaiting (spec#5 §3.4 input box state)", () => {
  it("is awaiting when the stream ends on an unanswered clarification", () => {
    const msgs = [
      makeHuman("h0"),
      makeAIWithClarify("a1", "tc1"),
      makeTool("t1", "tc1"),
    ];
    expect(lastClarificationIsAwaiting(msgs)).toBe(true);
  });

  it("is NOT awaiting once the user has answered", () => {
    const msgs = [
      makeAIWithClarify("a1", "tc1"),
      makeTool("t1", "tc1"),
      makeHuman("h1"),
    ];
    expect(lastClarificationIsAwaiting(msgs)).toBe(false);
  });

  it("is NOT awaiting when the last AI message has no ask_clarification", () => {
    const aiNoClarify = {
      type: "ai",
      id: "a2",
      content: "done",
      tool_calls: [],
    } as Message;
    const msgs = [makeAIWithClarify("a1", "tc1"), makeHuman("h1"), aiNoClarify];
    expect(lastClarificationIsAwaiting(msgs)).toBe(false);
  });

  it("is NOT awaiting when there are no AI messages", () => {
    expect(lastClarificationIsAwaiting([makeHuman("h0")])).toBe(false);
    expect(lastClarificationIsAwaiting([])).toBe(false);
  });

  it("uses the MOST RECENT ask_clarification (multi-round)", () => {
    // First clarification answered, second one pending → still awaiting.
    const msgs = [
      makeAIWithClarify("a1", "tc1"),
      makeHuman("h1"),
      makeAIWithClarify("a2", "tc2"),
      makeTool("t2", "tc2"),
    ];
    expect(lastClarificationIsAwaiting(msgs)).toBe(true);
  });
});

describe("normalizeClarificationType (spec#5 §3.5)", () => {
  it("passes through valid types", () => {
    expect(normalizeClarificationType("risk_confirmation")).toBe(
      "risk_confirmation",
    );
    expect(normalizeClarificationType("suggestion")).toBe("suggestion");
    expect(normalizeClarificationType("missing_info")).toBe("missing_info");
  });

  it("returns undefined for unknown / non-string values (fallback to default tone)", () => {
    expect(normalizeClarificationType("nope")).toBeUndefined();
    expect(normalizeClarificationType(undefined)).toBeUndefined();
    expect(normalizeClarificationType(42)).toBeUndefined();
    expect(normalizeClarificationType(null)).toBeUndefined();
  });
});

describe("answeredOptionIndex (spec#5 §3.1 closed-loop highlight)", () => {
  const opts = ["Center=center, edge=periphery", "the other way", "neither"];

  it("returns null while still awaiting (no human reply yet)", () => {
    const msgs = [
      makeAIWithClarify("a1", "tc1"),
      makeTool("t1", "tc1"),
    ];
    expect(answeredOptionIndex(msgs, "tc1", opts)).toBeNull();
  });

  it("returns the index of the option whose text matches the reply", () => {
    const msgs = [
      makeAIWithClarify("a1", "tc1"),
      makeTool("t1", "tc1"),
      {
        type: "human",
        id: "h1",
        content: "the other way",
      } as Message,
    ];
    expect(answeredOptionIndex(msgs, "tc1", opts)).toBe(1);
  });

  it("returns null when the user typed a custom (non-matching) reply", () => {
    const msgs = [
      makeAIWithClarify("a1", "tc1"),
      makeTool("t1", "tc1"),
      {
        type: "human",
        id: "h1",
        content: "actually let me explain…",
      } as Message,
    ];
    expect(answeredOptionIndex(msgs, "tc1", opts)).toBeNull();
  });

  it("trims whitespace when matching", () => {
    const msgs = [
      makeAIWithClarify("a1", "tc1"),
      makeTool("t1", "tc1"),
      {
        type: "human",
        id: "h1",
        content: "  neither  ",
      } as Message,
    ];
    expect(answeredOptionIndex(msgs, "tc1", opts)).toBe(2);
  });
});
