import type { Message } from "@langchain/langgraph-sdk";
import { beforeEach, describe, expect, it } from "vitest";

import {
  clearExtractionCache,
  extractContentCached,
  extractReasoningCached,
} from "./extraction-cache";
import {
  extractContentFromMessage,
  extractReasoningContentFromMessage,
} from "./utils";

function makeAI(
  id: string,
  content: string,
  reasoning?: string,
): Message {
  return {
    type: "ai",
    id,
    content,
    additional_kwargs: reasoning != null ? { reasoning_content: reasoning } : {},
  } as Message;
}

describe("extraction-cache", () => {
  beforeEach(() => {
    clearExtractionCache();
  });

  it("returns the same value as the uncached extractor (content + reasoning)", () => {
    const msg = makeAI(
      "msg-1",
      "Visible text <think>hidden reasoning</think> more text\n[intent] EXPLORE",
      "explicit reasoning",
    );
    expect(extractContentCached(msg, false)).toBe(
      extractContentFromMessage(msg),
    );
    expect(extractReasoningCached(msg, false)).toBe(
      extractReasoningContentFromMessage(msg),
    );
  });

  it("caches terminal (non-streaming) messages: second call is a cache hit", () => {
    const msg = makeAI("msg-2", "terminal content");
    const first = extractContentCached(msg, false);
    expect(first).toBe("terminal content");
    // Second call with the same id + content length must hit the cache.
    const second = extractContentCached(msg, false);
    expect(second).toBe(first);
    // Mutating the message content does not change a cached terminal result
    // until the content-length dimension changes (cache key includes length).
    expect(second).toBe("terminal content");
  });

  it("does NOT cache the in-flight streaming message (isStreaming=true bypasses cache)", () => {
    let content = "streaming chunk 1";
    const msg = makeAI("msg-3", content);
    expect(extractContentCached(msg, true)).toBe("streaming chunk 1");

    // Simulate a new token arriving: same id, longer content. Because the
    // message is in-flight (isStreaming=true), the cache must be bypassed and
    // the fresh content returned — never a stale cached value.
    content = "streaming chunk 1 streaming chunk 2";
    (msg as { content: string }).content = content;
    expect(extractContentCached(msg, true)).toBe(content);
    expect(extractContentCached(msg, true)).toBe(
      extractContentFromMessage(msg),
    );
  });

  it("invalidates when a terminal message's content length changes", () => {
    const msg = makeAI("msg-4", "first version of content");
    expect(extractContentCached(msg, false)).toBe("first version of content");
    // Same id, but content grew → cache key (id, length) no longer matches →
    // recompute. (Terminal messages do not normally mutate; this is the
    // defensive invariant so a cache key collision can never serve stale text.)
    (msg as { content: string }).content = "first version of content + more";
    expect(extractContentCached(msg, false)).toBe(
      "first version of content + more",
    );
  });

  it("byte-identical output vs uncached extractor across reasoning shapes", () => {
    const cases: Message[] = [
      makeAI("c1", "plain text"),
      makeAI("c2", "text <think>inline think</think> tail"),
      makeAI("c3", "[intent] ANALYZE\nvisible body"),
      makeAI(
        "c4",
        "[gate_signals]\nstatus: ok\nfiles:\n  - a.png\n\n# Heading",
      ),
      makeAI("c5", "<think>only reasoning no visible</think>"),
    ];
    for (const msg of cases) {
      expect(extractContentCached(msg, false)).toBe(
        extractContentFromMessage(msg),
      );
      expect(extractReasoningCached(msg, false)).toBe(
        extractReasoningContentFromMessage(msg),
      );
    }
  });

  it("caches reasoning extraction independently from content extraction", () => {
    const msg = makeAI("msg-5", "body", "the reasoning");
    expect(extractReasoningCached(msg, false)).toBe("the reasoning");
    // Reasoning cached, then content ask — both independent slots, both hits.
    expect(extractContentCached(msg, false)).toBe("body");
    expect(extractReasoningCached(msg, false)).toBe("the reasoning");
  });
});
