import type { Message } from "@langchain/langgraph-sdk";

import {
  extractContentFromMessage,
  extractReasoningContentFromMessage,
} from "./utils";

/**
 * Per-message extraction cache (spec 2026-06-29-streaming-render-perf, Step 1).
 *
 * Problem this solves: the streaming render path re-runs
 * {@link extractContentFromMessage} / {@link extractReasoningContentFromMessage}
 * on the *entire* message content every deferred batch (4–6 calls per message,
 * each an O(n) regex scan over the accumulated content). As a message grows
 * the per-batch cost is O(n) and the whole stream amortises to O(n²) — the
 * confirmed #1 cause of the "browser single tab jank" during long streaming.
 *
 * The win is on *completed historical* messages: in a long thread they are the
 * overwhelming majority, their content never changes, yet they were being
 * re-scanned on every batch. The cache absorbs that.
 *
 * ## Correctness rule (the part that must not be wrong)
 *
 * The single in-flight streaming message's content IS mutating, so caching it
 * would serve stale text. Therefore:
 *
 * - `isStreaming === true`  → **bypass the cache entirely**, compute fresh.
 *   The streaming message pays the scan each batch (it has to — its content
 *   changed), but there is only ever ONE such message. Every other (terminal)
 *   message in the thread is served from cache.
 * - `isStreaming === false` → cache by `(message.id, contentLength)` so a
 *   defensive content-length dimension guarantees a terminal message whose
 *   content somehow mutated can never collide with a prior cached entry.
 *
 * This mirrors the design intent of the spec: "只对已终态（非流式）消息长缓存；
 * 流式中那一条不缓存".
 *
 * Output is byte-identical to the uncached extractor (see extraction-cache.test
 * "byte-identical output vs uncached extractor"). The cache is purely a memo;
 * it never transforms.
 */

type ExtractionKind = "content" | "reasoning";

interface CacheEntry {
  contentLength: number;
  content: string;
  reasoning: string | null;
}

// Module-level Map: the cache lives for the page session. Bounded by the
// number of messages in the open thread (cleared on thread switch via
// {@link clearExtractionCache}). Keys are message ids; values carry both
// extraction kinds so one cache miss populates both slots.
const cache = new Map<string, CacheEntry>();

function messageContentLength(message: Message): number {
  const content = message.content;
  if (typeof content === "string") {
    return content.length;
  }
  if (Array.isArray(content)) {
    // Approximate length for the content-length dimension. Exact value is not
    // required — it only needs to change when the content materially changes,
    // which array `.length` does as parts are added during streaming.
    return content.length;
  }
  return 0;
}

function computeEntry(message: Message): CacheEntry {
  return {
    contentLength: messageContentLength(message),
    content: extractContentFromMessage(message),
    reasoning: extractReasoningContentFromMessage(message),
  };
}

function getOrCompute(
  message: Message,
  isStreaming: boolean,
  kind: ExtractionKind,
): string | null {
  // The in-flight streaming message is never cached: its content is mutating
  // this very batch. Compute fresh and return — do not populate the cache.
  if (isStreaming) {
    return kind === "content"
      ? extractContentFromMessage(message)
      : extractReasoningContentFromMessage(message);
  }

  const id = typeof message.id === "string" ? message.id : undefined;
  // No stable id → cannot cache safely (would collide under an undefined key).
  // Fall back to the uncached extractor.
  if (!id) {
    return kind === "content"
      ? extractContentFromMessage(message)
      : extractReasoningContentFromMessage(message);
  }

  const existing = cache.get(id);
  // Cache hit only when the content-length dimension still matches — defends
  // against a terminal message whose content mutated under the same id.
  if (existing?.contentLength === messageContentLength(message)) {
    return kind === "content" ? existing.content : existing.reasoning;
  }

  const entry = computeEntry(message);
  cache.set(id, entry);
  return kind === "content" ? entry.content : entry.reasoning;
}

/** Cached equivalent of {@link extractContentFromMessage}. */
export function extractContentCached(
  message: Message,
  isStreaming: boolean,
): string {
  // extractContentFromMessage returns "" (never null/undefined) for messages
  // with no content; the cache stores the same string.
  return getOrCompute(message, isStreaming, "content") ?? "";
}

/** Cached equivalent of {@link extractReasoningContentFromMessage}. */
export function extractReasoningCached(
  message: Message,
  isStreaming: boolean,
): string | null {
  return getOrCompute(message, isStreaming, "reasoning");
}

/**
 * Drop every cached entry. Call on thread switch so a long session does not
 * retain entries from closed threads (and so the cache cannot leak content
 * across threads).
 */
export function clearExtractionCache(): void {
  cache.clear();
}
