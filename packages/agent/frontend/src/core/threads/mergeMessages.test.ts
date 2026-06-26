import type { Message } from "@langchain/langgraph-sdk";
import { describe, expect, it } from "vitest";

import canonical from "./__fixtures__/e9837b33-canonical.json";
import { mergeMessages } from "./hooks";

// Real dogfood thread e9837b33 (5 runs, 40 messages) — dumped from the
// LangGraph checkpoint store. Used to lock the rejoin disorder fix.
//
// Spec 2026-06-26-rejoin-thread-history-merge-disorder-fix — §三 forensics
// (root-cause assumptions REFUTED by this data):
//   • All 40 messages have a STABLE identity (0 undefined, 0 collisions) → the
//     spec's "messageIdentity returns undefined → early break" hypothesis is
//     FALSE for this thread.
//   • The disorder is Mechanism 2: mergeMessages' overlap scan assumes
//     `threadMessages` is always a superset of `historyMessages`' tail. During
//     the rejoin race window `thread.messages` (useStream state) can be STALER
//     (older/shorter) than `historyMessages` (useThreadHistory per-run loads).
//     Then history's tail is not in thread → the scan `break`s immediately →
//     cutoff = history.length → concat = [...full history, ...stale thread] →
//     dedupe keeps the LAST (thread) copy of each shared identity, which sits
//     at the END → those messages (incl. the first input) RELOCATE to the tail.
//   • No message is ever DELETED by mergeMessages — the "outputs vanish" symptom
//     is the reorder pushing them below the viewport, not deletion. mergedLen
//     stays 40 in every scenario.

const CANONICAL = canonical as unknown as Message[];

function canonIdx(m: Message): number {
  return CANONICAL.findIndex((c) => c.id === m.id);
}

function messageIdentityRaw(message: Message): string | undefined {
  if (
    "tool_call_id" in message &&
    typeof message.tool_call_id === "string" &&
    message.tool_call_id.length > 0
  ) {
    return `tool:${message.tool_call_id}`;
  }
  if (typeof message.id === "string" && message.id.length > 0) {
    return `message:${message.id}`;
  }
  return undefined;
}

function makeMsg(id: string, type: Message["type"] = "human"): Message {
  return { type, id, content: id } as Message;
}

function makeToolMsg(id: string, toolCallId: string): Message {
  return {
    type: "tool",
    id,
    tool_call_id: toolCallId,
    content: id,
  } as Message;
}

describe("mergeMessages — rejoin disorder (spec 2026-06-26)", () => {
  // RED test 1 — the actual e9837b33 failure: thread STALER than history.
  // UseThreadHistory already loaded the full history while useStream's state
  // still reflects only the first 4 messages. The first input (canonical idx 0)
  // must remain at the TOP of the merged view, not jump to position 16+.
  it("R1: stale thread (canonical 0..3) + full history → first input stays first, canonical order preserved", () => {
    const staleThread = CANONICAL.slice(0, 4);
    const merged = mergeMessages(CANONICAL, staleThread, []);

    expect(merged.length).toBe(40);
    // No reorder: merged order == canonical order.
    expect(merged.map(canonIdx)).toEqual(
      Array.from({ length: 40 }, (_, i) => i),
    );
    // The original input (first upload message) stays first.
    expect(merged[0]?.id).toBe(CANONICAL[0]?.id);
  });

  it("R2: stale thread at multiple split points keeps canonical order (no input-to-middle)", () => {
    for (const N of [4, 12, 24, 34]) {
      const staleThread = CANONICAL.slice(0, N);
      const merged = mergeMessages(CANONICAL, staleThread, []);
      expect(merged.length).toBe(40);
      expect(merged.map(canonIdx)).toEqual(
        Array.from({ length: 40 }, (_, i) => i),
      );
      expect(merged[0]?.id).toBe(CANONICAL[0]?.id);
    }
  });

  // Synthetic, identity-stable messages — the minimal reproducer independent of
  // the 40-msg fixture. history = newer full set [h1,a1,h2,a2]; thread = older
  // prefix [h1,a1]. The bug relocates h1/a1 to the tail.
  it("R3: synthetic stale-thread reorder — history=[h1,a1,h2,a2], thread=[h1,a1] → order h1,a1,h2,a2", () => {
    const h1 = makeMsg("h1");
    const a1 = makeMsg("a1", "ai");
    const h2 = makeMsg("h2");
    const a2 = makeMsg("a2", "ai");
    const history = [h1, a1, h2, a2];
    const staleThread = [h1, a1];

    const merged = mergeMessages(history, staleThread, []);
    // stable expectation:
    expect(merged.map((m) => m.id)).toEqual(["h1", "a1", "h2", "a2"]);
  });

  // Tool messages dedupe by tool_call_id (identity = tool:<id>), not message id.
  // The fix must still collapse a history tool msg and a thread tool msg that
  // share tool_call_id, keeping canonical order.
  it("R4: tool messages collapse by tool_call_id, order preserved", () => {
    const t1 = makeToolMsg("tool-1", "call_abc");
    const a1 = makeMsg("a1", "ai");
    const history = [t1, a1];
    // thread has the SAME tool result under a different message id (e.g.
    // re-emitted by the live stream) — must dedupe to one, keep canonical order.
    const t1Dup = makeToolMsg("tool-1-dup", "call_abc");
    const staleThread = [t1Dup];
    const merged = mergeMessages(history, staleThread, []);
    // Two messages, tool result collapsed to one (identity tool:call_abc), in
    // canonical order: tool result first, then a1. The surviving copy may be
    // either the history or thread object (same content), so assert by identity.
    expect(merged.length).toBe(2);
    expect(messageIdentityRaw(merged[0]!)).toBe("tool:call_abc");
    expect(messageIdentityRaw(merged[1]!)).toBe("message:a1");
  });
});

describe("mergeMessages — regressions (must stay green)", () => {
  // The classic valid case: thread is the full superset, history is a tail
  // subset. Must keep canonical order, no dupes.
  it("G1: history=tail subset (34..39), thread=full → canonical order, len 40", () => {
    const merged = mergeMessages(CANONICAL.slice(34), CANONICAL, []);
    expect(merged.length).toBe(40);
    expect(merged.map(canonIdx)).toEqual(
      Array.from({ length: 40 }, (_, i) => i),
    );
  });

  it("G2: history=full, thread=full → canonical order, no dupes", () => {
    const merged = mergeMessages(CANONICAL, CANONICAL, []);
    expect(merged.length).toBe(40);
    expect(merged.map(canonIdx)).toEqual(
      Array.from({ length: 40 }, (_, i) => i),
    );
  });

  // Optimistic messages always render last (in-flight human upload bubble).
  it("G3: optimistic appended after merged history+thread", () => {
    const opt = makeMsg("opt-human");
    const merged = mergeMessages(CANONICAL.slice(34), CANONICAL, [opt]);
    expect(merged.length).toBe(41);
    expect(merged[40]?.id).toBe("opt-human");
  });

  // Hidden control messages: a hidden message that SHARES an identity with a
  // visible one (same id / same tool_call_id) collapses to the visible copy.
  // A hidden message with its OWN distinct identity survives. A hidden message
  // whose identity is ONLY shared with other hidden copies keeps the last copy.
  it("G4: hidden sharing visible identity collapses; hidden with own identity survives", () => {
    const visible = makeMsg("v1"); // identity message:v1
    const hiddenSameId = {
      ...makeMsg("v1"),
      additional_kwargs: { hide_from_ui: true },
    } as Message; // same identity message:v1 → collapses to visible
    const hiddenOwn = {
      ...makeMsg("own-hidden"),
      name: "loop_warning",
    } as Message; // distinct identity message:own-hidden → survives

    const merged = mergeMessages([hiddenOwn], [visible, hiddenSameId], []);
    const ids = merged.map((m) => m.id);
    expect(ids).toContain("v1"); // visible kept
    expect(ids.filter((id) => id === "v1").length).toBe(1); // only one v1
    expect(ids).toContain("own-hidden"); // distinct-identity hidden survives
  });

  // No-flicker invariant: consecutive calls with growing thread (streaming)
  // must not move an already-rendered message to a different group position.
  // Here we assert order stability across a thread that grows by one message.
  it("G5: streaming growth — appending to thread keeps prior order stable", () => {
    const history = [makeMsg("h1"), makeMsg("a1", "ai")];
    const before = mergeMessages(history, [makeMsg("h1"), makeMsg("a1", "ai"), makeMsg("m1", "ai")], []);
    const after = mergeMessages(history, [makeMsg("h1"), makeMsg("a1", "ai"), makeMsg("m1", "ai"), makeMsg("m2", "ai")], []);
    // The shared prefix order is unchanged; the new message is appended.
    expect(after.slice(0, before.length).map((m) => m.id)).toEqual(
      before.map((m) => m.id),
    );
    expect(after[after.length - 1]?.id).toBe("m2");
  });

  // Thread-with-gap: thread skipped an older prefix that history still has
  // (e.g. summarization archived it, or thread hydrated a later checkpoint).
  // The history-only prefix must render BEFORE the thread in canonical order,
  // never relocated to the end. This is the inverse of the R1 race and locks
  // that the interleave places older messages correctly too.
  it("G6: thread missing older prefix that history has → prefix renders before thread (canonical order)", () => {
    const history = [makeMsg("old1"), makeMsg("old2"), makeMsg("cur1"), makeMsg("cur2", "ai")];
    // thread only has the newer two — Phase 1 emits old1/old2 (not in thread),
    // breaks at cur1 (in thread), Phase 2 emits the thread.
    const thread = [makeMsg("cur1"), makeMsg("cur2", "ai")];
    const merged = mergeMessages(history, thread, []);
    expect(merged.map((m) => m.id)).toEqual(["old1", "old2", "cur1", "cur2"]);
  });
});
