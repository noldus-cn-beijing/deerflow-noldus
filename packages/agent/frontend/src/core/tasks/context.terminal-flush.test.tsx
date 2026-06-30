// @vitest-environment jsdom
//
// spec 2026-06-30-subtask-card-stuck-in-progress-after-run-success-fix §四 TDD.
//
// Bug: report-writer is the LAST pipeline step. Its "Task Succeeded"
// ToolMessage arrives in thread.messages; MessageList's render-time walk
// calls `updateSubtask({ id, status: "completed" })`. But that call carries
// no `latestMessage`, so it hits the in-place-mutate branch of
// useUpdateSubtask — which mutates `tasks[id]` WITHOUT setTasks, betting
// that "a later MessageList re-render (triggered by some OTHER message
// change) will flush this terminal state to the UI". For the last step there
// is no later message change (run already success, stream stopped, no
// ask_clarification), so that flush never happens → the SubtaskCard keeps
// spinning `animate-spin` forever even though the backend run succeeded.
//
// Fix (修法 A): when status transitions from a non-terminal state to
// completed/failed (a REAL state transition), flush via setTasks — but
// deferred out of the render phase (queueMicrotask) so we don't reintroduce
// the "Cannot update while rendering" warning that the in-place branch
// exists to avoid.
//
// These tests mount the REAL SubtasksProvider + a card subscribing to
// useSubtask, drive useUpdateSubtask directly, and assert on the rendered
// status. They fail if the terminal flush is absent or neutered (vacuous
// self-check per spec §四).
import { act, render, screen } from "@testing-library/react";
import { useEffect } from "react";
import { describe, expect, it } from "vitest";

import { SubtasksProvider, useFinalizeRunning, useSubtask, useUpdateSubtask } from "@/core/tasks/context";
import type { Subtask } from "@/core/tasks/types";

/** A card that subscribes to one subtask and renders its status verbatim. */
function SubtaskStatusCard({ id }: { id: string }) {
  const task = useSubtask(id);
  // Render the status text + a sentinel for the in-progress spinner so the
  // tests can assert terminal-vs-spinning without coupling to icon internals.
  return (
    <div>
      <span data-testid={`status-${id}`}>{task?.status ?? "missing"}</span>
      {task?.status === "in_progress" ? (
        <span data-testid={`spin-${id}`} className="animate-spin" />
      ) : null}
    </div>
  );
}

/**
 * A controller that mirrors the latest `updateSubtask` into a ref via effect,
 * so the test always invokes the freshest closure (which captures the latest
 * `tasks`). Reading happens through `getUpdate()` to avoid stale-closure races.
 */
function Controller({
  updateRef,
}: {
  updateRef: { current: ((u: Partial<Subtask> & { id: string }) => void) | null };
}) {
  const updateSubtask = useUpdateSubtask();
  useEffect(() => {
    updateRef.current = updateSubtask;
  });
  return null;
}

function mountFixture(id: string) {
  const updateRef: { current: ((u: Partial<Subtask> & { id: string }) => void) | null } = {
    current: null,
  };
  render(
    <SubtasksProvider>
      <SubtaskStatusCard id={id} />
      <Controller updateRef={updateRef} />
    </SubtasksProvider>,
  );
  // Wrapper that always dispatches through the freshest closure (capturing
  // the latest `tasks`). Controller's effect keeps updateRef.current in sync
  // after every commit, so callers never hit a stale-closure race.
  const update = (u: Partial<Subtask> & { id: string }) => {
    const fn = updateRef.current;
    if (!fn) throw new Error("updateSubtask not registered yet");
    fn(u);
  };
  return update;
}

const aMessage = (id: string) =>
  ({ id, type: "ai", content: "thinking…" }) as unknown as Subtask["messages"][number];

describe("useUpdateSubtask — terminal flush after Task Succeeded (spec 2026-06-30)", () => {
  it("flushes completed to subscribers even with no later message change (修法 A)", async () => {
    const id = "task-rw";
    const update = mountFixture(id);

    // 1. SSE task_running establishes the subtask in_progress (carries a
    //    latestMessage → flushes via setTasks; the card is now spinning).
    act(() => {
      update({
        id,
        status: "in_progress",
        subagent_type: "report-writer",
        description: "report",
        prompt: "p",
        latestMessage: aMessage("m1"),
      } as Partial<Subtask> & { id: string });
    });
    expect(screen.getByTestId(`status-${id}`).textContent).toBe("in_progress");
    expect(screen.getByTestId(`spin-${id}`)).toBeTruthy();

    // 2. MessageList render-time walk hits the "Task Succeeded" ToolMessage
    //    and calls updateSubtask({ status: "completed" }) with NO
    //    latestMessage. THIS is the bug path: in-place mutate, no flush.
    act(() => {
      update({ id, status: "completed", result: "## 最终结果\n报告 OK" });
    });

    // 3. CRUCIAL: no further update happens. report-writer is the last step —
    //    there is no "later message change" to bail us out. The terminal
    //    state must reach subscribers on its own (修法 A: deferred flush).
    // Let any deferred microtask flush drain.
    await act(async () => {
      await Promise.resolve();
    });

    expect(screen.getByTestId(`status-${id}`).textContent).toBe("completed");
    expect(screen.queryByTestId(`spin-${id}`)).toBeNull();
  });

  it("still spins while genuinely in_progress (no terminal event) — no regression", () => {
    const id = "task-da";
    const update = mountFixture(id);

    act(() => {
      update({
        id,
        status: "in_progress",
        subagent_type: "data-analyst",
        description: "d",
        prompt: "p",
        latestMessage: aMessage("m1"),
      } as Partial<Subtask> & { id: string });
    });

    // No terminal event arrives — the card must keep spinning.
    expect(screen.getByTestId(`status-${id}`).textContent).toBe("in_progress");
    expect(screen.getByTestId(`spin-${id}`)).toBeTruthy();
  });

  it("does not re-flush when status is already terminal (idempotent, no churn)", async () => {
    const id = "task-ce";
    const update = mountFixture(id);

    act(() => {
      update({
        id,
        status: "in_progress",
        subagent_type: "code-executor",
        description: "c",
        prompt: "p",
        latestMessage: aMessage("m1"),
      } as Partial<Subtask> & { id: string });
    });
    act(() => {
      update({ id, status: "completed", result: "done" });
    });
    await act(async () => {
      await Promise.resolve();
    });
    expect(screen.getByTestId(`status-${id}`).textContent).toBe("completed");

    // A second terminal walk (re-render with the same ToolMessage still in
    // thread.messages) must not throw or loop — same terminal → no-op.
    act(() => {
      update({ id, status: "completed", result: "done" });
    });
    await act(async () => {
      await Promise.resolve();
    });
    expect(screen.getByTestId(`status-${id}`).textContent).toBe("completed");
  });
});

describe("finalizeRunning — run-terminal fallback (spec 2026-06-30 兜底)", () => {
  /** Card that subscribes to one subtask and exposes its status via testid. */
  function StatusChip({ id }: { id: string }) {
    const task = useSubtask(id);
    return <span data-testid={`status-${id}`}>{task?.status ?? "missing"}</span>;
  }

  /** Wires updateSubtask + finalizeRunning out via effect-synced refs. */
  function Harness({
    ids,
    updateRef,
    finalizeRef,
  }: {
    ids: string[];
    updateRef: { current: ((u: Partial<Subtask> & { id: string }) => void) | null };
    finalizeRef: { current: ((s: "completed" | "failed") => void) | null };
  }) {
    const updateSubtask = useUpdateSubtask();
    const finalizeRunning = useFinalizeRunning();
    useEffect(() => {
      updateRef.current = updateSubtask;
      finalizeRef.current = finalizeRunning;
    });
    return (
      <>
        {ids.map((id) => (
          <StatusChip key={id} id={id} />
        ))}
      </>
    );
  }

  function mountMulti(ids: string[]) {
    const updateRef: { current: ((u: Partial<Subtask> & { id: string }) => void) | null } = {
      current: null,
    };
    const finalizeRef: { current: ((s: "completed" | "failed") => void) | null } = {
      current: null,
    };
    render(
      <SubtasksProvider>
        <Harness ids={ids} updateRef={updateRef} finalizeRef={finalizeRef} />
      </SubtasksProvider>,
    );
    const update = (u: Partial<Subtask> & { id: string }) => {
      const fn = updateRef.current;
      if (!fn) throw new Error("updateSubtask not registered yet");
      fn(u);
    };
    const finalize = (s: "completed" | "failed") => {
      const fn = finalizeRef.current;
      if (!fn) throw new Error("finalizeRunning not registered yet");
      fn(s);
    };
    return { update, finalize };
  }

  const seedRunning = (
    update: (u: Partial<Subtask> & { id: string }) => void,
    id: string,
    subagent_type = "report-writer",
  ) =>
    update({
      id,
      status: "in_progress",
      subagent_type,
      description: "d",
      prompt: "p",
      latestMessage: aMessage(`m-${id}`),
    } as Partial<Subtask> & { id: string });

  it("on run success, flips every still-spinning subtask to completed (兜底)", async () => {
    const { update, finalize } = mountMulti(["a", "b", "c"]);
    act(() => {
      seedRunning(update, "a", "code-executor");
      seedRunning(update, "b", "report-writer");
    });
    // 'c' was already completed before the run ended — must stay completed.
    act(() => seedRunning(update, "c", "data-analyst"));
    await act(async () => {
      update({ id: "c", status: "completed", result: "done" });
      // 修法 A flushes via queueMicrotask — drain it before asserting.
      await Promise.resolve();
    });
    expect(screen.getByTestId("status-a").textContent).toBe("in_progress");
    expect(screen.getByTestId("status-b").textContent).toBe("in_progress");
    expect(screen.getByTestId("status-c").textContent).toBe("completed");

    // Run reaches success → finalizeRunning("completed").
    act(() => finalize("completed"));

    expect(screen.getByTestId("status-a").textContent).toBe("completed");
    expect(screen.getByTestId("status-b").textContent).toBe("completed");
    expect(screen.getByTestId("status-c").textContent).toBe("completed");
  });

  it("on run error, flips every still-spinning subtask to failed (兜底)", () => {
    const { update, finalize } = mountMulti(["a", "b"]);
    act(() => {
      seedRunning(update, "a", "code-executor");
      seedRunning(update, "b", "report-writer");
    });

    act(() => finalize("failed"));

    expect(screen.getByTestId("status-a").textContent).toBe("failed");
    expect(screen.getByTestId("status-b").textContent).toBe("failed");
  });

  it("is a no-op when nothing is in_progress (no spurious state churn)", async () => {
    const { update, finalize } = mountMulti(["a"]);
    act(() => seedRunning(update, "a", "report-writer"));
    await act(async () => {
      update({ id: "a", status: "completed", result: "done" });
      await Promise.resolve();
    });
    expect(screen.getByTestId("status-a").textContent).toBe("completed");

    expect(() => act(() => finalize("completed"))).not.toThrow();
    expect(screen.getByTestId("status-a").textContent).toBe("completed");
  });
});
