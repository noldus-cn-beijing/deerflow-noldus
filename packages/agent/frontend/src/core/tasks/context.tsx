import type { AIMessage } from "@langchain/langgraph-sdk";
import { createContext, useCallback, useContext, useState } from "react";

import type { Subtask } from "./types";

export interface SubtaskContextValue {
  tasks: Record<string, Subtask>;
  setTasks: (tasks: Record<string, Subtask>) => void;
  /**
   * Deterministic last-resort (spec 2026-06-30 兜底): flip every subtask still
   * `in_progress` to `finalStatus`. Called when the run reaches a terminal
   * state (success → completed, error → failed) so no card can keep spinning
   * after the run is over, regardless of whether 修法 A's per-task flush fired.
   */
  finalizeRunning: (finalStatus: "completed" | "failed") => void;
}

export const SubtaskContext = createContext<SubtaskContextValue>({
  tasks: {},
  setTasks: () => {
    /* noop */
  },
  finalizeRunning: () => {
    /* noop */
  },
});

export function SubtasksProvider({ children }: { children: React.ReactNode }) {
  const [tasks, setTasks] = useState<Record<string, Subtask>>({});
  // Function-style update so this reads the freshest tasks at call time
  // (onFinish/onError fire from useStream callbacks and may hold a stale
  // `tasks` closure). Only mutates entries that are genuinely in_progress,
  // so already-terminal cards are untouched.
  const finalizeRunning = useCallback((finalStatus: "completed" | "failed") => {
    setTasks((prev) => {
      let changed = false;
      const next: Record<string, Subtask> = {};
      for (const [id, task] of Object.entries(prev)) {
        if (task.status === "in_progress") {
          next[id] = { ...task, status: finalStatus };
          changed = true;
        } else {
          next[id] = task;
        }
      }
      return changed ? next : prev;
    });
  }, []);
  return (
    <SubtaskContext.Provider value={{ tasks, setTasks, finalizeRunning }}>
      {children}
    </SubtaskContext.Provider>
  );
}

export function useSubtaskContext() {
  const context = useContext(SubtaskContext);
  if (context === undefined) {
    throw new Error(
      "useSubtaskContext must be used within a SubtaskContext.Provider",
    );
  }
  return context;
}

export function useSubtask(id: string) {
  const { tasks } = useSubtaskContext();
  return tasks[id];
}

export function useUpdateSubtask() {
  const { tasks, setTasks } = useSubtaskContext();
  const updateSubtask = useCallback(
    (update: Partial<Subtask> & { id: string }) => {
      const existing = tasks[update.id];
      const incoming = update.latestMessage;

      // SSE path: a new AIMessage arrived via task_running. Append to the
      // accumulated messages array (dedup on id) and commit via setTasks —
      // SubtaskCard needs to re-render to show streaming progress.
      // setTasks is safe here: the SSE handler runs outside render.
      if (incoming) {
        const prevMessages = existing?.messages ?? [];
        const existingIndex =
          incoming.id != null
            ? prevMessages.findIndex((m) => m.id === incoming.id)
            : -1;
        let nextMessages: AIMessage[];
        if (existingIndex >= 0) {
          // Same id, updated content (streaming chunk). Replace in place so
          // partial text/tool_call state reflects the latest server-side
          // view instead of getting frozen at the first chunk.
          nextMessages = prevMessages.slice();
          nextMessages[existingIndex] = incoming;
        } else {
          nextMessages = [...prevMessages, incoming];
        }
        tasks[update.id] = {
          ...(existing ?? ({} as Subtask)),
          ...update,
          messages: nextMessages,
          latestMessage: nextMessages[nextMessages.length - 1],
        } as Subtask;
        setTasks({ ...tasks });
        return;
      }

      // All other paths — render-time metadata sync from MessageList walking
      // thread.messages, AND SSE task_completed/task_failed events that don't
      // carry a latestMessage — are committed in-place WITHOUT setTasks.
      //
      // Why no setTasks for terminal status events? Because the matching
      // ToolMessage ("Task Succeeded. Result: …") has either already arrived
      // in thread.messages or is about to. When MessageList re-renders for
      // that message change, this same code path runs with status:"completed"
      // and the in-place mutation makes the card render its terminal state.
      //
      // setTasks here would warn "Cannot update SubtasksProvider while
      // rendering MessageList" because MessageList calls updateSubtask
      // synchronously during render to mirror tool_call metadata into the
      // subtask store. (See upstream deerflow useUpdateSubtask — same shape.)
      const prevStatus = existing?.status;
      tasks[update.id] = {
        ...(existing ?? ({ messages: [] } as unknown as Subtask)),
        ...update,
        messages: existing?.messages ?? [],
      } as Subtask;

      // 修法 A (spec 2026-06-30): the in-place mutation above relies on "a
      // LATER MessageList re-render (triggered by some other message change)
      // flushing this terminal state to the UI". That assumption breaks for
      // the LAST pipeline step (e.g. report-writer): its "Task Succeeded"
      // ToolMessage arrives, the run is already success, the stream stops,
      // and no further message change ever triggers that bail-out re-render
      // → the card keeps spinning `animate-spin` forever.
      //
      // Fix: when status makes a REAL transition into a terminal state
      // (non-terminal → completed/failed), schedule ONE deferred setTasks so
      // the terminal state reaches subscribers without relying on a later
      // re-render. queueMicrotask runs after the current render commits, so
      // it does NOT trip the "Cannot update while rendering" warning that
      // the in-place branch exists to avoid. Idempotent: a second walk with
      // the same terminal status is not a transition, so it schedules nothing.
      const nextStatus = update.status ?? prevStatus;
      const isTerminalTransition =
        !!update.status &&
        (nextStatus === "completed" || nextStatus === "failed") &&
        prevStatus !== "completed" &&
        prevStatus !== "failed";
      if (isTerminalTransition) {
        const snapshot = tasks;
        queueMicrotask(() => {
          setTasks({ ...snapshot });
        });
      }
    },
    [tasks, setTasks],
  );
  return updateSubtask;
}

/** Accessor for the deterministic run-terminal fallback (spec 2026-06-30 兜底). */
export function useFinalizeRunning() {
  const { finalizeRunning } = useSubtaskContext();
  return finalizeRunning;
}
