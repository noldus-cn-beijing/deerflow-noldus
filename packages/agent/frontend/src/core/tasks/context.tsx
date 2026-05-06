import type { AIMessage } from "@langchain/langgraph-sdk";
import { createContext, useCallback, useContext, useState } from "react";

import type { Subtask } from "./types";

export interface SubtaskContextValue {
  tasks: Record<string, Subtask>;
  setTasks: (tasks: Record<string, Subtask>) => void;
}

export const SubtaskContext = createContext<SubtaskContextValue>({
  tasks: {},
  setTasks: () => {
    /* noop */
  },
});

export function SubtasksProvider({ children }: { children: React.ReactNode }) {
  const [tasks, setTasks] = useState<Record<string, Subtask>>({});
  return (
    <SubtaskContext.Provider value={{ tasks, setTasks }}>
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

      // Terminal-status SSE path: task_completed / task_failed arrive without
      // a new AIMessage but must trigger a re-render so the card flips from
      // in_progress to its final state. Guarded by status-changed check so
      // the render-time path below stays safe — MessageList calls this with
      // status: "completed" on every render after the tool message arrives,
      // and setTasks during a parent's render would warn/loop.
      if (
        (update.status === "completed" || update.status === "failed") &&
        existing?.status !== update.status
      ) {
        tasks[update.id] = {
          ...(existing ?? ({ messages: [] } as unknown as Subtask)),
          ...update,
          messages: existing?.messages ?? [],
        } as Subtask;
        setTasks({ ...tasks });
        return;
      }

      // Render-time path: MessageList walks thread.messages during render and
      // calls updateSubtask to reflect task_init / task_completed metadata
      // derived from lead-side tool_calls. Mutating in place (without
      // setTasks) avoids a render-loop; the next SSE event will propagate
      // the merged state to subscribers.
      tasks[update.id] = {
        ...(existing ?? ({ messages: [] } as unknown as Subtask)),
        ...update,
        messages: existing?.messages ?? [],
      } as Subtask;
    },
    [tasks, setTasks],
  );
  return updateSubtask;
}
