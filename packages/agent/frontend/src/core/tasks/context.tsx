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
