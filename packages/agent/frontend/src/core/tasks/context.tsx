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

      if (incoming) {
        const prevMessages = existing?.messages ?? [];
        const alreadyHas =
          incoming.id != null &&
          prevMessages.some((m) => m.id === incoming.id);
        const nextMessages = alreadyHas
          ? prevMessages
          : [...prevMessages, incoming];
        const next = {
          ...(existing ?? ({} as Subtask)),
          ...update,
          messages: nextMessages,
          latestMessage: nextMessages[nextMessages.length - 1],
        } as Subtask;
        if (existing && isSameSubtask(existing, next)) {
          return;
        }
        tasks[update.id] = next;
        setTasks({ ...tasks });
        return;
      }

      const next = {
        ...(existing ?? ({ messages: [] } as unknown as Subtask)),
        ...update,
        messages: existing?.messages ?? [],
      } as Subtask;
      if (existing && isSameSubtask(existing, next)) {
        return;
      }
      tasks[update.id] = next;
      setTasks({ ...tasks });
    },
    [tasks, setTasks],
  );
  return updateSubtask;
}

/**
 * Shallow-compare two Subtask objects so callers that re-fire the same update
 * (e.g. render-time `updateSubtask({ id, status: "in_progress" })`) don't
 * trigger a Context setState storm. `messages` is compared by reference — the
 * update path above guarantees the array is reused when nothing new arrived.
 */
function isSameSubtask(a: Subtask, b: Subtask): boolean {
  return (
    a.id === b.id &&
    a.status === b.status &&
    a.subagent_type === b.subagent_type &&
    a.description === b.description &&
    a.prompt === b.prompt &&
    a.result === b.result &&
    a.error === b.error &&
    a.messages === b.messages &&
    a.latestMessage === b.latestMessage
  );
}
