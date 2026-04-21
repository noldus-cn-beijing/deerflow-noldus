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
        tasks[update.id] = {
          ...(existing ?? ({} as Subtask)),
          ...update,
          messages: nextMessages,
          latestMessage: nextMessages[nextMessages.length - 1],
        } as Subtask;
        setTasks({ ...tasks });
        return;
      }

      tasks[update.id] = {
        ...(existing ?? ({ messages: [] } as unknown as Subtask)),
        ...update,
        messages: existing?.messages ?? [],
      } as Subtask;
      setTasks({ ...tasks });
    },
    [tasks, setTasks],
  );
  return updateSubtask;
}
