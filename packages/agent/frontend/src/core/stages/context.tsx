"use client";

// A2 stage narration store — consumes A1 backend stage events (kind: stage_plan /
// stage_update) from the custom SSE track and exposes them to components.
//
// Design: separate from SubtaskContext because stage state is keyed by stage-name
// (不是 task_id), is driven purely by A1 backend events (never inferred), and its
// lifecycle is per-run (reset on new run start), not per-subtask.

import { createContext, useCallback, useContext, useState } from "react";

export interface StageEntry {
  status: "active" | "completed" | "pending";
  narration: string;
}

export interface StagePlan {
  // Ordered list of stage names from A1 stage_plan event.
  stages: string[];
  // Stages skipped this run (e.g. data-analyst when n<2).
  skipped: string[];
}

export interface StageContextValue {
  plan: StagePlan | null;
  // Map from stage name → current entry (only present for stages that have
  // received at least one stage_update event).
  entries: Record<string, StageEntry>;
  setPlan: (plan: StagePlan) => void;
  updateStage: (stage: string, status: "active" | "completed", narration: string) => void;
  resetStages: () => void;
}

export const StageContext = createContext<StageContextValue>({
  plan: null,
  entries: {},
  setPlan: () => {
    /* noop */
  },
  updateStage: () => {
    /* noop */
  },
  resetStages: () => {
    /* noop */
  },
});

export function StagesProvider({ children }: { children: React.ReactNode }) {
  const [plan, setPlanState] = useState<StagePlan | null>(null);
  const [entries, setEntries] = useState<Record<string, StageEntry>>({});

  const setPlan = useCallback((newPlan: StagePlan) => {
    setPlanState(newPlan);
    // Reset entries on new plan (new run).
    setEntries({});
  }, []);

  const updateStage = useCallback((stage: string, status: "active" | "completed", narration: string) => {
    setEntries((prev) => ({
      ...prev,
      [stage]: { status, narration },
    }));
  }, []);

  const resetStages = useCallback(() => {
    setPlanState(null);
    setEntries({});
  }, []);

  return (
    <StageContext.Provider value={{ plan, entries, setPlan, updateStage, resetStages }}>
      {children}
    </StageContext.Provider>
  );
}

export function useStageContext() {
  return useContext(StageContext);
}

export function useStages() {
  const { plan, entries } = useStageContext();
  if (!plan) return null;
  return plan.stages.map((name) => ({
    name,
    skipped: plan.skipped.includes(name),
    entry: entries[name] ?? null,
  }));
}

/**
 * A2: returns the narration of the currently-`active` stage (the one the
 * pipeline is working on right now), or null when no stage is active.
 *
 * subtask-card renders this in its `in_progress` label so the user sees the
 * backend's own narration ("正在计算 EPM 行为指标…") instead of a frontend
 * query-table translation. Source = A1 events only, never inferred.
 */
export function useActiveStageNarration(): string | null {
  const { plan, entries } = useStageContext();
  if (!plan) return null;
  for (const name of plan.stages) {
    const entry = entries[name];
    if (entry?.status === "active" && entry.narration) {
      return entry.narration;
    }
  }
  return null;
}

export function useSetPlan() {
  return useStageContext().setPlan;
}

export function useUpdateStage() {
  return useStageContext().updateStage;
}

export function useResetStages() {
  return useStageContext().resetStages;
}
