"use client";

/**
 * useWorkflowStages —— 进度轨 7 阶段状态的 React 派生 hook（spec#4 §3.1）。
 *
 * 它是 deriveWorkflowStages（纯函数）的薄壳：useMemo 从 useRunTrace 的 TraceEvent[] +
 * thread.messages 派生 StageState[7]。**只读**——不写 state、不碰 submit/merge/dedupe
 * （spec §六红线）。
 *
 * 数据**同源 spec#2**（spec §3.1 / §六风险2）：trace 事件来自 useRunTrace（单一解析），
 * 本 hook 不另写解析。clarification/upload 两条轻量信号直接从 thread.messages 取，与 trace
 * 不漂移。
 *
 * t（i18n 文案）由调用方传入——hook 自身不引 useI18n，保持 core/workflow 与 i18n 单向依赖
 * （与 useRunTrace 同款约束）。单测直接测 deriveWorkflowStages 纯函数，本 hook 只做 useMemo
 * 包裹，无需 React 测试环境。
 */

import type { Message } from "@langchain/langgraph-sdk";
import { useMemo } from "react";

import { useRunTrace } from "@/core/trace";
import type { RunTraceTranslations } from "@/core/trace";

import { deriveWorkflowStages } from "./derive-workflow-stages";
import type { StageState, StageStatus } from "./derive-workflow-stages";

export interface UseWorkflowStagesArgs {
  messages: Message[];
  t: RunTraceTranslations;
}

/** 派生 7 阶段状态（spec §3.1）。 */
export function useWorkflowStages({ messages, t }: UseWorkflowStagesArgs): StageState[] {
  // trace 事件来自 spec#2 的 useRunTrace（同源），不在本 hook 另写解析。
  const events = useRunTrace({ messages, t });
  return useMemo(
    () => deriveWorkflowStages(events, messages),
    [events, messages],
  );
}

/** 轨上当前焦点阶段（active/waiting/failed；全 pending 时为 undefined）。 */
export function useWorkflowFocus(stages: StageState[]): StageState | undefined {
  return useMemo(
    () => stages.find((s) => s.status === "active" || s.status === "waiting" || s.status === "failed"),
    [stages],
  );
}

export type { StageState, StageStatus };
