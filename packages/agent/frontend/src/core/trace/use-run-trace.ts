"use client";

/**
 * useRunTrace —— 运行轨迹的 React 聚合 hook（spec §3.1）。
 *
 * 它是 buildRunTrace（纯函数）的薄壳：useMemo 从 thread.messages + SubtaskContext
 * 派生 TraceEvent[]。**只读**——不写 state、不碰 submit/merge/dedupe（spec §六红线）。
 *
 * 数据源全部现成（spec §一）：thread.messages（lead 层事件）+ useSubtaskContext()。
 * 不改 core/threads/hooks.ts 的事件处理。
 *
 * t（i18n 文案）由调用方从 useI18n() 取后传入——hook 自身不引 useI18n，保持 core/trace
 * 与 i18n/hooks 的依赖单向（i18n 不依赖 trace，避免潜在环）。单测直接测 buildRunTrace
 * 纯函数，本 hook 只做 useMemo 包裹，无需 React 测试环境。
 */

import type { Message } from "@langchain/langgraph-sdk";
import { useMemo } from "react";

import { useSubtaskContext } from "@/core/tasks/context";

import { buildRunTrace } from "./build-run-trace";
import type { RunTraceTranslations, TraceEvent, TraceEventStatus } from "./types";

export interface UseRunTraceArgs {
  messages: Message[];
  t: RunTraceTranslations;
}

export function useRunTrace({ messages, t }: UseRunTraceArgs): TraceEvent[] {
  const { tasks } = useSubtaskContext();
  return useMemo(
    () => buildRunTrace({ messages, subtasks: tasks }, t),
    [messages, tasks, t],
  );
}

/**
 * 入口按钮徽章用的概要：进行中步数、总步数、是否有错。
 * 同样是只读派生（spec §3.4 入口与默认态）。
 */
export interface RunTraceSummary {
  total: number;
  running: number;
  hasError: boolean;
  /** 入口按钮态：error（有 failed/gate-red）/ running（有进行中）/ idle */
  triggerState: "error" | "running" | "idle";
}

export function summarizeTrace(events: TraceEvent[]): RunTraceSummary {
  const total = events.length;
  const running = events.filter((e) => e.status === "running" || e.status === "waiting").length;
  const hasError = events.some((e) => e.status === "failed");
  const triggerState: RunTraceSummary["triggerState"] = hasError
    ? "error"
    : running > 0
      ? "running"
      : "idle";
  return { total, running, hasError, triggerState };
}

export function useRunTraceSummary(events: TraceEvent[]): RunTraceSummary {
  return useMemo(() => summarizeTrace(events), [events]);
}

export type { TraceEventStatus };
