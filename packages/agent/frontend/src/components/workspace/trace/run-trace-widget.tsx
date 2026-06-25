"use client";

/**
 * RunTraceWidget —— 运行轨迹入口按钮 + 抽屉的组合壳（spec §3.4 / §四 Step 3-4）。
 *
 * 把「入口按钮态」「抽屉开关」「useRunTrace 派生」收在一处，调用方只需传 messages。
 *
 * 入口默认态（spec §3.4）：
 *   - 默认关闭（不占空间，不打扰非技术研究员）。
 *   - 运行中：徽章「N 步进行中」。
 *   - 完成：徽章「N 步」。
 *   - 出错（任一 gate=红 / subagent failed）：按钮变 danger 色 + 轻脉动，引导点开看「卡在哪」。
 *
 * 与消息流里现有 SubtaskCard 并存（轨迹是第二视角，消息流不变）。
 * 与 artifacts 侧栏不互斥（overlay 抽屉，不进同一 ResizablePanelGroup）。
 */

import type { Message } from "@langchain/langgraph-sdk";
import { ListTreeIcon } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Tooltip } from "@/components/workspace/tooltip";
import { useI18n } from "@/core/i18n/hooks";
import { useRunTrace, useRunTraceSummary } from "@/core/trace";
import { cn } from "@/lib/utils";

import { RunTraceDrawer } from "./run-trace-drawer";

export function RunTraceWidget({ messages }: { messages: Message[] }) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const events = useRunTrace({ messages, t });
  const summary = useRunTraceSummary(events);

  // 没有 agent 行为且不在运行 → 不渲染入口（默认关、不占空间，spec §3.4）。
  // 一旦本次 run 产生过任何步骤（哪怕已完成），保留入口供回看。
  const hasEverHadSteps = summary.total > 0;

  const badgeText =
    summary.triggerState === "error"
      ? t.runTrace.hasError
      : summary.triggerState === "running"
        ? t.runTrace.runningSteps(summary.running)
        : t.runTrace.stepCount(summary.total);

  return (
    <>
      <Tooltip content={t.runTrace.triggerLabel}>
        <Button
          variant="ghost"
          aria-label={t.runTrace.triggerLabel}
          aria-expanded={open}
          aria-haspopup="dialog"
          onClick={() => setOpen((v) => !v)}
          className={cn(
            "text-muted-foreground hover:text-foreground gap-1.5 transition-colors duration-base ease-brand-out",
            // 出错态：danger 色 + 轻脉动（error-recovery：清晰恢复路径）
            summary.triggerState === "error" && "text-status-danger hover:text-status-danger",
            // 进行中且入口未开：轻脉动给一个钩子（不强推）
            summary.triggerState === "running" && !open && "animate-pulse-soft",
          )}
        >
          <ListTreeIcon className="size-4" />
          {hasEverHadSteps && (
            <span
              className={cn(
                "text-xs tabular-nums",
                summary.triggerState === "error" && "text-status-danger",
              )}
            >
              {badgeText}
            </span>
          )}
        </Button>
      </Tooltip>
      <RunTraceDrawer open={open} onOpenChange={setOpen} messages={messages} />
    </>
  );
}
