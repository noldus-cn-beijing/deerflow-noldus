"use client";

/**
 * AnalysisRail —— chat 区顶部常驻的 7 阶段工作流进度轨（spec#4 §3.2）。
 *
 * 形态：
 * - 宽屏：顶部横向 stepper（7 节点 + 1px 极细连线），sticky。
 * - 窄屏（<768px）：退化为「当前阶段 · 状态 + N/7」紧凑指示，点开看全部（content-priority /
 *   adaptive-navigation）。
 *
 * 数据**同源 spec#2**：useWorkflowStages 从 useRunTrace 派生（spec §3.1 / §六风险2），零后端改动。
 * 纯只读展示 + 锚点滚动（点阶段 → 平滑滚到消息流对应位置，spec §3.3）。
 *
 * 与 spec#2 运行轨迹抽屉并存：进度轨 = 宏观 7 阶段地图（常驻轻量），轨迹 = 微观每步流水（按需抽屉）。
 */

import { type Message } from "@langchain/langgraph-sdk";
import { useState } from "react";

import { type Translations } from "@/core/i18n";
import { useI18n } from "@/core/i18n/hooks";
import {
  type StageStatus,
  useWorkflowFocus,
  useWorkflowStages,
  WORKFLOW_STAGES,
} from "@/core/workflow";
import { cn } from "@/lib/utils";

import { compactStatusLabel, StageNode } from "./stage-node";

/** 数据属性：消息流 group wrapper 上挂的锚点（见 message-list.tsx）。 */
const MESSAGE_ANCHOR_ATTR = "data-message-id";

/** 滚动到消息流里对应 messageId 的元素（spec §3.3，smooth / reduced-motion auto）。 */
function scrollToMessage(messageId: string) {
  const reduced =
    typeof window !== "undefined" &&
    window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  const el = document.querySelector<HTMLElement>(`[${MESSAGE_ANCHOR_ATTR}="${CSS.escape(messageId)}"]`);
  el?.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "center" });
}

export interface AnalysisRailProps {
  /** 传入 messages 供 hook 派生（与 RunTraceWidget 同款从 thread.messages 取）。 */
  messages: Message[];
  /** 窄屏紧凑态额外类名（供挂载点控制布局）。 */
  className?: string;
}

export function AnalysisRail({ messages, className }: AnalysisRailProps) {
  const { t } = useI18n();
  const stages = useWorkflowStages({ messages, t });
  const focus = useWorkflowFocus(stages);
  const [compactOpen, setCompactOpen] = useState(false);

  // 无任何进展（空 thread）时不渲染——避免知识问答线程占垂直空间（spec §3.2 adaptive-navigation）。
  const hasProgress = stages.some((s) => s.status !== "pending");
  if (!hasProgress) return null;

  const focusOrdinal = focus ? stages.indexOf(focus) + 1 : 0;

  const rail = (
    <nav aria-label={t.workflowStages.navLabel} className={cn("w-full", className)}>
      {/* 宽屏横向 stepper（md+ 显示） */}
      <ol className="hidden items-start justify-between gap-1 md:flex">
        {stages.map((stage, index) => {
          const def = WORKFLOW_STAGES[index]!;
          return (
            <li key={stage.id} className="flex min-w-0 flex-1 items-start">
              <StageNode
                ordinal={index + 1}
                name={t.workflowStages.names[def.nameKey]}
                status={stage.status}
                hint={t.workflowStages.whatItDoes[def.nameKey]}
                statusLabel={statusLabel(stage.status, t)}
                waitingHint={t.workflowStages.waitingHint}
                icon={def.icon}
                colorToken={def.colorToken}
                showConnector={index > 0}
                anchorMessageId={stage.anchorMessageId}
                onScrollToAnchor={scrollToMessage}
              />
            </li>
          );
        })}
      </ol>

      {/* 窄屏紧凑态（<md 显示）：当前阶段 · 状态 + N/7 + 展开看全部 */}
      <div className="flex flex-col gap-1 md:hidden">
        <button
          type="button"
          onClick={() => setCompactOpen((v) => !v)}
          aria-expanded={compactOpen}
          aria-label={
            compactOpen ? t.workflowStages.compactCollapse : t.workflowStages.compactExpand
          }
          className="flex items-center gap-2 self-start rounded-md px-1 py-0.5 text-xs outline-none transition-colors duration-base ease-brand-out hover:bg-muted/40 focus-visible:ring-2 focus-visible:ring-status-success/40"
        >
          {focus ? (
            <>
              <Dot status={focus.status} colorToken={WORKFLOW_STAGES[focusOrdinal - 1]!.colorToken} />
              <span className="font-medium text-foreground">
                {focusOrdinal}. {t.workflowStages.names[WORKFLOW_STAGES[focusOrdinal - 1]!.nameKey]}
              </span>
              <span className="text-muted-foreground">· {compactStatusLabel(focus.status, t.workflowStages)}</span>
            </>
          ) : (
            <span className="text-muted-foreground">{t.workflowStages.navLabel}</span>
          )}
          <span className="text-muted-foreground/70 tabular-nums">
            {t.workflowStages.compactOf(focusOrdinal || 1)}
          </span>
        </button>

        {compactOpen && (
          <ol className="mt-1 grid grid-cols-2 gap-x-2 gap-y-1 rounded-md bg-muted/20 p-2">
            {stages.map((stage, index) => {
              const def = WORKFLOW_STAGES[index]!;
              return (
                <li key={stage.id} className="flex min-w-0 items-start">
                  <StageNode
                    ordinal={index + 1}
                    name={t.workflowStages.names[def.nameKey]}
                    status={stage.status}
                    hint={t.workflowStages.whatItDoes[def.nameKey]}
                    statusLabel={statusLabel(stage.status, t)}
                    waitingHint={t.workflowStages.waitingHint}
                    icon={def.icon}
                    colorToken={def.colorToken}
                    showConnector={false}
                    anchorMessageId={stage.anchorMessageId}
                    onScrollToAnchor={(id) => {
                      setCompactOpen(false);
                      scrollToMessage(id);
                    }}
                  />
                </li>
              );
            })}
          </ol>
        )}
      </div>
    </nav>
  );

  return rail;
}

/** 进度轨用的状态文字标签（与 stage-node 的 compactStatusLabel 同源，宽屏 stepper 用）。 */
function statusLabel(status: StageStatus, t: Translations): string {
  const w = t.workflowStages;
  switch (status) {
    case "done":
      return w.statusDone;
    case "active":
      return w.statusActive;
    case "waiting":
      return w.statusWaiting;
    case "warning":
      return w.statusWarning;
    case "failed":
      return w.statusFailed;
    default:
      return w.statusPending;
  }
}

/**
 * 紧凑态小圆点（color-not-only：色 + 紧跟的文字状态）。
 * done 用阶段色（inline style）；active/waiting 用状态色 + 脉动。
 */
function Dot({ status, colorToken }: { status: StageStatus; colorToken: string }) {
  const className = cn(
    "size-2 shrink-0 rounded-full",
    status === "active" && "bg-status-success animate-pulse-soft",
    status === "waiting" && "bg-status-warning animate-pulse-warm",
    status === "warning" && "bg-status-warning",
    status === "failed" && "bg-status-danger",
    status === "pending" && "bg-muted-foreground/40",
  );
  return (
    <span
      className={className}
      style={status === "done" ? { backgroundColor: colorToken } : undefined}
      aria-hidden="true"
    />
  );
}
