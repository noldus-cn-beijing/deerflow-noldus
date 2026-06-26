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
import { useMemo, useState } from "react";

import { type Translations } from "@/core/i18n";
import { useI18n } from "@/core/i18n/hooks";
import {
  CHART_STAGE_ID,
  stageDefOf,
  type CapabilityStageId,
  type StageStatus,
  useCapabilityPlan,
  useWorkflowStages,
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

/** 轨上一个可见阶段的渲染视图（def + 状态 + 锚点 + 在可见集里的序号）。 */
interface VisibleStage {
  id: CapabilityStageId;
  status: StageStatus;
  anchorMessageId?: string;
  ordinal: number;
}

export interface AnalysisRailProps {
  /** 传入 messages 供 hook 派生（与 RunTraceWidget 同款从 thread.messages 取）。 */
  messages: Message[];
  /** 窄屏紧凑态额外类名（供挂载点控制布局）。 */
  className?: string;
}

export function AnalysisRail({ messages, className }: AnalysisRailProps) {
  const { t } = useI18n();
  // 方案 B（spec 2026-06-26 §二/三）：显哪些阶段由 capability plan 动态决定；
  // 各阶段 active/done/waiting 状态仍由 7 阶段线性推导（useWorkflowStages）取。
  const plan = useCapabilityPlan({ messages, t });
  const stages = useWorkflowStages({ messages, t });
  const [compactOpen, setCompactOpen] = useState(false);

  // 把 plan（可见 id 集）与 stages（状态）合并成渲染视图。charts 不在 7 阶段数组里，
  // 其状态从 capability plan 是否触及推断：触及即 done（图已生成；chart-only run 主语义）。
  const visibleStages = useMemo<VisibleStage[]>(() => {
    return plan.map((entry, index) => {
      let status: StageStatus;
      let anchorMessageId = entry.anchorMessageId;
      if (entry.id === CHART_STAGE_ID) {
        // charts 是独立能力阶段，不在 deriveWorkflowStages 的 7 阶段里；触及即 done。
        status = "done";
      } else {
        const found = stages.find((s) => s.id === entry.id);
        status = found?.status ?? "done";
        anchorMessageId = anchorMessageId ?? found?.anchorMessageId;
      }
      return { id: entry.id, status, ...(anchorMessageId ? { anchorMessageId } : {}), ordinal: index + 1 };
    });
  }, [plan, stages]);

  // 无任何 pipeline 信号（知识问答 run）→ plan 空 → 不渲染轨（日式克制，不塞空轨）。
  if (visibleStages.length === 0) return null;

  const focus = visibleStages.find(
    (s) => s.status === "active" || s.status === "waiting" || s.status === "failed",
  );
  const focusOrdinal = focus ? focus.ordinal : 0;
  const totalCount = visibleStages.length;

  const rail = (
    <nav aria-label={t.workflowStages.navLabel} className={cn("w-full", className)}>
      {/* 宽屏横向 stepper（md+ 显示） */}
      <ol className="hidden items-start justify-between gap-1 md:flex">
        {visibleStages.map((stage) => {
          const def = stageDefOf(stage.id);
          return (
            <li key={stage.id} className="flex min-w-0 flex-1 items-start">
              <StageNode
                ordinal={stage.ordinal}
                name={t.workflowStages.names[def.nameKey as keyof typeof t.workflowStages.names] ?? def.nameKey}
                status={stage.status}
                hint={t.workflowStages.whatItDoes[def.nameKey as keyof typeof t.workflowStages.whatItDoes] ?? ""}
                statusLabel={statusLabel(stage.status, t)}
                waitingHint={t.workflowStages.waitingHint}
                icon={def.icon}
                colorToken={def.colorToken}
                showConnector={stage.ordinal > 1}
                anchorMessageId={stage.anchorMessageId}
                onScrollToAnchor={scrollToMessage}
              />
            </li>
          );
        })}
      </ol>

      {/* 窄屏紧凑态（<md 显示）：当前阶段 · 状态 + N/total + 展开看全部 */}
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
              <Dot status={focus.status} colorToken={stageDefOf(focus.id).colorToken} />
              <span className="font-medium text-foreground">
                {focus.ordinal}. {t.workflowStages.names[stageDefOf(focus.id).nameKey as keyof typeof t.workflowStages.names] ?? stageDefOf(focus.id).nameKey}
              </span>
              <span className="text-muted-foreground">· {compactStatusLabel(focus.status, t.workflowStages)}</span>
            </>
          ) : (
            <span className="text-muted-foreground">{t.workflowStages.navLabel}</span>
          )}
          <span className="text-muted-foreground/70 tabular-nums">
            {t.workflowStages.compactOf(focusOrdinal || 1, totalCount)}
          </span>
        </button>

        {compactOpen && (
          <ol className="mt-1 grid grid-cols-2 gap-x-2 gap-y-1 rounded-md bg-muted/20 p-2">
            {visibleStages.map((stage) => {
              const def = stageDefOf(stage.id);
              return (
                <li key={stage.id} className="flex min-w-0 items-start">
                  <StageNode
                    ordinal={stage.ordinal}
                    name={t.workflowStages.names[def.nameKey as keyof typeof t.workflowStages.names] ?? def.nameKey}
                    status={stage.status}
                    hint={t.workflowStages.whatItDoes[def.nameKey as keyof typeof t.workflowStages.whatItDoes] ?? ""}
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
