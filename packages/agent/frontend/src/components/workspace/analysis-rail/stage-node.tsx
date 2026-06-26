"use client";

/**
 * StageNode —— 进度轨上的单个阶段节点（spec#4 §3.2）。
 *
 * 视觉（日式克制 + spec#1 token）：
 * - 连接线 1px 极细（不是粗箭头）。
 * - 节点小圆点：done=✓ 阶段色实心 / active=品牌绿空心 + 描边呼吸（animate-pulse-soft）/
 *   waiting=琥珀脉冲 / pending=灰描边 / failed=红✗ / warning=黄。
 * - 当前阶段（active/waiting/failed）= 唯一视觉主角；其余降为背景层（低对比）。
 * - 「色 + 图标 + 文字」三件套（color-not-only）——色盲下靠图标/文字也能读。
 *
 * 交互（spec §3.3）：有 anchorMessageId 时可点击/Enter → scrollIntoView 平滑滚动到对应消息。
 * a11y：节点是 button（可 tab + Enter），当前焦点阶段 aria-current="step"，tooltip 显阶段说明。
 */

import { type LucideIcon, CheckIcon, XIcon } from "lucide-react";

import { Tooltip } from "@/components/workspace/tooltip";
import type { Translations } from "@/core/i18n";
import type { StageStatus } from "@/core/workflow";
import { cn } from "@/lib/utils";

export interface StageNodeProps {
  /** 阶段序号（1-based），用于紧凑态与 a11y。 */
  ordinal: number;
  name: string;
  status: StageStatus;
  /** 该阶段说明（tooltip）。 */
  hint: string;
  /** 状态文字（色+图标+文字三件套的文字部分）。 */
  statusLabel: string;
  /** 等待 HITL 微标文案。 */
  waitingHint: string;
  icon: LucideIcon;
  /** 阶段色 CSS var（spec#1 --color-stage-*）。 */
  colorToken: string;
  /** 是否画左侧连接线（首个阶段不画）。 */
  showConnector: boolean;
  /** 有锚点时可点击滚动；无锚点（pending 未触及）时为静态展示。 */
  anchorMessageId?: string;
  onScrollToAnchor?: (messageId: string) => void;
}

/** 状态 → 节点圈样式（color-not-only：色+图标+文字）。 */
function ringClass(status: StageStatus): string {
  switch (status) {
    case "done":
      return "border-transparent text-white";
    case "active":
      // 品牌绿空心 + 描边呼吸（spec#1 animate-pulse-soft）
      return "border-status-success/60 bg-status-success/10 text-status-success animate-pulse-soft";
    case "waiting":
      // 琥珀脉冲（HITL 等待）
      return "border-status-warning/60 bg-status-warning/10 text-status-warning animate-pulse-warm";
    case "warning":
      return "border-status-warning/50 bg-status-warning/15 text-status-warning";
    case "failed":
      return "border-status-danger/50 bg-status-danger/10 text-status-danger";
    case "pending":
    default:
      return "border-border bg-transparent text-muted-foreground/50";
  }
}

export function StageNode({
  ordinal,
  name,
  status,
  hint,
  statusLabel,
  waitingHint,
  icon: Icon,
  colorToken,
  showConnector,
  anchorMessageId,
  onScrollToAnchor,
}: StageNodeProps) {
  const isFocus = status === "active" || status === "waiting" || status === "failed";
  const isDone = status === "done";
  const isWaiting = status === "waiting";
  const clickable = Boolean(anchorMessageId && onScrollToAnchor);

  const ring = ringClass(status);

  const node = (
    <span className="relative flex shrink-0 flex-col items-center gap-1">
      <span
        className={cn(
          "flex size-6 items-center justify-center rounded-full border transition-colors duration-base ease-brand-out",
          ring,
          // 已完成节点：实心染阶段色（覆盖 ringClass 的 bg）
          isDone && "border-transparent text-white",
        )}
        style={
          isDone
            ? { backgroundColor: colorToken, borderColor: "transparent" }
            : undefined
        }
        aria-hidden="true"
      >
        {isDone ? (
          <CheckIcon className="size-3.5" />
        ) : status === "failed" ? (
          <XIcon className="size-3.5" />
        ) : (
          <Icon className="size-3.5" />
        )}
      </span>
      {isWaiting && (
        <span className="text-[10px] leading-none font-medium text-status-warning tabular-nums">
          {waitingHint}
        </span>
      )}
    </span>
  );

  const labelBlock = (
    <span className="flex flex-col items-center gap-0.5 text-center">
      <span
        className={cn(
          "text-xs leading-tight whitespace-nowrap transition-opacity duration-base ease-brand-out",
          isFocus
            ? "text-foreground font-medium opacity-100"
            : isDone
              ? "text-muted-foreground opacity-80"
              : "text-muted-foreground/50 opacity-70",
        )}
      >
        {name}
      </span>
      <span
        className={cn(
          "text-[10px] leading-none tabular-nums",
          isFocus ? "opacity-80" : "opacity-50",
          status === "active" && "text-status-success",
          status === "waiting" && "text-status-warning",
          status === "failed" && "text-status-danger",
          status === "warning" && "text-status-warning",
        )}
      >
        {statusLabel}
      </span>
    </span>
  );

  const tooltipContent = (
    <div className="flex flex-col gap-0.5">
      <span className="text-xs font-medium">
        {ordinal}. {name}
      </span>
      <span className="text-muted-foreground text-xs">{hint}</span>
      <span className="text-muted-foreground text-[10px]">{statusLabel}</span>
    </div>
  );

  return (
    <Tooltip content={tooltipContent}>
      <div
        className={cn(
          "group/stage flex items-start gap-1.5",
          // 当前焦点阶段唯一主角；其余降为背景层（spec §3.2 visual-hierarchy）
          !isFocus && "opacity-80",
        )}
      >
        {showConnector && (
          <span
            className={cn("mt-3 h-px flex-1", isDone ? "opacity-40" : "opacity-100 bg-border")}
            style={isDone ? { backgroundColor: colorToken } : undefined}
            aria-hidden="true"
          />
        )}
        {clickable ? (
          <button
            type="button"
            onClick={() => anchorMessageId && onScrollToAnchor?.(anchorMessageId)}
            aria-current={isFocus ? "step" : undefined}
            aria-label={`${ordinal}. ${name} — ${statusLabel}. ${hint}`}
            className="flex shrink-0 cursor-pointer flex-col items-center gap-1 rounded-md p-0.5 outline-none transition-opacity duration-base ease-brand-out hover:opacity-100 focus-visible:ring-2 focus-visible:ring-status-success/40"
          >
            {node}
            {labelBlock}
          </button>
        ) : (
          <span
            aria-current={isFocus ? "step" : undefined}
            aria-label={`${ordinal}. ${name} — ${statusLabel}. ${hint}`}
            className="flex shrink-0 cursor-default flex-col items-center gap-1 p-0.5"
          >
            {node}
            {labelBlock}
          </span>
        )}
      </div>
    </Tooltip>
  );
}

/** 状态 → 紧凑文字标签（窄屏「当前阶段 · 状态」用）。 */
export function compactStatusLabel(
  status: StageStatus,
  t: Translations["workflowStages"],
): string {
  switch (status) {
    case "done":
      return t.statusDone;
    case "active":
      return t.statusActive;
    case "waiting":
      return t.statusWaiting;
    case "warning":
      return t.statusWarning;
    case "failed":
      return t.statusFailed;
    default:
      return t.statusPending;
  }
}
