"use client";

/**
 * TraceEventItem —— 运行轨迹时间线上的单个节点（spec §3.2）。
 *
 * 视觉：极细竖线（border-l 1px）+ 小圆点节点（日式克制，不堆方框）。
 * 状态：色 + 图标 + 文字三件套（color-not-only），用 spec1 的 --color-status-*。
 *   running=brand 脉动 / ok=success / warning=warning / failed=danger / waiting=warning 脉动。
 *
 * progressive-disclosure：
 *   - dispatch 节点可展开，露出该 subagent 内部 tool/reasoning 子步骤（缩进二级时间线）。
 *   - gate 节点可展开，见 DataQualityWarning 明细（severity/code/message/blocks_downstream）。
 *
 * 入场动效用 spec1 的 --animate-fade-in-up（ease-brand-out，从下方进 = hierarchy-motion）。
 * stagger 由父列表按 index 给 animation-delay（连续节点逐个 30-50ms 入场，不齐刷）。
 */

import {
  AlertTriangleIcon,
  CheckCircle2Icon,
  ChevronDownIcon,
  CircleDashedIcon,
  FileCheckIcon,
  FlaskConicalIcon,
  ListTreeIcon,
  MessageCircleQuestionIcon,
  MinusCircleIcon,
  TerminalIcon,
  WrenchIcon,
  XCircleIcon,
  type LucideIcon,
} from "lucide-react";
import { useState } from "react";

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { useI18n } from "@/core/i18n/hooks";
import type { TraceEvent, TraceEventKind, TraceEventStatus } from "@/core/trace";
import { cn } from "@/lib/utils";

const KIND_ICON: Record<TraceEventKind, LucideIcon> = {
  paradigm: FlaskConicalIcon,
  dispatch: ListTreeIcon,
  tool: TerminalIcon,
  gate: WrenchIcon,
  clarification: MessageCircleQuestionIcon,
  artifact: FileCheckIcon,
};

interface StatusVisual {
  /** 圆点 + 图标的颜色类（text-status-*，spec1 token） */
  color: string;
  /** 圆点底色类（bg-status-*，必须字面量给 Tailwind JIT） */
  dot: string;
  /** 软底色（bg-status-*-soft）—— 给当前行 hover/活跃态 */
  soft: string;
  /** 状态图标 */
  icon: LucideIcon;
  /** 是否脉动（running / waiting） */
  pulse: boolean;
}

function statusVisual(status: TraceEventStatus): StatusVisual {
  switch (status) {
    case "running":
      return {
        color: "text-brand",
        dot: "bg-brand",
        soft: "bg-brand/10",
        icon: CircleDashedIcon,
        pulse: true,
      };
    case "ok":
      return {
        color: "text-status-success",
        dot: "bg-status-success",
        soft: "bg-status-success-soft",
        icon: CheckCircle2Icon,
        pulse: false,
      };
    case "warning":
      return {
        color: "text-status-warning",
        dot: "bg-status-warning",
        soft: "bg-status-warning-soft",
        icon: AlertTriangleIcon,
        pulse: false,
      };
    case "failed":
      return {
        color: "text-status-danger",
        dot: "bg-status-danger",
        soft: "bg-status-danger-soft",
        icon: XCircleIcon,
        pulse: false,
      };
    case "waiting":
      return {
        color: "text-status-warning",
        dot: "bg-status-warning",
        soft: "bg-status-warning-soft",
        icon: MinusCircleIcon,
        pulse: true,
      };
  }
}

export function TraceEventItem({
  event,
  index,
  isLast,
}: {
  event: TraceEvent;
  /** 在父列表中的位置，用于 stagger 入场延迟 */
  index: number;
  isLast: boolean;
}) {
  const { t } = useI18n();
  const sv = statusVisual(event.status);
  const KindIcon = KIND_ICON[event.kind];
  const StatusIcon = sv.icon;
  const kindLabel = KIND_LABEL[event.kind](t.runTrace);
  const statusLabel = STATUS_LABEL[event.status](t.runTrace);

  const canExpand =
    (event.kind === "dispatch" && (event.subEvents?.length ?? 0) > 0) ||
    (event.kind === "gate" && (event.detail?.kind === "gate" && event.detail.warnings.length > 0));
  const [open, setOpen] = useState(event.kind === "dispatch" ? false : false);

  const delay = Math.min(index, 12) * 0.04; // stagger 40ms，封顶避免长列表等太久

  return (
    <Collapsible open={open} onOpenChange={setOpen} asChild>
      <li
        className={cn(
          "relative pl-5",
          // 极细竖线（border-l 1px，不是粗轴）—— 仅在非末节点拉到下一个节点
          !isLast && "before:absolute before:top-3 before:bottom-0 before:left-[3px] before:w-px before:bg-border",
        )}
        style={{ animationDelay: `${delay}s` }}
      >
        {/* 小圆点节点（彩色，覆盖在竖线上） */}
        <span
          className={cn(
            "absolute top-2 left-0 size-1.5 rounded-full ring-2 ring-background",
            sv.dot,
            sv.pulse && "animate-pulse-soft",
          )}
          aria-hidden="true"
        />
        <div
          className={cn(
            "animate-fade-in-up -ml-0.5 flex flex-col gap-0.5 rounded-md px-1.5 py-1 transition-colors duration-base ease-brand-out",
            open && sv.soft,
          )}
        >
          <div className="flex items-center gap-1.5">
            <KindIcon className={cn("size-3.5 shrink-0 text-muted-foreground")} />
            <span className="min-w-0 flex-1 truncate text-sm">{event.title}</span>
            {/* 状态三件套之图标 + 文字（色已由 sv.color 给图标；文字 muted 表 statusLabel） */}
            <StatusIcon className={cn("size-3.5 shrink-0", sv.color, sv.pulse && "animate-pulse-soft")} />
            <span className={cn("shrink-0 text-xs", sv.color)}>{statusLabel}</span>
            {canExpand && (
              <CollapsibleTrigger
                className={cn(
                  "shrink-0 rounded p-0.5 text-muted-foreground transition-transform duration-base ease-brand-out hover:bg-accent",
                  open && "rotate-180",
                )}
                aria-label={
                  event.kind === "dispatch" ? t.runTrace.showSubSteps : t.runTrace.showGateDetail
                }
                aria-expanded={open}
              >
                <ChevronDownIcon className="size-3.5" />
              </CollapsibleTrigger>
            )}
          </div>
          {/* kind 类别标签（色 + 图标 + 文字三件套的「文字」补充，screen reader 友好） */}
          <span className="sr-only">{kindLabel}</span>

          {/* 一行 inline 明细：范式 / 产物文件名 / clarification 选项（不展开就瞥见） */}
          <InlineDetail event={event} />

          {canExpand && (
            <CollapsibleContent className="mt-1">
              {event.kind === "dispatch" && event.subEvents && (
                <ul className="ml-1 border-l border-border/60 pl-3">
                  {event.subEvents.map((sub, i) => (
                    <TraceEventItem
                      key={sub.id}
                      event={sub}
                      index={i}
                      isLast={i === event.subEvents!.length - 1}
                    />
                  ))}
                </ul>
              )}
              {event.kind === "gate" && event.detail?.kind === "gate" && (
                <GateDetail warnings={event.detail.warnings} />
              )}
            </CollapsibleContent>
          )}
        </div>
      </li>
    </Collapsible>
  );
}

function InlineDetail({ event }: { event: TraceEvent }) {
  const detail = event.detail;
  if (!detail) return null;
  if (detail.kind === "paradigm") {
    const parts = [detail.paradigm, detail.ev19Template].filter(Boolean);
    if (parts.length === 0) return null;
    return (
      <div className="text-muted-foreground pl-5 text-xs">{parts.join(" · ")}</div>
    );
  }
  if (detail.kind === "artifact") {
    if (detail.filepaths.length === 0) return null;
    return (
      <div className="text-muted-foreground truncate pl-5 text-xs" title={detail.filepaths.join(", ")}>
        {detail.filepaths.join(", ")}
      </div>
    );
  }
  if (detail.kind === "clarification" && detail.options && detail.options.length > 0) {
    return (
      <div className="text-muted-foreground truncate pl-5 text-xs">
        {detail.options.join(" / ")}
      </div>
    );
  }
  return null;
}

function GateDetail({
  warnings,
}: {
  warnings: NonNullable<Extract<TraceEvent["detail"], { kind: "gate" }>["warnings"]>;
}) {
  const { t } = useI18n();
  return (
    <ul className="space-y-1">
      {warnings.map((w, i) => {
        const tone =
          w.severity === "critical"
            ? "text-status-danger"
            : w.severity === "warning"
              ? "text-status-warning"
              : "text-status-info";
        return (
          <li key={`${w.code}-${i}`} className="text-xs">
            <span className={cn("font-medium", tone)}>
              [{w.severity} · {w.code}]
              {w.blocks_downstream ? " · blocks" : ""}
            </span>{" "}
            <span className="text-muted-foreground">{w.message}</span>
          </li>
        );
      })}
      <span className="sr-only">{t.runTrace.gateTitle}</span>
    </ul>
  );
}

const KIND_LABEL: Record<TraceEventKind, (r: ReturnType<typeof useI18n>["t"]["runTrace"]) => string> = {
  paradigm: (r) => r.kindParadigm,
  dispatch: (r) => r.kindDispatch,
  tool: (r) => r.kindTool,
  gate: (r) => r.kindGate,
  clarification: (r) => r.kindClarification,
  artifact: (r) => r.kindArtifact,
};

const STATUS_LABEL: Record<TraceEventStatus, (r: ReturnType<typeof useI18n>["t"]["runTrace"]) => string> = {
  running: (r) => r.statusRunning,
  ok: (r) => r.statusOk,
  warning: (r) => r.statusWarning,
  failed: (r) => r.statusFailed,
  waiting: (r) => r.statusWaiting,
};
