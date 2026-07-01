"use client";

import {
  AlertTriangleIcon,
  HelpCircleIcon,
  LightbulbIcon,
  PauseIcon,
  ShieldAlertIcon,
  type LucideIcon,
} from "lucide-react";

import { useI18n } from "@/core/i18n/hooks";
import { cn } from "@/lib/utils";

import { statusBarClass, statusTextClass, type Status } from "../kit/status-badge";

import { ClarificationOptions } from "./clarification-options";
import { MarkdownContent } from "./markdown-content";

/**
 * DecisionCard —— 把 `ask_clarification` 从「消息流里一段不起眼的 markdown + 按钮」
 * 升级成**显眼的决策卡**（spec 2026-06-24-frontend-phase0-5）。
 *
 * 视觉结构（spec §3.1，日式克制）：
 *   左 accent bar（状态色细竖条，非整卡变色）
 *   ├ 标题：图标 + 状态色文字（color-not-only 三件套）
 *   ├ 决策依据（context）：muted 块 + 「为什么问」前缀（缺失不显）
 *   ├ question：主文案（markdown）
 *   └ ClarificationOptions：主/次 + 键盘 1-9 + 已答闭环
 *
 * 设计纪律：
 * - **不改后端 / 不改「点选项=发消息」机制**（spec §二非目标 / §五工程纪律）。本组件只
 *   渲染，`onSelect` 透传给 `ClarificationOptions`（最终转发为下条 user message）。
 * - **waiting 信号与进度轨同源**：本组件不重算工作流阶段（SSOT 在 `core/workflow/`），
 *   仅据「这条 clarification 是否已被回答」决定 accent 颜色 / 脉动 / 选项禁用。
 * - accent 色复用 spec#1 的 `--color-status-*`（danger/warning/info/success）+ spec#1
 *   曲线 / `animate-pulse-warm`（与 spec#4 进度轨 waiting 节点同款脉动）。
 * - reduced-motion：脉动在 `prefers-reduced-motion` 下降级为静态（由 globals.css 的
 *   `animate-pulse-warm` 媒体查询负责，本组件不另写）。
 */

/**
 * clarification_type → 视觉调性映射（spec §3.5）。
 *
 * 状态色（accent bar / 标题文字 / 柔和底色）统一由 kit 的 `Status` SSOT 派生
 * （`statusBarClass` / `statusTextClass`），不在本卡里手写 `bg-status-*` 字符串。
 * 本卡只保留自己的 type-specific 图标（ShieldAlertIcon 等，非 kit `STATUS_ICON`），
 * 因为各 clarification_type 的语义图标不同（spec D2 Task5：decision-card keeps
 * its own `icon` field; only the color goes through kit）。
 */
interface Tone {
  /** kit 状态 SSOT —— accent / 标题色 / 柔和底色都从它派生。 */
  status: Status;
  /** lucide 图标（color-not-only 三件套之一；type-specific，非 kit SSOT）。 */
  icon: LucideIcon;
  /** 是否脉动（suggestion 弱信号不脉动，spec §3.5）。 */
  pulse: boolean;
  /** risk_confirmation 用更强标题，其余统一「分析已暂停」。 */
  titleKey: "cardRiskTitle" | "cardPausedTitle";
}

function toneFor(type: string | undefined): Tone {
  switch (type) {
    case "risk_confirmation":
      return { status: "danger", icon: ShieldAlertIcon, pulse: true, titleKey: "cardRiskTitle" };
    case "suggestion":
      return { status: "info", icon: LightbulbIcon, pulse: false, titleKey: "cardPausedTitle" };
    case "approach_choice":
      return { status: "warning", icon: HelpCircleIcon, pulse: true, titleKey: "cardPausedTitle" };
    case "missing_info":
    case "ambiguous_requirement":
    default:
      return { status: "warning", icon: AlertTriangleIcon, pulse: true, titleKey: "cardPausedTitle" };
  }
}

/**
 * @param question      反问正文（已 strip 掉编号选项列表的 markdown）。
 * @param context       决策依据（agent 为什么问 / 基于哪些列）。缺失则不显该块。
 * @param options       ask_clarification 的选项（可空 → 仅渲染自定义输入提示）。
 * @param answeredIndex 已选选项下标（≥0 表已答）；null 表仍在等待。
 * @param onSelect      点选项 / 按数字键的回调（转发为下条 user message）。
 */
export function DecisionCard({
  question,
  context,
  clarificationType,
  options,
  answeredIndex,
  onSelect,
  isLoading = false,
  threadId,
  className,
}: {
  question: string;
  context?: string;
  clarificationType?: string;
  options: string[] | undefined;
  answeredIndex: number | null;
  onSelect: (option: string) => void;
  isLoading?: boolean;
  threadId?: string;
  className?: string;
}) {
  const { t } = useI18n();
  const tone = toneFor(clarificationType);
  const answered = answeredIndex !== null;
  const Icon = answered ? PauseIcon : tone.icon;
  // 等待中才脉动；已答 / suggestion（弱信号）不脉动。
  const pulse = !answered && tone.pulse;

  // 状态色统一由 kit `Status` SSOT 派生：已答转 success（闭环反馈），否则用 tone.status。
  const status: Status = answered ? "success" : tone.status;
  // accent bar 是绝对定位覆盖条（card 用 padding 布局，非 flex），故保留本卡的定位
  // DOM（spec D2 Task5：decision-card 守自己的 accent bar 结构，test 契约不变），
  // 仅把颜色走 kit 的 statusBarClass/statusTextClass SSOT。
  const accentClass = statusBarClass(status);
  const titleColorClass = statusTextClass(status);
  // 柔和底色 = 状态 token 的 /10 透明度（D1 状态色 token 支持 alpha 变体）。
  const titleSoftClass = `${statusBarClass(status)}/10`;

  const contextText =
    typeof context === "string" ? context.trim() : "";

  return (
    <section
      className={cn(
        // 入场：spec#1 fade-in-up + ease-brand-out。
        "animate-fade-in-up relative overflow-hidden rounded-xl border bg-card/80",
        answered ? "border-status-success/40" : "border-border",
        className,
      )}
      // 不设 aria-label：下面的 <h4> 标题即为本 section 的可访问名（避免与可见标题
      // 文本重复，造成测试选择歧义 / 屏幕阅读器重复朗读）。role="group" 把内部选项
      // 聚合成一个导航单元。
      role="group"
    >
      {/* 左侧 accent bar —— 日式克制的「强信号」（细竖条，非整卡变色）。脉动与
          spec#4 进度轨 waiting 节点同款（animate-pulse-warm，reduced-motion 降级）。 */}
      <div
        className={cn(
          "absolute inset-y-0 left-0 w-1",
          accentClass,
          pulse && "animate-pulse-warm",
        )}
        aria-hidden="true"
      />

      <div className="flex flex-col gap-3 py-4 pr-4 pl-5">
        {/* 标题：图标 + 状态色文字（color-not-only 三件套）。 */}
        <div className="flex items-center gap-2">
          <span
            className={cn(
              "flex size-6 shrink-0 items-center justify-center rounded-full",
              titleSoftClass,
              titleColorClass,
            )}
            aria-hidden="true"
          >
            <Icon className="size-4" />
          </span>
          <h4
            className={cn(
              "text-sm font-semibold leading-tight",
              titleColorClass,
            )}
          >
            {answered ? t.clarification.answeredBadge : t.clarification[tone.titleKey]}
          </h4>
        </div>

        {/* 决策依据（context）—— muted 块 + 「为什么问」前缀。
            服务 memory feedback_identify_zone_info_not_persisted「带列依据反问」。
            缺失则不显。 */}
        {contextText && (
          <div
            className={cn(
              "rounded-md p-3 text-sm leading-relaxed",
              "bg-muted/60 text-muted-foreground",
            )}
          >
            <span className="text-foreground/70 mr-1 font-medium">
              {t.clarification.contextPrefix}：
            </span>
            {contextText}
          </div>
        )}

        {/* question —— 主文案，markdown 渲染（保留既有渲染行为，不退化纯文本）。 */}
        {question && (
          <MarkdownContent
            content={question}
            isLoading={isLoading}
            threadId={threadId}
          />
        )}

        {/* 选项 + 「或自定义输入」。等待中（非 loading、未答）启用键盘数字键。 */}
        <ClarificationOptions
          options={options}
          onSelect={onSelect}
          answeredIndex={answeredIndex}
          disabled={answered}
          keyboardActive={!answered}
        />
      </div>
    </section>
  );
}
