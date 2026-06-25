"use client";

/**
 * RunTracePanel —— 运行轨迹抽屉里的时间线主体（spec §3.2 / §四 Step 2）。
 *
 * 只读消费 useRunTrace 派生的 TraceEvent[]（spec §六：不改流式核心）。
 *
 * 性能（spec 五 a11y/性能）：
 *   - 单 run 节点 >50 → 折叠的子步骤懒渲染（Collapsible 关闭时不挂子树，已是 Radix 默认）。
 *   - 大量节点滚动流畅：外层 ScrollArea；本 spec 不强制虚拟化（virtualize-lists 的 >50 阈值
 *     指顶层节点，子步骤已折叠懒渲染即可满足，避免引入虚拟化库的复杂度）。
 *
 * 实时追加：新节点用 TraceEventItem 的 animate-fade-in-up + stagger 入场（spec1 曲线）。
 * 当前进行中的节点脉动（animate-pulse-soft），保证「实时进展可见，无 10s+ 空转 spinner」。
 */

import type { Message } from "@langchain/langgraph-sdk";
import { Loader2Icon } from "lucide-react";

import { ScrollArea } from "@/components/ui/scroll-area";
import { useI18n } from "@/core/i18n/hooks";
import { useRunTrace } from "@/core/trace";
import { cn } from "@/lib/utils";

import { TraceEventItem } from "./trace-event-item";

export function RunTracePanel({
  messages,
  className,
}: {
  messages: Message[];
  className?: string;
}) {
  const { t } = useI18n();
  const events = useRunTrace({ messages, t });

  if (events.length === 0) {
    return (
      <div className={cn("text-muted-foreground flex flex-col items-center gap-2 px-6 py-10 text-center text-sm", className)}>
        <Loader2Icon className="size-4 animate-spin opacity-40" />
        <span>{t.runTrace.empty}</span>
      </div>
    );
  }

  return (
    <ScrollArea className={cn("h-full", className)}>
      <ol className="flex flex-col gap-0.5 py-2 pr-2">
        {events.map((event, index) => (
          <TraceEventItem
            key={event.id}
            event={event}
            index={index}
            isLast={index === events.length - 1}
          />
        ))}
      </ol>
    </ScrollArea>
  );
}
