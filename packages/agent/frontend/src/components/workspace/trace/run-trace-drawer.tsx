"use client";

/**
 * RunTraceDrawer —— 运行轨迹 overlay 抽屉容器（spec §3.3 / §3.4 / §四 Step 3）。
 *
 * 关键决策（spec §3.3）：运行轨迹是「过程监督」（瞥一眼进度，瞬时），与 artifacts 侧栏
 * （「产物消费」，需要大空间、常驻）心智/时机不同。故 trace 走 overlay 抽屉，**不进
 * ResizablePanelGroup**——避免三栏 resize 复杂度 + 避免和 artifacts 互斥。
 *
 * 实现：Radix Dialog（非模态变体——modal={false} 在宽屏，让背景可点；窄屏带 scrim 模态）。
 *   - 从右侧滑入：translate-x，用 spec1 的 ease-brand-out + duration-slow（modal-motion）。
 *   - 宽度 clamp(360px, 33vw, 480px)。
 *   - scrim 仅窄屏 (<1024px) 出现（覆盖式）；宽屏非模态浮层（adaptive-navigation）。
 *   - Radix Dialog 自带 focus trap + ESC 关（modal-escape / escape-routes），满足 a11y。
 *
 * 状态：开关由父组件（RunTraceTrigger）持有，受控传入 open/onOpenChange。
 * 这里不存自己的 state——抽屉是无状态壳，数据全在 RunTracePanel 的 useRunTrace。
 */

import type { Message } from "@langchain/langgraph-sdk";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { XIcon } from "lucide-react";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { useI18n } from "@/core/i18n/hooks";
import { cn } from "@/lib/utils";

import { RunTracePanel } from "./run-trace-panel";

export interface RunTraceDrawerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  messages: Message[];
}

export function RunTraceDrawer({ open, onOpenChange, messages }: RunTraceDrawerProps) {
  const { t } = useI18n();
  // 窄屏 (<1024px) 才显示 scrim 并锁背景；宽屏作非模态浮层。
  // SSR 安全：初始按窄屏渲染（无 scrim 不影响功能），挂载后读 matchMedia 校正。
  const [isNarrow, setIsNarrow] = useState(true);
  useEffect(() => {
    const mq = window.matchMedia("(max-width: 1023px)");
    const update = () => setIsNarrow(mq.matches);
    update();
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);

  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange} modal={isNarrow}>
      {/* scrim：仅窄屏。宽屏无 overlay（非模态浮层，不锁背景）。 */}
      {isNarrow && (
        <DialogPrimitive.Overlay
          className={cn(
            "fixed inset-0 z-40 bg-black/30 backdrop-blur-[1px]",
            "data-[state=open]:animate-fade-in-up",
          )}
        />
      )}
      <DialogPrimitive.Content
        aria-describedby={undefined}
        className={cn(
          // fixed overlay，盖在 chat 之上，不进 ResizablePanelGroup
          "fixed top-0 right-0 bottom-0 z-50 flex h-full flex-col",
          "w-[clamp(360px,33vw,480px)] max-w-[92vw]",
          "border-l bg-elevated/95 backdrop-blur shadow-modal",
          // 从右侧滑入（spec1 ease-brand-out + duration-slow）
          "transition-transform duration-slow ease-brand-out",
          "data-[state=open]:translate-x-0 data-[state=closed]:translate-x-full",
          // outline 取代焦点环（Radix 默认 focus-visible）
          "outline-none",
        )}
      >
        <DialogPrimitive.Title className="sr-only">{t.runTrace.drawerTitle}</DialogPrimitive.Title>
        <header className="flex h-12 shrink-0 items-center justify-between border-b px-4">
          <span className="text-sm font-medium">{t.runTrace.drawerTitle}</span>
          <DialogPrimitive.Close asChild>
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label={t.runTrace.close}
            >
              <XIcon className="size-4" />
            </Button>
          </DialogPrimitive.Close>
        </header>
        <div className="min-h-0 flex-1">
          <RunTracePanel messages={messages} />
        </div>
      </DialogPrimitive.Content>
    </DialogPrimitive.Root>
  );
}
