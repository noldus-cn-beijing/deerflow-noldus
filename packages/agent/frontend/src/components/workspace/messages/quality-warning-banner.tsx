"use client";

import { ChevronDownIcon, ExclamationTriangleIcon } from "@radix-ui/react-icons";
import { useState } from "react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";

import { statusTextClass, type Status } from "../kit/status-badge";

export interface QualityWarning {
  severity: "critical" | "warning" | "info";
  code: string;
  metric: string;
  message: string;
  evidence?: Record<string, unknown>;
  blocks_downstream?: boolean;
}

/**
 * severity → 视觉调性。颜色统一由 kit `Status` SSOT 派生（statusTextClass），
 * 不再手写 text-red/orange/yellow/blue 原始 Tailwind 调色板——D2 收敛：对话业务
 * 卡片的状态色一律走 D1 `--color-status-*` 色盲安全 token。
 *
 * severity→Status 映射（spec D2 Task5）：critical（含 blocks_downstream）→ danger，
 * warning → warning，info → info。`variant` 是 shadcn Alert 壳的样式档（destructive
 * 给 critical 加重边框），与状态色正交、保留。
 */
function warningStyle(w: QualityWarning): {
  variant: "destructive" | "default";
  status: Status;
  label: string;
} {
  if (w.severity === "critical") {
    return {
      variant: "destructive",
      status: "danger",
      label: w.blocks_downstream ? "阻断级警告" : "严重警告",
    };
  }
  if (w.severity === "warning") {
    return {
      variant: "default",
      status: "warning",
      label: "提示",
    };
  }
  return {
    variant: "default",
    status: "info",
    label: "信息",
  };
}

export function QualityWarningBanner({
  warnings,
}: {
  warnings: QualityWarning[];
}) {
  const [expanded, setExpanded] = useState(false);

  if (!warnings || warnings.length === 0) return null;

  const critical = warnings.filter(
    (w) => w.severity === "critical" && w.blocks_downstream,
  );
  const style =
    critical.length > 0
      ? warningStyle(critical[0]!)
      : warningStyle(warnings[0]!);
  const colorClass = statusTextClass(style.status);

  return (
    <Alert variant={style.variant} className="my-2">
      <ExclamationTriangleIcon className={cn("size-4", colorClass)} />
      <AlertTitle className={cn("text-sm font-medium", colorClass)}>
        {warnings.length} 条数据质量警告
        {critical.length > 0 && `（含 ${critical.length} 条阻断级）`}
      </AlertTitle>
      <AlertDescription>
        <Collapsible open={expanded} onOpenChange={setExpanded}>
          <CollapsibleTrigger className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground">
            <ChevronDownIcon
              className={cn(
                "size-3 transition-transform",
                expanded && "rotate-180",
              )}
            />
            {expanded ? "收起详情" : "展开详情"}
          </CollapsibleTrigger>
          <CollapsibleContent>
            <ul className="mt-2 space-y-1.5">
              {warnings.map((w, i) => {
                const ws = warningStyle(w);
                return (
                  <li key={i} className="text-xs">
                    <span className={cn("font-medium", statusTextClass(ws.status))}>
                      [{ws.label} {w.code}]
                    </span>{" "}
                    <span className="text-muted-foreground">{w.message}</span>
                  </li>
                );
              })}
            </ul>
          </CollapsibleContent>
        </Collapsible>
      </AlertDescription>
    </Alert>
  );
}
