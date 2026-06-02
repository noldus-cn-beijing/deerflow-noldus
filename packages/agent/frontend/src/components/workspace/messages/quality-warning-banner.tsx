"use client";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";
import { ChevronDownIcon, ExclamationTriangleIcon } from "@radix-ui/react-icons";
import { useState } from "react";

export interface QualityWarning {
  severity: "critical" | "warning" | "info";
  code: string;
  metric: string;
  message: string;
  evidence?: Record<string, unknown>;
  blocks_downstream?: boolean;
}

function warningStyle(w: QualityWarning): {
  variant: "destructive" | "default";
  colorClass: string;
  label: string;
  iconClass: string;
} {
  if (w.severity === "critical" && w.blocks_downstream) {
    return {
      variant: "destructive",
      colorClass: "text-red-600",
      label: "阻断级警告",
      iconClass: "text-red-500",
    };
  }
  if (w.severity === "critical") {
    return {
      variant: "destructive",
      colorClass: "text-orange-600",
      label: "严重警告",
      iconClass: "text-orange-500",
    };
  }
  if (w.severity === "warning") {
    return {
      variant: "default",
      colorClass: "text-yellow-600",
      label: "提示",
      iconClass: "text-yellow-500",
    };
  }
  return {
    variant: "default",
    colorClass: "text-blue-600",
    label: "信息",
    iconClass: "text-blue-500",
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

  return (
    <Alert variant={style.variant} className="my-2">
      <ExclamationTriangleIcon className={cn("size-4", style.iconClass)} />
      <AlertTitle className={cn("text-sm font-medium", style.colorClass)}>
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
                    <span className={cn("font-medium", ws.colorClass)}>
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
