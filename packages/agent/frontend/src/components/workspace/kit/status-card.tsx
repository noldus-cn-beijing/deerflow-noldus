import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

import { AccentBar, type Status } from "./status-badge";

export function StatusCard({
  status,
  title,
  children,
  pulse = false,
  className,
}: {
  status: Status;
  title?: ReactNode;
  children?: ReactNode;
  pulse?: boolean;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex gap-3 rounded-lg bg-background p-3 shadow-float",
        pulse && "animate-pulse-warm",
        className,
      )}
    >
      <AccentBar status={status} />
      <div className="min-w-0 flex-1">
        {title != null && <div className="text-sm font-medium">{title}</div>}
        {children}
      </div>
    </div>
  );
}
