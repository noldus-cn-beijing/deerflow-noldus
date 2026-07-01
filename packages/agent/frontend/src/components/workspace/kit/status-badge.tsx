import { AlertTriangleIcon, CheckCircle2Icon, InfoIcon, XCircleIcon, type LucideIcon } from "lucide-react";

import { cn } from "@/lib/utils";

export type Status = "success" | "warning" | "danger" | "info";

/** Icon per status — SSOT so every card shows the same glyph (color-not-only). */
export const STATUS_ICON: Record<Status, LucideIcon> = {
  success: CheckCircle2Icon,
  warning: AlertTriangleIcon,
  danger: XCircleIcon,
  info: InfoIcon,
};

// status → D1 status token utility (text/bg). NEVER a raw hex; NEVER red-green only.
// Exported as helpers so cards with bespoke layouts (e.g. an absolute-positioned
// accent overlay in decision-card) can route their color through this SSOT
// without adopting the <AccentBar>/<StatusBadge> DOM.
export const statusBarClass = (status: Status): string => BAR_CLASS[status];
export const statusTextClass = (status: Status): string => TEXT_CLASS[status];

const TEXT_CLASS: Record<Status, string> = {
  success: "text-status-success",
  warning: "text-status-warning",
  danger: "text-status-danger",
  info: "text-status-info",
};
const BAR_CLASS: Record<Status, string> = {
  success: "bg-status-success",
  warning: "bg-status-warning",
  danger: "bg-status-danger",
  info: "bg-status-info",
};

export function StatusBadge({ status, label, size = "md" }: { status: Status; label?: string; size?: "sm" | "md" }) {
  const Icon = STATUS_ICON[status];
  return (
    <span className={cn("inline-flex items-center gap-1", TEXT_CLASS[status], size === "sm" ? "text-xs" : "text-sm")}>
      <Icon className={size === "sm" ? "size-3.5" : "size-4"} aria-hidden />
      {label != null && <span className="font-medium">{label}</span>}
    </span>
  );
}

/** Thin left accent bar — status color as a slim vertical strip, never a full-card fill. */
export function AccentBar({ status }: { status: Status }) {
  return <div className={cn("w-1 shrink-0 self-stretch rounded-full", BAR_CLASS[status])} aria-hidden />;
}
