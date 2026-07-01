import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
}: {
  icon?: LucideIcon;
  title: string;
  description?: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center gap-2 py-8 text-center">
      {Icon != null && <Icon className="text-muted-foreground size-6" aria-hidden />}
      <p className="text-sm font-medium">{title}</p>
      {description != null && <p className="text-muted-foreground text-xs">{description}</p>}
      {action != null && <div className="mt-1">{action}</div>}
    </div>
  );
}
