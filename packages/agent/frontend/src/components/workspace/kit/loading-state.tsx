import { Loader2Icon } from "lucide-react";

import { cn } from "@/lib/utils";

export function LoadingState({ variant, label }: { variant: "spinner" | "skeleton" | "dots"; label?: string }) {
  return (
    <div data-variant={variant} className="text-muted-foreground flex items-center gap-2 text-sm">
      {variant === "spinner" && <Loader2Icon className="size-4 animate-spin" aria-hidden />}
      {variant === "skeleton" && <span className={cn("h-4 w-24 rounded bg-muted", "animate-skeleton-entrance")} aria-hidden />}
      {variant === "dots" && <span className="animate-suggestion-in inline-flex gap-0.5" aria-hidden>•••</span>}
      {label != null && <span>{label}</span>}
    </div>
  );
}
