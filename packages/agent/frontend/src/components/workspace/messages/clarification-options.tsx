"use client";

import { Button } from "@/components/ui/button";
import { useI18n } from "@/core/i18n/hooks";
import { cn } from "@/lib/utils";

/**
 * Renders a vertical button group under an ask_clarification message.
 *
 * Each option from the model's `ask_clarification(options=[...])` call is
 * rendered as a clickable button. Clicking sends the option text as the next
 * user message — identical to typing it manually in the input bar.
 *
 * Hidden entirely when:
 * - `options` is empty/undefined (the model called ask_clarification without
 *   a pre-populated choice list — user must free-form reply)
 * - `disabled` is true (already answered or stream is running)
 */
export function ClarificationOptions({
  options,
  onSelect,
  disabled = false,
  className,
}: {
  options: string[] | undefined;
  onSelect: (option: string) => void;
  disabled?: boolean;
  className?: string;
}) {
  const { t } = useI18n();

  if (!options || options.length === 0) {
    return null;
  }

  return (
    <div
      className={cn("mt-3 flex flex-col gap-2", className)}
      role="group"
      aria-label={t.clarification.chooseOption}
    >
      {options.map((option, idx) => (
        <Button
          key={`${idx}-${option}`}
          type="button"
          variant="outline"
          size="sm"
          disabled={disabled}
          className="h-auto justify-start whitespace-normal py-2 text-left"
          onClick={() => onSelect(option)}
        >
          <span className="text-muted-foreground mr-2 shrink-0">
            {idx + 1}.
          </span>
          <span>{option}</span>
        </Button>
      ))}
      <p className="text-muted-foreground text-xs">
        {t.clarification.orTypeCustom}
      </p>
    </div>
  );
}
