"use client";

import { PauseIcon } from "lucide-react";
import { useEffect, useRef } from "react";

import { Button } from "@/components/ui/button";
import { useI18n } from "@/core/i18n/hooks";
import { cn } from "@/lib/utils";

/**
 * Renders the option list of an `ask_clarification` decision card (spec#5 §3.2).
 *
 * Behaviour contract (do not change without updating the decision-card spec):
 * - Clicking an option sends that option text as the next user message — identical to
 *   typing it. The "click option = send message" mechanism is unchanged from before
 *   spec#5 (spec §二非目标 / §五工程纪律).
 * - **Keyboard**: while the card is awaiting an answer and the text input is NOT
 *   focused, pressing digit keys `1`-`9` selects the matching option (≤9 options).
 *   When the input box is focused, digit keys type normally
 *   (spec §六 risk: 数字键与输入框冲突).
 * - **Primary/secondary hierarchy** (spec §3.1 primary-action): the first option is
 *   rendered with the brand outline as the recommended primary action; the rest are
 *   secondary. Every option meets the ≥44px touch target (spec §五 a11y 红线).
 * - **Answered state**: once `answeredIndex >= 0`, the chosen option is highlighted
 *   (success feedback) and all options are disabled (closed-loop, spec §3.1/§五).
 *
 * Hidden entirely when `options` is empty/undefined (the model called
 * ask_clarification without a pre-populated choice list — user must free-form reply).
 */
export function ClarificationOptions({
  options,
  onSelect,
  disabled = false,
  /**
   * Index of the option the user already picked, or `null` while still waiting.
   * Drives the green closed-loop highlight + disable-all (spec §3.1).
   */
  answeredIndex = null,
  className,
  /**
   * When true, this options block listens for digit-key shortcuts. The decision
   * card sets this only while it is awaiting an answer (spec §3.2); the listener
   * itself additionally ignores keys while focus is in a text field.
   */
  keyboardActive = false,
}: {
  options: string[] | undefined;
  onSelect: (option: string) => void;
  disabled?: boolean;
  answeredIndex?: number | null;
  className?: string;
  keyboardActive?: boolean;
}) {
  const { t } = useI18n();
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Digit-key selection (1-9). Only active when the card is awaiting an answer,
  // the list is not disabled, and focus is not inside a text input/textarea/
  // contenteditable (so typing into the reply box still works — spec §六 risk row).
  useEffect(() => {
    if (!keyboardActive || disabled) return;
    function onKey(e: KeyboardEvent) {
      if (e.defaultPrevented) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName?.toLowerCase();
      if (tag === "input" || tag === "textarea" || target?.isContentEditable) {
        return;
      }
      // Only 1-9 map to options (≤9 items). `0` and other keys are ignored.
      const n = Number(e.key);
      if (!Number.isInteger(n) || n < 1 || n > 9) return;
      const idx = n - 1;
      const opt = options?.[idx];
      if (!opt) return;
      e.preventDefault();
      onSelect(opt);
    }
    const node = containerRef.current ?? document;
    node.addEventListener("keydown", onKey as EventListener);
    return () => node.removeEventListener("keydown", onKey as EventListener);
  }, [keyboardActive, disabled, options, onSelect]);

  if (!options || options.length === 0) {
    return null;
  }

  return (
    <div
      ref={containerRef}
      className={cn("mt-3 flex flex-col gap-2", className)}
      role="group"
      aria-label={t.clarification.chooseOption}
      // tabIndex -1 so the container can host the keydown listener without joining
      // the Tab order (Tab still lands on each button — spec §3.2).
      tabIndex={-1}
    >
      {options.map((option, idx) => {
        const isPrimary = idx === 0;
        const isAnswered = answeredIndex === idx;
        const othersAnswered =
          answeredIndex !== null && answeredIndex !== idx;
        return (
          <Button
            key={`${idx}-${option}`}
            type="button"
            variant={isPrimary ? "default" : "outline"}
            disabled={disabled || answeredIndex !== null}
            aria-keyshortcuts={idx < 9 ? String(idx + 1) : undefined}
            // ≥44px touch target (spec §五 a11y 红线). `h-auto` + `min-h-11` keeps
            // multi-line options tall enough while growing with content.
            className={cn(
              "h-auto min-h-11 w-full justify-start whitespace-normal py-2.5 text-left",
              isPrimary &&
                "border-brand bg-brand/10 text-brand hover:bg-brand/15",
              isAnswered &&
                "border-status-success bg-status-success/15 text-status-success",
              othersAnswered && "opacity-50",
            )}
            onClick={() => onSelect(option)}
          >
            <span
              className={cn(
                "mr-2 shrink-0 rounded px-1.5 py-0.5 text-xs font-semibold",
                isPrimary
                  ? "bg-brand/15 text-brand"
                  : "bg-muted text-muted-foreground",
                isAnswered && "bg-status-success/20 text-status-success",
              )}
              aria-hidden="true"
            >
              {idx + 1}
            </span>
            <span className="flex-1">{option}</span>
          </Button>
        );
      })}
      <p className="text-muted-foreground flex items-center gap-1 text-xs">
        <PauseIcon className="size-3" aria-hidden="true" />
        {t.clarification.orTypeCustom}
      </p>
    </div>
  );
}
