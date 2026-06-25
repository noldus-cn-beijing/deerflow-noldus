"use client";

import { XIcon } from "lucide-react";
import { type ComponentProps, useState } from "react";

import { useI18n } from "@/core/i18n/hooks";
import { type PromptInputFilePart } from "@/core/uploads";
import { getFileIcon } from "@/core/utils/files";
import { cn } from "@/lib/utils";

export type Attachment = PromptInputFilePart & { id: string };

export interface AttachmentChipProps extends ComponentProps<"div"> {
  data: Attachment;
  onRemove?: (id: string) => void;
  /**
   * When true the remove (✕) control is always visible (fan-out list context).
   * When false (default flat row) it reveals on hover/focus only.
   */
  alwaysShowRemove?: boolean;
}

/**
 * Single attachment chip — reused in the flat row and inside the fan-out list.
 * Reuses the existing `getFileIcon` helper so file-type glyphs stay consistent
 * with the rest of the app. Does NOT own attachment state; removal is delegated
 * to the parent via `onRemove` (which calls the shared attachments store).
 */
export function AttachmentChip({
  data,
  onRemove,
  alwaysShowRemove = false,
  className,
  ...props
}: AttachmentChipProps) {
  const { t } = useI18n();
  const [imageFailed, setImageFailed] = useState(false);
  const filename = data.filename ?? "";
  const isImage =
    data.mediaType?.startsWith("image/") && !!data.url && !imageFailed;

  return (
    <div
      className={cn(
        "group border-border hover:bg-accent hover:text-accent-foreground dark:hover:bg-accent/50",
        "relative flex h-8 min-w-0 max-w-60 items-center gap-1.5 rounded-md border px-1.5",
        "text-sm font-medium transition-all select-none",
        "motion-safe:duration-fast motion-safe:ease-brand-out",
        "active:scale-[0.97]",
        className,
      )}
      {...props}
    >
      <div className="relative size-5 shrink-0">
        <div className="bg-background absolute inset-0 flex size-5 items-center justify-center overflow-hidden rounded transition-opacity group-hover:opacity-0 group-focus-within:opacity-0">
          {isImage ? (
            <img
              alt={filename || t.inputBox.attachmentImage}
              className="size-5 object-cover"
              height={20}
              onError={() => setImageFailed(true)}
              src={data.url}
              width={20}
            />
          ) : (
            <span className="text-muted-foreground flex size-5 items-center justify-center">
              {getFileIcon(filename || "", "size-3 shrink-0")}
            </span>
          )}
        </div>
        {onRemove && (
          <button
            aria-label={t.inputBox.removeAttachment.replace("{name}", filename)}
            className={cn(
              "bg-background absolute inset-0 flex size-5 cursor-pointer items-center justify-center rounded p-0 transition-opacity",
              alwaysShowRemove
                ? "opacity-100"
                : "opacity-0 group-hover:opacity-100 group-focus-within:opacity-100",
            )}
            onClick={(e) => {
              e.stopPropagation();
              onRemove(data.id);
            }}
            type="button"
          >
            <XIcon className="size-2.5" />
          </button>
        )}
      </div>
      <span className="flex-1 truncate">{filename}</span>
    </div>
  );
}
