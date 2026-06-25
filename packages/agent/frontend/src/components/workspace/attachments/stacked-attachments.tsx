"use client";

import { useState } from "react";

import { usePromptInputAttachments } from "@/components/ai-elements/prompt-input";
import { cn } from "@/lib/utils";

import { AttachmentChip } from "./attachment-chip";
import { AttachmentStack } from "./attachment-stack";
import { FanOutList } from "./fan-out-list";
import { partitionAttachments } from "./partition-attachments";

/**
 * Noldus stacked-attachments renderer (Phase0#8).
 *
 * Replaces the generated `PromptInputAttachments` usage in `input-box.tsx`.
 * Files ≤ `STACK_THRESHOLD` render flat (matching the original `flex-wrap` row);
 * overflow collapses into a single `AttachmentStack` ("+N") that fans out on
 * desktop hover OR touch tap. All attachment state/upload/removal flows through
 * the existing `usePromptInputAttachments()` store — this component only swaps
 * rendering, it does not touch the upload pipeline (spec §1.2 / §3.5).
 *
 * Must be rendered inside `<PromptInput>` so the store context resolves.
 */
export function StackedAttachments({ className }: { className?: string }) {
  const attachments = usePromptInputAttachments();
  const [open, setOpen] = useState(false);

  const files = attachments.files;
  const { flat, stacked, stackedCount } = partitionAttachments(files);

  if (files.length === 0) {
    return null;
  }

  return (
    <div
      className={cn("flex w-full flex-wrap items-center gap-2 p-3", className)}
    >
      {flat.map((file) => (
        <AttachmentChip
          data={file}
          key={file.id}
          onRemove={attachments.remove}
        />
      ))}
      {stackedCount > 0 ? (
        <AttachmentStack
          count={stackedCount}
          onOpenChange={setOpen}
          open={open}
        >
          <FanOutList items={stacked} onRemove={attachments.remove} />
        </AttachmentStack>
      ) : null}
    </div>
  );
}
