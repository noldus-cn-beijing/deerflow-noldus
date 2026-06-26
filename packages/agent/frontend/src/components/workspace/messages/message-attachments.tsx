"use client";

import { Loader2Icon } from "lucide-react";
import { useState } from "react";

import { Badge } from "@/components/ui/badge";
import { useI18n } from "@/core/i18n/hooks";
import type { FileInMessage } from "@/core/messages/utils";
import { cn } from "@/lib/utils";

import { AttachmentChip } from "../attachments/attachment-chip";
import { AttachmentStack } from "../attachments/attachment-stack";
import { FanOutList } from "../attachments/fan-out-list";
import { partitionAttachments } from "../attachments/partition-attachments";

import { getFileTypeLabel } from "./file-type-label";
import { toAttachments } from "./message-attachments-utils";

/**
 * Message-flow attachments renderer (Phase0#8 续 — 发送后附件复用堆叠).
 *
 * Replaces the old `RichFilesList` (which flat-wrapped every file and flooded
 * the bubble when researchers uploaded dozens–hundreds of EPM files). Now the
 * post-send attachments reuse the SAME stack + fan-out as the #8 input box
 * (`AttachmentStack` / `FanOutList` / `AttachmentChip` / `partitionAttachments`),
 * so the experience stays continuous across send: stacked → hover/tap to fan
 * out, identical to before-send.
 *
 * Read-only — a sent message's files cannot be removed, so no `onRemove` is
 * threaded into the chips / fan-out (spec §1 关键差异②).
 *
 * Upload-in-progress files (`status === "uploading"`) are deliberately kept
 * OUT of the stack: they render flat so the spinner / progress stays visible
 * (spec §3.1 — stacking must not swallow upload progress). Only fully uploaded
 * files collapse into the "+N" stack.
 */
export function MessageAttachments({
  files,
  threadId,
}: {
  files: FileInMessage[];
  threadId: string;
}) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);

  if (files.length === 0) return null;

  const uploading = files.filter((f) => f.status === "uploading");
  const uploaded = files.filter((f) => f.status !== "uploading");

  const attachments = toAttachments(uploaded, threadId);
  const { flat, stacked, stackedCount } = partitionAttachments(attachments);

  // When uploads are in flight, surface the existing "uploading" label on the
  // stack so the researcher can tell the stack is still filling (spec §3.1).
  const progressLabel = uploading.length > 0 ? t.uploads.uploading : undefined;

  return (
    <div className="mb-2 flex w-full flex-wrap justify-end gap-2">
      {uploading.map((file, index) => (
        <UploadingFileCard
          file={file}
          key={`uploading-${file.filename}-${index}`}
        />
      ))}
      {flat.map((attachment) => (
        <AttachmentChip data={attachment} key={attachment.id} />
      ))}
      {stackedCount > 0 ? (
        <AttachmentStack
          count={stackedCount}
          onOpenChange={setOpen}
          open={open}
          progressLabel={progressLabel}
        >
          {/* Read-only fan-out: no onRemove → no ✕ controls render. */}
          <FanOutList items={stacked} />
        </AttachmentStack>
      ) : null}
    </div>
  );
}

/**
 * Flat card for an upload-in-progress file. Mirrors the old `RichFileCard`
 * uploading branch (spinner + type badge + "上传中") so progress stays visible
 * and is never collapsed into the stack. Kept local to keep the old visual for
 * this transient state rather than inventing a spinner variant of `AttachmentChip`.
 */
function UploadingFileCard({ file }: { file: FileInMessage }) {
  return (
    <div
      className={cn(
        "bg-background border-border/40 flex max-w-50 min-w-30 flex-col gap-1 rounded-lg border p-3 opacity-60 shadow-sm",
      )}
    >
      <div className="flex items-start gap-2">
        <Loader2Icon className="text-muted-foreground mt-0.5 size-4 shrink-0 animate-spin" />
        <span
          className="text-foreground truncate text-sm font-medium"
          title={file.filename}
        >
          {file.filename}
        </span>
      </div>
      <div className="flex items-center justify-between gap-2">
        <Badge
          variant="secondary"
          className="rounded px-1.5 py-0.5 text-[10px] font-normal"
        >
          {getFileTypeLabel(file.filename)}
        </Badge>
      </div>
    </div>
  );
}
