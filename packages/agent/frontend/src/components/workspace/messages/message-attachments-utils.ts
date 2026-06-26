/**
 * Pure helpers for the message-flow attachments renderer (Phase0#8 续).
 *
 * `toAttachment` adapts a {@link FileInMessage} (the post-send shape that rides
 * in `message.additional_kwargs.files`) into the `Attachment` shape the #8
 * display components (`AttachmentChip` / `FanOutList`) consume. Keeping this as
 * a pure module — separate from the React component, mirroring how #8 splits
 * `partition-attachments.ts` (pure) from `stacked-attachments.tsx` (render) —
 * lets the mapping contract be unit tested independently of rendering.
 */
import { resolveArtifactURL } from "@/core/artifacts/utils";
import type { FileInMessage } from "@/core/messages/utils";

import type { Attachment } from "../attachments/attachment-chip";

const IMAGE_EXTENSIONS = [
  "png",
  "jpg",
  "jpeg",
  "gif",
  "webp",
  "svg",
  "bmp",
] as const;

/**
 * Derive an IANA media type from a filename's extension so `AttachmentChip`
 * takes its image-thumbnail branch (`data.mediaType?.startsWith("image/")`).
 * Anything non-image maps to `application/octet-stream` — the chip then renders
 * its generic file-glyph branch via `getFileIcon`.
 */
export function mediaTypeFromFilename(filename: string): string {
  const ext = filename.split(".").pop()?.toLowerCase() ?? "";
  return IMAGE_EXTENSIONS.includes(ext as (typeof IMAGE_EXTENSIONS)[number])
    ? `image/${ext === "jpg" ? "jpeg" : ext}`
    : "application/octet-stream";
}

/**
 * Map a post-send file into the #8 `Attachment` shape.
 *
 * - `url` is resolved through `resolveArtifactURL` (same path the old
 *   `RichFileCard` used) so image thumbnails / downloads keep working.
 * - `mediaType` is inferred from the extension (no MIME is stored on
 *   `FileInMessage`), so image files render thumbnails exactly like the
 *   input-box attachments.
 * - `id` is the filename — purely a React key; message-flow attachments are
 *   read-only, there is no remove-by-id.
 */
export function toAttachment(
  file: FileInMessage,
  threadId: string,
): Attachment {
  return {
    type: "file",
    id: file.filename,
    filename: file.filename,
    // `Attachment` (a `FileUIPart`) requires `url: string`. Uploaded files
    // always have a path (the caller filters `status === "uploading"` out
    // before mapping); an empty-string fallback keeps the type honest and lets
    // `AttachmentChip`'s `!!data.url` guard correctly skip the image branch.
    url: file.path ? resolveArtifactURL(file.path, threadId) : "",
    mediaType: mediaTypeFromFilename(file.filename),
  };
}

/**
 * Map a list of post-send files into #8 `Attachment` shapes, disambiguating
 * React keys by index when filenames repeat within one message.
 */
export function toAttachments(
  files: readonly FileInMessage[],
  threadId: string,
): Attachment[] {
  const seen = new Map<string, number>();
  return files.map((file) => {
    const occurrences = seen.get(file.filename) ?? 0;
    seen.set(file.filename, occurrences + 1);
    const attachment = toAttachment(file, threadId);
    return occurrences > 0
      ? { ...attachment, id: `${file.filename}#${occurrences}` }
      : attachment;
  });
}
