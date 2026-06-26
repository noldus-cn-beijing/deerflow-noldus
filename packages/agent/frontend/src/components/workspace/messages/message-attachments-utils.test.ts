import { describe, expect, it, vi } from "vitest";

// `resolveArtifactURL` reads the backend base URL from env; stub it so the
// mapping contract is asserted without depending on runtime config. Importing
// the real module would also drag `getBackendBaseURL()` env reads into the test.
vi.mock("@/core/artifacts/utils", () => ({
  resolveArtifactURL: (path: string, threadId: string) =>
    `<BASE>/api/threads/${threadId}/artifacts${path}`,
}));

import {
  mediaTypeFromFilename,
  toAttachment,
  toAttachments,
} from "@/components/workspace/messages/message-attachments-utils";
import type { FileInMessage } from "@/core/messages/utils";


function makeFile(over: Partial<FileInMessage> = {}): FileInMessage {
  return { filename: "Trial1.xlsx", size: 1024, path: "/mnt/x/Trial1.xlsx", ...over };
}

describe("mediaTypeFromFilename", () => {
  it("maps common image extensions to image/*", () => {
    expect(mediaTypeFromFilename("a.png")).toBe("image/png");
    expect(mediaTypeFromFilename("a.JPG")).toBe("image/jpeg");
    expect(mediaTypeFromFilename("a.jpeg")).toBe("image/jpeg");
    expect(mediaTypeFromFilename("a.gif")).toBe("image/gif");
    expect(mediaTypeFromFilename("a.webp")).toBe("image/webp");
  });

  it("maps non-image extensions to octet-stream so the chip renders a file glyph", () => {
    expect(mediaTypeFromFilename("Trial1.xlsx")).toBe("application/octet-stream");
    expect(mediaTypeFromFilename("data.csv")).toBe("application/octet-stream");
    expect(mediaTypeFromFilename("noext")).toBe("application/octet-stream");
  });
});

describe("toAttachment", () => {
  it("carries the filename as id + filename", () => {
    const a = toAttachment(makeFile({ filename: "EPM_ctrl.xlsx" }), "t1");
    expect(a.id).toBe("EPM_ctrl.xlsx");
    expect(a.filename).toBe("EPM_ctrl.xlsx");
  });

  it("resolves the url through resolveArtifactURL(file.path, threadId)", () => {
    const a = toAttachment(makeFile({ path: "/mnt/x/Trial1.xlsx" }), "t42");
    expect(a.url).toBe("<BASE>/api/threads/t42/artifacts/mnt/x/Trial1.xlsx");
  });

  it("falls back to an empty url while a file is still uploading (no path yet)", () => {
    const a = toAttachment(makeFile({ path: undefined, status: "uploading" }), "t1");
    // Empty string is falsy → AttachmentChip's `!!data.url` skips the image branch.
    expect(a.url).toBe("");
  });

  it("infers image mediaType so AttachmentChip takes its thumbnail branch", () => {
    const a = toAttachment(makeFile({ filename: "plot.png", path: "/mnt/p.png" }), "t1");
    expect(a.mediaType).toBe("image/png");
  });

  it("infers octet-stream mediaType for non-image files", () => {
    const a = toAttachment(makeFile({ filename: "Trial1.xlsx" }), "t1");
    expect(a.mediaType).toBe("application/octet-stream");
  });
});

describe("toAttachments", () => {
  it("maps each file and preserves order", () => {
    const out = toAttachments(
      [
        makeFile({ filename: "A.xlsx" }),
        makeFile({ filename: "B.csv" }),
      ],
      "t1",
    );
    expect(out.map((a) => a.filename)).toEqual(["A.xlsx", "B.csv"]);
  });

  it("disambiguates React keys when filenames repeat within one message", () => {
    const out = toAttachments(
      [
        makeFile({ filename: "dup.xlsx" }),
        makeFile({ filename: "dup.xlsx" }),
        makeFile({ filename: "dup.xlsx" }),
      ],
      "t1",
    );
    expect(out.map((a) => a.id)).toEqual(["dup.xlsx", "dup.xlsx#1", "dup.xlsx#2"]);
  });
});
