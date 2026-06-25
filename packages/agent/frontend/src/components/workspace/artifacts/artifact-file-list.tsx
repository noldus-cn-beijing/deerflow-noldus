import { DownloadIcon, LoaderIcon, PackageIcon } from "lucide-react";
import { useCallback, useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardAction,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import type { ArtifactMeta } from "@/core/artifacts/types";
import { urlOfArtifact } from "@/core/artifacts/utils";
import { useI18n } from "@/core/i18n/hooks";
import { installSkill } from "@/core/skills/api";
import {
  getFileExtensionDisplayName,
  getFileIcon,
  getFileName,
} from "@/core/utils/files";
import { cn } from "@/lib/utils";

import { useArtifacts } from "./context";

const IMAGE_EXTENSIONS = new Set([
  ".png",
  ".jpg",
  ".jpeg",
  ".gif",
  ".svg",
  ".webp",
  ".bmp",
]);

function isImageFile(meta: ArtifactMeta): boolean {
  const ext = meta.path.slice(meta.path.lastIndexOf(".")).toLowerCase();
  return IMAGE_EXTENSIONS.has(ext);
}

export function ArtifactFileList({
  className,
  files,
  threadId,
}: {
  className?: string;
  /** ArtifactMeta[]（spec phase0-3）；缩略图优先 thumb_path（治渲染成本 ①）。 */
  files: ArtifactMeta[];
  threadId: string;
}) {
  const { t } = useI18n();
  const { select: selectArtifact, setOpen } = useArtifacts();
  const [installingFile, setInstallingFile] = useState<string | null>(null);

  const handleClick = useCallback(
    (filepath: string) => {
      selectArtifact(filepath);
      setOpen(true);
    },
    [selectArtifact, setOpen],
  );

  const handleInstallSkill = useCallback(
    async (e: React.MouseEvent, filepath: string) => {
      e.stopPropagation();
      e.preventDefault();

      if (installingFile) return;

      setInstallingFile(filepath);
      try {
        const result = await installSkill({
          thread_id: threadId,
          path: filepath,
        });
        if (result.success) {
          toast.success(result.message);
        } else {
          toast.error(result.message || "Failed to install skill");
        }
      } catch (error) {
        console.error("Failed to install skill:", error);
        toast.error("Failed to install skill");
      } finally {
        setInstallingFile(null);
      }
    },
    [threadId, installingFile],
  );

  const imageFiles = files.filter(isImageFile);
  const otherFiles = files.filter((f) => !isImageFile(f));

  return (
    <div className={cn("flex w-full flex-col gap-4", className)}>
      {imageFiles.length > 0 && (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {imageFiles.map((file) => {
            const path = file.path;
            const thumbUrl = file.thumb_path
              ? urlOfArtifact({ filepath: file.thumb_path, threadId })
              : null;
            const imgUrl = urlOfArtifact({
              filepath: path,
              threadId,
            });
            return (
              <div
                key={path}
                className="group relative cursor-pointer overflow-hidden rounded-lg border bg-muted/30"
                onClick={() => handleClick(path)}
              >
                <img
                  src={thumbUrl ?? imgUrl}
                  alt={getFileName(path)}
                  className="w-full aspect-square object-contain"
                  loading="lazy"
                  decoding="async"
                />
                <div className="flex items-center justify-between border-t bg-background/80 px-3 py-2 text-xs backdrop-blur-sm">
                  <span className="truncate font-medium">
                    {getFileName(path)}
                  </span>
                  <a
                    href={urlOfArtifact({
                      filepath: path,
                      threadId,
                      download: true,
                    })}
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <Button variant="ghost" size="sm" className="h-6 px-2">
                      <DownloadIcon className="size-3" />
                    </Button>
                  </a>
                </div>
              </div>
            );
          })}
        </div>
      )}
      {otherFiles.length > 0 && (
        <ul className="flex w-full flex-col gap-4">
          {otherFiles.map((file) => {
            const path = file.path;
            return (
            <Card
              key={path}
              className="relative cursor-pointer p-3"
              onClick={() => handleClick(path)}
            >
              <CardHeader className="grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1 pr-2 pl-1">
                <CardTitle className="relative min-w-0 pl-8 leading-tight [overflow-wrap:anywhere] break-words">
                  <div className="min-w-0">{getFileName(path)}</div>
                  <div className="absolute top-2 -left-0.5">
                    {getFileIcon(path, "size-6")}
                  </div>
                </CardTitle>
                <CardDescription className="min-w-0 pl-8 text-xs">
                  {getFileExtensionDisplayName(path)} file
                </CardDescription>
                <CardAction className="row-span-1 self-center">
                  {path.endsWith(".skill") && (
                    <Button
                      variant="ghost"
                      disabled={installingFile === path}
                      onClick={(e) => handleInstallSkill(e, path)}
                    >
                      {installingFile === path ? (
                        <LoaderIcon className="size-4 animate-spin" />
                      ) : (
                        <PackageIcon className="size-4" />
                      )}
                      {t.common.install}
                    </Button>
                  )}
                  <Button variant="ghost" asChild>
                    <a
                      href={urlOfArtifact({
                        filepath: path,
                        threadId: threadId,
                        download: true,
                      })}
                      target="_blank"
                      rel="noopener noreferrer"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <DownloadIcon className="size-4" />
                      {t.common.download}
                    </a>
                  </Button>
                </CardAction>
              </CardHeader>
            </Card>
            );
          })}
        </ul>
      )}
    </div>
  );
}
