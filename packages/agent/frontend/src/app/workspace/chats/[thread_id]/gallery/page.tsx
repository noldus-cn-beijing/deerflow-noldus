"use client";

import { ArrowLeftIcon } from "lucide-react";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { ArtifactGallery } from "@/components/workspace/artifacts/gallery/artifact-gallery";
import { getAPIClient } from "@/core/api";
import { normalizeArtifacts, type ArtifactMeta } from "@/core/artifacts/types";
import { useI18n } from "@/core/i18n/hooks";
import type { AgentThreadState } from "@/core/threads";

export default function GalleryPage() {
  const { t } = useI18n();
  const router = useRouter();
  const params = useParams<{ thread_id: string }>();
  const threadId = params.thread_id;

  const [artifacts, setArtifacts] = useState<ArtifactMeta[]>([]);
  const [chartsStatus, setChartsStatus] = useState<AgentThreadState["charts_status"]>(undefined);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const client = getAPIClient();
        const state = await client.threads.getState(threadId);
        const values = (state.values ?? {}) as Partial<AgentThreadState>;
        if (!cancelled) {
          setArtifacts(normalizeArtifacts(values.artifacts));
          setChartsStatus(values.charts_status);
        }
      } catch {
        // 取状态失败时静默——画廊仍可下载 ZIP（第 1 层主路径不依赖 state）。
      } finally {
        if (!cancelled) setLoaded(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [threadId]);

  return (
    <div className="mx-auto flex size-full max-w-(--container-width-md) flex-col gap-4 p-6">
      <header className="flex items-center justify-between">
        <h1 className="text-lg font-medium">{t.gallery.title}</h1>
        <Button variant="ghost" size="sm" onClick={() => router.push(`/workspace/chats/${threadId}`)}>
          <ArrowLeftIcon className="size-4" />
          {t.gallery.backToChat}
        </Button>
      </header>

      {loaded && artifacts.length === 0 && (
        <p className="text-muted-foreground text-sm">{t.gallery.noArtifacts}</p>
      )}

      {loaded && artifacts.length > 0 && (
        <ArtifactGallery
          artifacts={artifacts}
          threadId={threadId}
          chartsStatus={chartsStatus}
        />
      )}
    </div>
  );
}
