"use client";

import { ArrowLeftIcon, DownloadIcon } from "lucide-react";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { InlineArtifactSummary } from "@/components/workspace/artifacts";
import { getAPIClient } from "@/core/api";
import { normalizeArtifacts, type ArtifactMeta } from "@/core/artifacts/types";
import { archiveArtifactsURL } from "@/core/artifacts/utils";
import { useI18n } from "@/core/i18n/hooks";
import type { AgentThreadState } from "@/core/threads";

/**
 * 产物画廊独立路由（spec §3.4 方案 A：可深链 / 可分享 / 可前进后退）。
 *
 * Phase 0 占位：渲染 inline 摘要（代表图 + 下载全部 ZIP + 失败/截断）+ 返回对话。
 * 完整画廊（分面筛选 + 虚拟化网格 + 小倍数对比 + lightbox）下一迭代上线（spec §四 Step 4-5）。
 */
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

      {/* 第 1 层主路径：零渲染带走全部图（spec §3.1.7） */}
      {artifacts.length > 0 && (
        <Button variant="secondary" size="sm" className="self-start" asChild>
          <a href={archiveArtifactsURL(threadId)} aria-label={t.gallery.downloadAll(artifacts.length)}>
            <DownloadIcon className="size-4" />
            {t.gallery.downloadAll(artifacts.length)}
          </a>
        </Button>
      )}

      {loaded && (
        <InlineArtifactSummary
          threadId={threadId}
          artifacts={artifacts}
          chartsStatus={chartsStatus}
        />
      )}

      <p className="text-muted-foreground text-sm">{t.gallery.galleryPlaceholder}</p>
    </div>
  );
}
