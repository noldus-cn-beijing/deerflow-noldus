"use client";

import { ArrowLeftIcon } from "lucide-react";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { ArtifactGallery } from "@/components/workspace/artifacts/gallery/artifact-gallery";
import { getAPIClient } from "@/core/api";
import { fetch as fetchWithAuth } from "@/core/api/fetcher";
import { normalizeArtifacts, type ArtifactMeta } from "@/core/artifacts/types";
import { chartsArtifactsURL } from "@/core/artifacts/utils";
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
      // spec 2026-06-26-artifact-bubbling §1.1/§1.2：画廊全量图走「磁盘 + plan_charts.json」端点，
      // 不再依赖 LangGraph state.artifacts（subagent→lead 边界会丢 artifacts，113 张只活 2 张）。
      // charts_status（失败/截断告警）仍读 state——它不在丢失链路上且对画廊有用。
      let charts: ArtifactMeta[] = [];
      try {
        const resp = await fetchWithAuth(chartsArtifactsURL(threadId));
        if (resp.ok) {
          charts = normalizeArtifacts((await resp.json()) as ArtifactMeta[]);
        }
      } catch {
        // 端点不可达（如旧后端未部署）：下方 state 回退兜底，画廊仍能显示 lead present 的代表图。
      }
      // 回退：端点为空或失败时，退回 state.artifacts（lead present 的 box 代表图仍在 state）。
      if (charts.length === 0) {
        try {
          const client = getAPIClient();
          const state = await client.threads.getState(threadId);
          const values = (state.values ?? {}) as Partial<AgentThreadState>;
          charts = normalizeArtifacts(values.artifacts);
          if (!cancelled) setChartsStatus(values.charts_status);
        } catch {
          // 取状态也失败时静默——画廊仍可下载 ZIP（第 1 层主路径不依赖 state）。
        }
      }
      if (!cancelled) {
        setArtifacts(charts);
        setLoaded(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [threadId]);

  // spec §三 现象3：画廊「返回对话」用 router.back() 回到对话页既有实例（不重挂载、
  // 不丢滚动位置）——page.tsx 自己注释警告过 router.push 会重挂载几百消息致卡顿。
  // 深链直进画廊（无 history）时 fallback 到 push。
  function handleBack() {
    if (typeof window !== "undefined" && window.history.length > 1) {
      router.back();
    } else {
      router.push(`/workspace/chats/${threadId}`);
    }
  }

  return (
    <div className="mx-auto flex size-full max-w-(--container-width-md) flex-col gap-4 p-6">
      <header className="flex items-center justify-between">
        <h1 className="text-lg font-medium">{t.gallery.title}</h1>
        <Button variant="ghost" size="sm" onClick={handleBack}>
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
