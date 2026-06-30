import { useQuery } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useState } from "react";

import { useThread } from "@/components/workspace/messages/context";
import { fetch as fetchWithAuth } from "@/core/api/fetcher";

import { loadArtifactContent, loadArtifactContentFromToolCall } from "./loader";
import type { ArtifactMeta } from "./types";
import { normalizeArtifacts } from "./types";
import { chartsArtifactsURL, dataArtifactsURL, reportsArtifactsURL } from "./utils";

export function useArtifactContent({
  filepath,
  threadId,
  enabled,
}: {
  filepath: string;
  threadId: string;
  enabled?: boolean;
}) {
  const isWriteFile = useMemo(() => {
    return filepath.startsWith("write-file:");
  }, [filepath]);
  const { thread, isMock } = useThread();
  const content = useMemo(() => {
    if (isWriteFile) {
      return loadArtifactContentFromToolCall({ url: filepath, thread });
    }
    return null;
  }, [filepath, isWriteFile, thread]);

  const { data, isLoading, error } = useQuery({
    queryKey: ["artifact", filepath, threadId, isMock],
    queryFn: () => {
      return loadArtifactContent({ filepath, threadId, isMock });
    },
    enabled,
    // Cache artifact content for 5 minutes to avoid repeated fetches (especially for .skill ZIP extraction)
    staleTime: 5 * 60 * 1000,
  });
  return {
    content: isWriteFile ? content : data?.content,
    url: isWriteFile ? undefined : data?.url,
    isLoading,
    error,
  };
}

/**
 * 拉取「磁盘 + plan_charts.json」全量图端点（spec 2026-06-26-artifact-bubbling §1.1）。
 *
 * 纯函数（不引 React）：/gallery 独立页与对话流内嵌图廊（useChartArtifacts）**同源**，
 * 不复制 fetch 逻辑（守 SSOT，memory `feedback_single_source_of_truth`）。磁盘是唯一真相
 * ——LangGraph state.artifacts 在 subagent→lead 边界会丢（chart-maker 画的 113 张只活 2 张），
 * 故两入口都走磁盘端点取图，state 只作回退兜底（lead present 的代表图仍在 state）。
 *
 * 返回归一化后的 ArtifactMeta[]；端点不可达/非 200 时抛错（由 hook 的 catch 转回退 +
 * 标 error，供对话流失败告警）；HTTP 200 但数组真空（图尚未落盘）返回 []。
 */
export async function fetchChartArtifactsFromDisk(threadId: string): Promise<ArtifactMeta[]> {
  const resp = await fetchWithAuth(chartsArtifactsURL(threadId));
  if (!resp.ok) {
    // 响应性失败（500 / 网络）——响亮冒泡，让消费方走回退并标记告警，不静默吞。
    throw new Error(`charts artifacts endpoint ${resp.status}`);
  }
  const json = (await resp.json()) as ArtifactMeta[];
  return normalizeArtifacts(json);
}

/** useChartArtifacts 的状态回退源（lead present 的代表图，仍在 state）。 */
function readStateArtifacts(
  fallback: ArtifactMeta[] | null | undefined,
): ArtifactMeta[] {
  return normalizeArtifacts(fallback);
}

export interface UseChartArtifactsArgs {
  /** state 回退：磁盘端点空/失败时，至少显示 lead present 的代表图。 */
  fallbackArtifacts?: ArtifactMeta[] | null;
}

export interface UseChartArtifactsResult {
  artifacts: ArtifactMeta[];
  /** 首次拉取完成（成功或回退），供消费方区分「加载中」与「真空」。 */
  loaded: boolean;
  /** 端点报错（已回退，消费方仍拿到数据，但可据此告警）。 */
  error: boolean;
  /** 重新拉取磁盘端点——run 完成（onFinish）后调一次确保全量（图陆续落盘）。 */
  refetch: () => Promise<void>;
}

/**
 * 对话流内嵌图廊的数据源 hook（spec §一修复）。
 *
 * 与 /gallery 独立页同走磁盘端点（fetchChartArtifactsFromDisk），磁盘空/失败时回退
 * fallbackArtifacts（state 代表图）。挂载时拉一次；refetch 供 run 完成后补全量。
 *
 * **性能**：只在挂载/refetch 时拉，不每帧拉、不轮询（spec §一「触发时机」）。对话流靠
 * present-files 消息出现 + run onFinish 两个事件驱动 refetch（见消费侧）。
 *
 * 与 useArtifactContent（单文件内容缓存）正交：那个拉单文件文本/图，这个拉全量图元数据列表。
 */
export function useChartArtifacts(
  threadId: string,
  args: UseChartArtifactsArgs = {},
): UseChartArtifactsResult {
  const { fallbackArtifacts } = args;
  const [artifacts, setArtifacts] = useState<ArtifactMeta[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState(false);

  const load = useCallback(async () => {
    let charts: ArtifactMeta[] = [];
    let failed = false;
    try {
      charts = await fetchChartArtifactsFromDisk(threadId);
    } catch {
      failed = true;
    }
    // 回退：磁盘空或失败时退回 state 代表图（同 /gallery）。
    if (charts.length === 0) {
      charts = readStateArtifacts(fallbackArtifacts);
    }
    setArtifacts(charts);
    setError(failed);
    setLoaded(true);
  }, [threadId, fallbackArtifacts]);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      await load();
      if (cancelled) return;
    })();
    return () => {
      cancelled = true;
    };
  }, [load]);

  return { artifacts, loaded, error, refetch: load };
}

/**
 * 报告产物磁盘端点（thread 资产面板）：拉 outputs/ 下 .md/.html 文档产物。
 * 与 fetchChartArtifactsFromDisk 对称——磁盘为真相，不依赖 state.artifacts 冒泡。
 * 端点不可达/非 200 抛错（由 useThreadAssets 的 catch 转空数组，报告区缺省不显示）。
 */
export async function fetchReportArtifactsFromDisk(threadId: string): Promise<ArtifactMeta[]> {
  const resp = await fetchWithAuth(reportsArtifactsURL(threadId));
  if (!resp.ok) {
    throw new Error(`reports artifacts endpoint ${resp.status}`);
  }
  const json = (await resp.json()) as ArtifactMeta[];
  return normalizeArtifacts(json);
}

/**
 * 指标结果表（data 类产物）磁盘端点（spec 2026-06-30 C1 模块2）。
 *
 * 拉 outputs/metrics_table.json 作为「指标结果表」data artifact，与 charts/reports 对称。
 * 端点不可达/非 200 抛错（由 useThreadAssets 的 catch 转空数组，指标表区缺省不显示）。
 * 旧后端无此端点时只让该类为空，不整体崩（charts/reports 仍显示）。
 */
export async function fetchDataArtifactsFromDisk(threadId: string): Promise<ArtifactMeta[]> {
  const resp = await fetchWithAuth(dataArtifactsURL(threadId));
  if (!resp.ok) {
    throw new Error(`data artifacts endpoint ${resp.status}`);
  }
  const json = (await resp.json()) as ArtifactMeta[];
  return normalizeArtifacts(json);
}

export interface UseThreadAssetsResult {
  /** 图表产物（磁盘 + plan_charts.json）。 */
  charts: ArtifactMeta[];
  /** 报告/文档产物（磁盘 .md/.html）。 */
  reports: ArtifactMeta[];
  /** 指标结果表 data 产物（磁盘 metrics_table.json，spec 2026-06-30 C1）。 */
  data: ArtifactMeta[];
  /** 首次拉取完成（成功或失败兜底），供消费方区分加载中 vs 真空。 */
  loaded: boolean;
  /** 重新拉取——run 完成（onFinish）后调一次确保全量（产物陆续落盘）。 */
  refetch: () => Promise<void>;
}

/**
 * thread 级「产出物」数据源 hook（资产面板）。
 *
 * 并行拉磁盘端点（charts + reports + data），合并成稳定的 thread 资产视图。**完全不读
 * streaming 消息流 / LangGraph state.artifacts**——产物 UI 因此不随 present_files 消息、
 * 流式 re-render、切 tab 而出现/消失/漂移（本会话 dogfood 反复暴露的不稳定家族根治）。
 *
 * 性能：挂载拉一次 + refetchSignal（run 完成）补拉一次；不每帧拉、不轮询。
 * 任一端点失败只让那一类产物为空，不整体崩（指标表端点旧后端没有时图/报告仍显示）。
 */
export function useThreadAssets(
  threadId: string,
  args: { refetchSignal?: number } = {},
): UseThreadAssetsResult {
  const { refetchSignal } = args;
  const [charts, setCharts] = useState<ArtifactMeta[]>([]);
  const [reports, setReports] = useState<ArtifactMeta[]>([]);
  const [data, setData] = useState<ArtifactMeta[]>([]);
  const [loaded, setLoaded] = useState(false);

  const load = useCallback(async () => {
    const [chartsRes, reportsRes, dataRes] = await Promise.allSettled([
      fetchChartArtifactsFromDisk(threadId),
      fetchReportArtifactsFromDisk(threadId),
      fetchDataArtifactsFromDisk(threadId),
    ]);
    setCharts(chartsRes.status === "fulfilled" ? chartsRes.value : []);
    setReports(reportsRes.status === "fulfilled" ? reportsRes.value : []);
    setData(dataRes.status === "fulfilled" ? dataRes.value : []);
    setLoaded(true);
  }, [threadId]);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      await load();
      if (cancelled) return;
    })();
    return () => {
      cancelled = true;
    };
  }, [load]);

  useEffect(() => {
    if (refetchSignal !== undefined && refetchSignal > 0) {
      void load();
    }
  }, [refetchSignal, load]);

  return { charts, reports, data, loaded, refetch: load };
}
