"use client";

import { ChevronDownIcon, ChevronRightIcon, DownloadIcon, TableIcon } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { fetch as fetchWithAuth } from "@/core/api/fetcher";
import { type ArtifactMeta } from "@/core/artifacts/types";
import { dataTableExportURL, metricsTableJSONURL } from "@/core/artifacts/utils";
import { useI18n } from "@/core/i18n/hooks";
import { cn } from "@/lib/utils";

/**
 * spec 2026-06-30 C1 模块3：指标结果表卡（overview-first）。
 *
 * 数据来自 /artifacts/metrics-table 端点（run_metric_plan 确定性导出的剥内脏 JSON）。
 * 与 report-card 同型：折叠态只渲染卡头，**首次展开**才 fetch JSON（懒加载，不给无指标
 * 的 thread 增加挂载开销）。
 *
 * **概览优先 + 性能红线（被测）**：
 * - 展开后默认**只渲染组综述层**（≈组数行，十几行，不卡）。
 * - 某组的 subject 明细仅在**该组展开时条件渲染**（非 CSS hide）——折叠/未展开组时
 *   DOM 不含上百 subject 行（行为学数据单范式常有 50-100+ subject）。
 * - 不上虚拟化（spec：50 行内无需；默认根本不渲染上百行）。
 *
 * **离群轻量标记（日式克制）**：IQR 判异（后端已算好 outlier_flags）→ 离群行圆点标记 +
 * 100% 不透明，非离群 ~70% 不透明。单一色盲安全强调色（#0072B2 蓝），绝不红绿、绝不只靠颜色。
 *
 * 本卡只做最小可用挂载（C1），完整画廊 layout 重设计/留白/层级归 C2。
 */

/** 离群强调色（Wong 色盲安全蓝）。 */
const OUTLIER_ACCENT = "#0072B2";

/** metrics_table.json 的干净结构（与后端 metrics_table_export.py 同构）。 */
interface MetricsTableJSON {
  paradigm?: string;
  metric_names?: string[];
  groups?: GroupSummary[];
  per_subject?: SubjectRow[];
}

interface GroupSummary {
  group: string;
  n: number;
  metrics?: Record<string, { mean?: number | null; std?: number | null; n?: number }>;
}

interface SubjectRow {
  subject: string;
  group: string;
  values?: Record<string, number | null>;
  outlier_flags?: Record<string, boolean>;
}

/** 一行是否在该 metric 上被标为离群（任一 metric 离群即整行高亮，便于扫读）。 */
function isRowOutlier(row: SubjectRow): boolean {
  return Object.values(row.outlier_flags ?? {}).some(Boolean);
}

/** 紧凑数值渲染（避免浮点尾部长串，3 位有效小数足够扫读；CSV 仍是全精度原始值）。 */
function fmtNum(v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  return Number.isFinite(v) ? (Math.round(v * 1000) / 1000).toString() : "—";
}

export function MetricsTableCard({ meta, threadId }: { meta: ArtifactMeta; threadId: string }) {
  const { t } = useI18n();
  const [expanded, setExpanded] = useState(false);
  const [table, setTable] = useState<MetricsTableJSON | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState(false);
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set());

  async function handleToggle() {
    const next = !expanded;
    setExpanded(next);
    if (next && table === null && !loadError) {
      setLoading(true);
      try {
        const resp = await fetchWithAuth(metricsTableJSONURL(threadId));
        if (!resp.ok) throw new Error(`metrics-table endpoint ${resp.status}`);
        setTable((await resp.json()) as MetricsTableJSON);
      } catch {
        setLoadError(true);
      } finally {
        setLoading(false);
      }
    }
  }

  function toggleGroup(group: string) {
    setExpandedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(group)) {
        next.delete(group);
      } else {
        next.add(group);
      }
      return next;
    });
  }

  const metricNames = table?.metric_names ?? [];
  const groups = table?.groups ?? [];
  const perSubject = table?.per_subject ?? [];

  return (
    <div className="rounded-lg border bg-background p-3" data-testid="metrics-table-card">
      <div className="flex items-center gap-2">
        <TableIcon className="size-4 shrink-0 text-primary" />
        <span className="text-sm font-medium">{meta.filename ?? t.gallery.metricsTableTitle}</span>
        <div className="ml-auto flex items-center gap-2">
          <Button type="button" variant="ghost" size="sm" className="gap-1" onClick={handleToggle}>
            {expanded ? <ChevronDownIcon className="size-4" /> : <ChevronRightIcon className="size-4" />}
            {t.gallery.metricsTableExpand}
          </Button>
          <Button asChild type="button" variant="outline" size="sm" className="gap-1">
            <a href={dataTableExportURL(threadId)} target="_blank" rel="noopener noreferrer">
              <DownloadIcon className="size-4" />
              {t.gallery.downloadCsv}
            </a>
          </Button>
        </div>
      </div>

      {expanded && (
        <div className="mt-3 border-t pt-3">
          {loading ? (
            <p className="text-muted-foreground text-sm">{t.gallery.metricsTableLoading}</p>
          ) : loadError || groups.length === 0 ? (
            <p className="text-muted-foreground text-sm">{t.gallery.metricsTableEmpty}</p>
          ) : (
            <div className="flex flex-col gap-3">
              {/* 综述层：默认渲染此层（≈组数行）。subject 明细仅在下方某组展开时条件渲染。 */}
              {groups.map((g) => {
                const gName = g.group;
                const isOpen = expandedGroups.has(gName);
                const rows = perSubject.filter((r) => r.group === gName);
                return (
                  <div key={gName} className="rounded-md border p-2">
                    <button
                      type="button"
                      onClick={() => toggleGroup(gName)}
                      className="flex w-full items-center gap-2 text-left"
                      data-testid={`metrics-group-${gName}`}
                    >
                      {isOpen ? (
                        <ChevronDownIcon className="size-3.5 shrink-0 text-muted-foreground" />
                      ) : (
                        <ChevronRightIcon className="size-3.5 shrink-0 text-muted-foreground" />
                      )}
                      <span className="text-sm font-medium">{gName}</span>
                      <span className="text-muted-foreground text-xs tabular-nums">
                        {t.gallery.metricsTableGroupN(g.n)}
                      </span>
                    </button>

                    {/* 组级指标综述（mean ± std）——始终渲染，便宜、行数=组数×指标数。 */}
                    <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 pl-5">
                      {metricNames.map((m) => {
                        const stat = g.metrics?.[m];
                        const mean = stat?.mean;
                        const std = stat?.std;
                        return (
                          <div key={m} className="text-xs">
                            <span className="text-muted-foreground">{m}: </span>
                            <span className="tabular-nums">{fmtNum(mean)}</span>
                            {std !== null && std !== undefined && (
                              <span className="text-muted-foreground tabular-nums"> ± {fmtNum(std)}</span>
                            )}
                          </div>
                        );
                      })}
                    </div>

                    {/* subject 明细：仅该组展开时条件渲染（性能红线）。 */}
                    {isOpen && rows.length > 0 && (
                      <ul className="mt-2 flex flex-col gap-0.5 pl-5" data-testid={`metrics-group-rows-${gName}`}>
                        {rows.map((r) => {
                          const outlier = isRowOutlier(r);
                          return (
                            <li
                              key={r.subject}
                              className={cn(
                                "flex items-center gap-2 text-xs tabular-nums",
                                outlier ? "opacity-100" : "opacity-70",
                              )}
                            >
                              {outlier ? (
                                <span
                                  title={t.gallery.outlierFlag}
                                  aria-label={t.gallery.outlierFlag}
                                  className="inline-block size-1.5 shrink-0 rounded-full"
                                  style={{ backgroundColor: OUTLIER_ACCENT }}
                                  data-testid="outlier-marker"
                                />
                              ) : (
                                <span className="inline-block size-1.5 shrink-0" />
                              )}
                              <span className="tabular-nums">{r.subject}</span>
                              {metricNames.map((m) => (
                                <span key={m} className="text-muted-foreground tabular-nums">
                                  {fmtNum(r.values?.[m])}
                                </span>
                              ))}
                            </li>
                          );
                        })}
                      </ul>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
