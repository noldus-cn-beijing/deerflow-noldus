import { getBackendBaseURL } from "../config";
import type { AgentThread } from "../threads";

import type { ArtifactMeta } from "./types";
import { normalizeArtifacts } from "./types";

export function urlOfArtifact({
  filepath,
  threadId,
  download = false,
  isMock = false,
}: {
  filepath: string;
  threadId: string;
  download?: boolean;
  isMock?: boolean;
}) {
  if (isMock) {
    return `${getBackendBaseURL()}/mock/api/threads/${threadId}/artifacts${filepath}${download ? "?download=true" : ""}`;
  }
  return `${getBackendBaseURL()}/api/threads/${threadId}/artifacts${filepath}${download ? "?download=true" : ""}`;
}

/**
 * 提取并归一化 thread 的 artifacts（spec phase0-3）。
 *
 * thread.values.artifacts 现是 ArtifactInput[]（裸 string | ArtifactMeta）；一律经
 * normalizeArtifacts 兜底成 ArtifactMeta[]，老数据（裸 string）不崩。
 */
export function extractArtifactsFromThread(thread: AgentThread): ArtifactMeta[] {
  return normalizeArtifacts(thread.values.artifacts);
}

/** 第 1 层主路径「下载全部 ZIP」端点（spec §3.1.7，零渲染）。 */
export function archiveArtifactsURL(threadId: string): string {
  return `${getBackendBaseURL()}/api/threads/${threadId}/artifacts/archive`;
}

/**
 * 画廊全量图端点（spec 2026-06-26-artifact-bubbling §1.1）。
 *
 * 数据源 = 磁盘 + plan_charts.json，**不依赖 LangGraph state 冒泡**（subagent→lead
 * 边界会丢 state.artifacts，实测 113 张只活 2 张）。画廊调此端点拉全部图，磁盘有几张
 * 就显示几张。返回 ArtifactMeta[]（chart 元数据来自后端，前端不正则猜分类）。
 */
export function chartsArtifactsURL(threadId: string): string {
  return `${getBackendBaseURL()}/api/threads/${threadId}/artifacts/charts`;
}

/**
 * 报告产物清单端点（thread 资产面板）：磁盘 outputs/ 下 .md/.html → [{path,kind,filename,ext}]。
 * 与 chartsArtifactsURL 对称，同样磁盘为真相、不依赖 state 冒泡。
 */
export function reportsArtifactsURL(threadId: string): string {
  return `${getBackendBaseURL()}/api/threads/${threadId}/artifacts/reports`;
}

/** 数据表 CSV 导出端点占位（spec §四 Step 5）。 */
export function dataTableExportURL(threadId: string): string {
  return `${getBackendBaseURL()}/api/threads/${threadId}/artifacts/data-table`;
}

/**
 * 报告多格式导出端点（spec 2026-06-29-report-export-formats-impl）。
 *
 * 把线程 outputs/report.html 转成 pdf / docx / tex 返回 attachment。与
 * archiveArtifactsURL / reportsArtifactsURL 同基础（getBackendBaseURL + /api/threads/…）。
 */
export function reportExportURL(threadId: string, format: "pdf" | "docx" | "tex"): string {
  return `${getBackendBaseURL()}/api/threads/${threadId}/artifacts/report/export?format=${format}`;
}

/**
 * Normalize an image src from report.md into an artifact API filepath.
 *
 * 规范形态（SSOT，2026-06-18）：report.md 内图片路径一律为带前导斜杠的虚拟
 * 绝对路径 ``/mnt/user-data/outputs/<name>.png``。后端 seal 的两个产出点
 * （placeholder 解析 + path normalize）已统一到这一形态，前端只认这一种。
 * 详见 docs/superpowers/specs/2026-06-18-report-image-path-ssot-spec.md。
 *
 * - 规范 src ``/mnt/user-data/…`` → 原样返回（直接交给 resolveArtifactURL，
 *   拼成 ``/api/threads/{tid}/artifacts/mnt/user-data/…``，后端
 *   resolve_virtual_path 内部 lstrip 后命中 ``mnt/user-data/`` 前缀）。
 * - 外链 ``http(s)://`` → 返回 null（调用方应原样使用 src）。
 * - 其余非规范形态 → 返回 null（响亮失败：调用方原样渲染让其 404 暴露，
 *   不再猜测/兜底，便于发现 report-writer 写错路径）。
 *
 * 历史 case 2/3/4/5（host 绝对路径 / outputs/ / 裸名 / /outputs/）已删除——
 * 它们是路径约定漂移之源（spec §2.3）。
 */
export function normalizeArtifactImageSrc(src: string): string | null {
  if (/^https?:\/\//i.test(src)) return null;

  // 规范形态：/mnt/user-data/… → 原样作为 artifact filepath（带前导斜杠）。
  if (src.startsWith("/mnt/user-data/")) {
    return src;
  }

  // 非规范形态：不猜测、不兜底，返回 null 让调用方原样暴露。
  return null;
}

export function resolveArtifactURL(absolutePath: string, threadId: string) {
  return `${getBackendBaseURL()}/api/threads/${threadId}/artifacts${absolutePath}`;
}
