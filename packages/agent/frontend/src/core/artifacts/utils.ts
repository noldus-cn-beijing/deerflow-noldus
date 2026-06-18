import { getBackendBaseURL } from "../config";
import type { AgentThread } from "../threads";

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

export function extractArtifactsFromThread(thread: AgentThread) {
  return thread.values.artifacts ?? [];
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
