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

export function normalizeArtifactImageSrc(src: string): string | null {
  if (/^https?:\/\//i.test(src)) return null;

  // Case 1: /mnt/user-data/... → strip /mnt prefix, becomes /user-data/outputs/X.png
  //         then extract everything after /user-data → /outputs/X.png
  if (src.startsWith("/mnt/user-data/")) {
    const rest = src.slice("/mnt/user-data".length); // e.g. /outputs/X.png
    return rest.startsWith("/") ? rest : `/${rest}`;
  }

  // Case 2: Host absolute path containing /user-data/outputs/
  //         e.g. /home/wq/.../threads/<tid>/user-data/outputs/X.png
  const udIdx = src.indexOf("/user-data/outputs/");
  if (udIdx !== -1) {
    return src.slice(udIdx + "/user-data".length); // → /outputs/X.png
  }

  // Case 3: Relative path starting with outputs/
  if (src.startsWith("outputs/")) {
    return `/${src}`;
  }

  // Case 4: Bare filename like plot_X.png — assume it lives in /outputs/
  if (!src.startsWith("/") && src.length > 0) {
    return `/outputs/${src}`;
  }

  // Case 5: Already a virtual path like /outputs/X.png
  if (src.startsWith("/outputs/")) {
    return src;
  }

  return null;
}

export function resolveArtifactURL(absolutePath: string, threadId: string) {
  return `${getBackendBaseURL()}/api/threads/${threadId}/artifacts${absolutePath}`;
}
