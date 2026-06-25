/**
 * ArtifactMeta 契约（spec 2026-06-24-frontend-phase0-3-artifact-gallery §3.1，决策1=路 A）。
 *
 * artifacts 从 string[] 升级成 ArtifactMeta[]：chart 产物带 chart 元数据（来自后端
 * plan_charts.json SSOT），报告/技能等无元数据产物仍可经 normalizeArtifact 兜底成
 * { path }。向后兼容：老 thread state 里的裸 string 仍合法。
 *
 * SSOT 守则：chart 元数据只来自后端（present_file 关联 plan_charts.json），前端不正则猜分类。
 */

export type ArtifactKind = "chart" | "report" | "data" | "skill" | "other";

export type ChartOutputMode = "aggregate" | "per_subject";

export interface ArtifactMeta {
  /** 唯一标识 + URL 基础（必有，向后兼容锚点）。 */
  path: string;
  kind?: ArtifactKind;
  // 以下 chart 专属，来自 plan_charts.json
  chart_id?: string;
  output_mode?: ChartOutputMode;
  paradigm?: string;
  metric?: string;
  subject?: string;
  group?: string;
  chart_type?: string;
  /** 后端 Pillow 缩略图（spec §3.1.6）；缺则前端退化原 path + decoding=async。 */
  thumb_path?: string;
  /** 可选增强（按 run 分面，spec §3.4.1 方案 B）；Phase 0 默认不填。 */
  run_id?: string;
}

/** 向后兼容：老数据是 string → 归一化成 { path }。 */
export type ArtifactInput = string | ArtifactMeta;

/** 把任意 string | ArtifactMeta 归一化成 ArtifactMeta。漏改的消费方退化（旧 string 仍渲染，不崩）。 */
export function normalizeArtifact(a: ArtifactInput): ArtifactMeta {
  return typeof a === "string" ? { path: a } : a;
}

/** 把 ArtifactInput[] 归一化成 ArtifactMeta[]。 */
export function normalizeArtifacts(list: ArtifactInput[] | null | undefined): ArtifactMeta[] {
  return (list ?? []).map(normalizeArtifact);
}

/** 图文件扩展名集合（复用 ArtifactFileList 的 IMAGE_EXTENSIONS 语义）。 */
const IMAGE_EXTENSIONS = new Set([".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp"]);

/** 是否为图片产物（按 path 扩展名）。 */
export function isImageArtifact(meta: ArtifactMeta): boolean {
  const ext = meta.path.slice(meta.path.lastIndexOf(".")).toLowerCase();
  return IMAGE_EXTENSIONS.has(ext);
}

/**
 * 选取聊天流 inline 代表图（确定性规则，spec §3.2）：
 * - output_mode === "aggregate" 的图全部 inline（通常 ≤6）。
 * - 若无 aggregate，退化为「每个 (paradigm, metric) 的第一张」。
 * - 其余进画廊。
 * 纯前端按 ArtifactMeta 算，无需 agent 决策。
 */
export function selectRepresentativeCharts(metas: ArtifactMeta[]): ArtifactMeta[] {
  const images = metas.filter(isImageArtifact);
  const aggregate = images.filter((m) => m.output_mode === "aggregate");
  if (aggregate.length > 0) {
    return aggregate;
  }
  // 退化：每个 (paradigm, metric) 的第一张。
  const seen = new Set<string>();
  const picked: ArtifactMeta[] = [];
  for (const m of images) {
    const key = `${m.paradigm ?? ""}|${m.metric ?? ""}`;
    if (!seen.has(key)) {
      seen.add(key);
      picked.push(m);
    }
  }
  return picked;
}
