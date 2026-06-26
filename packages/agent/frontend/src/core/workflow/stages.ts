/**
 * 工作流 7 阶段定义 —— 进度轨（spec#4）/ 决策卡（spec#5）/ 画廊标阶段的**单一来源**。
 *
 * 见 spec `docs/superpowers/specs/2026-06-24-frontend-phase0-4-analysis-rail-spec.md` §一。
 *
 * 这里只定义「7 阶段是哪 7 个、叫什么、什么色、列对齐反问怎么认」——**纯静态元数据 + 一个
 * 纯函数启发**。阶段状态（pending/active/waiting/done/...）由 `use-workflow-stages.ts` 从
 * `useRunTrace` 派生，不在这里。
 *
 * 工程纪律（CLAUDE.md「同一份知识绝不双存」+ memory `feedback_single_source_of_truth`）：
 * 任何组件要枚举这 7 阶段，都从这里引，不在各处重复硬编码。阶段色 token 复用 spec#1 的
 * `--color-stage-*`（不在本文件重定义）。
 */

import {
  ChartColumnBigIcon,
  ClipboardListIcon,
  FlaskConicalIcon,
  GitCompareArrowsIcon,
  LineChartIcon,
  type LucideIcon,
  NotebookPenIcon,
  ScanSearchIcon,
  ShieldCheckIcon,
} from "lucide-react";

/** 7 个阶段 id（spec §一表）。顺序即工作流顺序，固定。 */
export const WORKFLOW_STAGE_IDS = {
  upload: 0,
  paradigm: 1,
  align: 2,
  compute: 3,
  qc: 4,
  interpret: 5,
  report: 6,
} as const;

export type WorkflowStageId = keyof typeof WORKFLOW_STAGE_IDS;

/**
 * 能力阶段「图表生成」id（spec 2026-06-26-conversation-gallery-empty §二/三，方案 B 动态能力进度）。
 *
 * 它对应 chart-maker / run_chart_plan 这条**独立能力**——可脱离端到端流水线单独发生
 * （追问画图、chart-only run）。它**不是固定 7 阶段流水线的一环**：那 7 阶段是端到端线性推导的
 * SSOT（derive-workflow-stages），加第 8 项会破坏既有线性化测试与 spec#4 的「当前焦点唯一」语义。
 * 故 charts 走 capability-plan 动态显示，不进 WORKFLOW_STAGE_IDS 的固定序。
 */
export const CHART_STAGE_ID = "charts" as const;

/** 所有可在能力进度轨上出现的阶段 id（固定 7 + 动态 charts），供类型收口。 */
export type CapabilityStageId = WorkflowStageId | typeof CHART_STAGE_ID;

/** 单个阶段的静态元数据。colorToken 引用 spec#1 的 CSS 变量，不在此重定义色值。 */
export interface WorkflowStageDef {
  id: WorkflowStageId;
  /** i18n 文案 key —— 走 t.workflowStages.names.<nameKey>，绝不在组件里硬编码中文。 */
  nameKey: (typeof WORKFLOW_STAGE_NAME_KEYS)[WorkflowStageId];
  /** lucide 图标（color-not-only：色+图标+文字三件套之一）。 */
  icon: LucideIcon;
  /** 该阶段的低饱和 hue，引用 spec#1 `--color-stage-*` 变量。 */
  colorToken: string;
}

/** i18n 文案 key 集合（与 WORKFLOW_STAGE_IDS 同序，供 t.workflowStages.names 索引）。 */
export const WORKFLOW_STAGE_NAME_KEYS = {
  upload: "upload",
  paradigm: "paradigm",
  align: "align",
  compute: "compute",
  qc: "qc",
  interpret: "interpret",
  report: "report",
} as const;

/**
 * 7 阶段 SSOT（spec §一表）。
 * 顺序固定 = 工作流时序；colorToken 复用 spec#1，icon 选用语义贴近的 lucide 图标。
 */
export const WORKFLOW_STAGES: readonly WorkflowStageDef[] = [
  {
    id: "upload",
    nameKey: "upload",
    icon: ClipboardListIcon,
    colorToken: "var(--color-stage-upload)",
  },
  {
    id: "paradigm",
    nameKey: "paradigm",
    icon: FlaskConicalIcon,
    colorToken: "var(--color-stage-paradigm)",
  },
  {
    id: "align",
    nameKey: "align",
    icon: GitCompareArrowsIcon,
    colorToken: "var(--color-stage-align)",
  },
  {
    id: "compute",
    nameKey: "compute",
    icon: LineChartIcon,
    colorToken: "var(--color-stage-compute)",
  },
  {
    id: "qc",
    nameKey: "qc",
    icon: ShieldCheckIcon,
    colorToken: "var(--color-stage-qc)",
  },
  {
    id: "interpret",
    nameKey: "interpret",
    icon: ScanSearchIcon,
    colorToken: "var(--color-stage-interpret)",
  },
  {
    id: "report",
    nameKey: "report",
    icon: NotebookPenIcon,
    colorToken: "var(--color-stage-report)",
  },
];

/**
 * 能力阶段「图表生成」静态元数据（spec 2026-06-26 §二/三 方案 B）。
 *
 * 与 WORKFLOW_STAGES 同构（id/nameKey/icon/colorToken），但**独立列出**——它由
 * capability-plan 动态加入显示集，不参与 derive-workflow-stages 的 7 阶段线性推导。
 * colorToken 复用 spec#1 的图表色变量；nameKey 走 t.workflowStages.names.charts（i18n 已加）。
 */
export interface CapabilityStageDef {
  id: CapabilityStageId;
  nameKey: string;
  icon: LucideIcon;
  colorToken: string;
}

export const CHART_STAGE_DEF: CapabilityStageDef = {
  id: CHART_STAGE_ID,
  nameKey: "charts",
  icon: ChartColumnBigIcon,
  colorToken: "var(--color-stage-interpret)",
};

/**
 * 按 CapabilityStageId 取静态阶段定义（spec 2026-06-26 方案 B）。
 * 固定 7 阶段走 WORKFLOW_STAGES；charts 走 CHART_STAGE_DEF。单一来源：组件枚举阶段定义
 * 只经此函数，不在各处硬编码。
 */
export function stageDefOf(id: CapabilityStageId): CapabilityStageDef {
  if (id === CHART_STAGE_ID) return CHART_STAGE_DEF;
  const def = WORKFLOW_STAGES.find((s) => s.id === id);
  if (!def) throw new Error(`unknown capability stage id: ${id}`);
  return def;
}

/**
 * 能力阶段在动态轨上的**显示顺序**（spec §二/三）。
 *
 * 固定 7 阶段按 WORKFLOW_STAGE_IDS 序；charts 排在 report 之前、interpret 之后（图表是
 * 解读的可视化产出）。capability-plan 按此序稳定排序，不随事件到达顺序漂移。
 */
export const CAPABILITY_STAGE_ORDER: Record<CapabilityStageId, number> = {
  ...WORKFLOW_STAGE_IDS,
  [CHART_STAGE_ID]: 5.5, // interpret(5) < charts(5.5) < report(6)
};

/**
 * 列语义对齐反问（阶段 ③）的关键词启发（spec §3.1 / §六风险1）。
 *
 * `ask_clarification` 的 `clarification_type` 是 5 类通用枚举（missing_info /
 * ambiguous_requirement / approach_choice / risk_confirmation / suggestion）——
 * 没有专门的「列对齐」类型。列对齐反问由 `ethoinsight-column-confirmation` skill 产生，
 * question/options 里会出现中英文的列/区/中心/边缘术语。本启发是**前端兜底**（spec §3.1），
 * 不改后端 Literal。
 *
 * 同时覆盖中英文：研究员可能用任一语言与 agent 对话。
 */
export const COLUMN_ALIGNMENT_HINTS = [
  // 中文（ethoinsight-column-confirmation skill 用语）
  "分析区",
  "区域",
  "列",
  "中心区",
  "边缘区",
  "中央",
  "开放臂",
  "闭合臂",
  // 英文
  "column",
  "zone",
  "arena",
  "center",
  "centre",
  "periphery",
  "open arm",
  "closed arm",
] as const;

/** ask_clarification 调用的 args 局部视图（只取判列对齐需要的字段）。 */
export interface ClarificationArgs {
  clarification_type?: unknown;
  question?: unknown;
  options?: unknown;
  context?: unknown;
}

/**
 * 判断一次 ask_clarification 是否属「列语义对齐」（阶段 ③ 的 waiting/done 信号）。
 *
 * 判定（spec §3.1：clarification_type + question/options 关键词双判）：
 *  - 把 question + options + context 拼成一段小写文本；
 *  - 命中任一 COLUMN_ALIGNMENT_HINTS 关键词 → true。
 *
 * 保守原则（spec §六风险1）：拿不准时返回 false——让阶段 ③ 维持 active（「还在进行」），
 * 绝不因误判把未完成标成 done。
 */
export function isColumnAlignmentClarification(args: ClarificationArgs): boolean {
  const parts: string[] = [];
  if (typeof args.question === "string") parts.push(args.question);
  if (typeof args.context === "string") parts.push(args.context);
  if (Array.isArray(args.options)) {
    for (const opt of args.options) {
      if (typeof opt === "string") parts.push(opt);
    }
  }
  const haystack = parts.join(" ").toLowerCase();
  if (!haystack) return false;
  return COLUMN_ALIGNMENT_HINTS.some((hint) => haystack.includes(hint.toLowerCase()));
}
