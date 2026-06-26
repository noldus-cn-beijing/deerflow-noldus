/**
 * deriveCapabilityPlan —— spec 2026-06-26-conversation-gallery-empty §二/三，方案 B「动态能力进度」。
 *
 * 与 derive-workflow-stages（固定 7 阶段线性推导）**并存、正交**：
 * - deriveWorkflowStages：端到端线性流水线，7 阶段 SSOT，「当前焦点唯一」语义（spec#4）。
 * - deriveCapabilityPlan（本函数）：决定**轨上要显示哪些阶段**——从本 thread 实际触发的 subagent
 *   dispatch / 关键 tool_call 动态推导，而非写死 7 阶段。
 *
 * 三态（spec §三方案 B）：
 *  - **知识问答 run**（无任何 pipeline 信号）→ 空 plan → AnalysisRail 不渲染（日式克制，不塞空轨）。
 *  - **chart-only run**（chart-maker dispatch / run_chart_plan，无 upload/paradigm/compute）→ 只显 charts。
 *  - **端到端 run** → 显本 run 实际触及的全部阶段（含 charts，若画过图）。
 *
 * 这天然治问题2（画图了还停在「指标计算」）：charts 阶段从实际 dispatch 推导，画图就推进到 charts。
 *
 * 工程纪律（守 spec#4 红线 + memory）：
 *  - **纯前端推导、零后端改动**：只读 TraceEvent[] + thread.messages，不碰 mergeMessages/流式核心。
 *  - **同源**：trace 事件来自 useRunTrace（spec#2 单一解析），本函数不另写工具语义解析——只把事件
 *    归到「这条 run 触发了哪些能力」。chart 信号与 derive-workflow-stages.isComputeSignal 同款判据，
 *    但拆出来独立成 charts 阶段（那里把 run_chart_plan 并进 compute 是问题2根因）。
 *  - **保守**：拿不准时宁可显（active），绝不假完成。
 *
 * 纯函数，刻意不依赖 React——单测喂 mock 直接断言。
 */

import type { Message } from "@langchain/langgraph-sdk";

import type { TraceEvent } from "@/core/trace";

import {
  CAPABILITY_STAGE_ORDER,
  CHART_STAGE_ID,
  type CapabilityStageId,
  isColumnAlignmentClarification,
} from "./stages";

/** 能力进度项：阶段 id（消费方用 stageDefOf 取静态元数据，用 deriveWorkflowStages 取状态）。 */
export interface CapabilityStageEntry {
  id: CapabilityStageId;
  /** 该阶段首个相关 message 的 id（点阶段 → scrollIntoView 定位，spec#4 §3.3）。 */
  anchorMessageId?: string;
}

/** dispatch 节点标题里携带的 subagent_type（与 derive-workflow-stages 同款解析）。 */
function subagentTypeOfTitle(title: string): string | undefined {
  const match = /\b([a-z][a-z-]+)\s*$/i.exec(title.trim());
  return match?.[1]?.toLowerCase();
}

/** 是否为图表生成信号（chart-maker dispatch / run_chart_plan 工具或 bash / chart 产物 png）。 */
function isChartSignal(event: TraceEvent): boolean {
  if (event.kind === "dispatch") {
    const type = subagentTypeOfTitle(event.title) ?? "";
    if (type.includes("chart-maker") || type.includes("chart_maker")) return true;
  }
  if (event.kind === "tool") {
    const args = (event.detail as { args?: Record<string, unknown> } | undefined)?.args;
    const name = (event.detail as { name?: string } | undefined)?.name;
    if (name === "run_chart_plan") return true;
    const command = typeof args?.command === "string" ? args.command : "";
    if (command.includes("run_chart_plan")) return true;
  }
  if (event.kind === "artifact") {
    // chart 产物 png 也算图表阶段信号（脱离 dispatch/tool 也能认到）。
    const filepaths = (event.detail as { filepaths?: string[] } | undefined)?.filepaths ?? [];
    return filepaths.some((p) => /\.png$/i.test(p));
  }
  return false;
}

/** thread 里是否出现过上传文件（与 derive-workflow-stages 同款）。 */
function threadHasUploadedFiles(messages: Message[]): { has: boolean; messageId?: string } {
  for (const message of messages) {
    if (message.type !== "human") continue;
    const ak = (message as { additional_kwargs?: Record<string, unknown> }).additional_kwargs;
    const files = ak?.files;
    if (Array.isArray(files) && files.length > 0) {
      return { has: true, messageId: message.id };
    }
    const content = typeof message.content === "string" ? message.content : "";
    if (content.includes("<uploaded_files>")) {
      return { has: true, messageId: message.id };
    }
  }
  return { has: false };
}

/** 列对齐反问是否触及（waiting 或 done 都算「触及」，显 align 阶段）。 */
function columnAlignmentTouched(
  messages: Message[],
): { touched: boolean; messageId?: string } {
  for (const message of messages) {
    if (message.type !== "ai") continue;
    const toolCalls =
      (message as { tool_calls?: { name: string; args: Record<string, unknown> }[] }).tool_calls ?? [];
    const clarify = toolCalls.find((tc) => tc.name === "ask_clarification");
    if (clarify && isColumnAlignmentClarification(clarify.args)) {
      return { touched: true, messageId: message.id };
    }
  }
  return { touched: false };
}

/** 取首个命中事件所在 message id（经 tool_call id 反查）。 */
function anchorOfEvent(
  events: TraceEvent[],
  predicate: (e: TraceEvent) => boolean,
  messageIndex: Map<string, string>,
): string | undefined {
  const event = events.find(predicate);
  if (!event) return undefined;
  const id = event.id;
  if (messageIndex.has(id)) return messageIndex.get(id);
  return undefined;
}

/** 是否为指标计算信号（与 derive-workflow-stages.isComputeSignal 同款，但排除 run_chart_plan）。 */
function isComputeOnlySignal(event: TraceEvent): boolean {
  if (event.kind === "dispatch") {
    const type = subagentTypeOfTitle(event.title) ?? "";
    return type.includes("code-executor") || type.includes("code_executor");
  }
  if (event.kind === "tool") {
    const args = (event.detail as { args?: Record<string, unknown> } | undefined)?.args;
    const name = (event.detail as { name?: string } | undefined)?.name;
    // 指标计算工具——排除 run_chart_plan（它归 charts）。
    if (name === "prep_metric_plan" || name === "run_metric_plan") return true;
    const command = typeof args?.command === "string" ? args.command : "";
    if (command.includes("ethoinsight.scripts.")) return true;
    if (/prep_metric_plan|run_metric_plan/.test(command)) return true;
  }
  if (event.kind === "artifact") {
    // 指标产物 json/csv（png 归 charts，这里排除）。
    const filepaths = (event.detail as { filepaths?: string[] } | undefined)?.filepaths ?? [];
    return filepaths.some((p) => /\.(json|csv)$/i.test(p));
  }
  return false;
}

function isReportSignal(event: TraceEvent): boolean {
  if (event.kind === "dispatch") {
    const type = subagentTypeOfTitle(event.title) ?? "";
    if (type.includes("report-writer") || type.includes("report_writer")) return true;
  }
  if (event.kind === "artifact") {
    const filepaths = (event.detail as { filepaths?: string[] } | undefined)?.filepaths ?? [];
    if (filepaths.some((p) => /report\.md$/i.test(p))) return true;
  }
  return false;
}

function isDataAnalystDispatch(event: TraceEvent): boolean {
  if (event.kind !== "dispatch") return false;
  const type = subagentTypeOfTitle(event.title) ?? "";
  return type.includes("data-analyst") || type.includes("data_analyst");
}

/** 建立 tool_call id / message id → 所在 message id 的索引（取首个命中）。 */
function buildMessageIndex(messages: Message[]): Map<string, string> {
  const index = new Map<string, string>();
  for (const message of messages) {
    if (!message.id) continue;
    index.set(message.id, message.id);
    const toolCalls = (message as { tool_calls?: { id?: string }[] }).tool_calls ?? [];
    for (const tc of toolCalls) {
      if (tc.id) index.set(tc.id, message.id);
    }
  }
  return index;
}

/**
 * 核心：从 TraceEvent[] + messages 派生**本 run 要显示的能力阶段集**（spec §三方案 B）。
 *
 * 规则：
 *  - 每条能力（upload/paradigm/align/compute/qc/interpret/charts/report）在本 thread 有信号即「触及」。
 *  - 知识问答（一条都没触及）→ 返回 []，消费方据此隐藏轨。
 *  - 返回按 CAPABILITY_STAGE_ORDER 稳定排序的阶段列表（不随事件到达顺序漂移）。
 *
 * @param events  来自 useRunTrace（spec#2 单一解析）的聚合事件
 * @param messages  thread.messages——upload/align 两条轻量信号 + 锚点反查
 */
export function deriveCapabilityPlan(
  events: TraceEvent[],
  messages: Message[],
): CapabilityStageEntry[] {
  const messageIndex = buildMessageIndex(messages);
  const touched: CapabilityStageEntry[] = [];

  // ① upload
  const upload = threadHasUploadedFiles(messages);
  if (upload.has) {
    touched.push({ id: "upload", ...(upload.messageId ? { anchorMessageId: upload.messageId } : {}) });
  }

  // ② paradigm
  const paradigmEvent = events.find((e) => e.kind === "paradigm");
  if (paradigmEvent) {
    const anchor = anchorOfEvent(events, (e) => e.kind === "paradigm", messageIndex);
    touched.push({ id: "paradigm", ...(anchor ? { anchorMessageId: anchor } : {}) });
  }

  // ③ align（列对齐反问触及即显）
  const align = columnAlignmentTouched(messages);
  if (align.touched) {
    touched.push({ id: "align", ...(align.messageId ? { anchorMessageId: align.messageId } : {}) });
  }

  // ④ compute（排除 run_chart_plan，它归 charts）
  if (events.some(isComputeOnlySignal)) {
    const anchor = anchorOfEvent(events, isComputeOnlySignal, messageIndex);
    touched.push({ id: "compute", ...(anchor ? { anchorMessageId: anchor } : {}) });
  }

  // ⑤ qc
  const gateEvent = events.find((e) => e.kind === "gate");
  if (gateEvent) {
    const anchor = anchorOfEvent(events, (e) => e.kind === "gate", messageIndex);
    touched.push({ id: "qc", ...(anchor ? { anchorMessageId: anchor } : {}) });
  }

  // ⑥ interpret
  if (events.some(isDataAnalystDispatch)) {
    const anchor = anchorOfEvent(events, isDataAnalystDispatch, messageIndex);
    touched.push({ id: "interpret", ...(anchor ? { anchorMessageId: anchor } : {}) });
  }

  // charts（能力阶段，独立判据——治问题2：画图了不再并进 compute）
  if (events.some(isChartSignal)) {
    const anchor = anchorOfEvent(events, isChartSignal, messageIndex);
    touched.push({ id: CHART_STAGE_ID, ...(anchor ? { anchorMessageId: anchor } : {}) });
  }

  // ⑦ report
  if (events.some(isReportSignal)) {
    const anchor = anchorOfEvent(events, isReportSignal, messageIndex);
    touched.push({ id: "report", ...(anchor ? { anchorMessageId: anchor } : {}) });
  }

  // 稳定排序（CAPABILITY_STAGE_ORDER），不随事件到达顺序漂移。
  return touched.sort(
    (a, b) => CAPABILITY_STAGE_ORDER[a.id] - CAPABILITY_STAGE_ORDER[b.id],
  );
}
