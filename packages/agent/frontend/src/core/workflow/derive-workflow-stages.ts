/**
 * deriveWorkflowStages —— 把 spec#2 的 TraceEvent[] + thread.messages **降维投影**成
 * 7 阶段进度状态（spec §3.1）。
 *
 * 这是进度轨（spec#4）的数据核心。设计纪律（spec §3.1 / §六红线）：
 * - **同源**：trace 事件由 `buildRunTrace`（spec#2 单一解析）产出，本函数**不另写解析**——
 *   只把几十个事件归并到 7 桶（spec §3.1「降维投影」）。仅有的两条额外信号（upload 是否有
 *   文件、列对齐反问是否已被回答）是对同份 messages 的两个轻量谓词，不重复 trace 的工具语义
 *   解析（否则两处算法漂移，memory `feedback_handoff_metrics_field_divergence` 类教训）。
 * - **只读派生**：不写 state、不碰 submit/merge/dedupe。
 * - **保守**（spec §六风险1）：拿不准时宁可显 active/「还在进行」，绝不假完成。
 *
 * 纯函数，刻意不依赖 React——单测喂 mock 直接断言（spec §四 Step 2）。
 */

import type { Message } from "@langchain/langgraph-sdk";

import type { TraceEvent } from "@/core/trace";

import { WORKFLOW_STAGES, isColumnAlignmentClarification } from "./stages";

/**
 * 阶段状态机（spec §一「阶段状态机」）。
 * - pending  未开始（灰）
 * - active   当前焦点（品牌绿 + 描边呼吸）
 * - waiting  等待 HITL（琥珀脉冲）
 * - done     已完成（✓）
 * - warning  该阶段有非阻断质检警告（黄）
 * - failed   阻断/失败（红）
 */
export type StageStatus = "pending" | "active" | "waiting" | "done" | "warning" | "failed";

/** 单个阶段的派生状态。anchorMessageId 供点击滚动定位（spec §3.3）。 */
export interface StageState {
  id: (typeof WORKFLOW_STAGES)[number]["id"];
  status: StageStatus;
  /** 该阶段首个相关 message 的 id（点阶段 → scrollIntoView 定位）。 */
  anchorMessageId?: string;
}

/** dispatch 节点标题里携带的 subagent_type（buildRunTrace 用 `dispatch <type>` 作标题）。 */
function subagentTypeOfTitle(title: string): string | undefined {
  // buildRunTrace 用 t.toolCalls.stageBroadcast.dispatchSubagent(type) 作标题；
  // mock/真实文案都是 "dispatch <type>" / "派遣 <type>" 形态。取末段作 type 名兜底匹配。
  const match = /\b([a-z][a-z-]+)\s*$/i.exec(title.trim());
  return match?.[1]?.toLowerCase();
}

/** 是否为指标计算相关 bash/工具信号（spec §一表 ④：prep/run_metric_plan、code-executor、.json/.png）。 */
function isComputeSignal(event: TraceEvent): boolean {
  if (event.kind === "dispatch") {
    const type = subagentTypeOfTitle(event.title) ?? "";
    return type.includes("code-executor") || type.includes("code_executor");
  }
  if (event.kind === "tool") {
    const args = (event.detail as { args?: Record<string, unknown> } | undefined)?.args;
    const name = (event.detail as { name?: string } | undefined)?.name;
    // 指标计算工具名直接命中
    if (name === "prep_metric_plan" || name === "run_metric_plan" || name === "run_chart_plan") {
      return true;
    }
    const command = typeof args?.command === "string" ? args.command : "";
    if (command.includes("ethoinsight.scripts.")) return true;
    if (/prep_metric_plan|run_metric_plan|run_chart_plan/.test(command)) return true;
  }
  if (event.kind === "artifact") {
    // .json/.png 产物也算指标阶段产物（spec §一表 ④）
    const filepaths = (event.detail as { filepaths?: string[] } | undefined)?.filepaths ?? [];
    return filepaths.some((p) => /\.(json|png|csv)$/i.test(p));
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

/**
 * 计算「这条 ask_clarification 是否已被用户回答」。
 *
 * ClarificationMiddleware 用 Command(goto=END) 中断——用户回答以**后续 human message** 形式
 * 抵达，而非 ToolMessage 结果（spec §3.1）。故「已回答」= 该 clarification 所在 AI message
 * 之后还存在 human message。
 *
 * 返回值按 AI message 在 messages 里的 index 索引，供 stage ③ 判 waiting/done。
 */
function answeredClarificationAfter(
  messages: Message[],
): { isColumnAlignment: boolean; answered: boolean; messageId?: string } {
  // 取最近一次「列对齐」类 ask_clarification（多次时以最近一次为准——阶段状态反映当前焦点）。
  let clarIndex = -1;
  let clarMessageId: string | undefined;
  for (let i = 0; i < messages.length; i++) {
    const message = messages[i]!;
    if (message.type !== "ai") continue;
    const toolCalls =
      (message as { tool_calls?: { name: string; args: Record<string, unknown> }[] }).tool_calls ?? [];
    const clarify = toolCalls.find((tc) => tc.name === "ask_clarification");
    if (clarify && isColumnAlignmentClarification(clarify.args)) {
      clarIndex = i;
      clarMessageId = message.id;
    }
  }

  if (clarIndex === -1) {
    return { isColumnAlignment: false, answered: false };
  }

  // 是否存在该 AI message 之后的 human message（用户回答）
  let answered = false;
  for (let i = clarIndex + 1; i < messages.length; i++) {
    if (messages[i]!.type === "human") {
      answered = true;
      break;
    }
  }

  return {
    isColumnAlignment: true,
    answered,
    messageId: clarMessageId,
  };
}

/** thread 里是否出现过上传文件（human message 带 files / additional_kwargs.files）。 */
function threadHasUploadedFiles(messages: Message[]): { has: boolean; messageId?: string } {
  for (const message of messages) {
    if (message.type !== "human") continue;
    const ak = (message as { additional_kwargs?: Record<string, unknown> }).additional_kwargs;
    const files = ak?.files;
    if (Array.isArray(files) && files.length > 0) {
      return { has: true, messageId: message.id };
    }
    // 旧式 <uploaded_files> 标签（与 message-list-item.tsx 同款兼容）
    const content = typeof message.content === "string" ? message.content : "";
    if (content.includes("<uploaded_files>")) {
      return { has: true, messageId: message.id };
    }
  }
  return { has: false };
}

/** 在 events 里按谓词找首个命中，返回其对应的 message id（经 messageIndex 反查，spec §3.3 锚点）。 */
function anchorFromEvent(
  events: TraceEvent[],
  predicate: (e: TraceEvent) => boolean,
  messageIndex: Map<string, string>,
): string | undefined {
  const event = events.find(predicate);
  return event ? messageIdOfEvent(event, messageIndex) : undefined;
}

/** 单阶段「原始 outcome」（线性化前的独立判定）。 */
type RawOutcome = StageStatus;

function rawOutcomes(
  events: TraceEvent[],
  messages: Message[],
): RawOutcome[] {
  const out: RawOutcome[] = new Array(WORKFLOW_STAGES.length).fill("pending");

  // ① upload
  const upload = threadHasUploadedFiles(messages);
  if (upload.has) out[0] = "done";

  // ② paradigm
  const paradigmEvent = events.find((e) => e.kind === "paradigm");
  if (paradigmEvent) out[1] = "done";

  // ③ align（列对齐）：waiting=未答 / done=已答；非列对齐反问不影响
  const clarify = answeredClarificationAfter(messages);
  if (clarify.isColumnAlignment) {
    out[2] = clarify.answered ? "done" : "waiting";
  }

  // ④ compute
  const computeEvent = events.find(isComputeSignal);
  if (computeEvent) out[3] = "done";

  // ⑤ qc：gate 节点 critical+blocks→failed / warning→warning / info→done
  const gateEvent = events.find((e) => e.kind === "gate");
  if (gateEvent) {
    out[4] = gateEvent.status === "failed" ? "failed" : gateEvent.status === "warning" ? "warning" : "done";
  }

  // ⑥ interpret：data-analyst dispatch running→active / failed→failed / ok→done
  const daEvent = events.find(isDataAnalystDispatch);
  if (daEvent) {
    out[5] =
      daEvent.status === "running" ? "active" : daEvent.status === "failed" ? "failed" : "done";
  }

  // ⑦ report：report-writer dispatch ok / report.md artifact → done；running→active；failed→failed
  const reportEvent = events.find(isReportSignal);
  if (reportEvent) {
    out[6] =
      reportEvent.status === "running"
        ? "active"
        : reportEvent.status === "failed"
          ? "failed"
          : "done";
  }

  return out;
}

/**
 * 把每阶段独立 outcome 线性化成「单一焦点」的进度状态（spec §3.2：当前阶段唯一视觉主角）。
 *
 * 模型：工作流是线性推进的——存在唯一的 focus（active / waiting / failed）。
 * - lastReached = 最深触及的阶段（任何非 pending outcome）。
 * - focus：
 *   - 若 lastReached 的 outcome 是 parked（waiting/failed）或 active → focus 停在该阶段。
 *   - 若 lastReached 是 done/warning → focus 推进到下一阶段（变 active）；若已是最后阶段则无 focus。
 * - focus 之前的阶段：保留 warning/failed 标记，其余 done（含被「跳过」的回填 done）。
 * - focus 之后：pending。
 * - 空 thread（无任何触及）：全部 pending，无 focus（spec：保守，不假完成）。
 */
function linearize(raw: RawOutcome[]): StageStatus[] {
  const status: StageStatus[] = new Array(WORKFLOW_STAGES.length).fill("pending");

  let lastReached = -1;
  for (let i = 0; i < raw.length; i++) {
    if (raw[i] !== "pending") lastReached = i;
  }

  if (lastReached === -1) {
    // 空线程：全 pending（无 active 焦点）
    return status;
  }

  const lastOutcome = raw[lastReached]!;
  const parked = lastOutcome === "waiting" || lastOutcome === "failed" || lastOutcome === "active";

  const finalizeReached = (i: number) =>
    raw[i] === "warning" || raw[i] === "failed" ? (raw[i] as StageStatus) : "done";

  if (parked) {
    // 焦点停在 lastReached（waiting/failed/active）：之前全 done（保留标记），之后 pending
    for (let i = 0; i < WORKFLOW_STAGES.length; i++) {
      if (i < lastReached) status[i] = finalizeReached(i);
      else if (i === lastReached) status[i] = lastOutcome;
      else status[i] = "pending";
    }
  } else {
    // lastReached 已完成（done/warning）：之前含自己全 done（保留标记），下一阶段 active
    for (let i = 0; i < WORKFLOW_STAGES.length; i++) {
      if (i <= lastReached) status[i] = finalizeReached(i);
      else if (i === lastReached + 1) status[i] = "active";
      else status[i] = "pending";
    }
  }

  return status;
}

/**
 * 把 trace 事件 id（多数是 tool_call id）解析回它所在的 AI **message** id。
 *
 * 进度轨点阶段 → 滚动定位需要 message id（消息流 group wrapper 的 data-message-id = group.id
 * = message.id，见 message-list.tsx / utils.ts getMessageGroups）。TraceEvent.id 对 paradigm /
 * clarification / artifact / tool 节点是 tool_call id，对 gate 是 `${messageId}-gate`——都不直接
 * 是 message id。这里用 messages 反查，避免改动 spec#2 的 TraceEvent 结构（仍同源）。
 */
function messageIdOfEvent(event: TraceEvent, index: Map<string, string>): string | undefined {
  // gate 节点 id 形如 `${messageId}-gate` 或 `${messageIndex}-gate`：
  // 若是 messageId 形态，直接从 index 兜底；否则按 tool_call id 反查。
  const id = event.id;
  if (index.has(id)) return index.get(id);
  // gate id `${m}-gate`：尝试还原（仅当 m 是真实 message id 时命中 index）
  if (id.endsWith("-gate")) {
    const stripped = id.slice(0, -"-gate".length);
    if (index.has(stripped)) return index.get(stripped);
  }
  return undefined;
}

/** 建立 tool_call id / message id → 所在 message id 的索引（取首个命中）。 */
function buildMessageIndex(messages: Message[]): Map<string, string> {
  const index = new Map<string, string>();
  for (const message of messages) {
    if (!message.id) continue;
    index.set(message.id, message.id);
    const toolCalls =
      (message as { tool_calls?: { id?: string }[] }).tool_calls ?? [];
    for (const tc of toolCalls) {
      if (tc.id) index.set(tc.id, message.id);
    }
  }
  return index;
}

/**
 * 核心：从 TraceEvent[] + messages 派生 7 阶段状态（spec §3.1）。
 *
 * @param events  来自 useRunTrace（spec#2 单一解析）的聚合事件
 * @param messages  thread.messages——仅用于 upload/列对齐已答 两条轻量信号 + 锚点反查
 */
export function deriveWorkflowStages(events: TraceEvent[], messages: Message[]): StageState[] {
  const raw = rawOutcomes(events, messages);
  const statuses = linearize(raw);
  const messageIndex = buildMessageIndex(messages);

  return WORKFLOW_STAGES.map((stage, index) => {
    const anchor = anchorMessageIdFor(stage.id, events, messages, messageIndex);
    return {
      id: stage.id,
      status: statuses[index]!,
      ...(anchor ? { anchorMessageId: anchor } : {}),
    };
  });
}

/** 每阶段首个相关事件的所在 message id（供点击滚动定位，spec §3.3）。 */
function anchorMessageIdFor(
  stageId: (typeof WORKFLOW_STAGES)[number]["id"],
  events: TraceEvent[],
  messages: Message[],
  messageIndex: Map<string, string>,
): string | undefined {
  switch (stageId) {
    case "upload":
      return threadHasUploadedFiles(messages).messageId;
    case "paradigm":
      return anchorFromEvent(events, (e) => e.kind === "paradigm", messageIndex);
    case "align": {
      const clarify = answeredClarificationAfter(messages);
      return clarify.isColumnAlignment ? clarify.messageId : undefined;
    }
    case "compute":
      return anchorFromEvent(events, isComputeSignal, messageIndex);
    case "qc":
      return anchorFromEvent(events, (e) => e.kind === "gate", messageIndex);
    case "interpret":
      return anchorFromEvent(events, isDataAnalystDispatch, messageIndex);
    case "report":
      return anchorFromEvent(events, isReportSignal, messageIndex);
    default:
      return undefined;
  }
}
