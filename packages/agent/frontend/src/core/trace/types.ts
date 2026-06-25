/**
 * 运行轨迹（Run Trace）数据模型 —— 档 A live trace。
 *
 * 见 spec `docs/superpowers/specs/2026-06-24-frontend-phase0-2-run-trace-live-spec.md`。
 *
 * TraceEvent 是「本次 run 里 agent 的一个行为步骤」的聚合视角模型。它**不是新数据**：
 * 全部从已有的 thread.messages（lead 层 tool_calls / reasoning / gate / clarification /
 * present_files）+ subtasks（来自 SubtaskContext，含每个 subagent 的 messages[]）**派生**。
 *
 * 设计纪律（spec §3.1 / §六）：本模型 + buildRunTrace 是**只读派生**——不写 state、不碰
 * thread.submit / mergeMessages / groupMessages / dedupe。它只是把「散在消息流里的若干卡片」
 * 换一个聚合视角（统一的、可纵览的时间线），与消息流里现有的 SubtaskCard 并存，不替换。
 *
 * 输入被刻意设计成可扩展（spec §七为档 B replay 留口）：当前 live 源是 `(messages, subtasks)`；
 * 将来档 B 增加「从 history 端点喂历史 run」的输入源时，本模型与 buildRunTrace 不动，只换输入。
 */

import type { AIMessage, Message } from "@langchain/langgraph-sdk";

import type { Translations } from "@/core/i18n";
import type { QualityWarning } from "@/core/messages/utils";
import type { Subtask } from "@/core/tasks/types";

/**
 * 轨迹节点的行为类别。对齐 spec §二目标 2 的六类节点。
 *
 * - `paradigm`     范式锁定（set_experiment_paradigm 工具调用）
 * - `dispatch`     子代理派遣（task 工具调用；可展开挂该 subagent 的内部步骤）
 * - `tool`         lead 层普通工具调用（bash / inspect_uploaded_file / prep_metric_plan …）
 * - `gate`         质检 gate（data-analyst handoff 的 quality_warnings；绿/黄/红）
 * - `clarification` HITL 反问（ask_clarification 工具调用）
 * - `artifact`     产物生成（present_files 工具调用）
 */
export type TraceEventKind =
  | "paradigm"
  | "dispatch"
  | "tool"
  | "gate"
  | "clarification"
  | "artifact";

/**
 * 节点状态。映射 spec §3.2 的状态色（running=brand 脉动 / ok=success / warning=warning /
 * failed=danger / waiting=warning 脉动）。**色 + 图标 + 文字三件套**（color-not-only）。
 */
export type TraceEventStatus = "running" | "ok" | "warning" | "failed" | "waiting";

/**
 * 一个 trace 节点。id 全局唯一（用于 React key / 折叠态），order 是逻辑时序（升序）。
 */
export interface TraceEvent {
  id: string;
  kind: TraceEventKind;
  /** 人类可读标题，走 stage-broadcast / i18n 得到，绝不在组件里硬编码。 */
  title: string;
  status: TraceEventStatus;
  /** 逻辑时序：用 message 在流里的顺序 + subtask message 顺序拼接。升序。 */
  order: number;
  /** 展开内容（按 kind 不同）：工具 args/result、gate warnings、子步骤等。 */
  detail?: TraceEventDetail;
  /**
   * 仅 dispatch 节点有：该 subagent 的内部 tool/reasoning 子步骤（复用 convertToSteps，
   * spec §3.1）。progressive-disclosure——默认折叠，点开露出缩进二级时间线。
   */
  subEvents?: TraceEvent[];
}

/**
 * 节点展开明细的判别联合。按 kind 区分携带什么。
 */
export type TraceEventDetail =
  | { kind: "tool"; args?: Record<string, unknown>; result?: string }
  | { kind: "gate"; warnings: QualityWarning[] }
  | { kind: "clarification"; question?: string; options?: string[] }
  | { kind: "artifact"; filepaths: string[] }
  | { kind: "paradigm"; paradigm?: string; ev19Template?: string };

/**
 * buildRunTrace 的输入（spec §3.1）。
 *
 * 刻意是个普通对象而非 React state，方便：
 * 1. 单测喂 mock 直接断言（不依赖 React 渲染）；
 * 2. 档 B replay 扩展时加 history 源字段，本类型演进，buildRunTrace 不崩。
 */
export interface RunTraceInput {
  /** thread.messages —— lead 层 tool_calls / reasoning / gate / clarification / present_files。 */
  messages: Message[];
  /** SubtaskContext.tasks —— 每个 subagent 的 messages[] + status。 */
  subtasks: Record<string, Subtask>;
}

/**
 * buildRunTrace 需要的 i18n 文案（窄接口，便于单测喂 mock translations）。
 * 复用现有 t.toolCalls.stageBroadcast.* / t.subtasks.* 模式（spec §四 Step 5），
 * 外加一组本 spec 新增的 t.runTrace.* 文案。
 */
export type RunTraceTranslations = Pick<Translations, "toolCalls" | "subtasks" | "runTrace">;

export type { AIMessage };
