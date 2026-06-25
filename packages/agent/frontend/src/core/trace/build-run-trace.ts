/**
 * buildRunTrace —— 把「散在消息流里的若干卡片」聚合成一条统一的运行时间线。
 *
 * 见 spec `docs/superpowers/specs/2026-06-24-frontend-phase0-2-run-trace-live-spec.md` §3.1。
 *
 * 这是一个**纯函数**：`(messages, subtasks, t) => TraceEvent[]`。刻意不依赖 React，
 * 以便单测喂 mock 直接断言时序/状态（spec §四 Step 1），也方便档 B replay 复用。
 *
 * 工程纪律（spec §3.1 / §六 红线）：
 * - **只读派生**：不写 state、不碰 submit/merge/dedupe。
 * - **复用消息流同款 util**（extractQualityWarnings / findToolCallResult / stage-broadcast
 *   / convertToSteps 的子步骤语义），不另写解析——保证与消息流渲染不漂移（spec §六风险2）。
 * - **时序来源**：用 message 在 thread.messages 里的顺序 + tool_call 在 message 内的顺序
 *   拼接 order。live 不需要墙钟（那是档 B replay 的「真实间隔」）——live 就是「到了就追加」。
 */

import type { Message } from "@langchain/langgraph-sdk";

import { SUBTASK_HIDDEN_TOOL_CALL_NAMES } from "@/components/workspace/messages/message-group";
import {
  extractQualityWarnings,
  findToolCallResult,
  type QualityWarning,
} from "@/core/messages/utils";
import type { Subtask } from "@/core/tasks/types";
import {
  getStageBroadcastForBash,
  getStageBroadcastForSubagent,
} from "@/core/tools/stage-broadcast";

import {
  type RunTraceInput,
  type RunTraceTranslations,
  type TraceEvent,
  type TraceEventStatus,
} from "./types";

// 复用消息流同款 hidden 集合（spec §3.1：subagent 内部步骤直接复用 convertToSteps 语义）。
// 引 SubtaskCard 同一份清单，避免轨迹与卡片隐藏规则漂移。

/** lead 层普通工具调用的标题：bash 走 stage-broadcast，其余走 description/useTool。 */
function toolCallTitle(
  name: string,
  args: Record<string, unknown>,
  t: RunTraceTranslations,
): string {
  if (name === "bash") {
    const command = typeof args.command === "string" ? args.command : "";
    return getStageBroadcastForBash(command, t) ?? t.toolCalls.executeCommand;
  }
  if (typeof args.description === "string" && args.description) {
    return args.description;
  }
  return t.toolCalls.useTool(name);
}

/** gate warnings → 单个 gate 节点的状态。spec §3.1：critical(+blocks)=红 / warning=黄 / info=绿。 */
function gateStatusFromWarnings(warnings: QualityWarning[]): TraceEventStatus {
  if (warnings.some((w) => w.severity === "critical")) return "failed";
  if (warnings.some((w) => w.severity === "warning")) return "warning";
  return "ok";
}

/** reasoning_content 的轻量本地读取（避免再 import 一条会拉额外依赖的解析链）。 */
function readReasoning(message: Message): string | null {
  if (message.type !== "ai") return null;
  const ak = message.additional_kwargs as Record<string, unknown> | undefined;
  if (ak && typeof ak.reasoning_content === "string" && ak.reasoning_content) {
    return ak.reasoning_content;
  }
  return null;
}

/**
 * 把一个 subagent 的内部 messages[] 折叠成二级 trace 子步骤。
 * 复用 convertToSteps 的语义（SubtaskCard 同款）——reasoning + 可见 tool_call，
 * 保证轨迹与消息流的子代理时间线一致（spec §3.1 / §六风险2）。
 *
 * 子步骤状态随 subtask 整体状态：in_progress→running / failed→failed / completed→ok。
 */
function buildSubEvents(subtask: Subtask, t: RunTraceTranslations, orderBase: number): TraceEvent[] {
  const events: TraceEvent[] = [];
  const subStatus: TraceEventStatus =
    subtask.status === "failed"
      ? "failed"
      : subtask.status === "in_progress"
        ? "running"
        : "ok";

  for (const [index, message] of subtask.messages.entries()) {
    if (message.type !== "ai") continue;

    if (readReasoning(message)) {
      events.push({
        id: `${subtask.id}-reasoning-${index}`,
        kind: "tool",
        title: t.subtasks.expertWorking,
        status: subStatus,
        order: orderBase + index * 0.01,
      });
    }

    for (const toolCall of message.tool_calls ?? []) {
      if (toolCall.name === "task") continue; // 子代理不嵌套派遣
      if (SUBTASK_HIDDEN_TOOL_CALL_NAMES.has(toolCall.name)) continue;
      events.push({
        id: toolCall.id ?? `${subtask.id}-subtool-${index}-${toolCall.name}`,
        kind: "tool",
        title: toolCallTitle(toolCall.name, toolCall.args, t),
        status: subStatus,
        order: orderBase + index * 0.01 + 0.001,
        detail: { kind: "tool", args: toolCall.args },
      });
    }
  }
  return events;
}

/** subtask status → dispatch 节点状态。 */
function dispatchStatus(subtask: Subtask | undefined): TraceEventStatus {
  if (!subtask) return "running"; // task 已调用但 SSE 还没建出 subtask → 视为运行中
  if (subtask.status === "completed") return "ok";
  if (subtask.status === "failed") return "failed";
  return "running";
}

function asStringArray(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const strs = value.filter((o): o is string => typeof o === "string");
  return strs.length > 0 ? strs : undefined;
}

/**
 * 核心聚合：遍历 thread.messages，按时序产出 TraceEvent[]。
 *
 * 时序：order = messageIndex + (callIndex+1)*0.001。同一 message 内多个 tool_call
 * 按出现顺序排；gate 节点挂在产出它的那条 message 上（与 quality_warnings 同位）。
 */
export function buildRunTrace(input: RunTraceInput, t: RunTraceTranslations): TraceEvent[] {
  const { messages, subtasks } = input;
  const events: TraceEvent[] = [];

  for (const [messageIndex, message] of messages.entries()) {
    if (message.type !== "ai") continue;

    // gate 节点：data-analyst handoff 的 quality_warnings 挂在 AIMessage additional_kwargs。
    // 复用 extractQualityWarnings（spec §3.1），不另写解析。
    const warnings = extractQualityWarnings(message as Record<string, unknown>);
    if (warnings.length > 0) {
      events.push({
        id: `${message.id ?? messageIndex}-gate`,
        kind: "gate",
        title: t.runTrace.gateTitle,
        status: gateStatusFromWarnings(warnings),
        order: messageIndex + 0.0005, // gate 略晚于同 message 的 tool_calls
        detail: { kind: "gate", warnings },
      });
    }

    const toolCalls = message.tool_calls ?? [];
    for (const [callIndex, toolCall] of toolCalls.entries()) {
      const order = messageIndex + (callIndex + 1) * 0.001;

      // 子代理派遣 → dispatch 节点（可展开挂子步骤）。匹配键：subtask.id === toolCall.id。
      if (toolCall.name === "task") {
        const id = toolCall.id ?? `${message.id}-task-${callIndex}`;
        const subtask = subtasks[id];
        const subagentType =
          typeof toolCall.args.subagent_type === "string"
            ? toolCall.args.subagent_type
            : (subtask?.subagent_type ?? "agent");
        events.push({
          id,
          kind: "dispatch",
          title: getStageBroadcastForSubagent(subagentType, t),
          status: dispatchStatus(subtask),
          order,
          subEvents: subtask ? buildSubEvents(subtask, t, order + 0.0001) : undefined,
        });
        continue;
      }

      // 范式锁定 → paradigm 节点。
      if (toolCall.name === "set_experiment_paradigm") {
        events.push({
          id: toolCall.id ?? `${message.id}-paradigm-${callIndex}`,
          kind: "paradigm",
          title:
            typeof toolCall.args.description === "string" && toolCall.args.description
              ? toolCall.args.description
              : t.toolCalls.useTool(toolCall.name),
          status: "ok",
          order,
          detail: {
            kind: "paradigm",
            paradigm: typeof toolCall.args.paradigm === "string" ? toolCall.args.paradigm : undefined,
            ev19Template:
              typeof toolCall.args.ev19_template === "string"
                ? toolCall.args.ev19_template
                : undefined,
          },
        });
        continue;
      }

      // HITL 反问 → clarification 节点（等待用户）。
      if (toolCall.name === "ask_clarification") {
        const question =
          typeof toolCall.args.question === "string" ? toolCall.args.question : undefined;
        const options = asStringArray(toolCall.args.options);
        events.push({
          id: toolCall.id ?? `${message.id}-clarify-${callIndex}`,
          kind: "clarification",
          title: t.toolCalls.stageBroadcast.askClarification,
          status: "waiting",
          order,
          detail: { kind: "clarification", question, options },
        });
        continue;
      }

      // 产物生成 → artifact 节点。
      if (toolCall.name === "present_files") {
        const filepaths = asStringArray(toolCall.args.filepaths) ?? [];
        const result = toolCall.id ? findToolCallResult(toolCall.id, messages) : undefined;
        events.push({
          id: toolCall.id ?? `${message.id}-artifact-${callIndex}`,
          kind: "artifact",
          title: t.toolCalls.presentFiles,
          status: result ? "ok" : "running",
          order,
          detail: { kind: "artifact", filepaths },
        });
        continue;
      }

      // lead 层其它工具（bash / inspect_uploaded_file / prep_metric_plan / …）→ tool 节点。
      const result = toolCall.id ? findToolCallResult(toolCall.id, messages) : undefined;
      events.push({
        id: toolCall.id ?? `${message.id}-tool-${callIndex}`,
        kind: "tool",
        title: toolCallTitle(toolCall.name, toolCall.args, t),
        status: result ? "ok" : "running",
        order,
        detail: { kind: "tool", args: toolCall.args, result },
      });
    }
  }

  // 升序排（live 即「到了就追加」，order 单调）。
  return events.sort((a, b) => a.order - b.order);
}
