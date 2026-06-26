/**
 * 决策卡 / 输入框态联动用的纯函数（spec 2026-06-24-frontend-phase0-5 §3.4）。
 *
 * 设计纪律：
 * - **不改后端**：`ask_clarification` 走 `ClarificationMiddleware` 的 `Command(goto=END)` 中断，
 *   用户回答以**后续 human message** 形式抵达（与 spec#4 `derive-workflow-stages` 的
 *   `answeredClarificationAfter` 同款判定，见 memory `feedback_seal_missing_root_cause…`
 *   族「同源」原则）。这里不重复 trace 的工具语义解析——只对同份 `messages` 做轻量谓词。
 * - **纯函数**：刻意不依赖 React，单测喂 mock 直接断言（spec §四 TDD）。
 * - 与 spec#4 的列对齐判定**职责不同**：spec#4 的 `isColumnAlignmentClarification` +
 *   `answeredClarificationAfter` 只盯「列对齐」这一类反问（驱动进度轨 ③ 阶段状态）；
 *   本函数面向**任意类型**的 ask_clarification（决策卡 / 输入框态关心的是「agent 在等
 *   任何决策」，不局限于列对齐）。两处算法独立，不互相漂移。
 */

import type { Message } from "@langchain/langgraph-sdk";

/** ask_clarification 的 clarification_type 枚举（镜像后端 Literal，纯前端视图）。 */
export type ClarificationType =
  | "missing_info"
  | "ambiguous_requirement"
  | "approach_choice"
  | "risk_confirmation"
  | "suggestion";

/** 从一条 message 的 tool_calls 里找出 ask_clarification 调用的 id（若有）。 */
function askClarificationToolCallId(message: Message): string | undefined {
  if (message.type !== "ai") return undefined;
  const tc = (message.tool_calls ?? []).find((c) => c.name === "ask_clarification");
  return tc?.id;
}

/**
 * 「这条 ask_clarification 是否已被用户回答」(spec §3.1 已答闭环反馈 + §3.4)。
 *
 * 判定：以该 AI message（含 ask_clarification tool_call）在消息流中的位置为锚，
 * 其后是否存在 human message。 ClarificationMiddleware 用 `Command(goto=END)` 中断，
 * 用户回答 = 后续 human message（而非 ToolMessage 结果）。
 *
 * @param messages     完整消息流。
 * @param toolCallId   ask_clarification 的 tool_call id（即决策卡对应 ToolMessage 的 tool_call_id）。
 */
export function isClarificationAnswered(
  messages: Message[],
  toolCallId: string | undefined,
): boolean {
  if (!toolCallId) return false;
  // 定位发起这次 ask_clarification 的 AI message 索引。
  const anchorIdx = messages.findIndex((m) => askClarificationToolCallId(m) === toolCallId);
  if (anchorIdx < 0) return false;
  for (let i = anchorIdx + 1; i < messages.length; i++) {
    if (messages[i]!.type === "human") return true;
  }
  return false;
}

/**
 * 「整条消息流是否止于一个未答的 ask_clarification」(spec §3.4 输入框态联动)。
 *
 * 用于输入框 placeholder：流非 loading 且最后一条 AI 消息发了 ask_clarification、
 * 且其后没有任何 human message（即仍在等待用户决策）。
 *
 * 取**最后一条** AI 消息上的 ask_clarification（多次反问时以最近一次为准——
 * 输入框态反映「当前焦点」，与 spec#4 的「最近一次列对齐」同款取法）。
 */
export function lastClarificationIsAwaiting(messages: Message[]): boolean {
  // 从尾向前找最后一条 AI 消息。
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i]!;
    if (m.type !== "ai") continue;
    const tcId = askClarificationToolCallId(m);
    if (!tcId) return false; // 最后一条 AI 消息没有 ask_clarification → 不在等待
    // 其后是否存在 human message。
    for (let j = i + 1; j < messages.length; j++) {
      if (messages[j]!.type === "human") return false;
    }
    return true;
  }
  return false;
}

/**
 * 把后端 `clarification_type` 原值规整成前端枚举（容错未知/缺失值）。
 *
 * 判不准时返回 undefined → 决策卡按默认（warning 等待调）渲染，不影响「显眼」主目标
 * （spec §六风险2：type 粒度不足时统一 warning 调兜底）。
 */
export function normalizeClarificationType(
  raw: unknown,
): ClarificationType | undefined {
  if (typeof raw !== "string") return undefined;
  const allowed: ClarificationType[] = [
    "missing_info",
    "ambiguous_requirement",
    "approach_choice",
    "risk_confirmation",
    "suggestion",
  ];
  return (allowed as string[]).includes(raw) ? (raw as ClarificationType) : undefined;
}

/**
 * 找出用户「点了哪个选项」（spec §3.1 已答闭环高亮）。
 *
 * 用户回答以 human message 形式抵达（点选项 = 发该选项文本，与 typing 等价）。
 * 取该 clarification 之后的第一条 human message，与 options 逐条精确比对（trim）：
 *  - 命中 → 返回该选项下标（高亮选中项）；
 *  - 未命中（用户自定义输入）→ 返回 null（只显「已确认」标题，不高亮任何选项）。
 *
 * @param messages     完整消息流。
 * @param toolCallId   ask_clarification 的 tool_call id。
 * @param options      选项列表。
 */
export function answeredOptionIndex(
  messages: Message[],
  toolCallId: string | undefined,
  options: readonly string[] | undefined,
): number | null {
  if (!toolCallId || !options || options.length === 0) return null;
  if (!isClarificationAnswered(messages, toolCallId)) return null;
  const anchorIdx = messages.findIndex(
    (m) => askClarificationToolCallId(m) === toolCallId,
  );
  if (anchorIdx < 0) return null;
  const reply = (() => {
    for (let i = anchorIdx + 1; i < messages.length; i++) {
      if (messages[i]!.type === "human") {
        return typeof messages[i]!.content === "string"
          ? (messages[i]!.content as string).trim()
          : "";
      }
    }
    return "";
  })();
  if (!reply) return null;
  const idx = options.findIndex((opt) => opt.trim() === reply);
  return idx >= 0 ? idx : null;
}
