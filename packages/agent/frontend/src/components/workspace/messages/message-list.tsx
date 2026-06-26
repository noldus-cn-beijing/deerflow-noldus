import type { BaseStream } from "@langchain/langgraph-sdk/react";
import { useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import type { StickToBottomContext } from "use-stick-to-bottom";

import {
  Conversation,
  ConversationContent,
} from "@/components/ai-elements/conversation";
import { useI18n } from "@/core/i18n/hooks";
import { answeredOptionIndex } from "@/core/messages/clarification-state";
import {
  extractContentFromMessage,
  extractQualityWarnings,
  extractTextFromMessage,
  findToolCallArgs,
  groupMessages,
  hasContent,
  hasPresentFiles,
  hasReasoning,
  hasToolCalls,
  stripClarificationOptionsFromContent,
} from "@/core/messages/utils";
import type { Subtask } from "@/core/tasks";
import { useUpdateSubtask } from "@/core/tasks/context";
import type { AgentThreadState } from "@/core/threads";
import { cn } from "@/lib/utils";

import { InlineArtifactSummary } from "../artifacts/inline-artifact-summary";
import { StreamingIndicator } from "../streaming-indicator";

import { DecisionCard } from "./decision-card";
import { MarkdownContent } from "./markdown-content";
import { MessageGroup } from "./message-group";
import { MessageListItem } from "./message-list-item";
import {
  PROGRESSIVE_MOUNT_MIN_TOTAL,
  useProgressiveMount,
} from "./progressive-mount";
import { QualityWarningBanner } from "./quality-warning-banner";
import { MessageListSkeleton } from "./skeleton";
import { SubtaskCard } from "./subtask-card";
import {
  VIRTUALIZATION_THRESHOLD,
  VirtualizedGroups,
} from "./virtualized-groups";

export const MESSAGE_LIST_DEFAULT_PADDING_BOTTOM = 160;
export const MESSAGE_LIST_FOLLOWUPS_EXTRA_PADDING_BOTTOM = 80;
// Breathing room added on top of the measured input-box height so the last
// message scrolls fully past the floating input box's top edge instead of
// sitting flush against it (spec §2.2: +1.5rem). Shared by both chat routes
// (chats/[thread_id] + agents/[agent_name]/chats/[thread_id]) so the dynamic
// paddingBottom computation can't drift between them — see catastrophic-
// forgetting self-check in the input-box-overlap spec.
export const INPUT_BOX_PADDING_BREATHING_ROOM_PX = 24;

export function MessageList({
  className,
  threadId,
  thread,
  messageRunIds,
  paddingBottom = MESSAGE_LIST_DEFAULT_PADDING_BOTTOM,
  onSelectClarificationOption,
}: {
  className?: string;
  threadId: string;
  thread: BaseStream<AgentThreadState>;
  messageRunIds?: Map<string, string>;
  paddingBottom?: number;
  /**
   * Optional callback fired when the user clicks one of the option buttons
   * rendered under an `ask_clarification` message. Receives the raw option
   * text, which the page is expected to forward as the next user message.
   */
  onSelectClarificationOption?: (optionText: string) => void;
}) {
  const { t } = useI18n();
  const updateSubtask = useUpdateSubtask();
  const messages = thread.messages;
  // Phase0#7 Step 4 — scroll context captured from the registry Conversation
  // (StickToBottom) so the virtualizer can read its scroll element. Declared
  // before any early return to satisfy the Rules of Hooks.
  const scrollContextRef = useRef<StickToBottomContext | null>(null);
  // Phase0#7 Step 1 — streaming throttle (render-layer only, zero touch to
  // hooks.ts SSE/merge core):
  //   1) `useDeferredValue` lets React interrupt/merge the high-frequency
  //      streaming re-renders so the main thread stays free for input/scroll
  //      (<100ms input latency). It only lowers re-render priority; the
  //      in-flight message still streams visibly.
  //   2) `useMemo` caches the O(n) `groupMessages(messages, mapper)` pass so
  //      it re-runs only when the deferred messages change — not on every
  //      unrelated render (e.g. `thread.isLoading` toggles, parent state).
  // `groupMessages` logic is unchanged (only wrapped). See spec §1.5/§3.2.
  const deferredMessages = useDeferredValue(messages);
  const isLoading = thread.isLoading;
  const artifacts = thread.values.artifacts;
  // 对话流图廊 refetch 信号（spec 2026-06-26-conversation-gallery-empty §一触发时机）：
  // run 完成（isLoading true→false）后递增，传给 InlineArtifactSummary 触发其重拉磁盘端点
  // 补全量图（chart-maker 陆续落盘，首轮可能不全）。present-files 出现即首次拉，这里只补 run 完。
  const wasLoadingRef = useRef(false);
  const [artifactsRefetchSignal, setArtifactsRefetchSignal] = useState(0);
  useEffect(() => {
    if (wasLoadingRef.current && !isLoading) {
      setArtifactsRefetchSignal((n) => n + 1);
    }
    wasLoadingRef.current = isLoading;
  }, [isLoading]);
  const chartsStatus = thread.values.charts_status;
  // 对话流是否已有 present-files 组（lead 显式 present 了产物 → InlineArtifactSummary 已就地
  // 内嵌）。若 lead 画完图只用文字汇报、从不调 present_files，则没有这种组 → 对话流零图廊入口
  // （用户「图库在哪」）。这时在对话流末尾兜底渲染一个 InlineArtifactSummary：它自己走磁盘端点
  // 取图，磁盘有图/报告就显示入口、空则返回 null（零副作用）。两处不并存（有 present 组就不兜底）。
  const hasPresentFilesGroup = useMemo(
    () => deferredMessages.some(hasPresentFiles),
    [deferredMessages],
  );
  const renderedGroups = useMemo(
    () =>
      groupMessages(deferredMessages, (group) => {
        // spec#4 进度轨锚点：每个 group 外层挂 data-message-id（= group.id = 种子
        // message.id），供点阶段节点时 querySelector + scrollIntoView 定位。虚拟化
        // (Phase0#7) 与非虚拟化两条渲染路径都透传这个属性，不影响分组/流式逻辑。
        if (group.type === "human" || group.type === "assistant") {
          return (
            <div key={group.id} data-message-id={group.id}>
              {group.messages.map((msg) => {
                return (
                  <MessageListItem
                    key={`${group.id}/${msg.id}`}
                    message={msg}
                    isLoading={isLoading}
                    threadId={threadId}
                    messageRunIds={messageRunIds}
                  />
                );
              })}
            </div>
          );
        } else if (group.type === "assistant:clarification") {
          const message = group.messages[0];
          if (message && hasContent(message)) {
            const toolCallId =
              message.type === "tool" ? message.tool_call_id : undefined;
            const toolArgs = toolCallId
              ? findToolCallArgs(toolCallId, deferredMessages)
              : undefined;
            const rawOptions = toolArgs?.options;
            // Options may arrive as a JSON-encoded string from certain LLMs
            // (see ClarificationMiddleware._format_clarification_message).
            let options: string[] | undefined;
            if (Array.isArray(rawOptions)) {
              options = rawOptions.filter(
                (opt): opt is string => typeof opt === "string",
              );
            } else if (typeof rawOptions === "string") {
              try {
                const parsed = JSON.parse(rawOptions);
                if (Array.isArray(parsed)) {
                  options = parsed.filter(
                    (opt): opt is string => typeof opt === "string",
                  );
                }
              } catch {
                options = [rawOptions];
              }
            }
            // spec#5 决策卡：clarification_type 驱动 accent 色/标题强度；context 是
            // 决策依据（「为什么问」）；answeredIndex 已答时高亮选中项（闭环反馈）。
            // 类型/context 均从 findToolCallArgs 取（message-list.tsx:90 既有 pattern）。
            const clarificationType =
              typeof toolArgs?.clarification_type === "string"
                ? toolArgs.clarification_type
                : undefined;
            const contextText =
              typeof toolArgs?.context === "string"
                ? toolArgs.context
                : undefined;
            const answeredIndex = answeredOptionIndex(
              deferredMessages,
              toolCallId,
              options,
            );
            return (
              <div key={group.id} data-message-id={group.id}>
                {onSelectClarificationOption ? (
                  <DecisionCard
                    question={stripClarificationOptionsFromContent(
                      extractContentFromMessage(message),
                      options ?? [],
                    )}
                    context={contextText}
                    clarificationType={clarificationType}
                    options={options}
                    answeredIndex={answeredIndex}
                    onSelect={onSelectClarificationOption}
                    isLoading={isLoading}
                    threadId={threadId}
                  />
                ) : (
                  // No select handler (e.g. read-only / mock) — still render the
                  // question as markdown so the pause is visible without buttons.
                  <MarkdownContent
                    content={stripClarificationOptionsFromContent(
                      extractContentFromMessage(message),
                      options ?? [],
                    )}
                    isLoading={isLoading}
                    threadId={threadId}
                  />
                )}
              </div>
            );
          }
          return null;
        } else if (group.type === "assistant:present-files") {
          // inline 代表图 + 入口（spec §3.2）：图元数据走磁盘端点 /artifacts/charts
          // （InlineArtifactSummary 内部 useChartArtifacts），不再吃 thread.values.artifacts
          // （state 冒泡恒空，chart-maker artifacts 不上行到 lead，见 spec §一）。
          // artifacts 仍作回退传下去（磁盘空/失败时至少显示 lead present 的代表图）。
          const hasPresentFileMessage = group.messages.some(hasPresentFiles);
          return (
            <div className="w-full" key={group.id} data-message-id={group.id}>
              {group.messages[0] && hasContent(group.messages[0]) && (
                <MarkdownContent
                  content={extractContentFromMessage(group.messages[0])}
                  isLoading={isLoading}
                  className="mb-4"
                  threadId={threadId}
                />
              )}
              {hasPresentFileMessage && (
                <InlineArtifactSummary
                  threadId={threadId}
                  artifacts={artifacts}
                  chartsStatus={chartsStatus}
                  refetchSignal={artifactsRefetchSignal}
                />
              )}
            </div>
          );
        } else if (group.type === "assistant:subagent") {
          const tasks = new Set<Subtask>();
          for (const message of group.messages) {
            if (message.type === "ai") {
              for (const toolCall of message.tool_calls ?? []) {
                if (toolCall.name === "task") {
                  const task: Subtask = {
                    id: toolCall.id!,
                    subagent_type: toolCall.args.subagent_type,
                    description: toolCall.args.description,
                    prompt: toolCall.args.prompt,
                    status: "in_progress",
                    messages: [],
                  };
                  updateSubtask(task);
                  tasks.add(task);
                }
              }
            } else if (message.type === "tool") {
              const taskId = message.tool_call_id;
              if (taskId) {
                const result = extractTextFromMessage(message);
                // 任何 tool_call_id 匹配的 ToolMessage 抵达都视为终态——in_progress 是 streaming 阶段才有效。
                // 后端短路返回的特殊 ToolMessage（如 GateEnforcementMiddleware 写 name="gate_enforcement"
                // 的"数据质量检查发现 critical 问题..."消息）不带 Task* 前缀，必须显式识别才不会让卡片永远卡在 in_progress。
                // 真实案例: 2026-05-26 FST 端到端 — data-analyst 第一次派遣被 gate_enforcement 短路，
                //         由于 ToolMessage 不以 "Task Succeeded/failed/timed out" 开头，
                //         以前 fallback 到 in_progress 让卡片永远不结束。
                if (result.startsWith("Task Succeeded")) {
                  // 兼容两种格式:
                  // 旧: "Task Succeeded. Result: <...>"
                  // 新 (5/22 起): "Task Succeeded.\n\n<timeline>\n## 最终结果\n<...>"
                  const newDelim = "## 最终结果\n";
                  const oldDelim = "Task Succeeded. Result:";
                  let resultText: string | undefined;
                  if (result.includes(newDelim)) {
                    resultText = result.split(newDelim)[1]?.trim();
                  } else if (result.startsWith(oldDelim)) {
                    resultText = result.split(oldDelim)[1]?.trim();
                  } else {
                    // "Task Succeeded.\n\n" without timeline 也算成功，整体当 result
                    resultText = result
                      .replace(/^Task Succeeded\.\s*/u, "")
                      .trim();
                  }
                  updateSubtask({
                    id: taskId,
                    status: "completed",
                    result: resultText,
                  });
                } else if (result.startsWith("Task failed.")) {
                  updateSubtask({
                    id: taskId,
                    status: "failed",
                    error: result.split("Task failed.")[1]?.trim(),
                  });
                } else if (result.startsWith("Task timed out")) {
                  updateSubtask({
                    id: taskId,
                    status: "failed",
                    error: result,
                  });
                } else if (result.startsWith("Task cancelled")) {
                  updateSubtask({
                    id: taskId,
                    status: "failed",
                    error: result,
                  });
                } else {
                  // 后端 middleware 短路（如 gate_enforcement）返回的 ToolMessage 不带 Task* 前缀。
                  // ToolMessage 既已抵达 thread.messages，说明 lead 收到了，此 task 已不可能再 streaming。
                  // 切到 completed 并把原文交给卡片展示（卡片可显示"被门拦截"的红字）。
                  updateSubtask({
                    id: taskId,
                    status: "completed",
                    result,
                  });
                }
              }
            }
          }
          const results: React.ReactNode[] = [];
          for (const message of group.messages.filter(
            (message) => message.type === "ai",
          )) {
            if (hasReasoning(message)) {
              results.push(
                <MessageGroup
                  key={"thinking-group-" + message.id}
                  messages={[message]}
                  isLoading={isLoading}
                />,
              );
            }
            // Render the lead's narrative content alongside the dispatched
            // subagents. Without this branch, any AIMessage that arrives with
            // both content and a `task` tool_call has its prose dropped —
            // during streaming, that means already-rendered text disappears
            // the moment the tool_call chunk lands, leaving an empty
            // container behind (the "横线 div" symptom seen in dogfood).
            if (hasContent(message)) {
              const narrative = extractContentFromMessage(message);
              if (narrative) {
                results.push(
                  <MarkdownContent
                    key={"narrative-" + message.id}
                    content={narrative}
                    isLoading={isLoading}
                    threadId={threadId}
                  />,
                );
              }
            }
            // Render quality warnings that may be packed into the same
            // AIMessage alongside task dispatches. When the lead agent
            // reports "n=1 cannot compute statistics" and dispatches the
            // next subagent in one message, the message lands in
            // assistant:subagent (because hasToolCalls → hasSubagent).
            // Without this branch, QualityWarningBanner is silently lost
            // (see §3.1 of 2026-06-04-frontend-info-architecture-fixes).
            const qw = extractQualityWarnings(
              message as unknown as Record<string, unknown>,
            );
            if (qw.length > 0) {
              results.push(
                <QualityWarningBanner
                  key={"qw-" + message.id}
                  warnings={qw}
                />,
              );
            }
            results.push(
              <div
                key="subtask-count"
                className="text-muted-foreground font-norma pt-2 text-sm"
              >
                {t.subtasks.executing(tasks.size)}
              </div>,
            );
            const taskIds = message.tool_calls
              ?.filter((toolCall) => toolCall.name === "task")
              .map((toolCall) => toolCall.id);
            for (const taskId of taskIds ?? []) {
              results.push(
                <SubtaskCard
                  key={"task-group-" + taskId}
                  taskId={taskId!}
                  threadId={threadId}
                  messageRunIds={messageRunIds}
                />,
              );
            }
          }
          return (
            <div
              key={"subtask-group-" + group.id}
              data-message-id={group.id}
              className="relative z-1 flex flex-col gap-2"
            >
              {results}
            </div>
          );
        } else if (group.type === "assistant:processing") {
          // Intermediate-step group: AIMessages that carry tool_calls (non-task,
          // non-present_files) like ask_clarification / set_viz_choice /
          // set_experiment_paradigm / prep_metric_plan / read_file. Without this
          // branch we'd fall back to <MessageGroup>, which buries content in
          // the CoT (thinking) collapse — that is the 2026-05-25 regression
          // where lead's "已收到 X 的结果..." 汇报 was invisible to the user
          // because cd512536 (5/21) packed report + ask_clarification into the
          // same AIMessage (see thread 9f77adcc-2a18-... vs 7456611e-... where
          // the final AIMessage had no tool_call and rendered as a main bubble).
          //
          // Strategy: render each AIMessage's narrative content as a main
          // bubble; keep reasoning in its own collapsible MessageGroup (so
          // users can still inspect thinking); tool_call results render via
          // their own dedicated groups elsewhere (assistant:clarification,
          // etc.).
          const results: React.ReactNode[] = [];
          for (const message of group.messages) {
            if (message.type !== "ai") {
              continue;
            }
            if (hasReasoning(message)) {
              results.push(
                <MessageGroup
                  key={"thinking-group-" + message.id}
                  messages={[message]}
                  isLoading={isLoading}
                />,
              );
            }
            if (hasContent(message)) {
              const narrative = extractContentFromMessage(message);
              if (narrative) {
                results.push(
                  <MarkdownContent
                    key={"narrative-" + message.id}
                    content={narrative}
                    isLoading={isLoading}
                    threadId={threadId}
                  />,
                );
              }
            }
            // Tool-call-only AIMessages (no reasoning, no content) still
            // carry useful signal — inspect_uploaded_file, prep_metric_plan,
            // set_experiment_paradigm etc. Without this branch they are
            // silently dropped because the handler only checks hasReasoning
            // and hasContent (§3.3 of 2026-06-04-frontend-info-architecture-fixes).
            if (
              !hasReasoning(message) &&
              !hasContent(message) &&
              hasToolCalls(message)
            ) {
              results.push(
                <MessageGroup
                  key={"tool-only-" + message.id}
                  messages={[message]}
                  isLoading={isLoading}
                />,
              );
            }
          }
          if (results.length === 0) {
            return null;
          }
          return (
            <div
              key={"processing-group-" + group.id}
              data-message-id={group.id}
              className="flex flex-col gap-2"
            >
              {results}
            </div>
          );
        }
        // All MessageGroup variants are handled above; this is unreachable.
        // Returning null keeps TypeScript's exhaustive narrowing happy.
        return null;
      }),
    // Deps: the deferred messages array identity is the primary key. The
    // remaining closed-over values that affect the rendered output are listed
    // explicitly so the cache invalidates when they change (not every render).
    // `updateSubtask`/`t` come from context hooks; `threadId`/`messageRunIds`
    // /`onSelectClarificationOption` are props. `isLoading`/`artifacts`/
    // `chartsStatus` are read off `thread` into stable locals above.
    [
      deferredMessages,
      isLoading,
      artifacts,
      chartsStatus,
      threadId,
      messageRunIds,
      updateSubtask,
      onSelectClarificationOption,
      artifactsRefetchSignal,
      t,
    ],
  );
  const shouldVirtualize = renderedGroups.length >= VIRTUALIZATION_THRESHOLD;
  // chat-render-jank-on-open Fix 2 — split the one-shot historical mount on the
  // NON-virtualized path (virtualized already window-mounts). Only when the
  // thread is not streaming and the list is heavy enough to be worth splitting.
  // The hook grows `visibleCount` from the initial slice toward the full count
  // across idle frames; see progressive-mount.ts. Called unconditionally
  // (before the loading-skeleton early return) to respect the Rules of Hooks;
  // it is a no-op when disabled.
  const progressiveEnabled =
    !shouldVirtualize && !isLoading && renderedGroups.length >= PROGRESSIVE_MOUNT_MIN_TOTAL;
  const visibleCount = useProgressiveMount({
    total: renderedGroups.length,
    enabled: progressiveEnabled,
  });
  if (thread.isThreadLoading && messages.length === 0) {
    return <MessageListSkeleton />;
  }
  const visibleGroups =
    progressiveEnabled && visibleCount < renderedGroups.length
      ? renderedGroups.slice(renderedGroups.length - visibleCount)
      : renderedGroups;
  const trailing = (
    <>
      {thread.isLoading && <StreamingIndicator className="my-4" />}
      {/* 图廊/报告入口兜底（修「图库在哪」）：lead 画完图未调 present_files 时对话流没有
          present-files 组 → 无内嵌入口。这里在流末尾补一个 InlineArtifactSummary，它自走磁盘
          端点取图，磁盘有图/报告即显示「打开画廊/下载/报告卡」，空则返回 null。仅在无 present
          组、非加载中、非新会话时渲染，避免与就地入口重复或在流式中途闪现。 */}
      {!hasPresentFilesGroup && !isLoading && messages.length > 0 && (
        <InlineArtifactSummary
          className="mx-auto w-full max-w-(--container-width-md)"
          threadId={threadId}
          artifacts={artifacts}
          chartsStatus={chartsStatus}
          refetchSignal={artifactsRefetchSignal}
        />
      )}
      <div style={{ height: `${paddingBottom}px` }} />
    </>
  );
  return (
    <Conversation
      // spec 2026-06-26-conversation-gallery-empty §四点五 ②：Conversation（ai-elements registry，
      // 禁改源）默认 overflow-y-hidden 致右侧滚动条不显示 + 内容溢出容器。消费侧传 overflow-y-auto
      // 覆盖（cn/tailwind-merge 让后者赢），零改 registry。StickToBottom 内部仍管理滚动行为。
      className={cn("flex size-full flex-col justify-center overflow-y-auto", className)}
      contextRef={scrollContextRef}
    >
      <ConversationContent className="mx-auto w-full max-w-(--container-width-md) gap-8 pt-12">
        {shouldVirtualize ? (
          <VirtualizedGroups
            groups={renderedGroups}
            scrollContext={scrollContextRef.current}
            paddingBottomPx={paddingBottom}
          >
            {trailing}
          </VirtualizedGroups>
        ) : (
          <>
            {visibleGroups}
            {trailing}
          </>
        )}
      </ConversationContent>
    </Conversation>
  );
}
