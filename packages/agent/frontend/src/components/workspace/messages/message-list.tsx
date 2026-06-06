import type { BaseStream } from "@langchain/langgraph-sdk/react";

import {
  Conversation,
  ConversationContent,
} from "@/components/ai-elements/conversation";
import { useI18n } from "@/core/i18n/hooks";
import {
  extractContentFromMessage,
  extractPresentFilesFromMessage,
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

import { ArtifactFileList } from "../artifacts/artifact-file-list";
import { StreamingIndicator } from "../streaming-indicator";

import { ClarificationOptions } from "./clarification-options";
import { MarkdownContent } from "./markdown-content";
import { MessageGroup } from "./message-group";
import { MessageListItem } from "./message-list-item";
import { QualityWarningBanner } from "./quality-warning-banner";
import { MessageListSkeleton } from "./skeleton";
import { SubtaskCard } from "./subtask-card";

export const MESSAGE_LIST_DEFAULT_PADDING_BOTTOM = 160;
export const MESSAGE_LIST_FOLLOWUPS_EXTRA_PADDING_BOTTOM = 80;

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
  if (thread.isThreadLoading && messages.length === 0) {
    return <MessageListSkeleton />;
  }
  return (
    <Conversation
      className={cn("flex size-full flex-col justify-center", className)}
    >
      <ConversationContent className="mx-auto w-full max-w-(--container-width-md) gap-8 pt-12">
        {groupMessages(
          messages,
          (group) => {
          if (group.type === "human" || group.type === "assistant") {
            return group.messages.map((msg) => {
              return (
                <MessageListItem
                  key={`${group.id}/${msg.id}`}
                  message={msg}
                  isLoading={thread.isLoading}
                  threadId={threadId}
                  messageRunIds={messageRunIds}
                />
              );
            });
          } else if (group.type === "assistant:clarification") {
            const message = group.messages[0];
            if (message && hasContent(message)) {
              const toolCallId =
                message.type === "tool" ? message.tool_call_id : undefined;
              const toolArgs = toolCallId
                ? findToolCallArgs(toolCallId, messages)
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
              return (
                <div key={group.id}>
                  <MarkdownContent
                    content={stripClarificationOptionsFromContent(
                      extractContentFromMessage(message),
                      options ?? [],
                    )}
                    isLoading={thread.isLoading}
                    threadId={threadId}
                  />
                  {onSelectClarificationOption && (
                    <ClarificationOptions
                      options={options}
                      onSelect={onSelectClarificationOption}
                      disabled={thread.isLoading}
                    />
                  )}
                </div>
              );
            }
            return null;
          } else if (group.type === "assistant:present-files") {
            const files: string[] = [];
            for (const message of group.messages) {
              if (hasPresentFiles(message)) {
                const presentFiles = extractPresentFilesFromMessage(message);
                files.push(...presentFiles);
              }
            }
            return (
              <div className="w-full" key={group.id}>
                {group.messages[0] && hasContent(group.messages[0]) && (
                  <MarkdownContent
                    content={extractContentFromMessage(group.messages[0])}
                    isLoading={thread.isLoading}
                    className="mb-4"
                    threadId={threadId}
                  />
                )}
                <ArtifactFileList files={files} threadId={threadId} />
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
                    isLoading={thread.isLoading}
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
                      isLoading={thread.isLoading}
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
                    isLoading={thread.isLoading}
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
                      isLoading={thread.isLoading}
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
                    isLoading={thread.isLoading}
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
                className="flex flex-col gap-2"
              >
                {results}
              </div>
            );
          }
          // All MessageGroup variants are handled above; this is unreachable.
          // Returning null keeps TypeScript's exhaustive narrowing happy.
          return null;
        },
        )}
        {thread.isLoading && <StreamingIndicator className="my-4" />}
        <div style={{ height: `${paddingBottom}px` }} />
      </ConversationContent>
    </Conversation>
  );
}
